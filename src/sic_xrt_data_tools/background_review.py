"""Hash-bound development-only background review; no automatic truth labels."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, gaussian_filter, label
from scipy.spatial import cKDTree

from .annotation_workspace import SourceReader
from .development_cohort import develop_points
from .source_registry import digest_file


def centers(width, height, points, count=48, seed=27):
    if width < 128 or height < 128 or count < 1:
        raise ValueError('Image too small or invalid count')
    rng = np.random.default_rng(seed)
    tree = cKDTree(np.asarray(points).reshape(-1, 2)) if len(points) else None
    selected = []
    for _ in range(count * 1000):
        x, y = int(rng.integers(64, width-63)), int(rng.integers(64, height-63))
        if tree is not None and tree.query([x, y])[0] < 128:
            continue
        if any((x-a)**2+(y-b)**2 < 192**2 for a, b in selected):
            continue
        selected.append((x, y))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Not enough separated candidates; do not relax silently')
    return selected


def line_proposal(patch):
    gray = patch.mean(axis=2, dtype=np.float32) / 255
    response = gaussian_filter(gray, 5) - gaussian_filter(gray, 1)
    cutoff = max(.02, float(np.median(np.abs(response - np.median(response)))) * 6)
    components, count = label(binary_closing(response > cutoff, iterations=2))
    candidates = []
    for index in range(1, count+1):
        yy, xx = np.nonzero(components == index)
        if len(xx) < 15:
            continue
        distance = np.hypot(xx - patch.shape[1]/2, yy - patch.shape[0]/2).min()
        if distance > 40:
            continue
        eigen = np.linalg.eigvalsh(np.cov(np.array([xx, yy])))
        if eigen[-1] < 30 or eigen[-1] / max(eigen[0], 1) < 3:
            continue
        candidates.append((float(distance), [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]))
    return {'status': 'automatic_proposal_not_ground_truth' if candidates else 'unresolved',
            'bbox_patch_xyxy': min(candidates)[1] if candidates else None,
            'training_eligible': False, 'physical_scale_known': False}


def build(source_root, registry, workspace, cohort, correction_path, output):
    source_root, registry, workspace, cohort, correction_path, output = map(Path, (source_root, registry, workspace, cohort, correction_path, output))
    if output.exists():
        raise ValueError('New output directory required')
    info = json.loads((cohort/'cohort.json').read_text(encoding='utf-8'))
    if digest_file(cohort/'samples.csv') != info['manifest_sha256']:
        raise ValueError('Cohort hash mismatch')
    rows = list(csv.DictReader((cohort/'samples.csv').open(encoding='utf-8-sig')))
    if any(r['wafer'] not in ('1', '2', '9') for r in rows):
        raise ValueError('Only development wafers 1/2/9 allowed')
    repair = json.loads(correction_path.read_text(encoding='utf-8'))
    if repair['evidence']['status'] != 'passed':
        raise ValueError('Coordinate evidence required')
    checks = json.loads((workspace/'output_hashes.json').read_text(encoding='utf-8'))
    if digest_file(workspace/'provider_points.jsonl') != checks['provider_points.jsonl']:
        raise ValueError('Provider point hash mismatch')
    points = develop_points([json.loads(s) for s in (workspace/'provider_points.jsonl').read_text(encoding='utf-8').splitlines()], repair['all_provider_corrections'])
    assets = [json.loads(s) for s in (registry/'assets.jsonl').read_text(encoding='utf-8').splitlines()]
    reader = SourceReader(source_root, assets)
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    (output/'bpd_context').mkdir()
    candidates, geometry = [], []
    selected_sources = {wafer: Counter(r['source_id'] for r in rows if r['wafer'] == wafer).most_common(1)[0][0] for wafer in ('1', '2', '9')}
    for wafer, source_id in selected_sources.items():
        asset = reader.by_id[source_id]
        image = reader.image(asset)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected original uint8 RGB')
        h, w = image.shape[:2]
        xy = [(p['x'], p['y']) for p in points if p['image_asset_id'] == source_id]
        board = Image.new('RGB', (6*144, 8*156), '#17202b')
        draw = ImageDraw.Draw(board)
        for i, (x, y) in enumerate(centers(w, h, xy, seed=27+int(wafer))):
            ident = f'background_w{wafer}_{i+1:02d}'
            path = output/'patches'/f'{ident}.tif'
            patch = image[y-64:y+64, x-64:x+64]
            tifffile.imwrite(path, patch, photometric='rgb')
            ox, oy = (i % 6)*144+8, (i//6)*156
            board.paste(Image.fromarray(patch), (ox, oy))
            draw.text((ox, oy+132), f'{i+1:02d} W{wafer}', fill='white')
            candidates.append({'id': ident, 'wafer': wafer, 'source_id': source_id, 'source_sha256': asset['sha256'],
                               'coordinate_space': 'raw_pixel_xy', 'x': x, 'y': y, 'path': f'patches/{ident}.tif',
                               'sha256': digest_file(path), 'status': 'pending_visual_review', 'human_verified': False})
        board.save(output/f'background_wafer{wafer}.png')
        for row in rows:
            if row['source_id'] != source_id or row['label'] != 'BPD':
                continue
            x, y = round(float(row['x'])), round(float(row['y']))
            if not (128 <= x <= w-128 and 128 <= y <= h-128):
                continue
            patch = image[y-128:y+128, x-128:x+128]
            path = output/'bpd_context'/f'{row["point_id"]}.tif'
            tifffile.imwrite(path, patch, photometric='rgb')
            geometry.append({'point_id': row['point_id'], 'wafer': wafer, 'source_id': source_id,
                             'source_sha256': asset['sha256'], 'path': str(path.relative_to(output)), 'sha256': digest_file(path),
                             'crop_origin_xy': [x-128, y-128], **line_proposal(patch)})
        del image
        print(f'wafer {wafer}: background candidates 48', flush=True)
    reader.verify_unchanged()
    manifest = {'schema': 'xrt_background_review_v1', 'source_root': str(source_root), 'cohort_root': str(cohort),
                'cohort_manifest_sha256': info['manifest_sha256'], 'coordinate_repair_sha256': digest_file(correction_path),
                'selected_sources': selected_sources, 'candidates': candidates, 'bpd_geometry_proposals': geometry,
                'expert_ground_truth': False, 'test_evaluated': False}
    (output/'candidates.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def finalize(output, observations):
    output = Path(output)
    data = json.loads((output/'candidates.json').read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in observations}
    if len(by_id) != len(observations) or set(by_id) != {r['id'] for r in data['candidates']}:
        raise ValueError('One explicit observation required per candidate')
    weak = []
    for row in data['candidates']:
        observation = by_id[row['id']]
        if set(observation) != {'id', 'decision', 'note'}:
            raise ValueError('Observations may not override provenance or eligibility')
        if observation['decision'] not in ('background_candidate', 'uncertain', 'artifact_candidate') or not observation.get('note'):
            raise ValueError('Explicit decision and observation required')
        path = (output/row['path']).resolve()
        if not path.is_relative_to(output.resolve()) or digest_file(path) != row['sha256']:
            raise ValueError('Reviewed patch changed')
        weak.append({**row, 'review_actor': 'Codex_AI', 'human_verified': False, 'expert_ground_truth': False,
                     'weak_training_eligible': observation['decision'] == 'background_candidate',
                     'status': 'ai_observed_not_expert_verified', **observation})
    manifest = {**data, 'schema': 'xrt_weak_background_v1', 'candidates': weak,
                'observation_scope': 'visible morphology only; no absence-of-defect guarantee'}
    with (output/'weak_background.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
