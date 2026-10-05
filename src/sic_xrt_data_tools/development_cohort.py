"""Provider-labelled development cohort; never imports reserved wafer 8 images."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from .annotation_workspace import SourceReader
from .coordinate_repair import mirrored_xy
from .provider_quality import POLICY, coordinate_flags, point_metrics
from .source_registry import digest_file

WAFERS = ('1', '2', '9')
CLASSES = ('BPD', 'TED', 'TSD')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def neighbor_policy(point, neighbors):
    """Only other types within the central 32-pixel square are target conflicts."""
    left, top = round(point['x'])-64, round(point['y'])-64
    different = [q for q in neighbors if q['point_id'] != point['point_id']
                 and q['base_label'] != point['base_label']
                 and left <= q['x'] < left+128 and top <= q['y'] < top+128]
    conflict = any(left+48 <= q['x'] < left+80 and top+48 <= q['y'] < top+80 for q in different)
    return conflict, len(different)


def develop_points(points, corrections):
    selected = []
    for original in points:
        if str(original['area_id']) not in WAFERS:
            continue
        point = dict(original)
        change = corrections.get(point['point_id'])
        if change:
            if [point['x'], point['y']] != change['old_xy'] or point['fine_label'] != change['provider_fine_label']:
                raise ValueError('Correction/provider mismatch')
            point['x'], point['y'] = change['new_xy']
        point['original_xy'] = [original['x'], original['y']]
        point['coordinate_corrected'] = bool(change)
        selected.append(point)
    if len({p['point_id'] for p in selected}) != len(selected):
        raise ValueError('Duplicate provider points')
    return selected


def build(source_root, registry, workspace, parent_dataset, output):
    source_root, registry, workspace, parent_dataset, output = (
        Path(p).resolve() for p in (source_root, registry, workspace, parent_dataset, output))
    if output.exists() or any(output.is_relative_to(p) for p in (source_root, registry, workspace, parent_dataset)):
        raise ValueError('New independent output required')
    protected = json.loads((parent_dataset/'output_hashes.json').read_text(encoding='utf-8'))
    for relative, expected in protected.items():
        path = (parent_dataset/relative).resolve()
        if not path.is_relative_to(parent_dataset) or digest_file(path) != expected:
            raise ValueError('Parent dataset hash mismatch')
    correction = json.loads((parent_dataset/'coordinate_repair.json').read_text(encoding='utf-8'))
    if correction['status'] != 'completed' or correction['evidence']['status'] != 'passed':
        raise ValueError('Verified coordinate evidence required')
    history = json.loads((parent_dataset/'기록/ai_review_history.json').read_text(encoding='utf-8'))
    workspace_hashes = json.loads((workspace/'output_hashes.json').read_text(encoding='utf-8'))
    if digest_file(workspace/'provider_points.jsonl') != workspace_hashes['provider_points.jsonl']:
        raise ValueError('Provider manifest hash mismatch')
    original = list(map(json.loads, (workspace/'provider_points.jsonl').read_text(encoding='utf-8').splitlines()))
    points = develop_points(original, correction['all_provider_corrections'])
    assets = list(map(json.loads, (registry/'assets.jsonl').read_text(encoding='utf-8').splitlines()))
    reader = SourceReader(source_root, assets)
    groups = defaultdict(list)
    for p in points:
        groups[p['image_asset_id']].append(p)
    conflicts = coordinate_flags(points)
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    rows, excluded, sources, roi_cache = [], [], [], {}
    try:
        for source_id, items in sorted(groups.items()):
            asset = reader.by_id[source_id]
            image = reader.image(asset)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('Expected raw uint8 RGB')
            if source_id == correction['source_id']:
                if asset['sha256'] != correction['evidence']['target_file_sha256']:
                    raise ValueError('Correction image hash mismatch')
                digest = hashlib.sha256()
                for start in range(0, len(image), 64):
                    digest.update(np.ascontiguousarray(image[start:start+64]).tobytes())
                if digest.hexdigest() != correction['evidence']['target_decoded_sha256']:
                    raise ValueError('Decoded correction image mismatch')
            tree = cKDTree([(p['x'], p['y']) for p in items])
            for p in items:
                for ref in p['roi_refs']:
                    aid = ref['roi_asset_id']
                    if aid not in roi_cache:
                        roi_cache[aid] = reader.points(reader.by_id[aid])
                    if not np.allclose(roi_cache[aid][ref['point_index']], p['original_xy'], rtol=0, atol=1e-5):
                        raise ValueError('Raw ROI differs from provider coordinates')
                if p['coordinate_corrected'] and (
                        source_id != correction['source_id'] or list(mirrored_xy(
                            *p['original_xy'], image.shape[1], image.shape[0])) != [p['x'], p['y']]):
                    raise ValueError('Unverified coordinate change')
                quality = point_metrics(image, p['x'], p['y'])
                reasons = list(quality['flags']) + list(conflicts[p['point_id']])
                if p['base_label'] not in CLASSES:
                    reasons.append('unknown_provider_type')
                reviews = history.get(p['point_id'], [])
                last = reviews[-1] if reviews else None
                if last and (last['status'] == 'hold' or last['proposed_base_label'] != p['base_label']):
                    reasons.append('prior_AI_hold_or_disagreement')
                nearby = [items[j] for j in tree.query_ball_point((p['x'], p['y']), 66, p=np.inf)]
                conflict, neighbor_count = neighbor_policy(p, nearby)
                if conflict:
                    reasons.append('different_type_in_center32')
                if reasons:
                    excluded.append({'point_id': p['point_id'], 'wafer': str(p['area_id']),
                                     'label': p['base_label'], 'reasons': sorted(set(reasons))})
                    continue
                x, y = round(p['x']), round(p['y'])
                patch = image[y-64:y+64, x-64:x+64].copy()
                if patch.shape != (128, 128, 3):
                    raise ValueError('Patch outside source')
                relative = f'patches/{p["point_id"]}_128.tif'
                path = (output/relative).resolve()
                if not path.is_relative_to(output/'patches'):
                    raise ValueError('Invalid point path')
                tifffile.imwrite(path, patch, photometric='rgb', metadata=None)
                rows.append({'patch_id': p['point_id']+'_128', 'point_id': p['point_id'],
                    'source_id': source_id, 'wafer': str(p['area_id']), 'split': 'development',
                    'label': p['base_label'], 'provider_fine_label': p['fine_label'], 'phase': p['phase'],
                    'x': p['x'], 'y': p['y'], 'left': x-64, 'top': y-64, 'size': 128, 'dtype': 'uint8',
                    'path': relative, 'sha256': digest_file(path), 'context_other_type_count': neighbor_count,
                    'coordinate_corrected': p['coordinate_corrected'], 'human_verified': False,
                    'label_basis': 'supplied_roi_filename_research_only'})
            sources.append({'source_id': source_id, 'wafer': str(items[0]['area_id']),
                            'sha256': asset['sha256'], 'locator': asset['locator']})
            print(f'{source_id}: cumulative selected={len(rows)} excluded={len(excluded)}', flush=True)
        reader.verify_unchanged()
        rows.sort(key=lambda r: r['patch_id'])
        if {r['wafer'] for r in rows} != set(WAFERS) or not rows:
            raise ValueError('Missing development wafer')
        with (output/'samples.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output/'excluded.json', excluded)
        write_json(output/'sources.json', sources)
        contract = {'schema': 'wafer_grouped_development_cohort', 'schema_version': 1,
                    'created_at': datetime.now(UTC).isoformat(), 'status': 'completed',
                    'manifest_sha256': digest_file(output/'samples.csv'), 'classes': list(CLASSES),
                    'allowed_wafers': list(WAFERS), 'forbidden_wafers': ['8'], 'counts': dict(Counter(
                        f'{r["wafer"]}/{r["label"]}' for r in rows)), 'samples': len(rows),
                    'quality_policy': POLICY, 'target_policy': 'center32_clear_of_other_type_ROI;context128_may_contain_neighbors',
                    'parent_manifest_sha256': digest_file(parent_dataset/'기록/samples.csv'),
                    'coordinate_evidence_sha256': digest_file(parent_dataset/'coordinate_repair.json'),
                    'provider_points_sha256': digest_file(workspace/'provider_points.jsonl'),
                    'original_labels_changed': False, 'expert_ground_truth': False,
                    'test_evaluated': False, 'research_only': True,
                    'wafer_identity_basis': 'user confirmed Areas are different wafers',
                    'excluded_count': len(excluded)}
        write_json(output/'cohort.json', contract)
        for relative, expected in protected.items():
            if digest_file(parent_dataset/relative) != expected:
                raise ValueError('Parent dataset changed')
        write_json(output/'output_hashes.json', {p.relative_to(output).as_posix(): digest_file(p)
                   for p in output.rglob('*') if p.is_file()})
        return contract
    except BaseException as error:
        write_json(output/'BUILD_FAILED.json', {'error': type(error).__name__, 'message': str(error)})
        raise
