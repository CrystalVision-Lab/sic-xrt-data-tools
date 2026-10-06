"""Source-bound actual detection review. Unmatched is never a negative label."""
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from .source_registry import digest_file


def read_source(spec):
    if 'archive' in spec:
        with zipfile.ZipFile(spec['archive']) as archive:
            payload = archive.read(spec['member'])
        if hashlib.sha256(payload).hexdigest() != spec['sha256']:
            raise ValueError('Source member hash mismatch')
        image = tifffile.imread(io.BytesIO(payload))
    else:
        path = Path(spec['path'])
        if digest_file(path) != spec['sha256']:
            raise ValueError('Source image hash mismatch')
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None  # Hash-bound local research image.
            with Image.open(path) as handle:
                image = np.asarray(handle.convert('RGB')).copy()
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected raw RGB uint8')
    return image


def build(queue_path, sources, output):
    output = Path(output)
    if output.exists():
        raise ValueError('New output required')
    with Path(queue_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [(r['wafer'], r['candidate_id']) for r in rows]
    if len(set(ids)) != len(ids) or not rows or any(r['wafer'] not in ('1', '2', '9') for r in rows):
        raise ValueError('Unique development candidates required; wafer8 forbidden')
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    (output/'contexts').mkdir()
    result = []
    for wafer in sorted({r['wafer'] for r in rows}):
        spec = sources[wafer]
        image = read_source(spec)
        h, w = image.shape[:2]
        selected = [r for r in rows if r['wafer'] == wafer]
        board = None
        for index, row in enumerate(selected):
            if row['source_image_sha256'] != spec['sha256']:
                raise ValueError('Coordinate/source identity mismatch')
            raw_x, raw_y = float(row['x']), float(row['y'])
            if not math.isfinite(raw_x) or not math.isfinite(raw_y):
                raise ValueError('Nonfinite coordinates')
            x, y = round(raw_x), round(raw_y)
            if not (64 <= x <= w-64 and 64 <= y <= h-64):
                raise ValueError('Insufficient training patch; do not pad or move')
            ident = f'w{wafer}_{index+1:03d}'
            patch = image[y-64:y+64, x-64:x+64]
            context = np.zeros((256, 256, 3), dtype=np.uint8)
            left, top, right, bottom = max(0, x-128), max(0, y-128), min(w, x+128), min(h, y+128)
            valid = [left-x+128, top-y+128, right-x+128, bottom-y+128]
            context[valid[1]:valid[3], valid[0]:valid[2]] = image[top:bottom, left:right]
            patch_path, context_path = output/'patches'/f'{ident}.tif', output/'contexts'/f'{ident}.tif'
            tifffile.imwrite(patch_path, patch, photometric='rgb')
            tifffile.imwrite(context_path, context, photometric='rgb')
            if index % 20 == 0:
                board = Image.new('RGB', (5*264, 4*284), '#17202b')
            ox, oy = index % 5*264, index % 20//5*284
            board.paste(Image.fromarray(context), (ox, oy))
            draw = ImageDraw.Draw(board)
            # Brackets identify the 128px training field without hiding its center.
            for cx, cy, dx, dy in [(64,64,1,1),(191,64,-1,1),(64,191,1,-1),(191,191,-1,-1)]:
                draw.line((ox+cx,oy+cy,ox+cx+dx*8,oy+cy), fill='#ffde59')
                draw.line((ox+cx,oy+cy,ox+cx,oy+cy+dy*8), fill='#ffde59')
            clipped = valid != [0, 0, 256, 256]
            draw.text((ox+4, oy+259), f'{ident} center=128,128'+(' EDGE' if clipped else ''), fill='white')
            if index % 20 == 19 or index == len(selected)-1:
                board.save(output/f'wafer{wafer}_sheet{index//20+1}.png')
            result.append({'id': ident, 'wafer': wafer, 'candidate_id': row['candidate_id'],
                           'x': raw_x, 'y': raw_y, 'coordinate_space': 'raw_pixel_xy',
                           'source_sha256': spec['sha256'], 'path': str(patch_path.relative_to(output)),
                           'sha256': digest_file(patch_path), 'context_path': str(context_path.relative_to(output)),
                           'context_sha256': digest_file(context_path), 'review_status': 'pending',
                           'context_valid_xyxy': valid, 'context_clipped': clipped,
                           'human_verified': False, 'expert_ground_truth': False, 'training_eligible': False})
        del image
    manifest = {'schema': 'xrt_detection_review_v1', 'queue_sha256': digest_file(queue_path),
                'sources': sources, 'candidates': result, 'test_evaluated': False,
                'selection_bias': 'highest existing type score and >40px from supplied points; not random population'}
    (output/'candidates.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def finalize(output, observations):
    output = Path(output).resolve()
    manifest = json.loads((output/'candidates.json').read_text(encoding='utf-8'))
    mapping = {r['id']: r for r in observations}
    if len(mapping) != len(observations) or set(mapping) != {r['id'] for r in manifest['candidates']}:
        raise ValueError('Exactly one observation per candidate required')
    decisions = []
    for row in manifest['candidates']:
        obs = mapping[row['id']]
        if (set(obs) != {'id', 'decision', 'note'} or obs['decision'] not in
                ('weak_background', 'possible_defect', 'uncertain') or not obs['note'].strip()):
            raise ValueError('Explicit limited observation required')
        for path_key, hash_key in [('path', 'sha256'), ('context_path', 'context_sha256')]:
            path = (output/row[path_key]).resolve()
            if not path.is_relative_to(output) or digest_file(path) != row[hash_key]:
                raise ValueError('Reviewed image changed')
        decisions.append({**row, **obs, 'review_status': 'ai_observed', 'review_actor': 'Codex_AI',
                          'human_verified': False, 'expert_ground_truth': False,
                          'training_eligible': obs['decision'] == 'weak_background',
                          'label_basis': 'weak_visual_background_not_expert_truth'})
    result = {**manifest, 'schema': 'xrt_detection_observations_v1', 'candidates': decisions}
    with (output/'observations.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result
