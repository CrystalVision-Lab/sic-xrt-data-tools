"""Explicit, evidence-backed coordinate repairs; never infer physical type labels."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .point_dataset import patch_bounds
from .source_registry import digest_file


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def exact_horizontal_mirror(reference, target):
    """Compare every decoded pixel, in bounded row blocks, before accepting a mirror."""
    if reference.shape != target.shape or reference.dtype != target.dtype:
        raise ValueError('Mirror pair shape/dtype mismatch')
    if reference.ndim not in (2, 3) or reference.shape[1] < 2:
        raise ValueError('Expected a spatial image')
    digest = hashlib.sha256()
    for start in range(0, len(reference), 64):
        block = np.ascontiguousarray(reference[start:start+64, ::-1])
        if not np.array_equal(block, target[start:start+64]):
            raise ValueError('Images are not an exact horizontal mirror')
        digest.update(block.tobytes())
    return {'method': 'every_decoded_pixel_exact_equality', 'status': 'passed',
            'shape': list(target.shape), 'dtype': str(target.dtype),
            'target_decoded_sha256': digest.hexdigest(),
            'transform': 'target_x = width - 1 - reference_x; target_y = reference_y'}


def mirrored_xy(x, y, width, height):
    if not np.isfinite([x, y]).all() or not (0 <= x <= width-1 and 0 <= y <= height-1):
        raise ValueError('Coordinate is outside pixel-center bounds')
    return width - 1 - x, y


def export_corrected_point_audit(image, source_id, corrections, evidence, output, size=128):
    """Export all supplied corrected points, excluding only out-of-image patches."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Audit output must be new')
    if (evidence.get('status') != 'passed'
            or hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            != evidence.get('target_decoded_sha256')):
        raise ValueError('Image does not match verified mirror evidence')
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    rows, excluded = [], []
    h, w = image.shape[:2]
    for point_id, item in sorted(corrections.items()):
        x, y = mirrored_xy(*item['old_xy'], w, h)
        if [x, y] != item['new_xy']:
            raise ValueError('Audit coordinate does not match mirror transform')
        bounds = patch_bounds(x, y, size, w, h)
        if bounds is None:
            excluded.append({'point_id': point_id, 'reason': 'patch_crosses_image_boundary'})
            continue
        left, top, right, bottom = bounds
        path = output/'patches'/f'{point_id}_{size}.tif'
        if not path.resolve().is_relative_to(output/'patches'):
            raise ValueError('Invalid point ID')
        tifffile.imwrite(path, image[top:bottom, left:right].copy(), photometric='rgb', metadata=None)
        rows.append({'point_id': point_id, 'patch_id': f'{point_id}_{size}', 'source_id': source_id,
                     'path': path.relative_to(output).as_posix(), 'sha256': digest_file(path),
                     'label': item['provider_fine_label'].split('_')[0],
                     'provider_fine_label': item['provider_fine_label'], 'dtype': str(image.dtype),
                     'size': str(size), 'x': x, 'y': y, 'split': 'val'})
    manifest = {'schema_version': 1, 'task': 'all_corrected_provider_points_audit',
                'source_id': source_id, 'provider_count': len(corrections), 'rows': rows, 'excluded': excluded,
                'selection': 'all supplied corrected points; boundary-only exclusion; no model score filtering',
                'evidence': evidence, 'human_verified': False, 'research_only': True, 'test_evaluated': False}
    write_json(output/'audit.json', manifest)
    return manifest


def repair_dataset(old_root, new_root, image, source_id, corrections, evidence):
    """Copy a schema-v1 dataset and repair explicitly enumerated points.

    Corrections carry old/new coordinates derived from audited provider ROI files.
    The selected cohort, labels, splits, and unselected files must remain unchanged.
    This is not a quality re-selection and does not establish expert ground truth.
    """
    old_root, new_root = Path(old_root).resolve(), Path(new_root).resolve()
    if new_root.exists() or new_root == old_root or old_root in new_root.parents:
        raise ValueError('Output must be a new directory outside the source dataset')
    if evidence.get('status') != 'passed' or evidence.get('method') != 'every_decoded_pixel_exact_equality':
        raise ValueError('Exact mirror evidence is required')
    if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != evidence['target_decoded_sha256']:
        raise ValueError('Repair image differs from verified target')
    protected = json.loads((old_root/'output_hashes.json').read_text(encoding='utf-8'))
    for relative, expected in protected.items():
        path = (old_root/relative).resolve()
        if not path.is_relative_to(old_root) or digest_file(path) != expected:
            raise ValueError('Original dataset integrity check failed')
    with (old_root/'기록/samples.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    plan = []
    height, width = image.shape[:2]
    for row in rows:
        c = corrections.get(row['point_id'])
        if c is None:
            continue
        if row['source_id'] != source_id or row['split'] == 'test':
            raise ValueError('Repair source mismatch or reserved test point')
        if [float(row['x']), float(row['y'])] != c['old_xy']:
            raise ValueError('Provider coordinate does not match dataset')
        xy = mirrored_xy(*c['old_xy'], width, height)
        if list(xy) != c['new_xy']:
            raise ValueError('Unverified coordinate transform')
        bounds = patch_bounds(*xy, int(row['size']), width, height)
        if bounds is None:
            raise ValueError('Corrected patch crosses image boundary; requires explicit reselection')
        destination = (new_root/row['path']).resolve()
        if not destination.is_relative_to(new_root):
            raise ValueError('Invalid patch path')
        plan.append((row, c, bounds))
    if not plan:
        raise ValueError('No selected points match the correction plan')
    shutil.copytree(old_root, new_root)
    try:
        changes = []
        for row, correction, (left, top, right, bottom) in plan:
            old_row = dict(row)
            path = new_root/row['path']
            patch = image[top:bottom, left:right].copy()
            tifffile.imwrite(path, patch, photometric='rgb' if patch.ndim == 3 else 'minisblack', metadata=None)
            row.update(x=str(correction['new_xy'][0]), y=str(correction['new_xy'][1]),
                       left=str(left), top=str(top), sha256=digest_file(path))
            # Prior visual reviews concern the old (wrong) patch, so must not transfer.
            row.update(ai_visual_reviewed='False', ai_status='coordinate_repaired_pending_type_review',
                       ai_proposed_base_label='', human_verified='False',
                       expert_semantic_confirmation='False', eligible_for_verified_evaluation='False')
            changes.append({'point_id': row['point_id'], 'patch_id': row['patch_id'],
                            'old': old_row, 'new': dict(row), 'roi_refs': correction['roi_refs']})
        with (new_root/'기록/samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        changed_ids = {r['patch_id'] for r, _, _ in plan}
        # Balanced training manifests must reflect any explicitly repaired training row.
        by_id = {r['patch_id']: r for r in rows}
        for path in (new_root/'기록').glob('train_balanced_*.csv'):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                balanced = list(csv.DictReader(stream))
            if any(r['patch_id'] in changed_ids for r in balanced):
                with path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(by_id[r['patch_id']] for r in balanced)
        for row in rows:
            if digest_file(new_root/row['path']) != row['sha256']:
                raise ValueError('Repaired dataset patch integrity failure')
        provenance = json.loads((new_root/'provenance.json').read_text(encoding='utf-8'))
        provenance.update(parent_dataset=str(old_root), parent_manifest_sha256=protected['기록/samples.csv'],
                          manifest_sha256=digest_file(new_root/'기록/samples.csv'), coordinates_changed=True,
                          selection_policy='inherited unchanged cohort; quality selection predates coordinate repair',
                          coordinate_repair='coordinate_repair.json', test_evaluated=False)
        write_json(new_root/'provenance.json', provenance)
        summary = json.loads((new_root/'summary.json').read_text(encoding='utf-8'))
        summary.update(coordinate_repairs=len(changes), quality_reselection_performed=False,
                       included_ai_reviewed=sum(r.get('ai_visual_reviewed') == 'True' for r in rows))
        write_json(new_root/'summary.json', summary)
        record = {'schema_version': 1, 'task': 'explicit_horizontal_mirror_coordinate_repair',
                  'status': 'completed', 'evidence': evidence, 'source_id': source_id,
                  'all_provider_corrections': corrections, 'selected_patch_changes': changes,
                  'selected_patch_count': len(changes), 'original_type_labels_changed': False,
                  'cohort_changed': False, 'human_verified': False, 'research_only': True,
                  'reserved_test_modified': False, 'test_evaluated': False}
        write_json(new_root/'coordinate_repair.json', record)
        write_json(new_root/'validation.json', {'status': 'passed', 'files_checked': len(rows),
                   'scope': 'structural and byte integrity only; not physical type confirmation'})
        (new_root/'COORDINATE_REPAIR.md').write_text(
            '# 좌표 보정 데이터셋\n\n타입 라벨과 표본은 유지하고 명시된 좌표만 보정했습니다. '
            '기존 검수와 제외 통계는 보정 전 선택 이력입니다. 보정한 패치의 기존 AI 검수는 승계하지 않습니다. '
            '전문가 정답 확정 자료가 아니며, 기존 데이터셋과 입력이 다르므로 점수 상승을 학습 개선으로 해석하면 안 됩니다.\n',
            encoding='utf-8')
        hashes = {p.relative_to(new_root).as_posix(): digest_file(p) for p in sorted(new_root.rglob('*'))
                  if p.is_file() and p.name != 'output_hashes.json'}
        write_json(new_root/'output_hashes.json', hashes)
        for relative, expected in protected.items():
            if digest_file(old_root/relative) != expected:
                raise RuntimeError('Original dataset changed during repair')
        return record
    except BaseException:
        write_json(new_root/'BUILD_FAILED.json', {'status': 'failed'})
        raise
