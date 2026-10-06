"""Verify import/coordinate provenance without claiming exhaustive defect truth."""
import math
from collections import defaultdict


def audit_roi_import(points, read_roi):
    """read_roi(asset_id) returns verified original coordinates as Nx2 values."""
    by_roi = defaultdict(list)
    for point in points:
        if str(point['area_id']) not in ('1', '2', '9'):
            raise ValueError('Reserved/unsupported wafer forbidden')
        for ref in point['roi_refs']:
            by_roi[ref['roi_asset_id']].append((ref['point_index'], point))
    results = []
    for asset_id, imported in sorted(by_roi.items()):
        original = read_roi(asset_id)
        seen, errors = set(), []
        for index, point in imported:
            if not 0 <= index < len(original):
                errors.append({'point_id': point['point_id'], 'reason': 'index_outside_roi'})
                continue
            if index in seen:
                errors.append({'point_id': point['point_id'], 'reason': 'duplicate_roi_index'})
            seen.add(index)
            expected = original[index]
            if any(not math.isclose(float(actual), float(target), rel_tol=0, abs_tol=.0001)
                   for actual, target in zip((point['x'],point['y']), expected, strict=True)):
                errors.append({'point_id': point['point_id'], 'reason': 'coordinate_changed'})
        missing = sorted(set(range(len(original)))-seen)
        results.append({'roi_asset_id':asset_id, 'original_count':len(original), 'imported_indices':len(seen),
                        'missing_indices':missing, 'errors':errors, 'passed':not missing and not errors})
    return {'passed':bool(results) and all(r['passed'] for r in results), 'rois':results,
            'original_coordinate_count':sum(r['original_count'] for r in results),
            'scope':'Selected associated ROI files only; not proof that all physical defects are annotated',
            'expert_ground_truth':False}


def audit_frame(points, references, image_sha256, expected_image_sha256, width, height,
                mirror_all_x=False, corrections=None, margin=64):
    if image_sha256 != expected_image_sha256:
        raise ValueError('Coordinate frame image hash mismatch')
    if width <= 2*margin or height <= 2*margin:
        raise ValueError('Invalid frame size')
    corrections = corrections or {}
    expected = {}
    outside = []
    for point in points:
        if str(point['area_id']) not in ('1', '2', '9'):
            raise ValueError('Reserved wafer forbidden')
        ident = point['point_id']
        if ident in expected or ident in outside:
            raise ValueError('Duplicate provider identity')
        x, y = float(point['x']), float(point['y'])
        change = corrections.get(ident)
        if change:
            if [x,y] != change['old_xy'] or point['fine_label'] != change['provider_fine_label']:
                raise ValueError('Correction/provider mismatch or double correction')
            x,y = change['new_xy']
        if mirror_all_x:
            x = width-1-x
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite coordinate')
        if not (margin <= round(x) <= width-margin and margin <= round(y) <= height-margin):
            outside.append(ident)
            continue
        expected[ident] = (x,y,point['base_label'])
    actual = {r['id']:r for r in references}
    if len(actual) != len(references):
        raise ValueError('Duplicate reference identity')
    changed = []
    for ident in actual.keys() & expected.keys():
        x,y,label = expected[ident]
        row = actual[ident]
        if (not math.isclose(row['x'],x,rel_tol=0,abs_tol=.0001)
                or not math.isclose(row['y'],y,rel_tol=0,abs_tol=.0001) or row['type'] != label):
            changed.append(ident)
    missing, extra = sorted(expected.keys()-actual.keys()), sorted(actual.keys()-expected.keys())
    return {'passed':not changed and not missing and not extra, 'provided_count':len(points),
            'expected_interior_count':len(expected), 'reference_count':len(actual),
            'boundary_excluded':len(outside), 'coordinate_or_type_mismatch':sorted(changed),
            'missing_interior_ids':missing, 'extra_ids':extra, 'image_sha256':image_sha256,
            'scope':'Faithful coordinate/type propagation only, not expert correctness or annotation completeness'}
