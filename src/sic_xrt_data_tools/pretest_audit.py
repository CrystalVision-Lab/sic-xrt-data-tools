"""Read-only source/ROI audit. Image contrast is a diagnostic, never a type label."""

import numpy as np
import tifffile

from .source_registry import digest_file


def audit_source_rows(root, rows, points, reader, image):
    """Check every selected training patch against its source and original ROI."""
    root = root.resolve()
    roi_cache = {}
    checked = 0
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('Only training rows are accepted')
        point = points[row['point_id']]
        if point['image_asset_id'] != row['source_id']:
            raise ValueError('Source mapping mismatch')
        xy = [float(row['x']), float(row['y'])]
        if not np.allclose(xy, [point['x'], point['y']], rtol=0, atol=1e-5):
            raise ValueError('Coordinate mismatch')
        if row['label'] != point['base_label'] or row['provider_fine_label'] != point['fine_label']:
            raise ValueError('Provider label changed')
        if not point['roi_refs']:
            raise ValueError('Missing provider ROI reference')
        for ref in point['roi_refs']:
            aid = ref['roi_asset_id']
            if aid not in roi_cache:
                roi_cache[aid] = reader.points(reader.by_id[aid])
            index = ref['point_index']
            if not isinstance(index, int) or not 0 <= index < len(roi_cache[aid]):
                raise ValueError('Invalid ROI point index')
            if not np.allclose(roi_cache[aid][index], xy, rtol=0, atol=1e-5):
                raise ValueError('Raw ROI coordinate mismatch')
        size = int(row['size'])
        # reviewed_research_dataset uses Python round (ties-to-even), not half-up.
        left, top = round(xy[0])-size//2, round(xy[1])-size//2
        bounds = (left, top, left+size, top+size)
        if (min(left, top) < 0 or bounds[2] > image.shape[1] or bounds[3] > image.shape[0]
                or [left, top] != [int(row['left']), int(row['top'])]):
            raise ValueError('Invalid crop bounds')
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root) or digest_file(path) != row['sha256']:
            raise ValueError('Patch hash/path mismatch')
        patch = tifffile.imread(path)
        left, top, right, bottom = bounds
        original = image[top:bottom, left:right]
        if patch.dtype != original.dtype or not np.array_equal(patch, original):
            raise ValueError('Patch pixels differ from source')
        checked += 1
    return {'patches_checked': checked, 'roi_files_checked': len(roi_cache),
            'pixel_match': True, 'roi_coordinate_match': True, 'labels_unchanged': True,
            'crop_rounding': 'python_round_ties_to_even', 'semantic_type_verified': False}


def orientation_probe(image, coordinates, limit=256):
    """Compare local contrast at fixed points and mirror/shift controls.

    Uses all or uniformly spaced points, independent of predictions or class correctness.
    A higher score alone does not authorize any coordinate transform.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected raw uint8 RGB image')
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Expected nonempty finite Nx2 coordinates')
    h, w = image.shape[:2]
    if np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h):
        raise ValueError('Coordinates outside image')
    xy = xy[np.linspace(0, len(xy)-1, min(limit, len(xy)), dtype=int)]
    results = {}
    for name in ['identity', 'mirror_x', 'mirror_y', 'rotate180', 'shift_control']:
        transformed = xy.copy()
        if name in ('mirror_x', 'rotate180'):
            transformed[:, 0] = w-1-transformed[:, 0]
        if name in ('mirror_y', 'rotate180'):
            transformed[:, 1] = h-1-transformed[:, 1]
        if name == 'shift_control':
            transformed = (transformed + [173, 137]) % [w, h]
        values, average = [], np.zeros((65, 65), dtype=float)
        for x, y in np.floor(transformed+.5).astype(int):
            if x < 32 or y < 32 or x+33 > w or y+33 > h:
                continue
            patch = image[y-32:y+33, x-32:x+33].mean(axis=2)/255
            residual = np.abs(patch-np.median(patch))
            values.append(float(np.quantile(residual[24:41, 24:41], .95)))
            average += residual
        if values:
            average /= len(values)
            py, px = np.unravel_index(np.argmax(average), average.shape)
            results[name] = {'n': len(values), 'median_center_contrast': float(np.median(values)),
                             'mean_center_contrast': float(np.mean(values)),
                             'average_peak_offset': [int(px-32), int(py-32)]}
        else:
            results[name] = {'n': 0, 'median_center_contrast': None}
    return {'method': 'fixed_point_local_contrast_with_mirror_and_shift_controls_v1',
            'hypotheses': results, 'automatic_transform_authorized': False}
