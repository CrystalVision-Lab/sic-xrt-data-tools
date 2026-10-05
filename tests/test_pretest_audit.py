import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from sic_xrt_data_tools.pretest_audit import audit_source_rows, orientation_probe


def test_raw_roi_and_pixel_mismatch_rejected(tmp_path):
    image = np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3)
    path = tmp_path/'p.tif'
    tifffile.imwrite(path, image[8:24, 8:24])
    row = {'split': 'train', 'point_id': 'p', 'source_id': 's', 'x': '16', 'y': '16',
           'label': 'TED', 'provider_fine_label': 'TED_a', 'size': '16', 'left': '8', 'top': '8',
           'path': 'p.tif', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    point = {'image_asset_id': 's', 'x': 16, 'y': 16, 'base_label': 'TED', 'fine_label': 'TED_a',
             'roi_refs': [{'roi_asset_id': 'r', 'point_index': 0}]}
    reader = SimpleNamespace(by_id={'r': {}}, points=lambda _: np.array([[16., 16.]]))
    assert audit_source_rows(tmp_path, [row], {'p': point}, reader, image)['patches_checked'] == 1
    with pytest.raises(ValueError, match='training'):
        audit_source_rows(tmp_path, [dict(row, split='test')], {'p': point}, reader, image)
    reader.points = lambda _: np.array([[17., 16.]])
    with pytest.raises(ValueError, match='Raw ROI'):
        audit_source_rows(tmp_path, [row], {'p': point}, reader, image)
    reader.points = lambda _: np.array([[16., 16.]])
    wrong_image = image.copy()
    wrong_image[16, 16] = 0
    with pytest.raises(ValueError, match='pixels'):
        audit_source_rows(tmp_path, [row], {'p': point}, reader, wrong_image)


def test_mirror_probe_is_diagnostic_only():
    image = np.full((300, 300, 3), 100, np.uint8)
    image[75:86, 65:76] = 240
    result = orientation_probe(image, [[229, 80]])
    scores = result['hypotheses']
    assert scores['mirror_x']['median_center_contrast'] > scores['identity']['median_center_contrast']
    assert result['automatic_transform_authorized'] is False
    with pytest.raises(ValueError):
        orientation_probe(image, [[float('nan'), 0]])
