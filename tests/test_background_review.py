import json

import numpy as np
import pytest

from sic_xrt_data_tools.background_review import centers, finalize, line_proposal
from sic_xrt_data_tools.source_registry import digest_file


def test_sampling_respects_annotations_bounds_and_spacing():
    points = [(256, 256)]
    result = centers(1024, 1024, points, count=8)
    assert result == centers(1024, 1024, points, count=8)
    for i, (x, y) in enumerate(result):
        assert 64 <= x <= 960 and 64 <= y <= 960
        assert np.hypot(x-256, y-256) >= 128
        assert all(np.hypot(x-a, y-b) >= 192 for a, b in result[:i])
    with pytest.raises(ValueError):
        centers(128, 128, [(64, 64)], count=1)


def test_geometry_never_becomes_ground_truth():
    blank = np.full((256, 256, 3), 128, np.uint8)
    assert line_proposal(blank)['status'] == 'unresolved'
    blank[80:180, 124:130] = 0
    proposal = line_proposal(blank)
    assert proposal['bbox_patch_xyxy'] is not None
    assert proposal['training_eligible'] is False
    assert proposal['physical_scale_known'] is False


def fixture_manifest(root):
    patch = root/'sample.tif'
    patch.write_bytes(b'only the hash is checked here')
    (root/'candidates.json').write_text(json.dumps({'candidates': [
        {'id': 'one', 'path': patch.name, 'sha256': digest_file(patch)}
    ]}), encoding='utf-8')
    return {'id': 'one', 'decision': 'background_candidate', 'note': 'visible texture only'}


def test_ai_observation_cannot_claim_human_review(tmp_path):
    observation = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match='override'):
        finalize(tmp_path, [{**observation, 'human_verified': True}])
    result = finalize(tmp_path, [observation])['candidates'][0]
    assert not result['human_verified'] and not result['expert_ground_truth']
    assert result['weak_training_eligible']


def test_missing_duplicate_or_modified_reviews_rejected(tmp_path):
    observation = fixture_manifest(tmp_path)
    for items in ([], [observation, observation]):
        with pytest.raises(ValueError):
            finalize(tmp_path, items)
    (tmp_path/'sample.tif').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        finalize(tmp_path, [observation])
