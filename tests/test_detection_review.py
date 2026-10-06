import csv
import json

import numpy as np
import pytest
from PIL import Image

from sic_xrt_data_tools.detection_review import build, finalize
from sic_xrt_data_tools.source_registry import digest_file


def setup_inputs(root, wafer='1', x=64):
    path = root/'image.png'
    Image.fromarray(np.full((400, 400, 3), 111, np.uint8)).save(path)
    source = {'path': str(path), 'sha256': digest_file(path)}
    row = {'wafer': wafer, 'candidate_id': 'one', 'x': x, 'y': 200,
           'source_image_sha256': source['sha256']}
    queue = root/'queue.csv'
    with queue.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return queue, {wafer: source}


def test_edge_context_explicit_training_pixels_unchanged(tmp_path):
    queue, sources = setup_inputs(tmp_path)
    result = build(queue, sources, tmp_path/'out')
    row = result['candidates'][0]
    assert row['context_clipped']
    assert row['context_valid_xyxy'] == [64, 0, 256, 256]
    assert row['training_eligible'] is False
    import tifffile
    patch = tifffile.imread(tmp_path/'out'/row['path'])
    assert patch.shape == (128, 128, 3) and np.all(patch == 111)


def test_holdout_and_source_mismatch_rejected(tmp_path):
    queue, sources = setup_inputs(tmp_path, wafer='8')
    with pytest.raises(ValueError, match='wafer8'):
        build(queue, sources, tmp_path/'forbidden')
    queue, sources = setup_inputs(tmp_path)
    sources['1']['sha256'] = 'invalid'
    with pytest.raises(ValueError, match='hash'):
        build(queue, sources, tmp_path/'invalid')


@pytest.mark.parametrize('decision,eligible', [('weak_background', True), ('possible_defect', False), ('uncertain', False)])
def test_review_never_promotes_ai_to_human_truth(tmp_path, decision, eligible):
    queue, sources = setup_inputs(tmp_path)
    out = tmp_path/'out'
    build(queue, sources, out)
    observation = {'id': 'w1_001', 'decision': decision, 'note': 'Observed morphology only'}
    with pytest.raises(ValueError):
        finalize(out, [{**observation, 'human_verified': True}])
    result = finalize(out, [observation])['candidates'][0]
    assert result['training_eligible'] is eligible
    assert not result['expert_ground_truth'] and not result['human_verified']


def test_changed_patch_or_missing_observation_rejected(tmp_path):
    queue, sources = setup_inputs(tmp_path)
    out = tmp_path/'out'
    build(queue, sources, out)
    with pytest.raises(ValueError):
        finalize(out, [])
    row = json.loads((out/'candidates.json').read_text(encoding='utf-8'))['candidates'][0]
    (out/row['context_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        finalize(out, [{'id': row['id'], 'decision': 'uncertain', 'note': 'Unclear'}])
