import json

import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy import ndimage

from sic_xrt_data_tools.annotation_workspace import build_workspace
from sic_xrt_data_tools.candidate_workbench import (
    build_tracks,
    build_workbench,
    contrast_candidates,
    line_geometry_candidate,
    neighbor_edges,
    phase_shift_candidate,
    tag_candidates,
    validate_outputs,
)
from sic_xrt_data_tools.source_registry import Registrar, digest_file


def spot_image():
    image = np.full((128, 128), 128.0, np.float32)
    y, x = np.mgrid[:128, :128]
    image += 100 * np.exp(-((x - 30)**2 + (y - 40)**2) / 8)
    image -= 100 * np.exp(-((x - 90)**2 + (y - 80)**2) / 8)
    return image


def test_contrast_polarities_are_unlabeled_proposals_and_blank_is_not_normal():
    points = contrast_candidates(spot_image(), max_candidates=8)
    assert any(p['polarity'] == 'bright' and abs(p['x'] - 30) <= 2 and abs(p['y'] - 40) <= 2 for p in points)
    assert any(p['polarity'] == 'dark' and abs(p['x'] - 90) <= 2 and abs(p['y'] - 80) <= 2 for p in points)
    assert contrast_candidates(np.full((128, 128), 100)) == []
    tagged = tag_candidates(points, {'asset_id': 'image', 'area_id': '1', 'phase_hint': 'before'})
    assert all(p['base_label'] is None and p['probability'] is None and not p['human_verified'] for p in tagged)
    assert all(p['split'] == 'unassigned' and not p['eligible_for_verified_evaluation'] for p in tagged)
    with pytest.raises(ValueError, match='nonfinite'):
        contrast_candidates(np.full((64, 64), np.nan))


def test_translation_moves_after_image_into_before_coordinates():
    rng = np.random.default_rng(7)
    before = rng.normal(size=(64, 64)).astype(np.float32)
    after = np.roll(before, (5, -7), axis=(0, 1))
    result = phase_shift_candidate(before, after)
    assert result['shift_yx'] == [-5, 7]
    assert result['normalized_correlation'] > 0.99
    assert not result['physical_field_of_view_confirmed']
    assert phase_shift_candidate(np.ones((64, 64)), np.ones((64, 64)))['status'] == 'unresolved'


def test_local_line_length_is_not_a_whole_line_or_physical_measurement():
    y, _ = np.mgrid[:256, :256]
    image = (128 - 90 * np.exp(-((y - 128)**2) / 3)).astype(np.float32)
    result = line_geometry_candidate(image, {'x': 128, 'y': 128, 'point_id': 'p', 'image_asset_id': 'i'})
    assert result['status'] == 'local_straight_segment_candidate'
    assert min(result['angle_raw_x_deg'], 180 - result['angle_raw_x_deg']) < 1
    assert result['local_segment_length_px'] > 100
    assert result['whole_line_length_confirmed'] is False
    assert result['length_um'] is result['curvature'] is result['step_flow_zero_direction'] is None
    round_point = {'x': 30, 'y': 40, 'point_id': 'p', 'image_asset_id': 'i'}
    assert line_geometry_candidate(spot_image(), round_point)['status'] == 'unresolved'


def test_same_polarity_mutual_nearest_paths_require_three_frames():
    frames = []
    for frame in range(3):
        frames.append([{'candidate_id': f'p{frame}', 'image_asset_id': 's', 'frame_index': frame,
                        'x': 20 + frame, 'y': 30, 'polarity': 'bright'}])
    assert neighbor_edges(frames[0], [frames[1][0] | {'polarity': 'dark'}]) == []
    assert neighbor_edges(frames[0], [frames[1][0] | {'x': 100}]) == []
    edges = neighbor_edges(frames[0], frames[1]) + neighbor_edges(frames[1], frames[2])
    assert build_tracks(frames[0] + frames[1], edges[:1]) == []
    tracks = build_tracks([p for frame in frames for p in frame], edges)
    assert len(tracks) == 1 and tracks[0]['candidate_ids'] == ['p0', 'p1', 'p2']
    assert tracks[0]['length_um'] is tracks[0]['conversion_depth_um'] is None
    assert tracks[0]['human_verified'] is False


def test_full_workbench_scan_preserves_sources_and_proposes_unassigned_fold_plans(tmp_path):
    raw = tmp_path / 'raw'
    before_folder = raw / '2D XRT/1 Area/before_1_x100'
    after_folder = raw / '2D XRT/1 Area/after_1_x100'
    before_folder.mkdir(parents=True)
    after_folder.mkdir(parents=True)
    image = spot_image().astype(np.uint8)
    Image.fromarray(image).save(before_folder / 'before_1.jpg')
    Image.fromarray(np.roll(image, (3, -2), (0, 1))).save(after_folder / 'after_1.jpg')
    stack_dir = raw / '3D XRT'
    stack_dir.mkdir()
    tifffile.imwrite(stack_dir / 'sample.tif', np.stack([ndimage.shift(image, (0, i)) for i in range(3)]),
                     photometric='minisblack', metadata={'axes': 'TYX'})
    originals = {p: digest_file(p) for p in raw.rglob('*') if p.is_file()}
    registry = tmp_path / 'registry'
    Registrar(raw, registry).run()
    workspace = tmp_path / 'workspace'
    build_workspace(registry, workspace, with_reviews=False)
    output = tmp_path / 'candidates'
    result = build_workbench(workspace, output, max_2d=8, max_3d=8)
    assert result['candidates_2d'] >= 4 and result['candidates_3d'] >= 6
    assert result['survey_records'] == 5 and result['before_after_pair_candidates'] == 1
    assert result['training_ready'] is False and result['split_status'] == 'unassigned'
    assert validate_outputs(output)
    assert all(digest_file(p) == sha for p, sha in originals.items())
    folds = json.loads((output / 'future_wafer_folds.json').read_text())
    assert len(folds['folds']) == 9 and all(not p['ready'] and len(p['training_areas']) == 8 for p in folds['folds'])
    assert (output / 'candidate_review_template.csv').exists()
    with pytest.raises(FileExistsError):
        build_workbench(workspace, output)
    (registry / 'assets.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='source_registry_changed'):
        build_workbench(workspace, tmp_path / 'changed')
