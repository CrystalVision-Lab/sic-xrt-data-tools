import numpy as np
import pytest

from sic_xrt_data_tools.coordinate_repair import exact_horizontal_mirror, mirrored_xy


def test_mirror_checks_all_pixels_including_late_rows():
    image = np.arange(130*11*3, dtype=np.uint16).reshape(130, 11, 3)
    target = image[:, ::-1].copy()
    assert exact_horizontal_mirror(image, target)['status'] == 'passed'
    target[-1, -1, -1] += 1
    with pytest.raises(ValueError, match='not an exact'):
        exact_horizontal_mirror(image, target)


def test_coordinate_transform_is_involutive_and_preserves_y():
    for x in [0, .25, 50.5, 100]:
        xy = mirrored_xy(x, 3.25, 101, 20)
        assert xy == (100-x, 3.25)
        assert mirrored_xy(*xy, 101, 20) == (x, 3.25)
    for x in [-1, 101, float('nan')]:
        with pytest.raises(ValueError):
            mirrored_xy(x, 5, 101, 20)


def test_repair_preserves_original_labels_and_test_bytes(tmp_path):
    import csv
    import json

    import tifffile

    from sic_xrt_data_tools.coordinate_repair import repair_dataset, write_json
    from sic_xrt_data_tools.source_registry import digest_file

    old = tmp_path/'old'
    (old/'기록').mkdir(parents=True)
    image = np.arange(24*24*3, dtype=np.uint8).reshape(24, 24, 3)
    evidence = exact_horizontal_mirror(image[:, ::-1], image)
    rows = []
    for split, x in [('train', 6), ('val', 7), ('test', 8)]:
        path = old/f'{split}.tif'
        tifffile.imwrite(path, image[4:12, x-4:x+4], photometric='rgb', metadata=None)
        rows.append({'patch_id': split, 'point_id': split, 'source_id': 'source',
                     'path': path.name, 'sha256': digest_file(path), 'split': split, 'label': 'TSD',
                     'size': '8', 'x': str(x), 'y': '8', 'left': str(x-4), 'top': '4',
                     'ai_visual_reviewed': 'True', 'ai_status': 'suggest', 'ai_proposed_base_label': 'TSD',
                     'human_verified': 'False', 'expert_semantic_confirmation': 'False',
                     'eligible_for_verified_evaluation': 'False'})
    with (old/'기록/samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ('summary.json', 'provenance.json', 'validation.json'):
        write_json(old/name, {})
    hashes = {p.relative_to(old).as_posix(): digest_file(p) for p in old.rglob('*') if p.is_file()}
    write_json(old/'output_hashes.json', hashes)
    corrections = {'val': {'old_xy': [7.0, 8.0], 'new_xy': [16.0, 8.0], 'roi_refs': []}}
    new = tmp_path/'new'
    record = repair_dataset(old, new, image, 'source', corrections, evidence)
    assert record['selected_patch_count'] == 1
    assert record['selected_patch_changes'][0]['new']['label'] == 'TSD'
    assert record['selected_patch_changes'][0]['new']['ai_visual_reviewed'] == 'False'
    assert np.array_equal(tifffile.imread(new/'val.tif'), image[4:12, 12:20])
    assert digest_file(new/'test.tif') == hashes['test.tif']
    assert all(digest_file(old/p) == sha for p, sha in hashes.items())
    assert json.loads((new/'provenance.json').read_text(encoding='utf-8'))['coordinates_changed']
    with pytest.raises(ValueError, match='new directory'):
        repair_dataset(old, new, image, 'source', corrections, evidence)


def test_full_point_audit_excludes_only_boundary_and_preserves_subtype(tmp_path):
    from sic_xrt_data_tools.coordinate_repair import export_corrected_point_audit

    image = np.zeros((24, 24, 3), dtype=np.uint8)
    evidence = exact_horizontal_mirror(image, image)
    points = {'middle': {'old_xy': [10, 10], 'new_xy': [13, 10], 'provider_fine_label': 'TSD_c'},
              'edge': {'old_xy': [0, 0], 'new_xy': [23, 0], 'provider_fine_label': 'TSD_a'}}
    result = export_corrected_point_audit(image, 'source', points, evidence, tmp_path/'audit', size=8)
    assert len(result['rows']) == 1
    assert result['rows'][0]['provider_fine_label'] == 'TSD_c'
    assert result['excluded'] == [{'point_id': 'edge', 'reason': 'patch_crosses_image_boundary'}]
