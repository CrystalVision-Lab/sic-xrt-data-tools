import pytest

from sic_xrt_data_tools.development_cohort import develop_points, neighbor_policy


def point(ident, x, label='BPD', wafer='1'):
    return {'point_id': ident, 'x': x, 'y': 100., 'area_id': wafer, 'base_label': label, 'fine_label': label}


def test_context_neighbor_is_allowed_but_center_conflict_is_not():
    p = point('a', 100.)
    assert neighbor_policy(p, [p, point('b', 140., 'TED')]) == (False, 1)
    assert neighbor_policy(p, [point('b', 110., 'TED')]) == (True, 1)
    assert neighbor_policy(p, [point('b', 110., 'BPD')]) == (False, 0)


def test_development_cannot_include_reserved_wafer_and_preserves_labels():
    p = point('a', 100., 'TSD', '2')
    corrected = {'a': {'old_xy': [100., 100.], 'new_xy': [899., 100.], 'provider_fine_label': 'TSD'}}
    rows = develop_points([p, point('test', 100., wafer='8')], corrected)
    assert len(rows) == 1 and rows[0]['x'] == 899 and rows[0]['base_label'] == 'TSD'
    assert p['x'] == 100
    corrected['a']['provider_fine_label'] = 'TED'
    with pytest.raises(ValueError):
        develop_points([p], corrected)
