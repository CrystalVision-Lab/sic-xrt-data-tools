import pytest

from sic_xrt_data_tools.reference_audit import audit_frame, audit_roi_import


def point(x=100., y=150., ident='a'):
    return {'point_id':ident,'area_id':'2','x':x,'y':y,'base_label':'TSD','fine_label':'TSD_a',
            'roi_refs':[{'roi_asset_id':'roi','point_index':0}]}


def test_import_missing_index_and_coordinate_change_are_distinct():
    actual = point()
    result = audit_roi_import([actual], lambda _: [(100.,150.),(200.,250.)])
    assert not result['passed']
    assert result['rois'][0]['missing_indices'] == [1]
    assert not result['rois'][0]['errors']
    actual['x'] += 1
    result = audit_roi_import([actual], lambda _: [(100.,150.)])
    assert result['rois'][0]['errors'][0]['reason'] == 'coordinate_changed'


def test_import_fidelity_is_not_expert_truth():
    result = audit_roi_import([point()], lambda _: [(100.,150.)])
    assert result['passed']
    assert result['expert_ground_truth'] is False


def test_source_hash_and_reserved_wafer_guard():
    with pytest.raises(ValueError, match='hash'):
        audit_frame([point()], [], 'different','expected',1000,1000)
    forbidden = {**point(),'area_id':'8'}
    with pytest.raises(ValueError, match='Reserved'):
        audit_roi_import([forbidden], lambda _: [])


def test_existing_mirror_repair_must_apply_exactly_once():
    p = point()
    change = {'a':{'old_xy':[100.,150.],'new_xy':[899.,150.],'provider_fine_label':'TSD_a'}}
    refs = [{'id':'a','x':899.,'y':150.,'type':'TSD'}]
    assert audit_frame([p], refs,'same','same',1000,1000,corrections=change)['passed']
    stale = [{**refs[0],'x':100.}]
    assert audit_frame([p], stale,'same','same',1000,1000,corrections=change)['coordinate_or_type_mismatch'] == ['a']
    with pytest.raises(ValueError, match='double correction'):
        audit_frame([{**p,'x':899.}], refs,'same','same',1000,1000,corrections=change)


def test_jpeg_mirror_and_boundary_exclusions():
    points = [point(), point(x=20, ident='edge')]
    refs = [{'id':'a','x':899.,'y':150.,'type':'TSD'}]
    result = audit_frame(points,refs,'same','same',1000,1000,mirror_all_x=True)
    assert result['passed'] and result['boundary_excluded'] == 1
    assert audit_frame(points,[],'same','same',1000,1000,mirror_all_x=True)['missing_interior_ids'] == ['a']
