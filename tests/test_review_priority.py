from sic_xrt_data_tools.review_priority import select_priority


def test_diagnostic_sampling_preserves_labels_and_includes_conflicts_and_unknowns():
    index = {str(n):{'item_id':str(n), 'source_kind':'provider', 'image_asset_id':'img',
        'frame_index':None, 'x':float(n), 'y':0., 'fine_label':'TED_a', 'human_verified':False} for n in range(10)}
    index['conflict'] = index['0'] | {'item_id':'conflict', 'fine_label':'TSD_a'}
    index['unknown'] = index['5'] | {'item_id':'unknown', 'fine_label':'unknown'}
    index['bad'] = index['6'] | {'item_id':'bad', 'coordinate_valid':False}
    index['auto'] = index['7'] | {'item_id':'auto', 'source_kind':'contrast'}
    before = {k:dict(v) for k,v in index.items()}
    rows,audit = select_priority(index)
    ids = {p['item_id'] for p in rows}
    assert {'0','9','conflict','unknown','bad'} <= ids and 'auto' not in ids
    assert audit['coordinate_label_conflicts'] == [['0','conflict'], ['5','unknown']]
    assert index == before
    assert select_priority(dict(reversed(list(index.items())))) == (rows,audit)
