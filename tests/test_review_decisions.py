import json

import pytest

from sic_xrt_data_tools.review_decisions import (
    event_digest,
    export_review,
    inspect_session,
    validate_events,
)
from sic_xrt_data_tools.source_registry import digest_file


def index():
    return {'pt': {'item_id': 'pt', 'image_asset_id': 'img', 'source_kind': 'provider',
                   'frame_index': None, 'x': 20.25, 'y': 40.5, 'fine_label': 'TED_a'}}


def event(**changes):
    data = {'sequence': 1, 'previous_event_sha256': None, 'event_id': 'e1', 'item_id': 'pt',
            'image_asset_id': 'img', 'frame_index': None, 'source_kind': 'provider', 'x': 20.25, 'y': 40.5,
            'decision': 'confirm', 'reviewed_label': 'TED_a', 'actual_actor': '검수자', 'actor_type': 'human',
            'directly_checked': True, 'reviewed_at': '2026-10-02T02:00:00+00:00'} | changes
    data['event_sha256'] = event_digest(data)
    return data


def test_decision_history_preserves_latest_hold_without_old_approval():
    first = event()
    second = event(sequence=2, previous_event_sha256=first['event_sha256'], event_id='e2', decision='hold', directly_checked=False)
    assert validate_events([first, second], index())['pt']['decision'] == 'hold'


@pytest.mark.parametrize('changes, reason', [
    ({'item_id': 'unknown'}, 'unknown_item'), ({'x': 21}, 'coordinate'),
    ({'frame_index': 1}, 'reference'), ({'reviewed_label': 'TED_g'}, 'label'),
    ({'reviewed_label': 'TSD_a'}, 'confirmation'), ({'actual_actor': ''}, 'actor'),
    ({'actual_actor': 'Codex_AI'}, 'ai_actor'), ({'directly_checked': False}, 'visual'),
    ({'reviewed_at': '2026-10-02'}, 'timestamp'), ({'sequence': 2}, 'chain'),
])
def test_bad_decisions_are_rejected(changes, reason):
    with pytest.raises(ValueError, match=reason):
        validate_events([event(**changes)], index())


def test_event_tampering_is_rejected_and_ai_proposal_is_not_human():
    data = event()
    data['actual_actor'] = '다른 사람'
    with pytest.raises(ValueError, match='event_hash'):
        validate_events([data], index())
    assert validate_events([event(actor_type='ai', actual_actor='Codex_AI')], index())['pt']['actor_type'] == 'ai'


@pytest.fixture
def session_inputs(tmp_path):
    workspace, candidates, session, source = [tmp_path/n for n in ('workspace', 'candidates', 'session', 'raw')]
    for p in (workspace, candidates, session, source):
        p.mkdir()

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def manifest(root):
        write(root/'output_hashes.json', {p.name: digest_file(p) for p in root.iterdir() if p.name != 'output_hashes.json'})

    write(workspace/'workspace.json', {'schema':'annotation_workspace', 'schema_version':2,
        'source_root':str(source), 'summary':{'requested_reviewer_name':'양희승'}})
    (workspace/'provider_points.jsonl').write_text(json.dumps(index()['pt'] | {'point_id':'pt'})+'\n', encoding='utf-8')
    manifest(workspace)
    write(candidates/'candidate_summary.json', {'schema':'candidate_workbench', 'schema_version':1,
        'workspace_manifest_sha256':digest_file(workspace/'output_hashes.json')})
    for n in ('candidates_2d.jsonl', 'candidates_3d.jsonl'):
        (candidates/n).write_bytes(b'')
    manifest(candidates)
    write(session/'review_session.json', {'schema':'review_decisions_session', 'schema_version':1,
        'workspace_manifest_sha256':digest_file(workspace/'output_hashes.json'),
        'candidate_manifest_sha256':digest_file(candidates/'output_hashes.json')})
    (session/'decisions.jsonl').write_bytes(b'')
    return workspace, candidates, session


def test_empty_session_and_verified_export_do_not_claim_complete_dataset(session_inputs, tmp_path):
    a,b,s = session_inputs
    reviewed, report = inspect_session(a,b,s)
    assert not reviewed and report['remaining_items'] == 1
    first = event()
    (s/'decisions.jsonl').write_text(json.dumps(first)+'\n', encoding='utf-8')
    output = tmp_path/'export'
    report = export_review(a,b,s,output)
    point = json.loads((output/'reviewed_points.jsonl').read_text(encoding='utf-8'))
    assert report['human_reviewed_points'] == 1 and point['point_human_reviewed']
    assert point['split'] == 'unassigned' and not point['eligible_for_verified_evaluation']
    assert not point['full_image_annotation_confirmed'] and not report['training_ready']
    assert not report['full_dataset_export_complete']
    assert report['review_log_sha256'] == digest_file(s/'decisions.jsonl')
    with pytest.raises(ValueError, match='independent'):
        export_review(a,b,s,output)
    with pytest.raises(ValueError, match='independent'):
        export_review(a,b,s,s/'export')


@pytest.mark.parametrize('changes', [
    {'decision':'hold', 'directly_checked':False},
    {'decision':'exclude', 'directly_checked':False},
    {'actor_type':'ai', 'actual_actor':'Codex_AI'},
])
def test_latest_hold_exclusion_or_ai_does_not_reuse_human_confirmation(session_inputs, changes):
    a,b,s = session_inputs
    first = event()
    second = event(sequence=2, event_id='e2', previous_event_sha256=first['event_sha256'], **changes)
    (s/'decisions.jsonl').write_text('\n'.join(json.dumps(p) for p in (first,second))+'\n', encoding='utf-8')
    points, report = inspect_session(a,b,s)
    assert not points[0]['point_human_reviewed'] and report['human_reviewed_points'] == 0
    assert report['review_events'] == 2 and report['latest_decisions'] == 1


def test_changed_session_header_or_partial_event_is_rejected(session_inputs):
    a,b,s = session_inputs
    (s/'decisions.jsonl').write_bytes(b'\n')
    with pytest.raises(ValueError, match='partial'):
        inspect_session(a,b,s)
    header = json.loads((s/'review_session.json').read_text(encoding='utf-8'))
    header['workspace_manifest_sha256'] = 'wrong'
    (s/'review_session.json').write_text(json.dumps(header), encoding='utf-8')
    with pytest.raises(ValueError, match='input_mismatch'):
        inspect_session(a,b,s)
