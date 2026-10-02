"""Validate append-only point review decisions without claiming full annotation."""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from .annotation_workspace import FINE_CLASSES, write_json, write_jsonl
from .candidate_workbench import validate_outputs
from .source_registry import digest_file

LABELS = set(FINE_CLASSES) | {'TED', 'TSD', 'normal', 'dust', 'scratch', 'unknown'}


def event_digest(event):
    content = {k: v for k, v in event.items() if k != 'event_sha256'}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def input_index(workspace, candidates):
    workspace, candidates = Path(workspace), Path(candidates)
    if not all(validate_outputs(p) for p in (workspace, candidates)):
        raise ValueError('input_hash_invalid')
    a = json.loads((workspace/'workspace.json').read_text(encoding='utf-8'))
    b = json.loads((candidates/'candidate_summary.json').read_text(encoding='utf-8'))
    if (a['schema'], a['schema_version'], b['schema'], b['schema_version']) != ('annotation_workspace', 2, 'candidate_workbench', 1):
        raise ValueError('unsupported_input_schema')
    if b['workspace_manifest_sha256'] != digest_file(workspace/'output_hashes.json'):
        raise ValueError('candidate_workspace_mismatch')
    index = {}
    for root, name, key, origin in [(workspace, 'provider_points.jsonl', 'point_id', 'provider'),
                                   (candidates, 'candidates_2d.jsonl', 'candidate_id', 'contrast'),
                                   (candidates, 'candidates_3d.jsonl', 'candidate_id', 'contrast')]:
        for line in (root/name).read_text(encoding='utf-8').splitlines():
            point = json.loads(line)
            item_id = point[key]
            if item_id in index:
                raise ValueError('duplicate_input_item_id')
            index[item_id] = point | {'item_id': item_id, 'source_kind': origin, 'frame_index': point.get('frame_index')}
    return index, a


def validate_events(events, index):
    latest, seen = {}, set()
    previous = None
    for seq, event in enumerate(events, 1):
        if type(event.get('sequence')) is not int or event.get('sequence') != seq or event.get('previous_event_sha256') != previous:
            raise ValueError('event_chain_invalid')
        if event.get('event_id') in seen or not isinstance(event.get('event_id'), str) or not event['event_id']:
            raise ValueError('event_id_invalid')
        seen.add(event['event_id'])
        if event_digest(event) != event.get('event_sha256'):
            raise ValueError('event_hash_invalid')
        previous = event['event_sha256']
        point = index.get(event.get('item_id'))
        if not point:
            raise ValueError('unknown_item_id')
        for field in ('image_asset_id', 'frame_index', 'source_kind'):
            if event.get(field) != point[field]:
                raise ValueError('item_reference_mismatch')
        for axis in ('x', 'y'):
            value = event.get(axis)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value-point[axis]) > 1e-8:
                raise ValueError('item_coordinate_mismatch')
        decision = event.get('decision')
        label = event.get('reviewed_label')
        if decision not in {'confirm', 'correct', 'exclude', 'hold'} or label not in LABELS | {None}:
            raise ValueError('invalid_review_decision_or_label')
        if decision in {'confirm', 'correct'} and label in {None, 'unknown'}:
            raise ValueError('label_required')
        if decision == 'confirm' and (point['source_kind'] != 'provider' or label != point['fine_label']):
            raise ValueError('confirmation_must_match_provider_label')
        actor = event.get('actual_actor')
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 120:
            raise ValueError('actual_actor_required')
        if event.get('actor_type') not in {'human', 'ai'} or type(event.get('directly_checked')) is not bool:
            raise ValueError('actor_or_check_invalid')
        if event['actor_type'] == 'human' and any(s in actor.casefold() for s in ('codex', 'chatgpt', 'gpt', 'ai')):
            raise ValueError('ai_actor_cannot_claim_human_review')
        if event['actor_type'] == 'human' and decision in {'confirm', 'correct'} and not event['directly_checked']:
            raise ValueError('direct_visual_review_required')
        try:
            timestamp = datetime.fromisoformat(event['reviewed_at'])
            if timestamp.tzinfo is None:
                raise ValueError('timezone_required')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('review_timestamp_invalid') from exc
        latest[point['item_id']] = event
    return latest


def inspect_session(workspace, candidates, session):
    workspace, candidates, session = Path(workspace), Path(candidates), Path(session)
    index, contract = input_index(workspace, candidates)
    header = json.loads((session/'review_session.json').read_text(encoding='utf-8'))
    if (header.get('schema'), header.get('schema_version')) != ('review_decisions_session', 1):
        raise ValueError('unsupported_review_schema')
    if header.get('workspace_manifest_sha256') != digest_file(workspace/'output_hashes.json') or \
            header.get('candidate_manifest_sha256') != digest_file(candidates/'output_hashes.json'):
        raise ValueError('review_session_input_mismatch')
    log_bytes = (session/'decisions.jsonl').read_bytes()
    lines = log_bytes.decode('utf-8').splitlines()
    if any(not s.strip() for s in lines):
        raise ValueError('blank_or_partial_review_event')
    events = [json.loads(s) for s in lines]
    latest = validate_events(events, index)
    reviewed = []
    for item_id, event in latest.items():
        point = index[item_id]
        human = event['actor_type'] == 'human' and event['directly_checked'] and event['decision'] in {'confirm', 'correct'}
        reviewed.append(point | {'review_event_id': event['event_id'], 'reviewed_label': event['reviewed_label'],
                                  'review_decision': event['decision'], 'actual_actor': event['actual_actor'],
                                  'reviewed_at': event['reviewed_at'], 'point_human_reviewed': human,
                                  'full_image_annotation_confirmed': False, 'expert_semantic_confirmation': False,
                                  'split': 'unassigned', 'eligible_for_verified_evaluation': False})
    return reviewed, {
        'schema': 'review_readiness', 'schema_version': 1, 'input_items': len(index), 'review_events': len(events),
        'latest_decisions': len(latest), 'remaining_items': len(index)-len(latest),
        'decision_counts': dict(Counter(e['decision'] for e in latest.values())),
        'review_log_sha256': hashlib.sha256(log_bytes).hexdigest(),
        'pending_or_excluded_items': sum(e['decision'] in {'hold', 'exclude'} for e in latest.values()),
        'human_reviewed_points': sum(p['point_human_reviewed'] for p in reviewed),
        'display_reviewer_name': contract['summary']['requested_reviewer_name'],
        'training_ready': False, 'full_dataset_export_complete': False,
        'blockers': ['all_wafer_complete_annotation_regions', 'class_semantic_definitions',
                     'cross_acquisition_physical_identity', 'reviewed_wafer_group_split',
                     'line_and_transition_and_3d_ground_truth', 'physical_calibration'],
    }


def export_review(workspace, candidates, session, output):
    output = Path(output).resolve()
    reviewed, report = inspect_session(workspace, candidates, session)
    source = Path(json.loads((Path(workspace)/'workspace.json').read_text(encoding='utf-8'))['source_root']).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output)
                              for p in (source, workspace, candidates, session)):
        raise ValueError('output_must_be_new_independent_directory')
    output.mkdir(parents=True)
    write_jsonl(output/'reviewed_points.jsonl', reviewed)
    write_json(output/'readiness.json', report)
    snapshots = {str(Path(p).resolve()): digest_file(p) for p in
               (Path(workspace)/'output_hashes.json', Path(candidates)/'output_hashes.json',
                Path(session)/'review_session.json')}
    snapshots[str((Path(session)/'decisions.jsonl').resolve())] = report['review_log_sha256']
    write_json(output/'input_hashes.json', snapshots)
    write_json(output/'output_hashes.json', {p.name: digest_file(p) for p in output.iterdir() if p.is_file()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'candidates', 'session', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_review(args.workspace, args.candidates, args.session, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
