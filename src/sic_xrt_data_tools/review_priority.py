"""Plan a small diagnostic review without approving unsampled provider labels."""
import argparse
import math
from collections import defaultdict
from pathlib import Path

from .annotation_workspace import write_json
from .review_decisions import input_index
from .source_registry import digest_file


def select_priority(index, per_group=3):
    if type(per_group) is not int or not 1 <= per_group <= 20:
        raise ValueError('invalid_sample_size')
    groups, positions, reasons = defaultdict(list), defaultdict(list), defaultdict(set)
    invalid, unknown = [], []
    provider = [p for p in index.values() if p['source_kind'] == 'provider']
    for p in provider:
        ident = p['item_id']
        finite = all(isinstance(p.get(k), (int, float)) and not isinstance(p[k], bool) and math.isfinite(p[k]) for k in ('x','y'))
        if not finite or p.get('coordinate_valid') is False:
            invalid.append(ident)
            reasons[ident].add('coordinate_problem')
            continue
        positions[(p['image_asset_id'], p['frame_index'], p['x'], p['y'])].append(p)
        if p.get('fine_label') in {None, 'unknown'}:
            unknown.append(ident)
            reasons[ident].add('unknown_provider_label')
        else:
            groups[(p['image_asset_id'], p['fine_label'])].append(p)
    duplicate_sets, conflicts = [], []
    for values in positions.values():
        if len(values) > 1:
            ids = sorted(p['item_id'] for p in values)
            duplicate_sets.append(ids)
            if len({p.get('fine_label') for p in values}) > 1:
                conflicts.append(ids)
                for ident in ids:
                    reasons[ident].add('coordinate_label_conflict')
    for values in groups.values():
        remaining = sorted(values, key=lambda p:p['item_id'])
        chosen = [remaining.pop(0)]
        while remaining and len(chosen) < per_group:
            point = max(remaining, key=lambda p:min((p['x']-q['x'])**2 + (p['y']-q['y'])**2 for q in chosen))
            chosen.append(point)
            remaining.remove(point)
        for p in chosen:
            reasons[p['item_id']].add('spatial_group_sample')
    rows = [{'item_id': ident, 'reasons': sorted(why)} for ident,why in reasons.items()]
    rows.sort(key=lambda p:(p['reasons'] == ['spatial_group_sample'], p['item_id']))
    return rows, {'provider_points':len(provider), 'known_image_label_groups':len(groups),
        'invalid_coordinates':sorted(invalid), 'unknown_labels':sorted(unknown),
        'duplicate_position_groups':sorted(duplicate_sets), 'coordinate_label_conflicts':sorted(conflicts)}


def build_plan(workspace, candidates, output, per_group=3):
    output = Path(output).resolve()
    index, contract = input_index(workspace, candidates)
    source = Path(contract['source_root']).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output)
                              for p in (source, workspace, candidates)):
        raise ValueError('output_must_be_new_independent_directory')
    rows, audit = select_priority(index, per_group)
    plan = {'schema':'review_priority_plan', 'schema_version':1,
        'workspace_manifest_sha256':digest_file(Path(workspace)/'output_hashes.json'),
        'candidate_manifest_sha256':digest_file(Path(candidates)/'output_hashes.json'),
        'items':rows, 'selected_count':len(rows), 'total_items':len(index), 'per_group':per_group,
        'selection':'per_image_fine_label_spatial_diagnostic_plus_known_errors', 'audit':audit,
        'provider_labels_preserved':True, 'human_verified_created':0, 'training_ready':False,
        'statistical_accuracy_estimate':None, 'unsampled_labels_approved':False,
        'automatic_candidates_deferred':sum(p['source_kind']=='contrast' for p in index.values()),
        'limitations':['diagnostic_sample_not_accuracy_or_complete_annotation',
                      'missing_wafer_annotations_and_subtype_definitions_remain',
                      'physical_identity_and_calibration_remain']}
    output.mkdir(parents=True)
    write_json(output/'review_plan.json', plan)
    write_json(output/'output_hashes.json', {'review_plan.json':digest_file(output/'review_plan.json')})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','candidates','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(args.workspace,args.candidates,args.output)
    print(f"First-pass review: {plan['selected_count']} / {plan['total_items']}")


if __name__ == '__main__':
    main()
