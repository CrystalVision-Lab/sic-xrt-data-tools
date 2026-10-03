"""Build isolated schema-v1 research patches from screened provider points and AI review history."""
import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from .annotation_workspace import SourceReader, write_jsonl
from .balanced_subset import create_subsets
from .candidate_workbench import validate_outputs
from .point_dataset import CLASSES, write_csv, write_json
from .source_registry import digest_file
from .validate_point_dataset import validate

SPLITS = {'1': 'train', '9': 'train', '2': 'val', '8': 'test'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def check_point(row, point, id_field):
    if row.get(id_field) != point['point_id'] or any(row.get(k) != point.get(k)
            for k in ('image_asset_id', 'area_id', 'phase', 'x', 'y', 'base_label', 'fine_label')):
        raise ValueError('point_reference_mismatch')


def review_history(points, roots, workspace_hash):
    index = {p['point_id']: p for p in points}
    history = defaultdict(list)
    for root in roots:
        if not validate_outputs(root):
            raise ValueError('ai_review_hash_invalid')
        panels = read_json(root / 'panels.json')
        result = read_json(root / 'AI_검수결과.json')
        if panels.get('workspace_manifest_sha256') != workspace_hash or \
                result.get('schema') != 'ai_visual_type_proposals' or result.get('schema_version') != 1:
            raise ValueError('ai_review_contract_mismatch')
        seen = set()
        for row in result['items']:
            ident = row['item_id']
            if ident in seen or ident not in index:
                raise ValueError('duplicate_or_unknown_review_point')
            seen.add(ident)
            check_point(row, index[ident], 'item_id')
            if row.get('review_actor') != 'Codex_AI' or row.get('ai_visual_reviewed') is not True or \
                    any(row.get(k) is not False for k in
                        ('human_verified', 'expert_semantic_confirmation', 'eligible_for_verified_evaluation')) or \
                    row.get('proposed_subtype') is not None:
                raise ValueError('invalid_provisional_ai_origin')
            if row['status'] not in ('hold', 'suggest') or \
                    (row['status'] == 'hold' and row['proposed_base_label'] is not None) or \
                    (row['status'] == 'suggest' and row['proposed_base_label'] not in CLASSES):
                raise ValueError('invalid_ai_proposal')
            stamp = datetime.fromisoformat(row['reviewed_at'])
            if stamp.tzinfo is None or (history[ident] and stamp <=
                    datetime.fromisoformat(history[ident][-1]['reviewed_at'])):
                raise ValueError('reviews_must_be_chronological')
            history[ident].append(row | {'review_root': str(root)})
    return dict(history)


def select_points(points, quality, latest, size=128):
    """Exclude flagged/held/disputed points and crops containing a differently labelled point."""
    if set(quality) != {p['point_id'] for p in points}:
        raise ValueError('quality_coverage_mismatch')
    grouped = defaultdict(list)
    for p in points:
        check_point(quality[p['point_id']], p, 'point_id')
        if all(isinstance(p[k], (float, int)) and not isinstance(p[k], bool) and
               math.isfinite(p[k]) for k in ('x', 'y')):
            grouped[p['image_asset_id']].append(p)
    trees = {k: cKDTree([(p['x'], p['y']) for p in rows]) for k, rows in grouped.items()}
    selected, excluded = [], []
    for p in points:
        reasons = []
        if p['base_label'] not in CLASSES:
            reasons.append('unknown_provider_type')
        if str(p['area_id']) not in SPLITS:
            reasons.append('wafer_without_fixed_split')
        q = quality[p['point_id']]
        if q.get('automatic_inspected') is not True:
            reasons.append('not_automatically_inspected')
        reasons.extend('quality:' + flag for flag in q['flags'])
        ai = latest.get(p['point_id'])
        if ai and ai['status'] == 'hold':
            reasons.append('ai_hold')
        elif ai and ai['proposed_base_label'] != p['base_label']:
            reasons.append('ai_provider_disagreement')
        if not reasons:
            left, top = round(p['x'])-size//2, round(p['y'])-size//2
            neighbors = trees[p['image_asset_id']].query_ball_point(
                (left+size/2, top+size/2), size/2+1, p=np.inf)
            if any(other['point_id'] != p['point_id'] and other['base_label'] != p['base_label']
                   and left <= other['x'] < left+size and top <= other['y'] < top+size
                   for other in (grouped[p['image_asset_id']][j] for j in neighbors)):
                reasons.append('different_provider_type_in_crop')
        if reasons:
            excluded.append({'point_id': p['point_id'], 'image_asset_id': p['image_asset_id'],
                             'area_id': p['area_id'], 'reasons': reasons})
        else:
            selected.append(p)
    return selected, excluded


def build(workspace, quality_root, review_roots, output):
    workspace, quality_root, output = map(Path, (workspace, quality_root, output))
    review_roots = [Path(p) for p in review_roots]
    if not review_roots or len({p.resolve() for p in review_roots}) != len(review_roots):
        raise ValueError('distinct_ordered_review_roots_required')
    if not all(validate_outputs(p) for p in (workspace, quality_root)):
        raise ValueError('input_hash_invalid')
    contract = read_json(workspace/'workspace.json')
    workspace_hash = digest_file(workspace/'output_hashes.json')
    audit = read_json(quality_root/'quality_summary.json')
    if (contract.get('schema'), contract.get('schema_version')) != ('annotation_workspace', 2) or \
            (audit.get('schema'), audit.get('schema_version')) != ('provider_point_quality_audit', 1) or \
            audit.get('workspace_manifest_sha256') != workspace_hash:
        raise ValueError('workspace_quality_contract_mismatch')
    points = read_rows(workspace/'provider_points.jsonl')
    if len({p['point_id'] for p in points}) != len(points):
        raise ValueError('duplicate_provider_point')
    q_rows = read_rows(quality_root/'point_quality.jsonl')
    quality = {p['point_id']: p for p in q_rows}
    if len(quality) != len(q_rows):
        raise ValueError('duplicate_quality_point')
    history = review_history(points, review_roots, workspace_hash)
    latest = {k: rows[-1] for k, rows in history.items()}
    selected, excluded = select_points(points, quality, latest)
    protected = [workspace, quality_root, *review_roots,
                 Path(contract['source_root']), Path(contract['registry_root'])]
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) or
            p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise ValueError('output_must_be_new_independent_directory')
    output.mkdir(parents=True)
    (output/'기록').mkdir()
    try:
        assets = read_rows(Path(contract['registry_root'])/'assets.jsonl')
        reader = SourceReader(contract['source_root'], assets)
        grouped = defaultdict(list)
        for p in selected:
            grouped[p['image_asset_id']].append(p)
        rows, sources = [], []
        for ident, items in grouped.items():
            asset = reader.by_id[ident]
            pixels = reader.image(asset)
            if pixels.dtype not in (np.uint8, np.uint16) or pixels.ndim not in (2, 3) or \
                    (pixels.ndim == 3 and pixels.shape[2] != 3):
                raise ValueError('unsupported_raw_image')
            wafer = str(items[0]['area_id'])
            split = SPLITS[wafer]
            sources.append({'source_id': ident, 'wafer': wafer, 'split': split,
                'shape': list(pixels.shape), 'dtype': str(pixels.dtype), 'sha256': asset['sha256'],
                'locator': asset['locator'], 'phase': items[0]['phase']})
            for p in items:
                if str(p['area_id']) != wafer or p['phase'] != items[0]['phase']:
                    raise ValueError('source_group_mismatch')
                left, top = round(p['x'])-64, round(p['y'])-64
                if left < 0 or top < 0 or left+128 > pixels.shape[1] or top+128 > pixels.shape[0]:
                    raise ValueError('selected_crop_outside_source')
                patch_id = p['point_id']+'_128'
                relative = f"patches_128/{split}/{p['base_label']}/{patch_id}.tif"
                path = output/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, pixels[top:top+128, left:left+128],
                    photometric='rgb' if pixels.ndim == 3 else 'minisblack')
                ai = latest.get(p['point_id'])
                rows.append({'patch_id': patch_id, 'path': relative, 'sha256': digest_file(path),
                    'point_id': p['point_id'], 'source_id': ident, 'wafer': wafer, 'split': split,
                    'label': p['base_label'], 'provider_fine_label': p['fine_label'], 'phase': p['phase'],
                    'size': 128, 'x': p['x'], 'y': p['y'], 'left': left, 'top': top,
                    'dtype': str(pixels.dtype), 'review_status': 'pending_expert_confirmation',
                    'label_basis': 'supplied_roi_filename_research_only',
                    'ai_visual_reviewed': bool(ai), 'ai_status': ai['status'] if ai else '',
                    'ai_proposed_base_label': ai['proposed_base_label'] if ai else '',
                    'review_actor': 'Codex_AI' if ai else '',
                    'reviewer_display_name': ai['reviewer_display_name'] if ai else '',
                    'human_verified': False, 'expert_semantic_confirmation': False,
                    'eligible_for_verified_evaluation': False})
            del pixels
        reader.verify_unchanged()
        if not rows:
            raise ValueError('no_research_candidates')
        rows.sort(key=lambda r: r['patch_id'])
        write_csv(output/'기록/samples.csv', rows, list(rows[0]))
        write_json(output/'기록/sources.json', sources)
        write_jsonl(output/'기록/excluded_points.jsonl', excluded)
        write_json(output/'기록/ai_review_history.json', history)
        counts = dict(Counter(f"128/{r['split']}/{r['label']}" for r in rows))
        summary = {'schema_version': 1, 'task': 'center_point_patch_classification_candidates',
            'classes': list(CLASSES), 'patch_sizes': [128], 'patch_count': len(rows), 'counts': counts,
            'review_status': 'research_pending_provider_and_AI_labels', 'research_only': True,
            'provider_points': len(points), 'excluded_points': len(excluded),
            'exclusion_counts': dict(Counter(r for p in excluded for r in p['reasons'])),
            'unique_ai_reviewed': len(latest), 'ai_review_observations': sum(map(len, history.values())),
            'latest_ai_status_counts': dict(Counter(r['status'] for r in latest.values())),
            'included_ai_reviewed': sum(r['ai_visual_reviewed'] for r in rows),
            'wafer_splits': SPLITS, 'wafers_without_provider_points': ['3', '4', '5', '6', '7'],
            'expert_ground_truth': False, 'original_labels_modified': False,
            'subtype_training': False, 'before_after_tracking_ground_truth': False}
        write_json(output/'summary.json', summary)
        write_json(output/'provenance.json', {'manifest_sha256': digest_file(output/'기록/samples.csv'),
            'workspace_manifest_sha256': workspace_hash,
            'quality_manifest_sha256': digest_file(quality_root/'output_hashes.json'),
            'ordered_ai_review_manifests': [digest_file(p/'output_hashes.json') for p in review_roots],
            'created_at': datetime.now(UTC).isoformat(), 'size': 128,
            'selection_policy': 'all quality flags, latest AI holds/disagreements, mixed-type crops excluded',
            'normal_examples_added': False, 'coordinates_changed': False,
            'raw_pixels': 'original dtype and 128x128 crop; no markers or display stretch',
            'research_only': True, 'test_evaluated': False})
        write_json(output/'validation.json', validate(output))
        if any(read_json(output/'validation.json')['missing_classes_by_split'].values()):
            raise ValueError('classes_missing_in_split')
        summary['balanced_training'] = create_subsets(output)['sizes']['128']
        write_json(output/'summary.json', summary)
        write_json(output/'output_hashes.json', {p.relative_to(output).as_posix(): digest_file(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'output_hashes.json'})
        return summary
    except Exception as exc:
        write_json(output/'BUILD_FAILED.json', {'error': str(exc), 'status': 'failed'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'quality', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ai-review', type=Path, action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.workspace, args.quality, args.ai_review, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
