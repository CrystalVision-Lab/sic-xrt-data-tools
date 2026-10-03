import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from sic_xrt_data_tools.ai_visual_review import prepare
from sic_xrt_data_tools.annotation_workspace import write_jsonl
from sic_xrt_data_tools.point_dataset import write_json
from sic_xrt_data_tools.reviewed_research_dataset import (
    build,
    review_history,
    select_points,
)
from sic_xrt_data_tools.source_registry import digest_file


def seal(root):
    write_json(root/'output_hashes.json', {p.relative_to(root).as_posix(): digest_file(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'})


def point(ident, label='TED', x=200, y=200, wafer='1', image='image'):
    return {'point_id': ident, 'base_label': label, 'fine_label': label+'_a',
            'image_asset_id': image, 'area_id': wafer, 'phase': 'before', 'x': x, 'y': y}


def quality(rows):
    return {p['point_id']: p | {'flags': [], 'automatic_inspected': True} for p in rows}


def ai(p, label=None, time='2026-10-03T00:00:00+00:00'):
    return p | {'item_id': p['point_id'], 'status': 'suggest' if label else 'hold',
        'proposed_base_label': label, 'proposed_subtype': None, 'review_actor': 'Codex_AI',
        'reviewer_display_name': '양희승', 'reviewed_at': time, 'ai_visual_reviewed': True,
        'human_verified': False, 'expert_semantic_confirmation': False,
        'eligible_for_verified_evaluation': False}


def test_screen_uses_latest_review_and_all_neighbors_including_excluded():
    rows = [point('keep'), point('hold', x=400), point('disagree', x=600),
            point('mixed', x=800), point('bad_neighbor', label='TSD', x=810)]
    q = quality(rows)
    q['bad_neighbor']['flags'] = ['color_band_suspected']
    selected, excluded = select_points(rows, q, {'hold': ai(rows[1]),
                                                'disagree': ai(rows[2], 'TSD')})
    assert [p['point_id'] for p in selected] == ['keep']
    reasons = {r['point_id']: r['reasons'] for r in excluded}
    assert reasons['hold'] == ['ai_hold']
    assert reasons['disagree'] == ['ai_provider_disagreement']
    assert reasons['mixed'] == ['different_provider_type_in_crop']
    with pytest.raises(ValueError, match='coverage'):
        select_points(rows, {}, {})
    q['keep']['x'] += 1
    with pytest.raises(ValueError, match='reference'):
        select_points(rows, q, {})


def test_review_reinspection_is_append_only_and_cannot_claim_human_or_subtype(tmp_path):
    p = point('p')
    roots = []
    for i, label in enumerate((None, 'TED')):
        root = tmp_path/str(i)
        root.mkdir()
        write_json(root/'panels.json', {'workspace_manifest_sha256': 'work'})
        write_json(root/'AI_검수결과.json', {'schema': 'ai_visual_type_proposals', 'schema_version': 1,
            'items': [ai(p, label, f'2026-10-03T00:0{i}:00+00:00')]})
        seal(root)
        roots.append(root)
    history = review_history([p], roots, 'work')
    assert len(history['p']) == 2 and history['p'][0]['status'] == 'hold'
    assert history['p'][-1]['status'] == 'suggest'
    with pytest.raises(ValueError, match='chronological'):
        review_history([p], roots[::-1], 'work')
    result = json.loads((roots[1]/'AI_검수결과.json').read_text(encoding='utf-8'))
    result['items'][0]['human_verified'] = True
    write_json(roots[1]/'AI_검수결과.json', result)
    seal(roots[1])
    with pytest.raises(ValueError, match='provisional'):
        review_history([p], roots, 'work')


def test_build_preserves_unmarked_raw_dtype_and_wafer_groups(tmp_path):
    source, registry, work, audit, review = [tmp_path/n for n in ('source','registry','work','audit','review')]
    for root in (source, registry, work, audit, review):
        root.mkdir()
    assets, rows = [], []
    pixels = np.arange(512*768, dtype=np.uint16).reshape(512, 768)
    for wafer in ('1', '2', '8', '9'):
        filename = wafer+'.tif'
        tifffile.imwrite(source/filename, pixels)
        assets.append({'asset_id': wafer, 'locator': filename, 'sha256': digest_file(source/filename),
                       'source_kind': 'file', 'suffix': '.tif', 'metadata': {'page_count': 1}})
        for i, label in enumerate(('BPD', 'TED', 'TSD')):
            rows.append(point(wafer+label, label=label, x=128+256*i, y=256, wafer=wafer, image=wafer))
    write_jsonl(registry/'assets.jsonl', assets)
    write_json(work/'workspace.json', {'schema': 'annotation_workspace', 'schema_version': 2,
               'source_root': str(source), 'registry_root': str(registry),
               'summary': {'requested_reviewer_name': '양희승'}})
    write_jsonl(work/'provider_points.jsonl', rows)
    seal(work)
    whash = digest_file(work/'output_hashes.json')
    write_json(audit/'quality_summary.json', {'schema': 'provider_point_quality_audit',
               'schema_version': 1, 'workspace_manifest_sha256': whash})
    write_jsonl(audit/'point_quality.jsonl', list(quality(rows).values()))
    seal(audit)
    write_json(review/'panels.json', {'workspace_manifest_sha256': whash})
    write_json(review/'AI_검수결과.json', {'schema': 'ai_visual_type_proposals', 'schema_version': 1,
               'items': [ai(rows[0], 'BPD')]})
    seal(review)
    output = tmp_path/'output'
    summary = build(work, audit, [review], output)
    assert summary['patch_count'] == 12 and summary['included_ai_reviewed'] == 1
    assert summary['balanced_training']['total'] == 6
    assert summary['wafer_splits'] == {'1': 'train', '9': 'train', '2': 'val', '8': 'test'}
    path = next((output/'patches_128/test/BPD').glob('*.tif'))
    patch = tifffile.imread(path)
    assert patch.dtype == np.uint16
    np.testing.assert_array_equal(patch, pixels[192:320, 64:192])
    assert json.loads((output/'validation.json').read_text())['files_checked'] == 12
    assert summary['expert_ground_truth'] is False
    with pytest.raises(ValueError, match='new_independent'):
        build(work, audit, [review], output)
    # Wider diagnostic panels must retain original coordinates and unmarked pixels at clipped edges.
    candidates, plan = tmp_path/'candidates', tmp_path/'plan'
    candidates.mkdir()
    plan.mkdir()
    for name in ('candidates_2d.jsonl', 'candidates_3d.jsonl'):
        write_jsonl(candidates/name, [])
    write_json(candidates/'candidate_summary.json', {'schema': 'candidate_workbench',
        'schema_version': 1, 'workspace_manifest_sha256': whash})
    seal(candidates)
    write_json(plan/'review_plan.json', {'schema': 'review_priority_plan', 'schema_version': 1,
        'workspace_manifest_sha256': whash, 'candidate_manifest_sha256': digest_file(candidates/'output_hashes.json'),
        'selected_count': 1, 'items': [{'item_id': rows[0]['point_id']}]})
    seal(plan)
    wide = tmp_path/'wide'
    prepare(work, candidates, plan, wide, context_size=512)
    panels = json.loads((wide/'panels.json').read_text(encoding='utf-8'))
    assert panels['context_size'] == 512 and panels['display_gray_stretch'] is True
    assert panels['items'][0]['x'] == 128 and panels['items'][0]['marker_x'] == 128
    patch = np.asarray(Image.open(wide/'patches/001.png'))
    display = (pixels[:, :384].astype(np.float32)*255/65535).astype(np.uint8)
    np.testing.assert_array_equal(patch, np.repeat(display[:, :, None], 3, axis=2))
