import json

import numpy as np
import pytest
from PIL import Image

from sic_xrt_data_tools.annotation_workspace import write_json, write_jsonl
from sic_xrt_data_tools.candidate_workbench import validate_outputs
from sic_xrt_data_tools.provider_quality import audit, coordinate_flags, point_metrics
from sic_xrt_data_tools.source_registry import digest_file


def test_contrast_alerts_distinguish_center_from_offset_without_moving_coordinates():
    base=np.full((160,160,3),100,dtype=np.uint8)
    center=base.copy()
    center[76:85,76:85]=255
    shifted=base.copy()
    shifted[76:85,99:108]=255
    stored=center.copy()
    c=point_metrics(center,80,80)
    s=point_metrics(shifted,80,80)
    f=point_metrics(base,80,80)
    assert 'center_contrast_not_clear' not in c['flags']
    assert c['nearest_contrast_peak_distance_px']<12
    assert 'contrast_offset_suspected' in s['flags']
    assert s['nearest_contrast_peak_distance_px']>12
    assert 'center_contrast_not_clear' in f['flags']
    assert f['nearest_contrast_peak_distance_px'] is None
    assert np.array_equal(center,stored)
    assert point_metrics(center.astype(np.uint16)*257,80,80)['center_contrast_abs']==pytest.approx(c['center_contrast_abs'],abs=1e-6)


def test_color_policy_does_not_flag_ordinary_brown_and_keeps_boundary_points():
    brown=np.full((160,160,3),(130,110,80),dtype=np.uint8)
    band=brown.copy()
    band[64:100]=(80,150,170)
    assert 'color_band_suspected' not in point_metrics(brown,80,80)['flags']
    assert 'color_band_suspected' in point_metrics(band,80,80)['flags']
    edge=point_metrics(brown,2.5,4.25)
    assert edge['crop_left']==0 and edge['crop_top']==0
    assert 'context_clipped' in edge['flags']
    assert point_metrics(brown,160,80)['flags']==['coordinate_invalid']
    assert point_metrics(brown,True,80)['flags']==['coordinate_invalid']
    with pytest.raises(ValueError,match='unsupported_raw'):
        point_metrics(brown.astype(np.float32),80,80)


def test_coordinate_conflicts_are_exact_annotation_alerts():
    rows=[{'point_id':'a','image_asset_id':'i','x':1,'y':2,'fine_label':'TED_a'},
          {'point_id':'b','image_asset_id':'i','x':1,'y':2,'fine_label':'TSD_a'},
          {'point_id':'c','image_asset_id':'i','x':2,'y':2,'fine_label':'TED_a'}]
    alerts=coordinate_flags(rows)
    assert alerts['a']==alerts['b']==['duplicate_position','coordinate_label_conflict']
    assert not alerts['c']


def test_whole_audit_is_read_only_and_does_not_promote_unseen_points(tmp_path):
    source,registry,workspace,visual,output=[tmp_path/n for n in ('source','registry','workspace','visual','output')]
    for p in (source,registry,workspace,visual):
        p.mkdir()
    image=source/'raw.png'
    pixels=np.full((160,160,3),100,dtype=np.uint8)
    pixels[76:85,76:85]=255
    Image.fromarray(pixels).save(image)
    original_sha=digest_file(image)
    asset={'asset_id':'image','source_kind':'file','locator':'raw.png','sha256':original_sha,'suffix':'.png','metadata':{'page_count':1}}
    write_jsonl(registry/'assets.jsonl',[asset])
    points=[{'point_id':str(n),'image_asset_id':'image','area_id':'1','phase':'before','x':x,'y':80,
             'fine_label':'TED_a','base_label':'TED','human_verified':False} for n,x in [(1,80),(2,120)]]
    write_jsonl(workspace/'provider_points.jsonl',points)
    write_json(workspace/'workspace.json',{'schema':'annotation_workspace','schema_version':2,'source_root':str(source),'registry_root':str(registry),'summary':{'point_count':2}})
    write_json(workspace/'output_hashes.json',{p.name:digest_file(p) for p in workspace.iterdir() if p.is_file()})
    observed=points[0]|{'item_id':'1','status':'suggest','proposed_base_label':'TED'}
    write_json(visual/'AI_검수결과.json',{'schema':'ai_visual_type_proposals','schema_version':1,'items':[observed]})
    write_json(visual/'panels.json',{'workspace_manifest_sha256':digest_file(workspace/'output_hashes.json')})
    write_json(visual/'output_hashes.json',{p.name:digest_file(p) for p in visual.iterdir() if p.is_file()})
    summary=audit(workspace,visual,output)
    rows=[json.loads(s) for s in (output/'point_quality.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['automatic_inspected']==2 and summary['ai_visual_reviewed']==1
    assert rows[0]['ai_visual_reviewed'] is True and rows[1]['ai_visual_reviewed'] is False
    assert all(not r['training_ready'] and not r['eligible_for_verified_evaluation'] and not r['human_verified'] for r in rows)
    assert [(r['x'],r['fine_label']) for r in rows]==[(r['x'],r['fine_label']) for r in points]
    assert digest_file(image)==original_sha and validate_outputs(output)
    with pytest.raises(ValueError,match='independent_directory'):
        audit(workspace,visual,output)
    (visual/'panels.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='input_hash_invalid'):
        audit(workspace,visual,tmp_path/'output-2')

