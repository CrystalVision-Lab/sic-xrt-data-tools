import json

import pytest
from PIL import Image

from sic_xrt_data_tools.ai_visual_review import finalize, validate_assessments
from sic_xrt_data_tools.annotation_workspace import write_json
from sic_xrt_data_tools.candidate_workbench import validate_outputs
from sic_xrt_data_tools.source_registry import digest_file


def test_ai_review_requires_every_observed_panel_and_keeps_unknown_types_held():
    rows=[{'blind_index':1},{'blind_index':2}]
    decisions=[{'blind_index':1,'status':'suggest','proposed_base_label':'TED','strength':'moderate','reason':'small localized contrast'},
               {'blind_index':2,'status':'hold','proposed_base_label':None,'strength':'low','reason':'ambiguous coordinate'}]
    assert len(validate_assessments(rows,decisions)) == 2
    with pytest.raises(ValueError,match='every_panel'):
        validate_assessments(rows,decisions[:1])
    with pytest.raises(ValueError,match='held_item'):
        validate_assessments(rows,[decisions[0],decisions[1] | {'proposed_base_label':'TSD'}])
    for changes in ({'human_verified':True},{'expert_semantic_confirmation':True},{'proposed_subtype':'a'}):
        with pytest.raises(ValueError,match='ai_cannot_claim'):
            validate_assessments(rows,[decisions[0] | changes,decisions[1]])
    for changes in ({'x':99}, {'fine_label':'TED_a'}, {'review_actor':'양희승'}, {'eligible_for_verified_evaluation':True}):
        with pytest.raises(ValueError,match='cannot_override'):
            validate_assessments(rows,[decisions[0] | changes,decisions[1]])
    for value in (True, '1', 1.0):
        with pytest.raises(ValueError,match='invalid_assessment_index'):
            validate_assessments(rows,[decisions[0] | {'blind_index':value},decisions[1]])


def test_finalized_ai_review_preserves_provider_labels_and_rejects_modified_panels(tmp_path):
    output=tmp_path/'output'
    (output/'patches').mkdir(parents=True)
    image=output/'patches/001.png'
    Image.new('RGB',(32,32),(120,100,80)).save(image)
    original_sha=digest_file(image)
    point={'blind_index':1,'item_id':'pt-a','area_id':'1','phase':'before','x':108,'y':208,
           'marker_x':8,'marker_y':8,'patch_file':'patches/001.png','patch_sha256':original_sha,
           'base_label':'TED','fine_label':'TED_a'}
    write_json(output/'panels.json',{'requested_reviewer_name':'양희승','items':[point]})
    write_json(output/'output_hashes.json',{p.relative_to(output).as_posix():digest_file(p) for p in output.rglob('*') if p.is_file()})
    decisions=tmp_path/'decisions.json'
    write_json(decisions,[{'blind_index':1,'status':'suggest','proposed_base_label':'TSD','strength':'low','reason':'round contrast'}])
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='panels_changed'):
        finalize(output,decisions)
    Image.new('RGB',(32,32),(120,100,80)).save(image)
    assert digest_file(image)==original_sha
    summary=finalize(output,decisions)
    result=json.loads((output/'AI_검수결과.json').read_text(encoding='utf-8'))['items'][0]
    assert summary['provider_base_type_disagreements']==[1]
    assert result['fine_label']=='TED_a' and result['base_label']=='TED'
    assert result['proposed_base_label']=='TSD' and result['proposed_subtype'] is None
    assert result['review_actor']=='Codex_AI' and result['reviewer_display_name']=='양희승'
    assert result['ai_visual_reviewed'] is True
    assert not result['human_verified'] and not result['expert_semantic_confirmation'] and not result['eligible_for_verified_evaluation']
    assert digest_file(image)==original_sha and validate_outputs(output)
    assert 'marked/001.png' in (output/'AI_타입검수.html').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='already_finalized'):
        finalize(output,decisions)
