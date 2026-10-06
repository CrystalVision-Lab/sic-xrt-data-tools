import copy
import json

import numpy as np
import pytest
import tifffile

from sic_xrt_data_tools.feedback_dataset import (
    build,
    digest,
    empty_target,
    save,
    validate_review,
)


def review(tmp_path, wafer='1'):
    source=tmp_path/'raw.tif'
    tifffile.imwrite(source,np.random.default_rng(42).integers(0,256,(512,512,3),dtype=np.uint8),photometric='rgb')
    return {'schema':'xrt_feedback_v1','session_id':'fixture-session','wafer_id':wafer,'events':[],
        'expert_ground_truth':False,'exhaustive_annotations':False,
        'source':{'file_path':str(source),'sha256':digest(source),'page_index':0,'width':512,'height':512,
                  'coordinate_space':'raw_pixel_xy','orientation':'encoded_no_exif_rotation'}}


def event(data,ident,action,original_x,original_y,**target_changes):
    previous=next((e for e in reversed(data['events']) if e['record_id']==ident),None)
    original=previous['original'] if previous else {'candidate_id':ident,'x':original_x,'y':original_y,'predicted_type':'TSD',
        'analysis_id':'fixture-run','model_sha256':'b'*64,'model_id':'fixture'}
    prior=copy.deepcopy(previous['target'] if previous else empty_target(original))
    target={**prior,**target_changes}
    data['events'].append({'id':str(len(data['events'])),'record_id':ident,'revision':previous['revision']+1 if previous else 1,
        'original':original,'previous_target':prior,'target':target,'action':action,'reviewer':'test reviewer',
        'actor':'human','created_at':'2026-10-06T00:00:00+00:00','note':''})


def test_explicit_decisions_unknown_types_corrected_offsets_and_duplicate_exclusion(tmp_path):
    data=review(tmp_path)
    event(data,'known','confirm',128.,128.,objectness=True)
    event(data,'known','location',128.,128.,x=134.5,y=127.,location_confirmed=True)
    event(data,'known','type',128.,128.,label='TED')
    event(data,'unknown','confirm',250.,250.,objectness=True)
    event(data,'background','background',350.,350.,objectness=False)
    event(data,'near','background',130.,130.,objectness=False)
    event(data,'hold','defer',400.,400.)
    event(data,'duplicate','duplicate',128.,128.,duplicate_of='known')
    path=tmp_path/'feedback.json';save(path,data)
    info=build([path],tmp_path/'dataset')
    samples=json.loads((tmp_path/'dataset/samples.json').read_text(encoding='utf-8'))
    assert info['sample_count']==4
    assert info['masks']=={'objectness_mask':4,'type_mask':2,'location_mask':2}
    anchor=next(r for r in samples if r['record_id'].endswith(':known') and r['role']=='original')
    assert anchor['offset_xy']==[6.5,-1.]
    unknown=next(r for r in samples if r['record_id'].endswith(':unknown'))
    assert unknown['type'] is None and unknown['type_mask'] is False and unknown['location_mask'] is False
    reasons=json.loads((tmp_path/'dataset/excluded.json').read_text(encoding='utf-8'))
    assert {r['reason'] for r in reasons}=={'duplicate','deferred_or_undone','background_near_confirmed_defect'}
    assert info['test_evaluated'] is False and info['expert_ground_truth'] is False
    with pytest.raises(ValueError,match='new directory'):
        build([path],tmp_path/'dataset')


def test_holdout_hash_and_fork_rejected(tmp_path):
    data=review(tmp_path,'8')
    with pytest.raises(ValueError,match='8'):
        validate_review(data)
    data['wafer_id']='1'
    event(data,'point','confirm',200.,200.,objectness=True)
    path=tmp_path/'one.json';save(path,data)
    fork=copy.deepcopy(data);fork['events'][0]['target']['objectness']=False;fork['events'][0]['action']='background'
    second=tmp_path/'two.json';save(second,fork)
    with pytest.raises(ValueError,match='Conflicting'):
        build([path,second],tmp_path/'fork')
    data['source']['sha256']='a'*64;save(path,data)
    with pytest.raises(ValueError,match='hash changed'):
        build([path],tmp_path/'wrong-hash')


def test_undo_add_and_tampered_history(tmp_path):
    data=review(tmp_path)
    original={'candidate_id':None,'x':200.,'y':200.,'predicted_type':None,'analysis_id':None,'model_sha256':None,'model_id':None}
    prior=empty_target(original)
    data['events']=[{'id':'e1','record_id':'manual','revision':1,'original':original,'previous_target':prior,
        'target':{**prior,'objectness':True,'location_confirmed':True},'action':'add','reviewer':'test reviewer','actor':'human',
        'created_at':'2026-10-06T00:00:00+00:00','note':''}]
    event(data,'manual','undo',200.,200.,objectness=None,location_confirmed=False)
    assert validate_review(data)['manual']['target']['objectness'] is None
    data['events'][-1]['target']['label']='TSD'
    with pytest.raises(ValueError,match='Contradictory'):
        validate_review(data)
