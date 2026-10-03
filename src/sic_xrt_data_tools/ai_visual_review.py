"""Prepare blind visual panels and record explicitly provisional AI base-type suggestions."""
import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .annotation_workspace import SourceReader, write_json
from .candidate_workbench import validate_outputs
from .review_decisions import input_index
from .source_registry import digest_file


def prepare(workspace, candidates, plan_root, output, context_size=256):
    if type(context_size) is not int or context_size not in (256,512):
        raise ValueError('context_size_must_be_256_or_512')
    workspace, candidates, plan_root, output = map(Path,(workspace,candidates,plan_root,output))
    index, contract = input_index(workspace,candidates)
    if not validate_outputs(plan_root):
        raise ValueError('plan_hash_invalid')
    plan = json.loads((plan_root/'review_plan.json').read_text(encoding='utf-8'))
    if (plan['schema'],plan['schema_version']) != ('review_priority_plan',1) or \
            plan['workspace_manifest_sha256'] != digest_file(workspace/'output_hashes.json') or \
            plan['candidate_manifest_sha256'] != digest_file(candidates/'output_hashes.json'):
        raise ValueError('plan_input_mismatch')
    ids = [p['item_id'] for p in plan['items']]
    if len(ids)!=len(set(ids)) or len(ids)!=plan['selected_count'] or not set(ids).issubset(index):
        raise ValueError('invalid_plan_ids')
    protected = (Path(contract['source_root']),workspace,candidates,plan_root,Path(contract['registry_root']))
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise ValueError('output_must_be_new_independent_directory')
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    assets = [json.loads(s) for s in (Path(contract['registry_root'])/'assets.jsonl').read_text(encoding='utf-8').splitlines()]
    reader = SourceReader(contract['source_root'],assets)
    grouped, rows = defaultdict(list), []
    for n, ident in enumerate(ids,1):
        row = index[ident] | {'blind_index':n}
        rows.append(row)
        grouped[row['image_asset_id']].append(row)
    for asset_id, points in grouped.items():
        pixels = reader.image(reader.by_id[asset_id])
        if pixels.dtype not in (np.uint8,np.uint16):
            raise ValueError('unsupported_raw_dtype')
        for p in points:
            x,y = p['x'],p['y']
            if not (math.isfinite(x) and math.isfinite(y) and 0<=x<pixels.shape[1] and 0<=y<pixels.shape[0]):
                raise ValueError('invalid_coordinate_for_panel')
            half=context_size//2
            raw = pixels[max(0,round(y)-half):min(pixels.shape[0],round(y)+half), max(0,round(x)-half):min(pixels.shape[1],round(x)+half)]
            left,top = max(0,round(x)-half),max(0,round(y)-half)
            view = raw if raw.dtype==np.uint8 else (raw.astype(np.float32)*255/65535).astype(np.uint8)
            image = Image.fromarray(view).convert('RGB')
            name = f"patches/{p['blind_index']:03d}.png"
            image.save(output/name)
            p.update(patch_file=name, patch_sha256=digest_file(output/name),
                     crop_left=left,crop_top=top,marker_x=x-left,marker_y=y-top)
        del pixels
    reader.verify_unchanged()
    for start in range(0,len(rows),16):
        tile_width=560 if context_size==512 else 416
        sheet = Image.new('RGB',(4*tile_width,4*316),'#111827')
        draw = ImageDraw.Draw(sheet)
        for slot,p in enumerate(rows[start:start+16]):
            ox,oy=(slot%4)*tile_width,(slot//4)*316
            draw.text((ox+8,oy+6),f"{p['blind_index']:03d} | W{p['area_id']} {p['phase']}",fill='white')
            image=Image.open(output/p['patch_file']).convert('RGB')
            marked=image.copy()
            pen=ImageDraw.Draw(marked)
            xx,yy=p['marker_x'],p['marker_y']
            for color,width in [('black',5),('#ffeb3b',1)]:
                for arm in [(xx-16,yy,xx-6,yy),(xx+6,yy,xx+16,yy),(xx,yy-16,xx,yy-6),(xx,yy+6,xx,yy+16)]:
                    pen.line(arm,fill=color,width=width)
            if context_size==512:
                marked=marked.resize((marked.width//2,marked.height//2),Image.Resampling.NEAREST)
            sheet.paste(marked,(ox+8,oy+32))
            zx,zy=max(0,round(xx)-32),max(0,round(yy)-32)
            zoom=image.crop((zx,zy,zx+64,zy+64)).resize((128,128),Image.Resampling.NEAREST)
            # Unmarked zoom preserves the actual center texture.
            sheet.paste(zoom,(ox+276,oy+32))
            draw.text((ox+276,oy+169),'64px crop / 2x',fill='#cbd5e1')
            if context_size==512:
                gray=np.asarray(zoom).mean(axis=2)
                lo,hi=np.percentile(gray,(1,99))
                stretched=np.clip((gray-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)
                sheet.paste(Image.fromarray(stretched).convert('RGB'),(ox+416,oy+32))
                draw.text((ox+416,oy+169),'DISPLAY stretch',fill='#cbd5e1')
                draw.text((ox+8,oy+271),'512px context / 0.5x display',fill='#cbd5e1')
            draw.text((ox+8,oy+293),f"original xy {p['x']:.2f},{p['y']:.2f}",fill='#cbd5e1')
        sheet.save(output/f"blind_{start//16+1:02d}.png")
    manifest={'schema':'ai_visual_review_panels','schema_version':1,'items':rows,
        'workspace_manifest_sha256':digest_file(workspace/'output_hashes.json'),
        'candidate_manifest_sha256':digest_file(candidates/'output_hashes.json'),
        'plan_manifest_sha256':digest_file(plan_root/'output_hashes.json'),
        'requested_reviewer_name':contract['summary']['requested_reviewer_name'],
        'blind_to_existing_labels_in_panels':True,'originals_changed':False,
        'context_size':context_size,'display_gray_stretch':context_size==512}
    write_json(output/'panels.json',manifest)
    write_json(output/'output_hashes.json',{p.relative_to(output).as_posix():digest_file(p) for p in output.rglob('*') if p.is_file()})
    return len(rows)


def validate_assessments(rows, decisions):
    if not isinstance(decisions, list) or any(not isinstance(d, dict) or type(d.get('blind_index')) is not int for d in decisions):
        raise ValueError('invalid_assessment_index')
    if {d.get('blind_index') for d in decisions}!={r['blind_index'] for r in rows} or len(decisions)!=len(rows):
        raise ValueError('every_panel_requires_one_assessment')
    lookup={d['blind_index']:d for d in decisions}
    for d in lookup.values():
        if d.get('status') not in {'suggest','hold'} or d.get('proposed_base_label') not in {None,'BPD','TED','TSD'}:
            raise ValueError('invalid_provisional_base_type')
        if (d['status']=='suggest') != (d['proposed_base_label'] is not None):
            raise ValueError('held_item_must_not_have_type')
        if d.get('strength') not in {'low','moderate'} or not isinstance(d.get('reason'),str) or not d['reason'].strip():
            raise ValueError('reason_and_nonprobabilistic_strength_required')
        if d.get('human_verified') or d.get('expert_semantic_confirmation') or d.get('proposed_subtype'):
            raise ValueError('ai_cannot_claim_verified_or_subtype')
        if set(d) - {'blind_index', 'status', 'proposed_base_label', 'strength', 'reason'}:
            raise ValueError('assessment_cannot_override_source_or_review_metadata')
    return lookup


def finalize(output, decisions_file):
    output=Path(output)
    if (output/'AI_검수결과.json').exists() or not validate_outputs(output):
        raise ValueError('panels_changed_or_already_finalized')
    panels=json.loads((output/'panels.json').read_text(encoding='utf-8'))
    decisions=json.loads(Path(decisions_file).read_text(encoding='utf-8'))
    lookup=validate_assessments(panels['items'],decisions)
    rows=[]
    for p in panels['items']:
        d=lookup[p['blind_index']]
        rows.append(p | d | {'review_actor':'Codex_AI','reviewer_display_name':panels['requested_reviewer_name'],
            'reviewed_at':datetime.now(UTC).isoformat(),'ai_visual_reviewed':True,
            'human_verified':False,'expert_semantic_confirmation':False,'eligible_for_verified_evaluation':False,
            'proposed_subtype':None,'type_identification':'provisional_image_morphology',
            'provider_agreement':None if d['status']=='hold' or p['base_label']=='unknown' else d['proposed_base_label']==p['base_label']})
    summary={'reviewed':len(rows),'suggested':sum(r['status']=='suggest' for r in rows),
        'held':sum(r['status']=='hold' for r in rows),'suggested_types':dict(Counter(r['proposed_base_label'] for r in rows if r['status']=='suggest')),
        'provider_base_type_disagreements':[r['blind_index'] for r in rows if r['provider_agreement'] is False],
        'human_verified_created':0,'original_labels_modified':False,'training_ready':False,
        'scope':'diagnostic provider points; not whole-wafer or 3D certification'}
    (output/'marked').mkdir()
    for r in rows:
        with Image.open(output/r['patch_file']) as raw:
            marked=raw.convert('RGB')
        pen=ImageDraw.Draw(marked)
        xx,yy=r['marker_x'],r['marker_y']
        for color,width in [('black',5),('#ffeb3b',1)]:
            for arm in [(xx-16,yy,xx-6,yy),(xx+6,yy,xx+16,yy),(xx,yy-16,xx,yy-6),(xx,yy+6,xx,yy+16)]:
                pen.line(arm,fill=color,width=width)
        r['marked_patch_file']=f"marked/{r['blind_index']:03d}.png"
        r['patch_width'],r['patch_height']=marked.size
        marked.save(output/r['marked_patch_file'])
    write_json(output/'AI_검수결과.json',{'schema':'ai_visual_type_proposals','schema_version':1,'summary':summary,'items':rows})
    with (output/'AI_검수결과.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['blind_index','item_id','area_id','phase','fine_label','status','proposed_base_label','strength','reason','provider_agreement','review_actor','reviewer_display_name','human_verified']
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    cards=[]
    for r in rows:
        e=html.escape
        phase={'before':'열처리 전','after':'열처리 후'}.get(r['phase'],r['phase'])
        strength={'low':'판독 근거 약함','moderate':'형태가 비교적 분명함'}[r['strength']]
        cards.append(f"<article id='item-{r['blind_index']}' data-status='{e(r['status'])}' data-agreement='{str(r['provider_agreement']).lower()}'><h3>{r['blind_index']:03d} · 웨이퍼 {e(str(r['area_id']))} · {e(phase)}</h3><p>제공자 {e(r['fine_label'])} → AI {e(r['proposed_base_label'] or '보류')} · {e(strength)}</p><div class='pictures'><figure><figcaption>원본 패치</figcaption><img width='{r['patch_width']}' src='{e(r['patch_file'])}'></figure><figure><figcaption>같은 패치 · 좌표 표시</figcaption><img width='{r['patch_width']}' src='{e(r['marked_patch_file'])}'></figure></div><p>원본 좌표 ({r['x']:.2f}, {r['y']:.2f})</p><p>{e(r['reason'])}</p><a href='blind_{(r['blind_index']-1)//16+1:02d}.png'>라벨을 숨기고 판독한 원본·확대 패널 보기</a><p><small>{e(r['item_id'])}</small></p></article>")
    body="<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI 타입 검수 제안</title><style>body{font-family:Malgun Gothic,sans-serif;background:#111827;color:white;margin:24px;line-height:1.6}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:#1f2937;padding:16px}article[hidden]{display:none}.pictures{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{max-width:100%;height:auto;image-rendering:pixelated}small{overflow-wrap:anywhere}a{color:#93c5fd}select{padding:8px;font-size:16px}</style>"
    body+=f"<h1>AI 타입 검수 제안 · {len(rows)}건</h1><p>표시 이름 {html.escape(panels['requested_reviewer_name'])} · 실제 판독 Codex_AI. 기본 종류의 영상 형태 제안이며 전문가 확정·사람 검수가 아닙니다. 기존 라벨·사람 검수 기록·학습 모델은 변경하지 않았습니다.</p>"
    body+=f"<p>종류 제안 {summary['suggested']}건 · 보류 {summary['held']}건 · 제공자 기본 타입과 다른 제안 {len(summary['provider_base_type_disagreements'])}건</p><p>대표 주석 표본의 판독 결과입니다. 웨이퍼 전체·3D·전후 추적과 세부 a~f의 검수 완료를 뜻하지 않습니다. 판독 강도는 정확도나 확률이 아닙니다. 좌우는 같은 패치이며 오른쪽의 노란 표시 중심이 대상입니다.</p><p>참고: <a href='https://www.sciencedirect.com/science/article/abs/pii/S0925963523005174'>Harada et al., 2023</a>. 현재 자료만으로 세부 문자와 물리 타입을 확정하지 않습니다.</p><label>보기 <select id='filter'><option value='all'>전체</option><option value='suggest'>종류 제안</option><option value='hold'>보류</option><option value='different'>제공자 기본 타입과 다른 제안</option></select></label><section>"+''.join(cards)+"</section><script>document.getElementById('filter').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>{a.hidden=e.target.value==='different'?a.dataset.agreement!=='false':e.target.value!=='all'&&a.dataset.status!==e.target.value;});});</script></html>"
    (output/'AI_타입검수.html').write_text(body,encoding='utf-8')
    write_json(output/'output_hashes.json',{p.relative_to(output).as_posix():digest_file(p) for p in output.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','finalize'])
    parser.add_argument('--context-size',type=int,default=256)
    for name in ('workspace','candidates','plan','decisions','output'):
        parser.add_argument('--'+name,type=Path,required=name=='output')
    a=parser.parse_args()
    if a.action=='prepare':
        print(prepare(a.workspace,a.candidates,a.plan,a.output,a.context_size))
    else:
        print(json.dumps(finalize(a.output,a.decisions),ensure_ascii=False))


if __name__=='__main__':
    main()
