"""Audit supplied point imagery without changing labels or claiming type verification."""
import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, laplace, maximum_filter

from .annotation_workspace import SourceReader, write_json, write_jsonl
from .candidate_workbench import validate_outputs
from .source_registry import digest_file

POLICY = {
    'context_half_size_px':64, 'contrast_half_size_px':32, 'center_radius_px':12,
    'cyan_purple_fraction':0.10, 'cyan_purple_max_row_fraction':0.50,
    'low_detail_laplacian_energy':0.00008, 'contrast_min_abs':0.015,
    'contrast_noise_multiplier':4.0,
}
FLAG_NAMES = {
    'coordinate_invalid':'좌표 범위 오류',
    'context_clipped':'패치가 원본 경계에 닿음',
    'color_band_suspected':'청록·보라색 영역/띠 의심',
    'low_local_detail':'국소 세부 대비 낮음',
    'center_contrast_not_clear':'중심 주변의 뚜렷한 대비를 찾지 못함',
    'contrast_offset_suspected':'검출된 대비가 중심에서 떨어짐',
    'duplicate_position':'같은 좌표에 주석 중복',
    'coordinate_label_conflict':'같은 좌표에 서로 다른 타입 주석',
    'unknown_provider_type':'제공 자료의 타입 미상',
}
WEIGHTS = {'coordinate_invalid':100, 'context_clipped':10, 'color_band_suspected':60,
           'low_local_detail':20, 'center_contrast_not_clear':35, 'contrast_offset_suspected':45,
           'duplicate_position':50, 'coordinate_label_conflict':90, 'unknown_provider_type':70}


def valid_coordinate(x,y,width,height):
    return all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (x,y)) and 0<=x<width and 0<=y<height


def point_metrics(pixels,x,y):
    if pixels.dtype not in (np.uint8,np.uint16) or pixels.ndim not in (2,3) or (pixels.ndim==3 and pixels.shape[2] not in (3,4)):
        raise ValueError('unsupported_raw_image')
    height,width=pixels.shape[:2]
    if not valid_coordinate(x,y,width,height):
        return {'flags':['coordinate_invalid'],'automatic_inspected':False}
    half=POLICY['context_half_size_px']
    left,top=max(0,round(x)-half),max(0,round(y)-half)
    right,bottom=min(width,round(x)+half),min(height,round(y)+half)
    patch=pixels[top:bottom,left:right].astype(np.float32)/(255 if pixels.dtype==np.uint8 else 65535)
    if patch.ndim==3:
        rgb=patch[:,:,:3]
        gray=rgb.mean(axis=2)
        r,g,b=np.moveaxis(rgb,-1,0)
        abnormal=((b-g>12/255)&(r-g>8/255))|((g-r>10/255)&(b-r>10/255))
        color_fraction=float(abnormal.mean())
        row_fraction=float(abnormal.mean(axis=1).max())
    else:
        gray=patch
        color_fraction=row_fraction=0.0
    flags=[]
    if gray.shape!=(2*half,2*half):
        flags.append('context_clipped')
    if color_fraction>=POLICY['cyan_purple_fraction'] and row_fraction>=POLICY['cyan_purple_max_row_fraction']:
        flags.append('color_band_suspected')
    detail=float(np.mean(laplace(gray)**2))
    if detail<POLICY['low_detail_laplacian_energy']:
        flags.append('low_local_detail')
    xx,yy=x-left,y-top
    half=POLICY['contrast_half_size_px']
    cx,cy=max(0,round(xx)-half),max(0,round(yy)-half)
    local=gray[cy:min(gray.shape[0],round(yy)+half),cx:min(gray.shape[1],round(xx)+half)]
    response=gaussian_filter(local,1.0)-gaussian_filter(local,4.0)
    magnitude=np.abs(response)
    noise=float(np.median(np.abs(response-np.median(response)))*1.4826)
    threshold=max(POLICY['contrast_min_abs'],POLICY['contrast_noise_multiplier']*noise)
    grid_y,grid_x=np.indices(local.shape)
    distances=np.hypot(grid_x+cx-xx,grid_y+cy-yy)
    center=distances<=POLICY['center_radius_px']
    support=float(magnitude[center].max()) if center.any() else 0.0
    if support<threshold:
        flags.append('center_contrast_not_clear')
    peak=(magnitude==maximum_filter(magnitude,size=7))&(magnitude>=threshold)
    py,px=np.nonzero(peak)
    if len(px):
        peak_distances=distances[py,px]
        near=int(np.argmin(peak_distances))
        nearest=float(peak_distances[near])
        peak_xy=[float(px[near]+cx+left),float(py[near]+cy+top)]
        if nearest>POLICY['center_radius_px']:
            flags.append('contrast_offset_suspected')
    else:
        nearest=None
        peak_xy=None
    return {'automatic_inspected':True,'flags':flags,'crop_left':left,'crop_top':top,
            'crop_width':right-left,'crop_height':bottom-top,
            'cyan_purple_fraction':color_fraction,'cyan_purple_max_row_fraction':row_fraction,
            'laplacian_energy':detail,'local_p95_p05':float(np.percentile(gray,95)-np.percentile(gray,5)),
            'contrast_threshold':float(threshold),'contrast_noise_mad':noise,'center_contrast_abs':support,
            'nearest_contrast_peak_distance_px':nearest,'nearest_contrast_peak_original_xy':peak_xy,
            'local_contrast_extrema_count':len(px),'center_radius_px':POLICY['center_radius_px']}


def coordinate_flags(points):
    groups=defaultdict(list)
    result=defaultdict(list)
    for p in points:
        if all(isinstance(p.get(k),(int,float)) and not isinstance(p[k],bool) and math.isfinite(p[k]) for k in ('x','y')):
            groups[(p['image_asset_id'],round(p['x'],4),round(p['y'],4))].append(p)
    for rows in groups.values():
        if len(rows)>1:
            for p in rows:
                result[p['point_id']].append('duplicate_position')
                if len({r['fine_label'] for r in rows})>1:
                    result[p['point_id']].append('coordinate_label_conflict')
    return result


def select_examples(rows,per_image=12):
    groups=defaultdict(list)
    for r in rows:
        if r['automatic_inspected']:
            groups[r['image_asset_id']].append(r)
    selected=[]
    for values in groups.values():
        # Include every flag available in each image and spread clear examples spatially.
        used=set()
        for flag in FLAG_NAMES:
            matches=sorted((r for r in values if flag in r['flags']),key=lambda r:r['point_id'])
            if matches:
                r=matches[len(matches)//2]
                used.add(r['point_id'])
                selected.append(r)
        chosen=[r for r in selected if r['image_asset_id']==values[0]['image_asset_id']]
        pool=sorted((r for r in values if r['point_id'] not in used),key=lambda r:(bool(r['flags']),r['point_id']))
        while pool and len(chosen)<per_image:
            eligible=[r for r in pool if bool(r['flags'])==bool(pool[0]['flags'])]
            r=max(eligible,key=lambda p:min((p['x']-q['x'])**2+(p['y']-q['y'])**2 for q in chosen)) if chosen else eligible[0]
            chosen.append(r);selected.append(r);pool.remove(r)
    return selected


def audit(workspace,ai_review,output):
    workspace,ai_review,output=map(Path,(workspace,ai_review,output))
    if not validate_outputs(workspace) or not validate_outputs(ai_review):
        raise ValueError('input_hash_invalid')
    contract=json.loads((workspace/'workspace.json').read_text(encoding='utf-8'))
    visual=json.loads((ai_review/'AI_검수결과.json').read_text(encoding='utf-8'))
    panels=json.loads((ai_review/'panels.json').read_text(encoding='utf-8'))
    if (contract['schema'],contract['schema_version'])!=('annotation_workspace',2) or (visual['schema'],visual['schema_version'])!=('ai_visual_type_proposals',1):
        raise ValueError('unsupported_input_schema')
    workspace_sha=digest_file(workspace/'output_hashes.json')
    ai_sha=digest_file(ai_review/'output_hashes.json')
    if panels['workspace_manifest_sha256']!=workspace_sha:
        raise ValueError('ai_workspace_mismatch')
    protected=[workspace,ai_review,Path(contract['source_root']),Path(contract['registry_root'])]
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise ValueError('output_must_be_new_independent_directory')
    points=[json.loads(s) for s in (workspace/'provider_points.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=[p['point_id'] for p in points]
    if len(ids)!=len(set(ids)) or len(ids)!=contract['summary']['point_count']:
        raise ValueError('provider_count_or_ids_invalid')
    observed={r['item_id']:r for r in visual['items']}
    if len(observed)!=len(visual['items']) or not set(observed).issubset(ids):
        raise ValueError('ai_point_ids_invalid')
    for p in points:
        if p['point_id'] in observed and any(p[k]!=observed[p['point_id']][k] for k in ('x','y','image_asset_id','fine_label','base_label')):
            raise ValueError('ai_original_metadata_mismatch')
    assets=[json.loads(s) for s in (Path(contract['registry_root'])/'assets.jsonl').read_text(encoding='utf-8').splitlines()]
    reader=SourceReader(contract['source_root'],assets)
    grouped=defaultdict(list)
    for p in points:
        grouped[p['image_asset_id']].append(p)
    extra=coordinate_flags(points)
    rows=[]
    output.mkdir(parents=True)
    (output/'examples').mkdir()
    example_rows=[]
    for asset_id,values in grouped.items():
        pixels=reader.image(reader.by_id[asset_id])
        current=[]
        for p in values:
            m=point_metrics(pixels,p['x'],p['y'])
            flags=sorted(set(m['flags']+extra[p['point_id']]+(['unknown_provider_type'] if p['base_label']=='unknown' else [])))
            previous=observed.get(p['point_id'])
            row=p|m|{'flags':flags,'priority_score':sum(WEIGHTS[f] for f in flags),
                     'inspection_origin':'automatic_image_metrics','ai_visual_reviewed':previous is not None,
                     'ai_visual_status':previous['status'] if previous else None,
                     'ai_proposed_base_label':previous['proposed_base_label'] if previous else None,
                     'human_verified':False,'expert_semantic_confirmation':False,
                     'eligible_for_verified_evaluation':False,'training_ready':False}
            current.append(row);rows.append(row)
        for r in select_examples(current):
            n=len(example_rows)+1
            crop=pixels[r['crop_top']:r['crop_top']+r['crop_height'],r['crop_left']:r['crop_left']+r['crop_width']]
            view=crop if crop.dtype==np.uint8 else (crop.astype(np.float32)*255/65535).astype(np.uint8)
            raw=Image.fromarray(view).convert('RGB')
            raw_name=f'examples/{n:03d}.png'
            raw.save(output/raw_name)
            marked=raw.copy()
            pen=ImageDraw.Draw(marked)
            x,y=r['x']-r['crop_left'],r['y']-r['crop_top']
            for color,width in [('black',4),('#ffeb3b',1)]:
                for line in [(x-12,y,x-4,y),(x+4,y,x+12,y),(x,y-12,x,y-4),(x,y+4,x,y+12)]:
                    pen.line(line,fill=color,width=width)
            marked_name=f'examples/{n:03d}_marked.png'
            marked.save(output/marked_name)
            example_rows.append(r|{'example_index':n,'raw_file':raw_name,'marked_file':marked_name})
        print(f'image {len({r["image_asset_id"] for r in rows})}/{len(grouped)} | {len(rows)}/{len(points)}',flush=True)
        del pixels
    reader.verify_unchanged()
    if digest_file(workspace/'output_hashes.json')!=workspace_sha or digest_file(ai_review/'output_hashes.json')!=ai_sha:
        raise ValueError('input_changed_during_audit')
    write_jsonl(output/'point_quality.jsonl',rows)
    fields=['point_id','image_asset_id','area_id','phase','fine_label','base_label','x','y','automatic_inspected','flags','priority_score','cyan_purple_fraction','laplacian_energy','center_contrast_abs','contrast_threshold','nearest_contrast_peak_distance_px','ai_visual_reviewed','ai_visual_status','ai_proposed_base_label','human_verified','training_ready']
    with (output/'point_quality.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r|{'flags':';'.join(r['flags'])} for r in rows)
    summary={'point_count':len(rows),'automatic_inspected':sum(r['automatic_inspected'] for r in rows),
             'point_images':len(grouped),'flagged':sum(bool(r['flags']) for r in rows),
             'unflagged':sum(not r['flags'] for r in rows),
             'flag_counts':dict(Counter(f for r in rows for f in r['flags'])),
             'by_wafer':dict(Counter(r['area_id'] for r in rows)),
             'by_image':{ident:{'point_count':len([r for r in rows if r['image_asset_id']==ident]),
                              'flagged':sum(bool(r['flags']) for r in rows if r['image_asset_id']==ident),
                              'area_id':values[0]['area_id'],'phase':values[0]['phase'],
                              'locator':reader.by_id[ident]['locator']}
                         for ident,values in grouped.items()},
             'ai_visual_reviewed':len(observed),'ai_visual_suggested':sum(r['status']=='suggest' for r in observed.values()),
             'ai_suggestions_with_quality_flags':sum(bool(r['flags']) for r in rows if r['ai_visual_status']=='suggest'),
             'human_verified_created':0,'original_labels_modified':False,'training_ready':False,
             'unflagged_means_verified':False,'scope':'all supplied 2D point annotations; not complete wafer detection or 3D certification'}
    write_json(output/'quality_summary.json',{'schema':'provider_point_quality_audit','schema_version':1,
               'workspace_manifest_sha256':workspace_sha,'ai_review_manifest_sha256':ai_sha,
               'policy':POLICY,'flag_names':FLAG_NAMES,'summary':summary,
               'limitations':['heuristic_quality_alerts_not_accuracy_or_type_confirmation','low_detail_may_be_acquisition_condition_or_clean_background',
                              'detected_extrema_are_not_physical_defect_counts','offset_is_not_an_automatic_coordinate_correction',
                              'no_label_or_training_changes','unsampled_images_and_missing_wafer_annotations_remain']})
    write_json(output/'examples.json',example_rows)
    render_report(output,summary,example_rows)
    write_json(output/'output_hashes.json',{p.relative_to(output).as_posix():digest_file(p) for p in output.rglob('*') if p.is_file()})
    return summary


def render_report(output,summary,examples):
    e=html.escape
    cards=[]
    for r in examples:
        flags=' · '.join(FLAG_NAMES[f] for f in r['flags']) or '설정한 검사에서 경고 없음'
        cards.append(f'<article><h3>{r["example_index"]:03d} · 웨이퍼 {e(str(r["area_id"]))} · {e(r["phase"])}</h3><p>원본 타입 {e(r["fine_label"])} · 좌표 ({r["x"]:.2f}, {r["y"]:.2f})</p><p>{e(flags)}</p><figure><img src="{e(r["raw_file"])}"><img src="{e(r["marked_file"])}"></figure><small>{e(r["point_id"])}</small></article>')
    counts=''.join(f'<li>{e(FLAG_NAMES[k])}: {v:,}건</li>' for k,v in summary['flag_counts'].items())
    body=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>전체 주석 품질 검사</title><style>body{{font-family:Malgun Gothic,sans-serif;background:#111827;color:white;margin:24px;line-height:1.6}}section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}}article{{padding:16px;background:#1f2937}}img{{image-rendering:pixelated;max-width:48%;height:auto;margin:2px}}figure{{margin:0}}small{{overflow-wrap:anywhere}}a{{color:#93c5fd}}</style>
<h1>전체 제공 주석 품질 검사</h1><p>{summary['point_count']:,}건 자동 검사 · 경고 {summary['flagged']:,}건 · 경고 없음 {summary['unflagged']:,}건</p>
<p>자동 검사 지표입니다. 경고 없음은 정답 타입 확인을 뜻하지 않으며 경고는 삭제 지시가 아닙니다. 원본 라벨·좌표·검수 기록은 보존했습니다.</p>
<p>사람/전문가 확정 0건 · 기존 AI 이미지 판독 {summary['ai_visual_reviewed']}건 · 이번 자동 검사로 타입 판독 완료 상태를 추가하지 않았습니다.</p>
<ul>{counts}</ul><p>아래는 영상별 경고 유형과 위치를 나눠 뽑은 예시입니다. 좌우는 같은 원본 패치이며 오른쪽에는 주석 좌표를 표시했습니다. 표시된 대비 위치로 원본 좌표를 이동시키지 않습니다.</p>
<p><a href="point_quality.csv">전체 항목 표 내려받기</a> · <a href="quality_summary.json">검사 기준과 요약</a></p><section>{''.join(cards)}</section></html>"""
    (output/'전체주석_품질검사.html').write_text(body,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','ai-review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(audit(a.workspace,a.ai_review,a.output),ensure_ascii=True))


if __name__=='__main__':
    main()

