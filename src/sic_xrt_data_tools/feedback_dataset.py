"""Convert explicit desktop review events into independently masked supervision.

No unlabeled region is a negative; no model prediction becomes a type target.
"""
import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

INPUT_SCHEMA = 'xrt_feedback_v1'
SCHEMA = 'xrt_feedback_dataset_v1'
CLASSES = ('BPD', 'TED', 'TSD')
WAFERS = ('1', '2', '3', '4', '5', '6', '7', '9')
MAX_OFFSET = 32


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def empty_target(original):
    return {'x':original['x'], 'y':original['y'], 'objectness':None, 'label':None, 'location_confirmed':False, 'duplicate_of':None}


def valid_xy(value, source):
    return all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and 0 <= value[k] < source[bound]
               for k, bound in (('x', 'width'), ('y', 'height')))


def validate_review(data):
    if (data.get('schema') != INPUT_SCHEMA or not isinstance(data.get('session_id'), str)
            or data.get('expert_ground_truth') is not False or data.get('exhaustive_annotations') is not False
            or data.get('wafer_id') not in WAFERS):
        raise ValueError('Development human review required; wafer 8 is forbidden')
    source = data['source']
    if (source.get('coordinate_space') != 'raw_pixel_xy' or source.get('orientation') != 'encoded_no_exif_rotation'
            or not isinstance(source.get('file_path'), str) or len(source.get('sha256', '')) != 64
            or any(c not in '0123456789abcdef' for c in source['sha256'])
            or type(source.get('page_index')) is not int or source['page_index'] < 0
            or any(type(source.get(k)) is not int or source[k] <= 0 for k in ('width', 'height'))):
        raise ValueError('Invalid source metadata')
    latest, ids = {}, set()
    for event in data['events']:
        if event.get('actor') != 'human' or not event.get('reviewer', '').strip() or event['id'] in ids:
            raise ValueError('Human reviewer and unique event IDs required')
        if len(event['reviewer']) > 100 or not isinstance(event.get('note'), str) or len(event['note']) > 2000:
            raise ValueError('Invalid reviewer/note')
        datetime.fromisoformat(event['created_at'])
        ids.add(event['id'])
        ident, original, target = event['record_id'], event['original'], event['target']
        prior = latest.get(ident)
        if not valid_xy(original, source) or not valid_xy(target, source):
            raise ValueError('Invalid original/target coordinates')
        expected = copy.deepcopy(prior['target'] if prior else empty_target(original))
        if (event['previous_target'] != expected or type(event['revision']) is not int
                or event['revision'] != (prior['revision']+1 if prior else 1)
                or prior and original != prior['original']):
            raise ValueError('Review history conflict')
        action = event['action']
        expected['duplicate_of'] = None
        if action == 'confirm':
            expected['objectness'] = True
        elif action == 'background':
            expected.update(objectness=False, label=None, location_confirmed=False)
        elif action == 'type':
            if target['label'] not in CLASSES:
                raise ValueError('Invalid confirmed type')
            expected.update(objectness=True, label=target['label'])
        elif action == 'type_unknown':
            expected.update(objectness=True, label=None)
        elif action in ('location', 'add'):
            expected.update(x=target['x'], y=target['y'], objectness=True, location_confirmed=True)
            if action == 'add' and target['label'] in CLASSES:
                expected['label'] = target['label']
            if action == 'add' and (prior or original.get('candidate_id') is not None or original.get('model_sha256') is not None):
                raise ValueError('Added point must be a new manual record')
        elif action == 'duplicate':
            if not target['duplicate_of'] or target['duplicate_of'] == ident:
                raise ValueError('Invalid duplicate link')
            expected.update(objectness=None, label=None, location_confirmed=False, duplicate_of=target['duplicate_of'])
        elif action == 'defer':
            expected.update(objectness=None, label=None, location_confirmed=False)
        elif action == 'undo' and prior:
            expected = prior['previous_target']
        else:
            raise ValueError('Unknown review action')
        if target != expected or type(target['location_confirmed']) is not bool or target['objectness'] is not None and type(target['objectness']) is not bool:
            raise ValueError('Contradictory target/mask')
        latest[ident] = event
    originals = {**data.get('candidate_snapshot', {}), **{k:v['original'] for k,v in latest.items()}}
    for event in latest.values():
        duplicate = event['target']['duplicate_of']
        if duplicate and duplicate not in originals:
            raise ValueError('Unresolved duplicate target')
    return latest


def read_image(source):
    path = Path(source['file_path'])
    if digest(path) != source['sha256']:
        raise ValueError('Source hash changed: '+str(path))
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        if source['page_index'] != 0:
            raise ValueError('JPEG must have page index 0')
        with Image.open(path) as image:
            pixels = np.array(image.convert('RGB'))  # Encoded orientation, never EXIF transpose.
    else:
        with tifffile.TiffFile(path) as image:
            pixels = image.pages[source['page_index']].asarray()
    if pixels.shape != (source['height'], source['width'], 3) or pixels.dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB image with matching dimensions')
    if digest(path) != source['sha256']:
        raise ValueError('Source changed during reading')
    return pixels


def build(review_paths, output):
    """Create a fresh package, rejecting ambiguous histories and final-test wafers."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be a new directory; old datasets are preserved')
    sessions = {}
    for path in review_paths:
        path = Path(path)
        raw = path.read_bytes()
        data = json.loads(raw)
        validate_review(data)
        source = data['source']
        key = (source['sha256'], source['page_index'])
        old = sessions.get(key)
        if old:
            previous = old[0]
            if data['session_id'] != previous['session_id'] or data['wafer_id'] != previous['wafer_id']:
                raise ValueError('Conflicting sessions/wafer for the same source page')
            shorter, longer = sorted((data, previous), key=lambda d: len(d['events']))
            if longer['events'][:len(shorter['events'])] != shorter['events']:
                raise ValueError('Conflicting review history')
            if longer is previous:
                continue
        sessions[key] = (data, path, hashlib.sha256(raw).hexdigest())
    if not sessions:
        raise ValueError('No review files')
    samples, records, exclusions, sources = [], [], [], []
    output.mkdir(parents=True)
    (output/'patches').mkdir()
    (output/'reviews').mkdir()
    for data, path, feedback_hash in sessions.values():
        latest = validate_review(data)
        pixels = read_image(data['source'])
        source = data['source']
        source_id = source['sha256']+':'+str(source['page_index'])
        copy_path = 'reviews/'+feedback_hash+'.json'
        # Bytes, not a reconstructed JSON: package provenance retains exact export hash.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != feedback_hash:
            raise ValueError('Review changed during packaging')
        (output/copy_path).write_bytes(raw)
        sources.append({'id':source_id, 'wafer':data['wafer_id'], 'source':source,
                        'review_path':copy_path, 'review_sha256':feedback_hash})
        confirmed = [e['target'] for e in latest.values() if e['target']['objectness'] is True and e['target']['location_confirmed']]
        for ident, event in latest.items():
            target, original = event['target'], event['original']
            record_key = data['session_id']+':'+ident
            records.append({'id':record_key,'source_id':source_id,'wafer':data['wafer_id'],
                            'review_sha256':feedback_hash,'event':event})
            reason = ('duplicate' if target['duplicate_of'] else 'deferred_or_undone' if target['objectness'] is None else None)
            if reason:
                exclusions.append({'record_id':record_key,'reason':reason})
                continue
            if target['objectness'] is False and any(math.hypot(t['x']-target['x'],t['y']-target['y']) <= MAX_OFFSET for t in confirmed):
                exclusions.append({'record_id':record_key,'reason':'background_near_confirmed_defect'})
                continue
            anchors = [('current',target['x'],target['y'])]
            if target['location_confirmed'] and (original['x'],original['y']) != (target['x'],target['y']):
                anchors.append(('original',original['x'],original['y']))
            for role, x, y in anchors:
                left, top = round(x)-64, round(y)-64
                center_x, center_y = left+64., top+64.
                dx,dy = target['x']-center_x,target['y']-center_y
                if left < 0 or top < 0 or left+128 > source['width'] or top+128 > source['height']:
                    exclusions.append({'record_id':record_key,'role':role,'reason':'incomplete_rgb128_patch'})
                    continue
                if max(abs(dx),abs(dy)) > MAX_OFFSET:
                    exclusions.append({'record_id':record_key,'role':role,'reason':'offset_exceeds_32px'})
                    continue
                ident_hash = hashlib.sha256((record_key+':'+role).encode()).hexdigest()
                patch_path = 'patches/'+ident_hash+'.tif'
                tifffile.imwrite(output/patch_path, pixels[top:top+128,left:left+128], photometric='rgb')
                samples.append({'id':ident_hash,'record_id':record_key,'source_id':source_id,'wafer':data['wafer_id'],
                    'path':patch_path,'sha256':digest(output/patch_path),'role':role,'crop_xy':[left,top],
                    'anchor_xy':[center_x,center_y],'objectness':int(target['objectness']),'objectness_mask':True,
                    'type':target['label'],'type_mask':target['objectness'] is True and target['label'] in CLASSES,
                    'offset_xy':[dx,dy] if target['location_confirmed'] else [0.,0.],
                    'location_mask':target['objectness'] is True and target['location_confirmed'],
                    'label_basis':'human_review','expert_ground_truth':False,'review_sha256':feedback_hash})
    save(output/'samples.json',samples)
    save(output/'records.json',records)
    save(output/'excluded.json',exclusions)
    manifest = {'schema':SCHEMA,'input_schema':INPUT_SCHEMA,'patch_contract':'rgb128_uint8_raw_pixel_xy',
        'classes':list(CLASSES),'max_offset_px':MAX_OFFSET,'samples_path':'samples.json','records_path':'records.json',
        'samples_sha256':digest(output/'samples.json'),'records_sha256':digest(output/'records.json'),
        'sources':sources,'sample_count':len(samples),'counts_by_wafer':dict(Counter(r['wafer'] for r in samples)),
        'masks':{key:sum(r[key] for r in samples) for key in ('objectness_mask','type_mask','location_mask')},
        'expert_ground_truth':False,'exhaustive_annotations':False,'test_evaluated':False,
        'automatic_model_promotion':False,'excluded_count':len(exclusions)}
    save(output/'dataset.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.reviews,args.output),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
