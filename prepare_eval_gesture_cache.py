"""Prepare strictly isolated validation/test first-four-session NPZ caches.

No labels or statistics are learned; uses the SAME raw parsers as training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import hmog_feta_gesture_dataset_prototype_v1 as pp
import analyze_hmog_feta_feta_criterion_v2_0 as v2


def build(args):
    if args.role not in ('validation','test'):raise ValueError('Only validation/test allowed')
    splits=Path(args.splits);doc=pp.read_splits(splits)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    modelmap=v2.feta_user_mapping(Path(args.userdata))
    entries=[];counts={};issues=[]
    for slug in pp.selected_scenarios(args.scenarios):
        ds,model,task=pp.SCENARIOS[slug]
        users=sorted(doc['splits']['FETA'][model][args.role] if ds=='FETA'
                     else doc['splits']['HMOG'][task][args.role])
        counts[slug]={'users_requested':len(users),'users_with_samples':0,'gestures':0}
        print(f'[{args.role}/{slug}] {len(users)} users',flush=True)
        for n,user in enumerate(users,1):
            if ds=='FETA' and modelmap.get(user)!=model:raise ValueError(f'Unexpected device: {user}')
            allowed=pp.first_four_sessions(doc,ds,model,task,user)
            try:
                if ds=='HMOG':
                    records,_,warnings=pp.extract_hmog_user(Path(args.hmog_root),user,task,allowed)
                else:
                    records,_,warnings=pp.extract_feta_user(Path(args.feta_root),user,model,task,
                                                             args.feta_coordinates,allowed)
                if not records:raise ValueError('No accepted gestures')
                if warnings and args.fail_on_errors:raise ValueError(warnings[0])
                issues.extend({'scenario':slug,'user':user,'warning':x} for x in warnings[:20])
            except Exception as exc:
                if args.fail_on_errors:raise
                issues.append({'scenario':slug,'user':user,'error':str(exc)})
                continue
            records.sort(key=lambda r:(str(r['meta']['session_id']),str(r['meta']['repetition']),
                                       r['meta']['start_ms'],r['meta']['end_ms']))
            filename=str(Path('shards')/slug/f'{user}.npz')
            pp.save_shard(output/filename,records)
            entries.append({'scenario':slug,'dataset':ds,'device_model':model,'task':task,
                            'role':args.role,'user_id':user,'file':filename,'count':len(records)})
            counts[slug]['users_with_samples']+=1;counts[slug]['gestures']+=len(records)
            if n%5==0 or n==len(users):print(f'  {n}/{len(users)} users, {counts[slug]["gestures"]} gestures',flush=True)
    index={'version':pp.VERSION,'role':args.role,'source_splits_sha256':pp.sha256(splits),
           'parser_version':v2.VERSION,'parser_file_sha256':pp.sha256(Path(v2.__file__)),
           'preprocessor_file_sha256':pp.sha256(Path(pp.__file__)),
           'settings':{'session_scope':'first-four','feta_coordinates':args.feta_coordinates,
                       'no_resampling':True,'max_users_per_scenario':0,'max_gestures_per_user':0},
           'feature_names':{'touch':list(pp.TOUCH_FEATURES),'accel':list(pp.MOTION_FEATURES),
                            'gyro':list(pp.MOTION_FEATURES)},
           'scenarios':counts,'shards':entries,'warnings_count':len(issues)}
    (output/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'issues.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.fail_on_errors and any(v['users_with_samples']!=v['users_requested'] for v in counts.values()):
        raise AssertionError('Some selected users were not extracted')
    print(f'DONE: {sum(x["count"] for x in entries)} {args.role} gestures -> {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('validation','test'),required=True)
    p.add_argument('--splits',type=Path,default=Path('experiments/splits/hmog_feta_splits_seed_20261112.json'))
    p.add_argument('--hmog-root',type=Path,default=Path('input_data/hmog_unarchived'))
    p.add_argument('--feta-root',type=Path,default=Path('input_data/data_files'))
    p.add_argument('--userdata',type=Path,default=Path('input_data/tables/tables/userdata.csv'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',default='all')
    p.add_argument('--feta-coordinates',choices=('points','pixels'),default='points')
    p.add_argument('--fail-on-errors',action='store_true')
    build(p.parse_args())
if __name__=='__main__':main()
