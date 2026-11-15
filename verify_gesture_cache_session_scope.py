#!/usr/bin/env python3
"""Audit which frozen HMOG/FETA session IDs actually occur in cached gesture NPZs.

No raw archive access required. Requires numpy. Optionally cross-checks every
complete first-four session count with the v2.0 session_task_counts.csv.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

MODELS = {
    'feta_6s_scroll': ('FETA', 'iPhone 6s', 'scroll'),
    'feta_6s_swipe': ('FETA', 'iPhone 6s', 'swipe'),
    'feta_7_scroll': ('FETA', 'iPhone 7', 'scroll'),
    'feta_7_swipe': ('FETA', 'iPhone 7', 'swipe'),
    'feta_x_scroll': ('FETA', 'iPhone X', 'scroll'),
    'feta_x_swipe': ('FETA', 'iPhone X', 'swipe'),
    'hmog_sitting': ('HMOG', 'Samsung Galaxy S4', 'reading_sitting'),
    'hmog_walking': ('HMOG', 'Samsung Galaxy S4', 'reading_walking'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--splits', type=Path, required=True)
    ap.add_argument('--audit-counts', type=Path, default=None,
                    help='Optional v2.0 session_task_counts.csv, for a full uncapped first-four cache')
    ap.add_argument('--output', type=Path, default=None)
    args=ap.parse_args()
    index=json.loads((args.cache/'index.json').read_text(encoding='utf-8'))
    splits=json.loads(args.splits.read_text(encoding='utf-8'))
    mode=index.get('settings',{}).get('session_scope','legacy-all-available')
    errors=[]; report={}; seen=Counter()
    for entry in index.get('shards',[]):
        slug=entry['scenario']
        if slug not in MODELS:
            errors.append(f'Unknown scenario: {slug}');continue
        dataset,model,task=MODELS[slug]
        user=str(entry['user_id'])
        if user not in splits['splits'][dataset][model if dataset=='FETA' else task]['train']:
            errors.append(f'Not a training user: {slug}/{user}')
        try:
            allowed=set(map(str,splits['session_index'][dataset][model][user][task]))
        except KeyError:
            errors.append(f'Missing frozen first-four IDs for {slug}/{user}')
            continue
        file=args.cache/Path(str(entry['file']).replace('\\','/'))
        if not file.is_file():
            errors.append(f'Missing shard file: {file}');continue
        with np.load(file,allow_pickle=False) as npz:
            metas=npz['meta_json']
        if len(metas)!=int(entry['count']):
            errors.append(f'Shard gesture count mismatch: {slug}/{user}: {len(metas)} vs {entry["count"]}')
        for line in metas:
            m=json.loads(str(line))
            if str(m['user_id'])!=user or m['dataset']!=dataset or m['task']!=task:
                errors.append(f'Wrong shard metadata: {slug}/{user}')
            sess=str(m['session_id'])
            seen[(slug,user,sess)]+=1
            out=report.setdefault(slug, {'gestures':0,'within_first_four':0,'outside_first_four':0,
                                         'sessions_seen':set(),'users_seen':set()})
            out['gestures']+=1
            out['sessions_seen'].add((user,sess));out['users_seen'].add(user)
            if sess in allowed:out['within_first_four']+=1
            else:
                out['outside_first_four']+=1
                if mode=='first-four':errors.append(f'OUTSIDE first-four: {slug}/{user}/{sess}')
    for slug,out in report.items():
        out['users_seen']=len(out['users_seen'])
        out['sessions_seen']=len(out['sessions_seen'])
        if out['gestures']!=int(index['scenarios'][slug]['gestures']):
            errors.append(f'Index scenario count mismatch: {slug}: {out["gestures"]} vs {index["scenarios"][slug]["gestures"]}')
    if args.audit_counts is not None:
        import pandas as pd
        df=pd.read_csv(args.audit_counts,dtype={'user_id':str,'session_id':str})
        for slug,(dataset,model,task) in MODELS.items():
            if slug not in index['scenarios']:continue
            if not (mode=='first-four' and index['settings'].get('max_users_per_scenario')==0
                    and index['settings'].get('max_gestures_per_user')==0):
                continue
            users=list(splits['splits'][dataset][model if dataset=='FETA' else task]['train'])
            subset=df[(df.dataset==dataset)&(df.device_model==model)&(df.task==task)
                      &(df.user_id.isin(users))]
            expected=Counter()
            for r in subset.itertuples():
                ids=splits['session_index'][dataset][model][str(r.user_id)][task]
                if str(r.session_id) in set(map(str,ids)):
                    expected[(slug,str(r.user_id),str(r.session_id))]+=int(r.gestures_strict_multimodal)
            actual={key:n for key,n in seen.items() if key[0]==slug}
            for key in set(expected)|set(actual):
                if actual.get(key,0)!=expected.get(key,0):
                    errors.append(f'v2.0 per-session count mismatch: {key}: cache={actual.get(key,0)} audit={expected.get(key,0)}')
    result={'status':'PASS' if not errors else 'FAIL','cache':str(args.cache),
            'session_scope':mode,'scenarios':report,'errors_count':len(errors),
            'errors':errors[:100]}
    serialized=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(serialized,encoding='utf-8')
    print(serialized)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
