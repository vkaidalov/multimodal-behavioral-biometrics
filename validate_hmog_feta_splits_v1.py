#!/usr/bin/env python3
"""Verify frozen, subject-disjoint HMOG/FETA splits against v2.0 gesture audit.

Default: stringent cross-check against all v2.0 audit CSVs, first-four temporal
order, per-session gesture counts, geometry/orientation configuration and split
membership. Optional --raw-check reparses ORIGINAL touch/accel/gyro CSVs for
included sessions with the exact SAME v2.0 extractor; this is recomputation,
not an independently implemented parser/algorithm.
Python >=3.10; --raw-check requires numpy and pandas (as the v2.0 audit does).
"""
from __future__ import annotations
import argparse, collections, csv, importlib.util, json, sys
from pathlib import Path

from make_hmog_feta_splits_v1 import (load_audit,MODELS,SPLITS,THRESHOLD,sha,REQUIRED_FILES)


def fail(msg):raise ValueError(msg)

def flat_splits(s):return {k:set(s[k]) for k in SPLITS}

def check_partition(record,expected,label):
    if not isinstance(record,dict) or set(record)!=set(SPLITS):
        fail(f'{label}: expected only train/validation/test keys')
    groups={}
    for part in SPLITS:
        ids=record[part]
        if not isinstance(ids,list) or any(not isinstance(x,str) or not x for x in ids):
            fail(f'{label}: invalid subject list {part}')
        if len(ids)!=len(set(ids)):fail(f'{label}: repeated user IDs in {part}')
        groups[part]=set(ids)
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if groups[a]&groups[b]:fail(f'{label}: {a} overlaps {b} = {groups[a]&groups[b]}')
    union=set().union(*groups.values())
    if union!=expected:
        fail(f'{label}: wrong cohort; missing={sorted(expected-union)[:10]}, unexpected={sorted(union-expected)[:10]}')
    return groups


def check_all(splits_json, audit_dir, allow_provisional=False, userdata_path=None):
    doc=json.loads(Path(splits_json).read_text(encoding='utf-8'))
    if doc.get('schema_version')!='1.0':fail('Unsupported JSON schema_version')
    status=doc.get('status')
    if status!='FROZEN_VERIFIED_AGAINST_V2_AUDIT':
        if not (allow_provisional and status=='PROVISIONAL_NEEDS_V2_AUDIT'):
            fail('JSON status is not FROZEN_VERIFIED_AGAINST_V2_AUDIT; generate final splits from v2 audit')
    cfg=doc.get('protocol',{})
    if cfg.get('threshold_gestures_per_each_first_four_sessions')!=20 or cfg.get('count_mode')!='strict':
        fail('Wrong threshold/count_mode (expected 20/strict)')
    if cfg.get('hmog_orientation')!='Phone_orientation=0 for EVERY touch sample and qualifying in-contact motion sample':
        fail('JSON does not declare strict HMOG portrait orientation')
    cohorts, counts, sessions, prov=load_audit(audit_dir)
    if status=='FROZEN_VERIFIED_AGAINST_V2_AUDIT':
        stored=doc.get('provenance',{}).get('audit_file_sha256',{})
        for filename in REQUIRED_FILES:
            actual=prov['audit_file_sha256'][filename]
            if stored.get(filename)!=actual:
                fail(f'Provenance mismatch: {filename} was changed since split was frozen')
    splits=doc.get('splits',{})
    if set(splits)!= {'FETA','HMOG'} or set(splits['FETA'])!=set(MODELS):
        fail('Wrong selected devices or dataset keys')
    expected_H=('reading_sitting','reading_walking','reading_combined')
    if set(splits['HMOG'])!=set(expected_H):fail('Wrong HMOG scenario keys')
    global_assign={}
    if userdata_path is not None:
        ud=Path(userdata_path)
        if not ud.is_file():fail('Missing original FETA userdata.csv: '+str(ud))
        with ud.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            indexed={''.join(c.lower() for c in k if c.isalnum()):k for k in reader.fieldnames}
            if 'uuid' not in indexed or 'phonemodel' not in indexed:
                fail('FETA userdata.csv missing uuid or phone_model column')
            users_map=collections.defaultdict(set)
            for r in reader:
                uid=str(r[indexed['uuid']]).strip();mod=str(r[indexed['phonemodel']]).strip()
                if uid:users_map[uid].add(mod)
        for model in MODELS:
            for uid in cohorts[('FETA',model,'scroll')]:
                if users_map.get(uid)!= {model}:
                    fail(f'Original userdata.csv contradicts cohort for {uid}: {users_map.get(uid)} versus {model}')
    for model in MODELS:
        expected=cohorts[('FETA',model,'scroll')]
        parts=check_partition(splits['FETA'][model],expected,'FETA/'+model)
        exp_quotas={'iPhone 7':(38,12,12),'iPhone X':(38,12,12),'iPhone 6s':(36,12,12)}[model]
        if tuple(map(lambda k:len(parts[k]),SPLITS))!=exp_quotas:
            fail('Unexpected FETA partition sizes for '+model)
        for part,users in parts.items():
            for user in users:
                if ('FETA',user) in global_assign:fail('FETA user assigned to two phone models: '+user)
                global_assign[('FETA',user)]=part
    sit=cohorts[('HMOG','Samsung Galaxy S4','reading_sitting')]
    walk=cohorts[('HMOG','Samsung Galaxy S4','reading_walking')]
    expect={'reading_sitting':sit,'reading_walking':walk,'reading_combined':sit&walk}
    h={task:check_partition(splits['HMOG'][task],roster,'HMOG/'+task) for task,roster in expect.items()}
    sizes={'reading_sitting':(33,11,12),'reading_walking':(46,16,15),'reading_combined':(32,11,11)}
    for task,parts in h.items():
        if tuple(len(parts[k]) for k in SPLITS)!=sizes[task]:fail('Incorrect HMOG partition sizes for '+task)
    hglobal={}
    for task in ('reading_sitting','reading_walking','reading_combined'):
        for split,users in h[task].items():
            for user in users:
                if user in hglobal and hglobal[user]!=split:
                    fail('HMOG user assigned to different partitions across tasks: '+user)
                hglobal[user]=split
    for split in SPLITS:
        if h['reading_combined'][split] != (h['reading_sitting'][split]&h['reading_walking'][split]):
            fail('Combined partition not exactly sitting ∩ walking for '+split)
    # Verify all recorded first-four IDs against exact source chronology.
    expected_index={}
    for (dataset,model,user,task),ids in sessions.items():
        expected_index.setdefault(dataset,{}).setdefault(model,{}).setdefault(user,{})[task]=ids
    user_index=doc.get('session_index',{})
    if status=='FROZEN_VERIFIED_AGAINST_V2_AUDIT':
        if user_index!=expected_index:fail('Frozen per-user session IDs do not match the audit')
    elif user_index:
        fail('Provisional JSON should not fabricate a session index')
    # FETA scroll and swipe must use identical whole session IDs per user.
    for model in MODELS:
        for u in cohorts[('FETA',model,'scroll')]:
            if sessions[('FETA',model,u,'scroll')]!=sessions[('FETA',model,u,'swipe')]:
                fail('FETA scroll/swipe first four sessions differ for '+model+'/'+u)
    report={'status':'PASS_AUDIT_CONSISTENCY','split_status':status,'threshold':THRESHOLD,
            'audit_version':prov['audit_version'],'cohort_sizes':{},
            'num_unique_users':{'FETA':len([k for k in global_assign if k[0]=='FETA']),
                                'HMOG':len(hglobal)},
            'caveat':'Audit-consistency validation does not independently prove the v2.0 parsing implementation is correct.'}
    for model in MODELS:
        report['cohort_sizes']['FETA/'+model]={p:len(splits['FETA'][model][p]) for p in SPLITS}
    for task in expected_H:
        report['cohort_sizes']['HMOG/'+task]={p:len(splits['HMOG'][task][p]) for p in SPLITS}
    # These are descriptive diagnostics, not a post-hoc optimization objective.
    # Stratification was chosen ONLY from first-two-session gesture counts.
    report['gesture_count_balance']={}
    for ds in ('FETA','HMOG'):
        model_tasks=([(m,('scroll','swipe')) for m in MODELS] if ds=='FETA' else
                     [('Samsung Galaxy S4',('reading_sitting','reading_walking'))])
        for model,tasks in model_tasks:
            for task in tasks:
                scenario=model if ds=='FETA' else task
                record=splits[ds][scenario]
                desc={}
                for part in SPLITS:
                    vals=[counts[(ds,model,u,task)] for u in record[part]]
                    desc[part]={'users':len(vals),
                                'mean_enrollment_gestures_2_sessions':round(sum(sum(v[:2]) for v in vals)/len(vals),2),
                                'mean_all_4_session_gestures':round(sum(sum(v) for v in vals)/len(vals),2),
                                'minimum_session_gestures':min(min(v) for v in vals)}
                report['gesture_count_balance'][ds+'/'+model+'/'+task]=desc
    return report,sessions,counts,doc,prov


def load_v2_module(path):
    path=Path(path)
    if not path.exists():fail('Cannot find v2 parser at '+str(path))
    spec=importlib.util.spec_from_file_location('hmog_feta_v2_audit_for_validation',str(path.resolve()))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def check_raw(sessions,counts,doc,parser_path,hmog_root,feta_root,userdata_path=None,limit_users=0):
    """Recompute all 4 sessions/two tasks for each eligible user (no model training)."""
    v2=load_v2_module(parser_path)
    info=doc['protocol']
    params={'min_sensor_samples':3,'edge_gap_ms':40.0,'internal_gap_ms':120.0,
            'pair_tolerance_ms':25.0,'min_sensor_pairs':1,'save_gestures':False}
    # Selected roster only. Enforce exact sample count by recomputing v2.
    samples={k for k in sessions if
             (k[0]=='FETA' and k[1] in MODELS) or (k[0]=='HMOG' and k[3] in ('reading_sitting','reading_walking'))}
    groups=collections.defaultdict(dict)
    for dataset,model,user,task in samples:groups[(dataset,model,user)][task]=sessions[(dataset,model,user,task)]
    subjects=sorted(groups)
    if limit_users>0:subjects=subjects[:limit_users]
    checked=0;by_dataset=collections.Counter()
    for ds,model,user in subjects:
        target=groups[(ds,model,user)]
        if ds=='FETA':
            home=Path(feta_root)/user
            if not home.is_dir():fail('Missing raw FETA user: '+str(home))
            for task,sids in target.items():
                for sid in sids:
                    folder=home/sid/task
                    if not folder.is_dir():fail('Missing raw FETA session/task: '+str(folder))
                    cnt=collections.Counter()
                    reps=sorted(p for p in folder.iterdir() if p.is_dir())
                    for rep in reps:
                        ident={'dataset':'FETA','device_model':model,'user_id':user,'session_id':sid,'task':task}
                        rr,gg,raw_rows,touchfiles,issues=v2.feta_rep(rep,ident,params)
                        cnt.update(rr)
                    expected_cnt=counts[('FETA',model,user,task)][sids.index(sid)]
                    if cnt['gestures_strict_multimodal']!=expected_cnt:
                        fail(f'FETA raw count changed: {user}/{sid}/{task}: raw={cnt["gestures_strict_multimodal"]}, audit={expected_cnt}')
                    checked+=1;by_dataset['FETA']+=1
        else:
            home=Path(hmog_root)/user
            if not home.is_dir():fail('Missing raw HMOG user: '+str(home))
            needed={(sid,task) for task,sids in target.items() for sid in sids}
            found={}
            # Need to resolve Activity.Session_number to folder name (may differ).
            for folder in sorted(p for p in home.iterdir() if p.is_dir()):
                rows,gesture_rows,issues=v2.hmog_session(folder,user,params)
                for r in rows:
                    key=(str(r['session_id']),r['task'])
                    if key in needed:
                        if key in found:fail('Duplicated HMOG raw session/task '+str((user,key)))
                        found[key]=int(r.get('gestures_strict_multimodal',0))
            if set(found)!=needed:fail(f'HMOG missing expected raw sessions for {user}: {needed-set(found)}')
            for key,cnt in found.items():
                sid,task=key
                expected_cnt=counts[('HMOG',model,user,task)][target[task].index(sid)]
                if cnt!=expected_cnt:fail(f'HMOG raw count changed: {user}/{key}: raw={cnt}, audit={expected_cnt}')
                checked+=1;by_dataset['HMOG']+=1
    return {'raw_check':'PASS_SAME_V2_PARSER','users_checked':len(subjects),
            'session_task_rows_recomputed':checked,'by_dataset':dict(by_dataset),
            'parser':'reused v2.0 audit function (not an independent implementation)'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,default=Path('output_data/hmog_feta_splits_seed_20261112.json'))
    p.add_argument('--audit-dir',type=Path,default=Path('output_data/hmog_feta_feta_criterion_v2_0'))
    p.add_argument('--report',type=Path,default=Path('output_data/hmog_feta_splits_validation.json'))
    p.add_argument('--allow-provisional',action='store_true')
    p.add_argument('--raw-check',action='store_true',help='Recount selected raw samples using v2.0 parser (slower)')
    p.add_argument('--limit-raw-users',type=int,default=0,help='Smoke-test only; 0 checks all selected users')
    p.add_argument('--v2-script',type=Path,default=Path('analyze_hmog_feta_feta_criterion_v2_0.py'))
    p.add_argument('--userdata',type=Path,default=Path('input_data/tables/tables/userdata.csv'))
    p.add_argument('--skip-userdata-check',action='store_true')
    p.add_argument('--hmog-input',type=Path,default=Path('input_data/hmog_unarchived'))
    p.add_argument('--feta-input',type=Path,default=Path('input_data/data_files'))
    args=p.parse_args()
    try:
        report,sessions,counts,doc,prov=check_all(args.splits,args.audit_dir,args.allow_provisional,
                                                 userdata_path=None if args.skip_userdata_check else args.userdata)
        if args.raw_check:
            report.update(check_raw(sessions,counts,doc,args.v2_script,args.hmog_input,args.feta_input,
                                    limit_users=args.limit_raw_users))
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('PASS:',args.report.resolve())
        print(json.dumps(report['cohort_sizes'],indent=2,ensure_ascii=False))
    except Exception as exc:
        print('FAIL: '+str(exc),file=sys.stderr)
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
