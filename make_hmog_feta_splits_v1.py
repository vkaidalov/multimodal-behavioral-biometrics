#!/usr/bin/env python3
"""Freeze subject-disjoint HMOG/FETA 60/20/20 splits at threshold=20.

Recommended: run with --audit-dir on the complete v2.0 audit output.
Fallback --provisional uses earlier uploaded intersection + legacy HMOG roster;
it is NOT a verified final split and must NOT be used for model evaluation.
Python >=3.10; no third-party packages.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path

SEED=20261112
THRESHOLD=20
MODELS=('iPhone 7','iPhone X','iPhone 6s')
SPLITS=('train','validation','test')
# Precommitted approximate 60/20/20 quotas. Same HMOG subjects stay in
# the same global partition across both activities and the future combined task.
HMOG_QUOTAS={'both':(32,11,11),'sitting_only':(1,0,1),'walking_only':(14,5,4)}
FETA_QUOTAS={'iPhone 7':(38,12,12),'iPhone X':(38,12,12),'iPhone 6s':(36,12,12)}
KEYS=('dataset','device_model','user_id','task')
REQUIRED_FILES=('summary.json','eligible_user_ids_by_threshold.csv',
                'session_task_counts.csv','first_four_session_manifest.csv',
                'task_intersection.csv')


def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def csv_key(row):return tuple(row[k].strip() for k in KEYS)

def assert_audit_parameters(info):
    if not str(info.get('version','')).startswith('hmog-feta-gesture-audit-2.0'):
        raise ValueError('Expected the HMOG/FETA v2.0 audit, found '+str(info.get('version')))
    par=info.get('parameters',{})
    if par.get('feta_criteria')!='min3_max_radial_strict_gt5' or par.get('hmog_portrait_all_touch_points') is not True:
        raise ValueError('Audit does not certify FETA >=3, radial >5 and HMOG portrait-only preprocessing')
    expected={'min_sensor_samples':3,'min_sensor_pairs':1,'edge_gap_ms':40.0,
              'internal_gap_ms':120.0,'pair_tolerance_ms':25.0}
    for key,val in expected.items():
        if float(par.get(key,float('nan')))!=float(val):
            raise ValueError(f'Audit motion setting {key} differs from frozen protocol: {par.get(key)!r} != {val!r}')
    if 20 not in info.get('thresholds',[]):raise ValueError('Audit missing threshold=20')


def load_audit(directory):
    d=Path(directory)
    missing=[p for p in REQUIRED_FILES if not (d/p).exists()]
    if missing:raise FileNotFoundError('Missing audit outputs: '+', '.join(missing))
    summary=json.loads((d/'summary.json').read_text(encoding='utf-8-sig'))
    assert_audit_parameters(summary)
    chosen=[r for r in read_csv(d/'eligible_user_ids_by_threshold.csv')
            if r['count_mode']=='strict' and int(r['threshold'])==THRESHOLD and
            ((r['dataset']=='HMOG' and r['task'] in ('reading_sitting','reading_walking')) or
             (r['dataset']=='FETA' and r['device_model'] in MODELS and r['task'] in ('scroll','swipe')))]
    cohorts=defaultdict(set)
    for r in chosen:
        key=(r['dataset'],r['device_model'],r['task'])
        if r['user_id'] in cohorts[key]:raise ValueError('Repeated eligible subject '+str(key))
        cohorts[key].add(r['user_id'])
    meta={}
    for row in read_csv(d/'session_task_counts.csv'):
        key=csv_key(row)
        meta.setdefault(key,[]).append(row)
    manifest=defaultdict(list)
    for r in read_csv(d/'first_four_session_manifest.csv'):
        manifest[csv_key(r)].append(r)
    counts={};sessions={}
    for (dataset,model,task),users in cohorts.items():
        for user in users:
            key=(dataset,model,user,task)
            all_rows=meta.get(key,[])
            if len(all_rows)<4:raise ValueError('Fewer than 4 observed sessions: '+str(key))
            if not all(int(r['order_verified'])==1 and math.isfinite(float(r['session_time'])) for r in all_rows):
                raise ValueError('Unverified chronology: '+str(key))
            times=[float(r['session_time']) for r in all_rows]
            if len(set(times))!=len(times):raise ValueError('Duplicate session timestamps: '+str(key))
            ordered=sorted(all_rows,key=lambda r:float(r['session_time']))[:4]
            if len({r['session_id'] for r in all_rows})!=len(all_rows):
                raise ValueError('Duplicate session IDs: '+str(key))
            rows=sorted(manifest.get(key,[]),key=lambda r:int(r['session_rank']))
            if len(rows)!=4 or [r['session_id'] for r in rows]!=[r['session_id'] for r in ordered] or [int(r['session_rank']) for r in rows]!=[1,2,3,4]:
                raise ValueError('Manifest does not match first 4 chronologically: '+str(key))
            for a,b in zip(ordered,rows):
                if int(a['gestures_strict_multimodal'])!=int(b['gestures_strict_multimodal']):
                    raise ValueError('Manifest/session count mismatch: '+str(key))
            sc=[int(r['gestures_strict_multimodal']) for r in ordered]
            if min(sc)<THRESHOLD:raise ValueError('Ineligible first-four session: '+str(key)+' '+str(sc))
            if any(int(r['gestures_feta_touch_usable'])<int(r['gestures_strict_multimodal']) for r in ordered):
                raise ValueError('Invalid strict>touch count: '+str(key))
            counts[key]=sc
            sessions[key]=[r['session_id'] for r in ordered]
    # FETA cohorts must overlap exactly, allowing the same device-level split.
    for model in MODELS:
        a=cohorts[('FETA',model,'scroll')];b=cohorts[('FETA',model,'swipe')]
        if a!=b:raise ValueError(f'FETA {model} scroll/swipe rosters differ; global device-level split requires a revised protocol')
    sit=cohorts[('HMOG','Samsung Galaxy S4','reading_sitting')]
    walk=cohorts[('HMOG','Samsung Galaxy S4','reading_walking')]
    if (len(sit),len(walk),len(sit&walk))!=(56,77,54):
        raise ValueError(f'HMOG cohort sizes have changed: sitting={len(sit)}, walking={len(walk)}, shared={len(sit&walk)}')
    for model,quotas in FETA_QUOTAS.items():
        if len(cohorts[('FETA',model,'scroll')])!=sum(quotas):
            raise ValueError(f'FETA {model}: unexpected group size, expected {sum(quotas)}')
    # Ensure metadata mapping and count consistency, including overlap CSV.
    intersections=read_csv(d/'task_intersection.csv')
    for dataset,model,taska,taskb in [('HMOG','Samsung Galaxy S4','reading_sitting','reading_walking')]+[
        ('FETA',m,'scroll','swipe') for m in MODELS]:
        matching=[r for r in intersections if r['dataset']==dataset and r['device_model']==model
                  and r['task_a']==taska and r['task_b']==taskb and r['count_mode']=='strict'
                  and int(r['threshold'])==20]
        if len(matching)!=1:raise ValueError('Missing unique intersection row for '+dataset+'/'+model)
        ground=set(filter(None,matching[0]['both_user_ids'].split('|')))
        calc=cohorts[(dataset,model,taska)] & cohorts[(dataset,model,taskb)]
        if ground!=calc or int(matching[0]['users_both'])!=len(calc):
            raise ValueError('Intersection CSV conflicts with eligible roster for '+dataset+'/'+model)
    return cohorts,counts,sessions,{'audit_version':summary['version'],
            'audit_file_sha256':{f:sha(d/f) for f in REQUIRED_FILES}}


def load_provisional(intersection_path,legacy_path):
    rows=read_csv(intersection_path)
    cohorts=defaultdict(set)
    for r in rows:
        if r['count_mode']!='strict' or int(r['threshold'])!=20:continue
        ds,model=r['dataset'],r['device_model']
        if ds=='FETA' and model in MODELS:
            ids=set(filter(None,r['both_user_ids'].split('|')))
            if len(ids)!=int(r['users_both']) or int(r['task_a_users'])!=len(ids) or int(r['task_b_users'])!=len(ids):
                raise ValueError('FETA intersection not equal to both tasks for '+model)
            cohorts[('FETA',model,'scroll')]=set(ids)
            cohorts[('FETA',model,'swipe')]=set(ids)
        elif ds=='HMOG':
            common=set(filter(None,r['both_user_ids'].split('|')))
    if not common or len(common)!=54:raise ValueError('Expected 54 shared HMOG users')
    old=read_csv(legacy_path)
    for task in ('reading_sitting','reading_walking'):
        ids={r['user_id'] for r in old if r['task']==task and
             int(r['distinct_session_numbers'])==4 and int(r['min_usable_in_session'])>=20}
        if not common<=ids:raise ValueError('Legacy v1.5 IDs disagree with the verified v2.0 intersection')
        cohorts[('HMOG','Samsung Galaxy S4',task)]=ids
    sit=cohorts[('HMOG','Samsung Galaxy S4','reading_sitting')]
    walk=cohorts[('HMOG','Samsung Galaxy S4','reading_walking')]
    if (len(sit),len(walk),len(sit&walk))!=(56,77,54):raise ValueError('Legacy HMOG rosters have wrong cohort size')
    for model,q in FETA_QUOTAS.items():
        if len(cohorts[('FETA',model,'scroll')])!=sum(q):raise ValueError('Wrong FETA size '+model)
    return cohorts,{}, {},{'provisional_sources':{
        'v2_task_intersection_sha256':sha(intersection_path),
        'legacy_v1_5_user_task_pairs_sha256':sha(legacy_path)}}


def stable_seed(base,label):
    return int.from_bytes(hashlib.sha256(f'{base}|{label}'.encode()).digest()[:8],'big')


def balance_score(groups,ids,counts,tasks):
    # Predeclared stratification based ONLY on the FIRST TWO (enrollment)
    # sessions, never the held-out genuine-test sessions 3 and 4.
    dim=[]
    for u in ids:
        vals=[]
        for task in tasks:
            c=counts[task,u]
            vals.extend((math.log1p(sum(c[:2])),math.log1p(min(c[:2]))))
        dim.append(vals)
    n=len(ids);d=len(dim[0])
    mean=[sum(v[j] for v in dim)/n for j in range(d)]
    sd=[max(0.15,math.sqrt(sum((v[j]-mean[j])**2 for v in dim)/n)) for j in range(d)]
    vec={u:dim[i] for i,u in enumerate(ids)}
    score=0.0
    for ss in groups:
        if not ss:continue
        for j in range(d):
            gap=(sum(vec[u][j] for u in ss)/len(ss)-mean[j])/sd[j]
            score+=gap**2
    return score


def allocate(ids,quota,seed,label,counts=None,tasks=(),tries=1500):
    ids=sorted(ids)
    if len(ids)!=sum(quota):raise ValueError(f'Quota mismatch for {label}: {len(ids)} vs {quota}')
    rng=random.Random(stable_seed(seed,label))
    if counts and tasks:
        features={(task,u):counts[task,u] for task in tasks for u in ids}
    else:features=None
    best=None;best_score=math.inf
    for attempt in range(tries if features else 1):
        perm=ids.copy();rng.shuffle(perm)
        groups=[];pos=0
        for n in quota:
            groups.append(sorted(perm[pos:pos+n]));pos+=n
        sc=balance_score(groups,ids,features,tasks) if features else 0
        if sc<best_score:
            best_score=sc;best=groups
    return {name:best[i] for i,name in enumerate(SPLITS)}, round(best_score,6) if features else None


def generate(cohorts,counts,sessions,prov,seed=SEED,tries=1500,provisional=False):
    splits={'FETA':{},'HMOG':{}}
    balance={}
    for model in MODELS:
        users=cohorts[('FETA',model,'scroll')]
        features={('scroll',u):counts[('FETA',model,u,'scroll')] for u in users} if counts else {}
        if counts:features.update({('swipe',u):counts[('FETA',model,u,'swipe')] for u in users})
        d,sc=allocate(users,FETA_QUOTAS[model],seed,'FETA/'+model,features,('scroll','swipe'),tries)
        splits['FETA'][model]=d;balance['FETA/'+model]=sc
    sit=cohorts[('HMOG','Samsung Galaxy S4','reading_sitting')]
    walk=cohorts[('HMOG','Samsung Galaxy S4','reading_walking')]
    subsets={'both':sit&walk,'sitting_only':sit-walk,'walking_only':walk-sit}
    assignments={}
    for group,users in subsets.items():
        tasks={'both':('reading_sitting','reading_walking'),
               'sitting_only':('reading_sitting',),'walking_only':('reading_walking',)}[group]
        features={(task,u):counts[('HMOG','Samsung Galaxy S4',u,task)] for u in users for task in tasks} if counts else {}
        d,sc=allocate(users,HMOG_QUOTAS[group],seed,'HMOG/'+group,features,tasks,tries)
        assignments[group]=d;balance['HMOG/'+group]=sc
    for task,subsetnames in [('reading_sitting',('both','sitting_only')),
                             ('reading_walking',('both','walking_only')),
                             ('reading_combined',('both',))]:
        splits['HMOG'][task]={split:sorted(u for grp in subsetnames for u in assignments[grp][split])
                              for split in SPLITS}
    session_index={}
    if sessions:
        for (ds,model,user,task),ids in sorted(sessions.items()):
            session_index.setdefault(ds,{}).setdefault(model,{}).setdefault(user,{})[task]=ids
    return {
        'schema_version':'1.0',
        'status':'PROVISIONAL_NEEDS_V2_AUDIT' if provisional else 'FROZEN_VERIFIED_AGAINST_V2_AUDIT',
        'seed':seed,
        'protocol':{
            'threshold_gestures_per_each_first_four_sessions':THRESHOLD,
            'count_mode':'strict','geometry':'DOWN-UP; >=3 touch points; max radial distance >5 native px',
            'hmog_orientation':'Phone_orientation=0 for EVERY touch sample and qualifying in-contact motion sample',
            'motion':'each accel/gyro >=3 samples; boundary<=40ms; internal gap<=120ms; pair<=25ms; >=1 pair',
            'split_type':'subject_disjoint_60_20_20_with_shared_subject_alignment',
            'session_roles':{'enrollment':[1,2],'genuine_test':[3,4]},
            'training_user_sessions':'all_available_sessions_for_the_assigned_task',
            'feta_models':list(MODELS),'hmog_combined':'intersection_only; not presently evaluated',
            'allocation_quota':{'FETA':{m:dict(zip(SPLITS,FETA_QUOTAS[m])) for m in MODELS},
                                'HMOG_strata':{g:dict(zip(SPLITS,q)) for g,q in HMOG_QUOTAS.items()}},
            'balancing':('best_of_'+str(tries)+'_predeclared_random_permutations_minimizing_enrollment_sessions_1_2_log_gesture_distribution'
                         if counts else 'deterministic_random_only_no_v2_per_user_gesture_counts')},
        'provenance':prov,'balance_objective':balance,'splits':splits,'session_index':session_index,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,default=Path('output_data/hmog_feta_feta_criterion_v2_0'))
    p.add_argument('--output',type=Path,default=Path('output_data/hmog_feta_splits_seed_20261112.json'))
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--candidates',type=int,default=1500)
    p.add_argument('--provisional',action='store_true',help='Use overlap from v2 + older HMOG v1.5 roster (NOT final)')
    p.add_argument('--intersection-csv',type=Path,default=Path('task_intersection.csv'))
    p.add_argument('--legacy-hmog-csv',type=Path,default=Path('user_task_pairs.csv'))
    a=p.parse_args()
    if a.candidates<1:raise ValueError('--candidates must be positive')
    if a.provisional:
        co,ct,ss,prov=load_provisional(a.intersection_csv,a.legacy_hmog_csv)
    else:co,ct,ss,prov=load_audit(a.audit_dir)
    result=generate(co,ct,ss,prov,a.seed,a.candidates,a.provisional)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Written:',a.output.resolve(),'\nStatus:',result['status'])
    for ds in ('FETA','HMOG'):
        for scen,sp in result['splits'][ds].items():
            print(f'{ds} | {scen}: '+', '.join(f'{k}={len(sp[k])}' for k in SPLITS))
    return 0
if __name__=='__main__':raise SystemExit(main())
