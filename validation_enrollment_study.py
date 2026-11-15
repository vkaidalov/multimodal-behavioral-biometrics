#!/usr/bin/env python3
"""Validation-only enrollment / score-calibration study for HMOG/FETA.

All templates and per-user standardization use only enrollment sessions 1,2.
For the cross-fitted operating threshold for subject U, exclude every row
involving U (as target OR source) from calibration, preventing subject leakage.
Models, embeddings, and data splits remain frozen; NO test data accessed.

All reported per-identity ROC/AUC/EER are based on the SAME fixed probe and
impostor samples across methods.  ROC is invariant to positive affine per-user
score normalization, but global operating thresholds are not.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

SCENARIOS = {
    'feta_6s_scroll': ('FETA','iPhone 6s','scroll'),
    'feta_6s_swipe': ('FETA','iPhone 6s','swipe'),
    'feta_7_scroll': ('FETA','iPhone 7','scroll'),
    'feta_7_swipe': ('FETA','iPhone 7','swipe'),
    'feta_x_scroll': ('FETA','iPhone X','scroll'),
    'feta_x_swipe': ('FETA','iPhone X','swipe'),
    'hmog_sitting': ('HMOG','Samsung Galaxy S4','reading_sitting'),
    'hmog_walking': ('HMOG','Samsung Galaxy S4','reading_walking'),
}
METHODS = ('session_balanced_centroid','pooled_centroid',
           'max_session_score','min_session_score','top5_template_score')
NORM_VARIANTS = ('raw', 'enrollment_cross_session_z')


def unit(a):
    a=np.asarray(a,dtype=np.float64)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def rocs(y,s):
    y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
    if len(np.unique(y))!=2:raise ValueError('ROC needs two classes')
    fpr,tpr,th=roc_curve(y,s,drop_intermediate=False)
    fnr=1-tpr;gap=fpr-fnr
    crossing=np.flatnonzero(gap>=0)
    if not len(crossing):eer=float((fpr[-1]+fnr[-1])/2)
    elif crossing[0]==0:eer=float((fpr[0]+fnr[0])/2)
    else:
        j=int(crossing[0]);i=j-1
        alpha=float(-gap[i]/(gap[j]-gap[i])) if gap[j]!=gap[i] else 0.0
        eer=float((fpr[i]+alpha*(fpr[j]-fpr[i])+fnr[i]+alpha*(fnr[j]-fnr[i]))/2)
    return float(roc_auc_score(y,s)), eer


def get_session_ids(splits, scenario, user):
    ds,model,task=SCENARIOS[scenario]
    return list(map(str,splits['session_index'][ds][model][str(user)][task]))


def permitted_users(splits,scenario):
    ds,model,task=SCENARIOS[scenario]
    if ds=='FETA':users=splits['splits']['FETA'][model]['validation']
    else:users=splits['splits']['HMOG'][task]['validation']
    return set(map(str,users))


def load_embeddings(path):
    with np.load(path,allow_pickle=False) as f:
        emb=f['embeddings'].astype(np.float64)
        attrs={key:f[key].copy() for key in
               ('scenario','dataset','user_id','session_id','repetition','start_ms','end_ms')}
        extra=json.loads(str(f['extra_json']))
    if emb.ndim!=2 or emb.shape[1]!=64 or not np.isfinite(emb).all():
        raise ValueError('Expected finite 64D embeddings')
    return unit(emb),attrs,extra


def prepare_groups(emb,fields,splits):
    grouped=defaultdict(lambda:defaultdict(list))
    for i,(sc,u) in enumerate(zip(fields['scenario'],fields['user_id'])):
        sc=str(sc);u=str(u)
        if sc not in SCENARIOS:raise ValueError(f'Unexpected scenario: {sc}')
        if u not in permitted_users(splits,sc):
            raise ValueError(f'Non-validation user in {sc}: {u}')
        grouped[sc][u].append(i)
    if set(grouped)!=set(SCENARIOS):raise ValueError('Expected eight validation scenarios')
    return grouped


def _topk_similarity(X,T,k=5):
    similarities=X@T.T
    n=min(k,similarities.shape[1])
    return np.partition(similarities,-n,axis=1)[:,-n:].mean(axis=1)


def templates(emb,e1,e2,z_std_floor=0.05):
    a=unit(emb[e1].mean(axis=0)).reshape(-1)
    b=unit(emb[e2].mean(axis=0)).reshape(-1)
    pooled=unit(emb[np.concatenate([e1,e2])].mean(axis=0)).reshape(-1)
    balanced=unit((a+b)/2).reshape(-1)
    all_templates=emb[np.concatenate([e1,e2])]
    # Method-matched, disjoint-session, enrollment-only score calibration.
    # For centroid methods the cross-session template is a single centroid.
    # For top5 the cross-session template is the five nearest *other-session*
    # enrollment gestures (avoids matching a gesture to itself).
    cross_centroid=np.concatenate([emb[e1]@b,emb[e2]@a])
    cross_top5=np.concatenate([
        _topk_similarity(emb[e1],emb[e2]),
        _topk_similarity(emb[e2],emb[e1])])
    stats={}
    for method in METHODS:
        sample=cross_top5 if method=='top5_template_score' else cross_centroid
        stats[method]={'mu':float(sample.mean()),
                       'std':float(max(sample.std(ddof=0),z_std_floor))}
    return {'balanced':balanced,'pooled':pooled,'s1':a,'s2':b,
            'enroll':all_templates,'stats':stats}


def method_scores(probes,tpl):
    s1=probes@tpl['s1'];s2=probes@tpl['s2']
    # Top-5 nearest enrollment samples; compare each probe only with genuine
    # enrollment templates for the target.  Do NOT use future probes as templates.
    top=_topk_similarity(probes,tpl['enroll'])
    return {
        'session_balanced_centroid':probes@tpl['balanced'],
        'pooled_centroid':probes@tpl['pooled'],
        'max_session_score':np.maximum(s1,s2),
        'min_session_score':np.minimum(s1,s2),
        'top5_template_score':top,
    }


def _group_probe_indices(field,source_indices,session_ids):
    by_group=defaultdict(list)
    for i in source_indices:
        sid=str(field['session_id'][i])
        if sid in session_ids[2:]:
            by_group[(sid,str(field['repetition'][i]))].append(i)
    out=[]
    for (session,rep),indexes in sorted(by_group.items()):
        indexes.sort(key=lambda i:(float(field['start_ms'][i]),float(field['end_ms'][i]),i))
        out.extend(indexes)
    return np.asarray(out,dtype=np.int64)


def build_scenario(emb,attrs,users,splitdoc,sc,seed,impostor_per_source,z_std_floor=0.05):
    session_ids={u:get_session_ids(splitdoc,sc,u) for u in users}
    enrollment={};probes={}
    for u,idx in sorted(users.items()):
        ids=session_ids[u]
        present={str(attrs['session_id'][i]) for i in idx}
        if set(ids)!=present:raise ValueError(f'Missing session {sc}/{u}: {present} != {set(ids)}')
        e1=np.asarray([i for i in idx if str(attrs['session_id'][i])==ids[0]],dtype=int)
        e2=np.asarray([i for i in idx if str(attrs['session_id'][i])==ids[1]],dtype=int)
        if len(e1)==0 or len(e2)==0:raise ValueError('Empty enrollment session')
        enrollment[u]=templates(emb,e1,e2,z_std_floor)
        probes[u]=_group_probe_indices(attrs,idx,ids)
        if len(probes[u])==0:raise ValueError('No probe gestures')
    # Exact same ordered sampling logic as the original evaluator for k=1.
    rng=np.random.default_rng(seed+1)
    raw=[]
    for target in sorted(users):
        source_vecs=[];source_names=[];y=[]
        for source in sorted(users):
            indexes=probes[source]
            if source!=target and impostor_per_source>0 and len(indexes)>impostor_per_source:
                selected=np.sort(rng.choice(len(indexes),size=impostor_per_source,replace=False))
                indexes=indexes[selected]
            source_vecs.extend(indexes.tolist())
            source_names.extend([source]*len(indexes))
            y.extend([int(source==target)]*len(indexes))
        X=emb[np.asarray(source_vecs,dtype=np.int64)]
        scores=method_scores(X,enrollment[target]); y=np.asarray(y,dtype=np.int8)
        source_names=np.asarray(source_names,dtype=np.str_)
        for meth,s in scores.items():
            raw.append((target,source_names,y,meth,s))
    return raw,enrollment


def analyze_scenario(sc,raw,enrollment,target_far):
    users=sorted(enrollment)
    data={}
    for method in METHODS:
        for variant in NORM_VARIANTS:
            per={}
            for target,source,y,met,base in raw:
                if met!=method:continue
                tpl=enrollment[target]
                s=np.asarray(base,dtype=np.float64)
                stats=tpl['stats'][method]
                if variant=='enrollment_cross_session_z':s=(s-stats['mu'])/stats['std']
                auc,eer=rocs(y,s)
                per[target]={'target':target,'source':source,'label':y,'score':s,
                             'auc':auc,'eer':eer,'mu':stats['mu'],'std':stats['std']}
            assert set(per)==set(users)
            # Cross-fitted threshold: for target U, use impostor scores for
            # targets != U and sources != U.  Neither verification scores nor
            # impostor probe scores involving U calibrate their own threshold.
            rows=[]
            for target in users:
                others=[v for t,p in per.items() if t!=target for v in
                        [p['score'][(p['label']==0)&(p['source']!=target)]]]
                impostor=np.concatenate(others)
                threshold=float(np.nextafter(np.quantile(impostor,1-target_far),np.inf))
                p=per[target];y=p['label'];s=p['score']
                far=float(np.mean(s[y==0]>=threshold))
                frr=float(np.mean(s[y==1]<threshold))
                rows.append({'scenario':sc,'method':method,'score_scale':variant,
                             'user_id':target,'auc':p['auc'],'eer':p['eer'],
                             'crossfit_far':far,'crossfit_frr':frr,
                             'crossfit_balanced_accuracy':1-(far+frr)/2,
                             'crossfit_threshold':threshold,
                             'genuine_count':int(np.count_nonzero(y==1)),
                             'impostor_count':int(np.count_nonzero(y==0)),
                             'enroll_cross_session_mu':p['mu'],
                             'enroll_cross_session_std':p['std']})
            imp_all=np.concatenate([p['score'][p['label']==0] for p in per.values()])
            # This is the ONLY threshold suitable to freeze after selecting
            # a method and applying to TEST; not reestimated on test.
            final_threshold=float(np.nextafter(np.quantile(imp_all,1-target_far),np.inf))
            data[(method,variant)]=(rows,final_threshold)
    return data

def paired_cluster_bootstrap(peruser, out, seed=20261112, draws=10000):
    """Paired cluster bootstrap over UNIQUE users, preserving repeated scenarios.

    FETA scroll/swipe of the same person and HMOG sitting/walking of the same
    person are sampled together.  Each scenario then receives equal weight.
    """
    base=peruser[(peruser.method=='session_balanced_centroid') &
                 (peruser.score_scale=='raw')]
    best=peruser[(peruser.method=='top5_template_score') &
                 (peruser.score_scale=='raw')]
    fields=['auc','eer','crossfit_far','crossfit_frr']
    merged=base.merge(best,on=['scenario','user_id'],suffixes=('_baseline','_top5'),validate='one_to_one')
    unique=sorted(set(merged.user_id));scenario_names=sorted(set(merged.scenario))
    n=len(unique);lookup={u:i for i,u in enumerate(unique)}
    diff_arrays={name:np.zeros((len(scenario_names),n)) for name in fields}
    mask=np.zeros((len(scenario_names),n))
    for j,sc in enumerate(scenario_names):
        for _,row in merged[merged.scenario==sc].iterrows():
            i=lookup[row.user_id];mask[j,i]=1
            for key in fields:
                diff_arrays[key][j,i]=float(row[f'{key}_top5']-row[f'{key}_baseline'])
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(n,np.full(n,1/n),size=draws).T
    denominators=mask@counts
    good=np.all(denominators>0,axis=0)
    estimates={}
    for key,values in diff_arrays.items():
        boot=np.mean((values@counts[:,good])/denominators[:,good],axis=0)
        observed=float(np.mean(values.sum(axis=1)/mask.sum(axis=1)))
        estimates[key]={'top5_minus_baseline':observed,
                        'bootstrap_95ci':[float(x) for x in np.quantile(boot,[0.025,0.975])],
                        'unit':'fraction','unique_users':n,'draws_valid':int(np.sum(good))}
    (out/'paired_cluster_bootstrap.json').write_text(json.dumps(estimates,indent=2),encoding='utf-8')
    pairs=[]
    for sc in scenario_names:
        a=merged[merged.scenario==sc]
        pairs.append({'scenario':sc,'users':len(a),
            **{f'delta_top5_minus_centroid_{k}':float((a[f'{k}_top5']-a[f'{k}_baseline']).mean())
               for k in fields}})
    pd.DataFrame(pairs).to_csv(out/'paired_scenario_differences.csv',index=False)
    return estimates


def write_report(peruser,summary,out,extra,args):
    out.mkdir(parents=True,exist_ok=True)
    peruser.to_csv(out/'per_user_validation.csv',index=False)
    summary.to_csv(out/'scenario_validation.csv',index=False)
    grouped=summary.groupby(['method','score_scale'],sort=True)
    agg=grouped.agg(mean_macro_auc=('macro_auc','mean'),
                    mean_macro_eer=('macro_eer','mean'),
                    mean_crossfit_far=('macro_crossfit_far','mean'),
                    mean_crossfit_frr=('macro_crossfit_frr','mean'),
                    mean_crossfit_balanced_accuracy=('macro_crossfit_balanced_accuracy','mean'),
                    scenarios=('scenario','nunique')).reset_index()
    agg.sort_values(['mean_macro_auc','mean_macro_eer'],ascending=[False,True],inplace=True)
    agg.to_csv(out/'method_comparison_validation.csv',index=False)
    baseline=agg[(agg.method=='session_balanced_centroid')&(agg.score_scale=='raw')]
    if len(baseline):
        base=baseline.iloc[0]
        print('[BASELINE] macro AUC',round(base['mean_macro_auc'],6),
              'macro EER',round(base['mean_macro_eer'],6),flush=True)
    estimates=paired_cluster_bootstrap(peruser,out,args.seed)
    print('[PAIRED] Top5 minus centroid, clustered 95% CIs:')
    print(json.dumps(estimates,indent=2))
    print('[COMPARISON] Rankings by 8-scenario macro AUC:')
    print(agg.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    meta={'extra_from_embeddings':extra,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          'validation_only':True,
          'threshold_crossfit':'leave-one-identity-out: remove matching target and source',
          'test_threshold':'Pooled validation impostor 95th percentile, frozen only after method selection',
          'per_user_auc_eer':'Computed on identical fixed probe and impostor sets for all methods',
          'score_scale_note':'Method-specific other-session genuine enrollment score calibration; positive affine per-user transformation leaves ROC/AUC/EER invariant',
          'sha256_validation_embeddings':hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        raw=agg[agg.score_scale=='raw'].sort_values('mean_macro_auc')
        fig,ax=plt.subplots(figsize=(8.4,4.7));ax.barh(raw.method,raw.mean_macro_auc)
        ax.set_xlabel('Validation mean macro AUC (8 equally weighted scenarios)')
        ax.set_xlim(0,1);ax.grid(axis='x',alpha=.22)
        fig.tight_layout();fig.savefig(out/'enrollment_auc.svg');plt.close(fig)
        frr=agg.pivot(index='method',columns='score_scale',values='mean_crossfit_frr')
        frr=frr.sort_values('raw')
        ax=frr.plot.barh(figsize=(9,4.8));ax.set_xlabel('Cross-fitted validation macro FRR (FAR target 5%)')
        ax.set_xlim(0,1);ax.grid(axis='x',alpha=.2)
        ax.figure.tight_layout();ax.figure.savefig(out/'crossfit_frr.svg');plt.close(ax.figure)
    except ImportError:
        print('[INFO] matplotlib not installed; skipping optional SVG',flush=True)
    return agg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path('experiments/splits/hmog_feta_splits_seed_20261112.json'))
    p.add_argument('--output',type=Path,default=Path('experiments/tuning/enrollment_study_global_temp005'))
    p.add_argument('--impostor-per-source',type=int,default=100)
    p.add_argument('--target-far',type=float,default=0.05)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--z-std-floor',type=float,default=0.05)
    args=p.parse_args()
    if not 0<args.target_far<1:raise ValueError('target-far must be (0,1)')
    if args.z_std_floor<=0:raise ValueError('z-std-floor must be positive')
    t0=time.perf_counter();print('[LOAD] Loading validation embeddings...',flush=True)
    emb,attrs,extra=load_embeddings(args.embeddings)
    if extra.get('role')!='validation':raise ValueError('TEST embeddings prohibited in this study')
    if extra.get('model_id')!='global':raise ValueError('Expected global encoder embeddings')
    sha=hashlib.sha256(args.splits.read_bytes()).hexdigest()
    if sha!=extra.get('splits_sha256'):
        raise ValueError('Frozen split hash mismatch: cannot combine these data')
    splits=json.loads(args.splits.read_text(encoding='utf8'))
    groups=prepare_groups(emb,attrs,splits)
    print(f'[LOAD] {len(emb):,} gestures, {len(groups)} scenarios, {time.perf_counter()-t0:.1f}s',flush=True)
    peruser=[];scenarios=[];frozen={}
    for sc,users in sorted(groups.items()):
        t=time.perf_counter()
        print(f'[START] {sc}: {len(users)} identities',flush=True)
        samples,templates_by_user=build_scenario(emb,attrs,users,splits,sc,args.seed,args.impostor_per_source,args.z_std_floor)
        variants=analyze_scenario(sc,samples,templates_by_user,args.target_far)
        for (method,variant),(rows,threshold) in variants.items():
            peruser.extend(rows)
            scenarios.append({'scenario':sc,'method':method,'score_scale':variant,
                              'users':len(rows),'macro_auc':float(np.mean([r['auc'] for r in rows])),
                              'macro_eer':float(np.mean([r['eer'] for r in rows])),
                              'macro_crossfit_far':float(np.mean([r['crossfit_far'] for r in rows])),
                              'macro_crossfit_frr':float(np.mean([r['crossfit_frr'] for r in rows])),
                              'macro_crossfit_balanced_accuracy':float(np.mean([r['crossfit_balanced_accuracy'] for r in rows])),
                              'frozen_validation_threshold':threshold})
            frozen[f'{sc}/{method}/{variant}']=threshold
        print(f'[DONE] {sc}: {time.perf_counter()-t:.1f}s; elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    peruser=pd.DataFrame(peruser)
    summary=pd.DataFrame(scenarios)
    write_report(peruser,summary,args.output,extra,args)
    (args.output/'all_validation_thresholds.json').write_text(json.dumps({
        'status':'CANDIDATE_CALIBRATIONS_UNSELECTED',
        'warning':'Select method using validation ONLY before applying thresholds to TEST',
        'thresholds':frozen,'role':'validation','model_id':'global',
        'embedding_sha256':hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),
        'splits_sha256':sha,'target_far':args.target_far},indent=2),encoding='utf8')
    print(f'[COMPLETE] {args.output} | elapsed {time.perf_counter()-t0:.1f}s',flush=True)


if __name__=='__main__':main()
