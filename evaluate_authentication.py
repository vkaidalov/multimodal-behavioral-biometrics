#!/usr/bin/env python3
"""HMOG/FETA authentication evaluator with centroid / top-5 enrollment.

Legacy invocation (no --methods): writes evaluation/<role> as before.
New --methods centroid,top5: writes evaluation/method_study/<role>/<method>/,
never overwriting the legacy results. No test without frozen calibration.

For each target, k-window score is the MEAN of k individual gesture scores:
centroid: cosine(gesture, normalized 2-session balanced centroid)
top5: mean of five largest cosines to enrolled gestures from sessions 1+2.
Both methods use IDENTICAL verification windows and impostor sampling.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
from gesture_pipeline_data import file_hash
from gesture_evaluation_core import (WINDOWS, load_embedding_table,
    build_role_scenarios, roc_metrics, operational_metrics)
from gesture_enrollment_methods import METHODS, PreparedScenario


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def quantile_summary(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(np.mean(x)),'median':float(np.median(x)),
            'p90':float(np.percentile(x,90)),'p95':float(np.percentile(x,95))}


def bootstrap_mean_ci(values,seed=20261112,draws=1000):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return [float('nan'),float('nan')]
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(samples,[2.5,97.5])]


def save_scores(path,rows):
    cols=list(zip(*rows))
    names=('target_user','source_user','label','score','start_ms','end_ms',
           'session_id','repetition','collection_latency_s')
    values={}
    for i,key in enumerate(names):
        typ=np.str_ if i in (0,1,6,7) else (np.int8 if i==2 else np.float64)
        values[key]=np.asarray(cols[i],dtype=typ)
    np.savez_compressed(path,**values)


def save_user_rocs(path,curves):
    keys=sorted(curves);offset=[0]
    cols={'fpr':[],'tpr':[],'thresholds':[]}
    for user in keys:
        for name in cols:cols[name].append(curves[user][name])
        offset.append(offset[-1]+len(curves[user]['fpr']))
    np.savez_compressed(path,user_ids=np.asarray(keys,dtype=np.str_),
        offsets=np.asarray(offset,dtype=np.int64),
        **{k:np.concatenate(v) for k,v in cols.items()})


def evaluate_records(sc,k,method,records,users,args,output,calibration):
    """Evaluates a fixed list of tuples, returns (per-user, scenario, threshold).

    Validation operational FAR/FRR at a pooled threshold are explicitly
    in-sample. Separate leave-one-identity-out threshold metrics are produced
    for the method study to prevent target/source-specific calibration leakage.
    """
    output.mkdir(parents=True,exist_ok=True)
    base={'model_id':args.model_id,'role':args.role,'scenario':sc,'k':k,
          'enrollment_method':method}
    key=f'{sc}/k{k}'
    if not records:
        if args.role=='validation':threshold=None
        else:threshold=calibration['thresholds'][key]
        per=[{**base,'user_id':u,'status':'insufficient_windows',
              'genuine_count':0,'impostor_count':0} for u in sorted(users)]
        scrow={**base,'status':'insufficient_windows','users':len(users),
               'evaluable_users':0,'coverage':0.,'macro_auc':float('nan'),
               'macro_eer':float('nan'),'pooled_auc':float('nan'),
               'pooled_eer':float('nan'),'threshold':threshold,
               'operating_threshold_status':'unavailable',
               'genuine_windows':0,'impostor_windows':0,
               'macro_crossfit_far':float('nan'),
               'macro_crossfit_frr':float('nan')}
        print(f'[COVERAGE] {sc} method={method} k={k}: insufficient windows',flush=True)
        return per,scrow,threshold
    target=np.asarray([r[0] for r in records],dtype=np.str_)
    source=np.asarray([r[1] for r in records],dtype=np.str_)
    labels=np.asarray([r[2] for r in records],dtype=np.int8)
    scores=np.asarray([r[3] for r in records],dtype=np.float64)
    delay=np.asarray([r[8] for r in records],dtype=np.float64)
    if args.role=='validation':
        impostor_scores=scores[labels==0]
        threshold=float(np.nextafter(np.quantile(impostor_scores,1-args.target_far),np.inf))
    else:
        raw=calibration['thresholds'][key]
        threshold=float(raw) if raw is not None else None
    per=[];per_user_rocs={}
    for user in sorted(users):
        take=target==user;y=labels[take];s=scores[take]
        n_genuine=int(np.count_nonzero(y==1));n_impostor=int(np.count_nonzero(y==0))
        if n_genuine==0 or n_impostor==0:
            per.append({**base,'user_id':user,'status':'insufficient_windows',
                        'genuine_count':n_genuine,'impostor_count':n_impostor})
            continue
        roc=roc_metrics(y,s)
        op=(operational_metrics(y,s,threshold) if threshold is not None else
            {name:float('nan') for name in
             ('far','frr','balanced_accuracy','genuine_accuracy','impostor_accuracy')})
        latency=quantile_summary(delay[take&(labels==1)])
        # Cross-fitted operating threshold on VALIDATION ONLY. Exclude each
        # user as target AND impostor source from that user's calibration.
        cross_th=None
        if args.role=='validation':
            calib=(labels==0)&(target!=user)&(source!=user)
            if np.any(calib):
                cross_th=float(np.nextafter(np.quantile(scores[calib],1-args.target_far),np.inf))
        cfop=(operational_metrics(y,s,cross_th) if cross_th is not None else
              {name:float('nan') for name in
               ('far','frr','balanced_accuracy','genuine_accuracy','impostor_accuracy')})
        r={**base,'user_id':user,'status':'ok',
           'auc':roc['auc'],'eer':roc['eer'],**op,'threshold':threshold,
           'operating_threshold_status':'calibrated' if threshold is not None else 'unavailable',
           'crossfit_threshold':cross_th,
           'crossfit_far':cfop['far'],'crossfit_frr':cfop['frr'],
           'crossfit_balanced_accuracy':cfop['balanced_accuracy'],
           'genuine_count':n_genuine,'impostor_count':n_impostor,
           'collection_latency_mean_s':latency['mean'],
           'collection_latency_median_s':latency['median'],
           'collection_latency_p90_s':latency['p90'],
           'collection_latency_p95_s':latency['p95']}
        per.append(r);per_user_rocs[user]=roc
    valid=[r for r in per if r['status']=='ok']
    if len(valid)<2:
        # Even when raw records exist, some users may be missing genuine data.
        # Keep explicit status rather than asking ROC for a single class.
        scrow={**base,'status':'insufficient_windows','users':len(users),
               'evaluable_users':len(valid),'coverage':len(valid)/len(users),
               'macro_auc':float('nan'),'macro_eer':float('nan'),
               'pooled_auc':float('nan'),'pooled_eer':float('nan'),
               'threshold':threshold,'operating_threshold_status':'unavailable',
               'genuine_windows':int((labels==1).sum()),'impostor_windows':int((labels==0).sum())}
        return per,scrow,threshold
    pooled=roc_metrics(labels,scores)
    grid=np.linspace(0.,1.,201)
    macro_tpr=np.mean([np.interp(grid,v['fpr'],v['tpr']) for v in per_user_rocs.values()],axis=0)
    aucs=[r['auc'] for r in valid];eers=[r['eer'] for r in valid]
    ci_auc=bootstrap_mean_ci(aucs,args.seed,args.bootstrap)
    ci_eer=bootstrap_mean_ci(eers,args.seed+3,args.bootstrap)
    cols=('far','frr','balanced_accuracy','genuine_accuracy','impostor_accuracy')
    scrow={**base,'status':'ok','users':len(users),'evaluable_users':len(valid),
           'coverage':len(valid)/len(users),
           'macro_auc':float(np.mean(aucs)),'macro_eer':float(np.mean(eers)),
           'pooled_auc':pooled['auc'],'pooled_eer':pooled['eer'],
           'macro_auc_ci95_low':ci_auc[0],'macro_auc_ci95_high':ci_auc[1],
           'macro_eer_ci95_low':ci_eer[0],'macro_eer_ci95_high':ci_eer[1],
           'threshold':threshold,
           'operating_threshold_status':'calibrated' if threshold is not None else 'unavailable',
           'genuine_windows':int((labels==1).sum()),
           'impostor_windows':int((labels==0).sum()),
           **{'macro_'+c:float(np.mean([r[c] for r in valid])) for c in cols},
           'macro_crossfit_far':float(np.mean([r['crossfit_far'] for r in valid])),
           'macro_crossfit_frr':float(np.mean([r['crossfit_frr'] for r in valid])),
           'macro_crossfit_balanced_accuracy':float(np.mean([r['crossfit_balanced_accuracy'] for r in valid])),
           'macro_latency_mean_s':float(np.mean([r['collection_latency_mean_s'] for r in valid])),
           'macro_latency_p95_s':float(np.mean([r['collection_latency_p95_s'] for r in valid])),
           'pooled_latency_mean_s':float(np.mean(delay[labels==1]))}
    save_scores(output/f'{sc}_k{k:02d}_scores.npz',records)
    save_user_rocs(output/f'{sc}_k{k:02d}_per_user_rocs.npz',per_user_rocs)
    np.savez_compressed(output/f'{sc}_k{k:02d}_scenario_roc.npz',
        fpr_grid=grid,macro_tpr=macro_tpr,pooled_fpr=pooled['fpr'],
        pooled_tpr=pooled['tpr'],pooled_thresholds=pooled['thresholds'])
    return per,scrow,threshold


def evaluate_scenario(task):
    sc,users,emb,meta,splits,args_d,outputs=task
    args=argparse.Namespace(**args_d)
    methods=args_d['_method_list'];windows=args_d['_parsed_windows']
    calibrations=args_d['_calibrations']
    print(f'[SCENARIO] {sc}: preparing {len(users)} users; '
          f'methods={methods}; k={len(windows)}',flush=True)
    prepared=PreparedScenario(emb,meta,users,splits,sc,methods,verbose=True)
    user_rows={m:[] for m in methods};scenario_rows={m:[] for m in methods}
    thresholds={m:{} for m in methods}
    for j,k in enumerate(windows,1):
        start=time.perf_counter()
        for method in methods:
            records=prepared.score_records(k,method,args.impostor_per_source,args.seed)
            if args.role=='test' and f'{sc}/k{k}' not in calibrations[method]['thresholds']:
                raise ValueError(f'Missing validation-calibrated threshold for {sc}/k{k}/{method}')
            urows,srow,thr=evaluate_records(sc,k,method,records,users,args,
                Path(outputs[method]),calibrations[method])
            srow['evaluation_seconds']=time.perf_counter()-start
            user_rows[method].extend(urows)
            scenario_rows[method].append(srow)
            if args.role=='validation':thresholds[method][f'{sc}/k{k}']=thr
            if srow['status']=='ok':
                print(f'[{args.role.upper()}] {sc} method={method} k={k}: '
                      f'EER={srow["macro_eer"]:.4f} AUC={srow["macro_auc"]:.4f} '
                      f'coverage={srow["coverage"]:.1%}',flush=True)
        print(f'[PROGRESS] {sc}: windows {j}/{len(windows)}; '
              f'k={k}, {time.perf_counter()-start:.1f}s',flush=True)
    return sc,user_rows,scenario_rows,thresholds


def scenario_task(sc,users,emb,meta,splits,args_d,outputs):
    indices=sorted(i for items in users.values() for i in items)
    mapping={old:new for new,old in enumerate(indices)}
    remapped={u:[mapping[i] for i in rows] for u,rows in users.items()}
    return sc,remapped,np.ascontiguousarray(emb[indices]),[meta[i] for i in indices],splits,args_d,outputs


def validation_comparison(per_method,outdir,windows):
    """Paired comparisons at scenario/k and balanced across scenarios.

    Coverage is reported separately; AUC/EER comparisons at k with missing
    users are NOT necessarily paired on identical sets of subjects.
    """
    if len(per_method)<2:return
    names=sorted(per_method)
    a,b=names[:2]
    rows=[];summary=[]
    amap={(r['scenario'],int(r['k'])):r for r in per_method[a]}
    bmap={(r['scenario'],int(r['k'])):r for r in per_method[b]}
    for key in sorted(set(amap)&set(bmap)):
        x=amap[key];y=bmap[key]
        if x['status']!='ok' or y['status']!='ok':continue
        rows.append({'scenario':key[0],'k':key[1],
                     'method_a':a,'method_b':b,
                     'auc_a':x['macro_auc'],'auc_b':y['macro_auc'],
                     'eer_a':x['macro_eer'],'eer_b':y['macro_eer'],
                     'coverage_a':x['coverage'],'coverage_b':y['coverage'],
                     'delta_auc_b_minus_a':y['macro_auc']-x['macro_auc'],
                     'delta_eer_b_minus_a':y['macro_eer']-x['macro_eer']})
    write_csv(outdir/'scenario_method_comparison.csv',rows)
    for k in windows:
        r=[v for v in rows if v['k']==k]
        if not r:continue
        summary.append({'k':k,'method_a':a,'method_b':b,'scenarios':len(r),
                        'mean_macro_auc_a':float(np.mean([x['auc_a'] for x in r])),
                        'mean_macro_auc_b':float(np.mean([x['auc_b'] for x in r])),
                        'mean_macro_eer_a':float(np.mean([x['eer_a'] for x in r])),
                        'mean_macro_eer_b':float(np.mean([x['eer_b'] for x in r])),
                        'mean_coverage_a':float(np.mean([x['coverage_a'] for x in r])),
                        'mean_coverage_b':float(np.mean([x['coverage_b'] for x in r]))})
    write_csv(outdir/'mean_scenario_comparison.csv',summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for field,title in [('auc','Mean validation macro AUC'),('eer','Mean validation macro EER'),
                            ('coverage','Mean coverage')]:
            fig,ax=plt.subplots(figsize=(8.3,4.8))
            for method,suffix in [(a,'a'),(b,'b')]:
                ax.plot([r['k'] for r in summary],
                        [r[f'mean_{"coverage" if field=="coverage" else "macro_"+field}_{suffix}']
                         for r in summary],marker='.',label=method)
            ax.set_xlabel('Consecutive gestures per window (k)')
            ax.set_ylabel(title)
            ax.legend();ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(outdir/f'{field}_versus_k.svg');plt.close(fig)
    except ImportError:
        print('[INFO] matplotlib unavailable; no SVG generated',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--role',choices=('validation','test'),required=True)
    p.add_argument('--results',default='experiments/results')
    p.add_argument('--splits',default='experiments/splits/hmog_feta_splits_seed_20261112.json')
    p.add_argument('--embeddings',default=None)
    p.add_argument('--calibration',default=None,help='TEST: validation thresholds JSON (single method only)')
    p.add_argument('--calibration-dir',default=None,help='TEST: directory with centroid/ and top5/ thresholds')
    p.add_argument('--methods',default=None,help='centroid,top5 (omit to retain original single-method layout)')
    p.add_argument('--impostor-per-source',type=int,default=100)
    p.add_argument('--target-far',type=float,default=0.05)
    p.add_argument('--windows',default=','.join(map(str,WINDOWS)))
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--workers',type=int,default=0,help='Scenario processes, 0=automatic up to 4')
    args=p.parse_args()
    methods=('centroid',) if args.methods is None else tuple(x.strip() for x in args.methods.split(','))
    if not methods or len(set(methods))!=len(methods) or any(x not in METHODS for x in methods):
        raise ValueError('Use --methods centroid,top5 (or either method)')
    if args.workers<0 or args.impostor_per_source<=0 or args.bootstrap<1:
        raise ValueError('workers >=0, impostor-per-source >0 and bootstrap >0 required')
    if not 0<args.target_far<1:
        raise ValueError('target-far must be in (0,1)')
    windows=tuple(int(v.strip()) for v in args.windows.split(','))
    if any(k<1 for k in windows) or len(set(windows))!=len(windows):
        raise ValueError('Window sizes must be distinct positive integers')
    if args.role=='test' and not(args.calibration or args.calibration_dir):
        raise ValueError('TEST requires frozen validation calibration; do not optimize on test')
    if args.role=='test' and len(methods)>1 and not args.calibration_dir:
        raise ValueError('Multiple TEST methods require --calibration-dir')
    t0=time.perf_counter()
    root=Path(args.results)/args.model_id
    path=Path(args.embeddings) if args.embeddings else root/'embeddings'/f'{args.role}.npz'
    print(f'[LOAD] Reading embeddings: {path}',flush=True)
    emb,meta,extra=load_embedding_table(path)
    print(f'[LOAD] {len(emb):,} gestures in {time.perf_counter()-t0:.1f}s',flush=True)
    if extra.get('role')!=args.role or extra.get('model_id')!=args.model_id:
        raise ValueError('Embedding role/model mismatch')
    splits_sha=file_hash(args.splits)
    if extra.get('splits_sha256')!=splits_sha:
        raise ValueError('Frozen split SHA mismatch')
    splits=json.loads(Path(args.splits).read_text(encoding='utf-8'))
    print('[INDEX] Validating role/identity/session membership...',flush=True)
    groups=build_role_scenarios(emb,meta,splits,args.role)
    if args.model_id!='global' and sorted(groups)!=[args.model_id]:
        raise ValueError('Specialized encoder evaluated on wrong scenario')
    study_mode=args.methods is not None
    base=(root/'evaluation'/'method_study'/args.role) if study_mode else root/'evaluation'/args.role
    outputs={m:base/m if study_mode else base for m in methods}
    for output in outputs.values():output.mkdir(parents=True,exist_ok=True)
    calibration={m:{} for m in methods}
    if args.role=='test':
        for method in methods:
            calpath=(Path(args.calibration_dir)/method/'thresholds_validation.json'
                     if args.calibration_dir else Path(args.calibration))
            doc=json.loads(calpath.read_text(encoding='utf-8'))
            expected=(doc['model_id']==args.model_id and doc['splits_sha256']==splits_sha
                      and doc['normalizer_sha256']==extra.get('normalizer_sha256')
                      and doc['checkpoint_epoch']==extra.get('checkpoint_epoch')
                      and doc['windows']==list(windows)
                      and doc['impostor_per_source']==args.impostor_per_source
                      and doc['seed']==args.seed
                      and np.isclose(doc['target_far'],args.target_far)
                      and doc.get('enrollment_method','centroid')==method)
            if not expected:raise ValueError(f'Incompatible validation calibration: {calpath}')
            calibration[method]=doc
    workers=args.workers or min(4,os.cpu_count() or 1,len(groups))
    workers=min(workers,len(groups))
    args_d=dict(vars(args))
    args_d['_parsed_windows']=windows
    args_d['_method_list']=methods
    args_d['_calibrations']=calibration
    tasks=[scenario_task(sc,users,emb,meta,splits,args_d,outputs)
           for sc,users in sorted(groups.items())]
    print(f'[START] {args.role}: scenarios={len(tasks)}, methods={methods}, '
          f'window sizes={len(windows)}, workers={workers}',flush=True)
    results={}
    if workers==1:
        for task in tasks:
            sc,ur,sr,th=evaluate_scenario(task)
            results[sc]=(ur,sr,th)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(evaluate_scenario,task):task[0] for task in tasks}
            pending=set(futures)
            while pending:
                done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                if not done:
                    print(f'[HEARTBEAT] {time.perf_counter()-t0:.0f}s elapsed; '
                          f'{len(results)}/{len(tasks)} scenarios complete',flush=True)
                for f in done:
                    sc,ur,sr,th=f.result();results[sc]=(ur,sr,th)
                    print(f'[PROGRESS] scenarios {len(results)}/{len(tasks)} complete',flush=True)
    all_sr={m:[] for m in methods};all_ur={m:[] for m in methods};all_th={m:{} for m in methods}
    for sc in sorted(results):
        ur,sr,th=results[sc]
        for m in methods:
            all_sr[m].extend(sr[m]);all_ur[m].extend(ur[m]);all_th[m].update(th[m])
    total=time.perf_counter()-t0
    for method in methods:
        out=outputs[method]
        write_csv(out/'per_user_metrics.csv',all_ur[method])
        write_csv(out/'scenario_metrics.csv',all_sr[method])
        if args.role=='validation':
            cfg={'model_id':args.model_id,'enrollment_method':method,
                 'score_aggregation':'mean of k individual gesture scores',
                 'splits_sha256':splits_sha,
                 'normalizer_sha256':extra.get('normalizer_sha256'),
                 'checkpoint_epoch':extra.get('checkpoint_epoch'),
                 'source_validation_embedding_sha256':file_hash(path),
                 'target_far':args.target_far,
                 'impostor_per_source':args.impostor_per_source,
                 'windows':list(windows),'seed':args.seed,
                 'thresholds':all_th[method],
                 'method':'validation impostor quantile, accept score >= threshold'}
            (out/'thresholds_validation.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        timing={'model_id':args.model_id,'role':args.role,'enrollment_method':method,
                'elapsed_seconds':total,'workers':workers,'scenarios':len(groups),
                'windows':list(windows),'impostor_per_source':args.impostor_per_source,
                'source_embeddings_sha256':file_hash(path),
                'validation_not_independent':args.role=='validation'}
        (out/'evaluation_timing.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
        print(f'[DONE] {method}: {len(all_sr[method])} scenario×k rows -> {out}',flush=True)
    if study_mode and args.role=='validation' and len(methods)==2:
        validation_comparison(all_sr,base,windows)
        print(f'[COMPARISON] -> {base / "mean_scenario_comparison.csv"}',flush=True)
    print(f'[COMPLETE] {args.role}: {total:.1f}s',flush=True)


if __name__=='__main__':main()
