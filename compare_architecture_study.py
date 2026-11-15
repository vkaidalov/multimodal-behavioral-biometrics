"""Compare frozen validation-only architecture experiments without modifying any runs.

The primary prespecified selection criterion is the equal-scenario average of
PER-USER macro AUC at k=1, using the existing evaluator's scenario_metrics.csv.
No test files are read. Run seeds are independent training repetitions.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

EXPECTED_SCENARIOS=tuple(sorted((
 'feta_6s_scroll','feta_6s_swipe','feta_7_scroll','feta_7_swipe',
 'feta_x_scroll','feta_x_swipe','hmog_sitting','hmog_walking')))


def read_metrics(path,method):
    f=path/'evaluation'/'method_study'/'validation'/method/'scenario_metrics.csv'
    if not f.is_file():raise FileNotFoundError(f)
    with f.open(newline='',encoding='utf-8-sig') as handle:
        data=list(csv.DictReader(handle))
    if not data:raise ValueError(f'No rows: {f}')
    rows=[r for r in data if int(r['k'])==1]
    scenarios=sorted(r['scenario'] for r in rows)
    if scenarios!=list(EXPECTED_SCENARIOS):
        raise ValueError(f'{f}: expected exactly 8 scenarios at k=1, found {scenarios}')
    for r in rows:
        if r.get('status','ok') not in ('ok','complete'):
            raise ValueError(f'{f}: non-OK scenario {r}')
        if float(r['coverage'])<0.999999:
            raise ValueError(f'{f}: k=1 coverage not full for {r["scenario"]}')
        for key in ('macro_auc','macro_eer'):
            if not np.isfinite(float(r[key])):raise ValueError(f'{f}: nonfinite {key}')
    return rows


def load_run(run_root,method):
    folder=run_root/'global'
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if cfg['scope']!='global':raise ValueError(f'Not global: {run_root}')
    if cfg['experiment_version']!='gesture-architecture-study-v1.0':
        raise ValueError(f'Unexpected experiment: {run_root}')
    rows=read_metrics(folder,method)
    bysc=[]
    for r in rows:
        bysc.append(dict(run=run_root.name,architecture=cfg['architecture']['name'],
                         objective=cfg['training'].get('objective','supcon'),
                         seed=int(cfg['training']['seed']),scenario=r['scenario'],
                         method=method,auc=float(r['macro_auc']),eer=float(r['macro_eer']),
                         users=int(r['evaluable_users'])))
    record={'run':run_root.name,'architecture':cfg['architecture']['name'],
            'objective':cfg['training'].get('objective','supcon'),
            'seed':int(cfg['training']['seed']),'method':method,
            'macro_auc':float(np.mean([r['auc'] for r in bysc])),
            'macro_eer':float(np.mean([r['eer'] for r in bysc])),
            'scenario_count':len(bysc),'run_root':str(run_root)}
    return record,bysc


def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(runs):
    groups=defaultdict(list)
    for r in runs:groups[(r['architecture'],r['objective'],r['method'])].append(r)
    result=[]
    for (arch,obj,method),data in sorted(groups.items()):
        auc=np.array([r['macro_auc'] for r in data],dtype=float)
        eer=np.array([r['macro_eer'] for r in data],dtype=float)
        result.append({'architecture':arch,'objective':obj,'method':method,
            'seeds':len(data),'macro_auc_mean':float(auc.mean()),
            'macro_auc_sd':float(auc.std(ddof=1)) if len(auc)>1 else float('nan'),
            'macro_eer_mean':float(eer.mean()),
            'macro_eer_sd':float(eer.std(ddof=1)) if len(eer)>1 else float('nan')})
    return sorted(result,key=lambda r:-r['macro_auc_mean'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',default='experiments/architecture_study')
    p.add_argument('--output',required=True,help='New, empty output directory; refuses overwrite')
    p.add_argument('--method',choices=('centroid','top5'),default='centroid')
    args=p.parse_args()
    source=Path(args.runs_root)
    runs=[];bysc=[]
    for run_dir in sorted(source.iterdir()):
        if not run_dir.is_dir() or not (run_dir/'global'/'config.json').is_file():continue
        r,s=load_run(run_dir,args.method)
        runs.append(r);bysc.extend(s)
    if not runs:raise ValueError(f'No finished runs found in {source}')
    names=[r['run'] for r in runs]
    if len(set(names))!=len(names):raise ValueError('Duplicate experiment names')
    summary=summarize(runs)
    dst=Path(args.output)
    if dst.exists():raise FileExistsError(f'Comparison directory already exists: {dst}')
    dst.mkdir(parents=True)
    save_csv(dst/'runs.csv',runs)
    save_csv(dst/'scenarios.csv',bysc)
    save_csv(dst/'architecture_summary.csv',summary)
    (dst/'manifest.json').write_text(json.dumps({'role':'validation','method':args.method,
        'k':1,'selection_metric':'equal-scenario per-user macro AUC',
        'runs':names,'test_role_used':False},indent=2),encoding='utf-8')
    print(f'[DONE] Compared {len(runs)} runs, {len(summary)} configurations -> {dst}',flush=True)
    for line in summary:
        print(f"  {line['architecture']:15s} {line['objective']:12s} n={line['seeds']} "
              f"AUC={line['macro_auc_mean']:.5f} "+
              (f"±{line['macro_auc_sd']:.5f}" if line['seeds']>1 else '(one seed)')+
              f" EER={line['macro_eer_mean']:.5f}",flush=True)

if __name__=='__main__':main()
