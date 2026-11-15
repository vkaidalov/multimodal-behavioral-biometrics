"""Validation vs independent test gap, both metrics in a single publication table.

Validation and test contain DIFFERENT users. Therefore differences are NOT
paired within identity, unlike specialized/global comparison within one role.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def save_csv(path,records):
    if not records:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='experiments/results')
    p.add_argument('--output',default='experiments/validation_vs_test')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261112)
    args=p.parse_args()
    root=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    rows=[]
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():continue
        er=model_dir/'evaluation'
        va=er/'validation'/'scenario_metrics.csv'
        te=er/'test'/'scenario_metrics.csv'
        vu=er/'validation'/'per_user_metrics.csv'
        tu=er/'test'/'per_user_metrics.csv'
        if not all(p.exists() for p in (va,te,vu,tu)):continue
        vmap={(x['scenario'],x['k']):x for x in read_csv(va) if x.get('status')=='ok'}
        tmap={(x['scenario'],x['k']):x for x in read_csv(te) if x.get('status')=='ok'}
        vusers=defaultdict(list);tusers=defaultdict(list)
        for x in read_csv(vu):
            if x.get('status')=='ok':vusers[(x['scenario'],x['k'])].append(x)
        for x in read_csv(tu):
            if x.get('status')=='ok':tusers[(x['scenario'],x['k'])].append(x)
        for key in sorted(vmap.keys()&tmap.keys()):
            v=vmap[key];t=tmap[key]
            r={'model_id':model_dir.name,'scenario':key[0],'k':int(key[1]),
               'validation_users':int(v['evaluable_users']),
               'test_users':int(t['evaluable_users']),
               'validation_coverage':float(v['coverage']),
               'test_coverage':float(t['coverage'])}
            for metric in ('eer','auc','balanced_accuracy'):
                col='macro_'+metric
                rv=float(v[col]);rt=float(t[col])
                r['validation_'+col]=rv;r['test_'+col]=rt
                r['gap_test_minus_validation_'+col]=rt-rv
                # Two INDEPENDENT groups of subjects; resample each separately.
                vvals=np.array([float(x[metric]) for x in vusers[key]])
                tvals=np.array([float(x[metric]) for x in tusers[key]])
                if args.bootstrap>0:
                    vb=vvals[rng.integers(len(vvals),size=(args.bootstrap,len(vvals)))].mean(axis=1)
                    tb=tvals[rng.integers(len(tvals),size=(args.bootstrap,len(tvals)))].mean(axis=1)
                    l,h=np.quantile(tb-vb,[.025,.975]);r['gap_'+col+'_ci95_low']=float(l)
                    r['gap_'+col+'_ci95_high']=float(h)
            rows.append(r)
    save_csv(out/'validation_vs_test_metrics.csv',rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['model_id'],r['scenario'])].append(r)
    for (model,scenario),items in sorted(groups.items()):
        items.sort(key=lambda r:r['k'])
        fig,ax=plt.subplots(figsize=(6.2,3.8))
        ax.plot([x['k'] for x in items],[x['validation_macro_eer'] for x in items],
                marker='o',label='Validation')
        ax.plot([x['k'] for x in items],[x['test_macro_eer'] for x in items],
                marker='o',label='Test')
        ax.set(xlabel='Number of consecutive gestures',ylabel='Macro EER',
               title=f'{model}: {scenario} — validation vs test')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(out/f'{model}_{scenario}_validation_vs_test_eer.svg');plt.close(fig)
    print(f'[DONE] {len(rows)} validation/test comparison rows -> {out}')
if __name__=='__main__':main()
