"""Paired comparison of global and specialized encoders using SAME subjects."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_rows(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='experiments/results')
    p.add_argument('--output',default='experiments/comparisons')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261112)
    args=p.parse_args()
    root=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];scenarios=['feta_6s_scroll','feta_6s_swipe','feta_7_scroll','feta_7_swipe',
                        'feta_x_scroll','feta_x_swipe','hmog_sitting','hmog_walking']
    for role in ('validation','test'):
        gp=root/'global'/'evaluation'/role/'per_user_metrics.csv'
        if not gp.exists():continue
        glob={(x['scenario'],x['k'],x['user_id']):x for x in read_rows(gp) if x.get('status')=='ok'}
        for scenario in scenarios:
            pp=root/scenario/'evaluation'/role/'per_user_metrics.csv'
            if not pp.exists():continue
            for x in read_rows(pp):
                if x.get('status')!='ok':continue
                key=(scenario,x['k'],x['user_id'])
                if key not in glob:continue
                g=glob[key]
                rows.append({'role':role,'scenario':scenario,'k':int(x['k']),
                     'user_id':x['user_id'],
                     'eer_specialized':float(x['eer']),'eer_global':float(g['eer']),
                     'delta_eer_global_minus_specialized':float(g['eer'])-float(x['eer']),
                     'auc_specialized':float(x['auc']),'auc_global':float(g['auc']),
                     'delta_auc_global_minus_specialized':float(g['auc'])-float(x['auc'])})
    write_rows(out/'paired_per_user.csv',rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['role'],row['scenario'],row['k'])].append(row)
    summary=[];rng=np.random.default_rng(args.seed)
    for (role,sc,k),part in sorted(groups.items()):
        r={'role':role,'scenario':sc,'k':k,'users':len(part)}
        for c in ('delta_eer_global_minus_specialized','delta_auc_global_minus_specialized'):
            values=np.asarray([x[c] for x in part])
            boot=values[rng.integers(0,len(values),size=(args.bootstrap,len(values)))].mean(axis=1)
            r[c+'_mean']=float(values.mean())
            r[c+'_ci95_low']=float(np.quantile(boot,.025))
            r[c+'_ci95_high']=float(np.quantile(boot,.975))
        summary.append(r)
    write_rows(out/'paired_scenario_comparison.csv',summary)
    for role in ('validation','test'):
        part=[x for x in summary if x['role']==role and x['k']==1]
        if not part:continue
        fig,ax=plt.subplots(figsize=(8,4))
        labels=[x['scenario'] for x in part]
        vals=[x['delta_eer_global_minus_specialized_mean'] for x in part]
        lower=[x['delta_eer_global_minus_specialized_ci95_low'] for x in part]
        upper=[x['delta_eer_global_minus_specialized_ci95_high'] for x in part]
        error=[np.array(vals)-lower,np.array(upper)-vals]
        ax.bar(range(len(part)),vals,yerr=error,capsize=3)
        ax.axhline(0,color='gray',linewidth=1)
        ax.set_xticks(range(len(part)),labels,rotation=35,ha='right')
        ax.set_ylabel('Global − specialized macro EER (per-user paired)')
        ax.set_title(f'Paired difference, {role}, one gesture')
        fig.tight_layout();fig.savefig(out/f'{role}_paired_delta_eer_k01.svg');plt.close(fig)
    print(f'[DONE] paired comparisons -> {out} ({len(rows)} user-matched records)')
if __name__=='__main__':main()
