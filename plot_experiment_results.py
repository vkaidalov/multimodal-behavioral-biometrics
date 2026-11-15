"""Publication-ready SVGs from saved CSV/NPZ only; never reruns encoder."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def records(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='experiments/results')
    p.add_argument('--output',default='experiments/figures')
    args=p.parse_args();base=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(base.glob('*/evaluation/*/scenario_metrics.csv')):
        rows.extend(records(path))
    if not rows:raise FileNotFoundError('No scenario_metrics.csv files; run evaluations first')
    scenarios=sorted({x['scenario'] for x in rows})
    for role in ('validation','test'):
        for sc in scenarios:
            subset=[r for r in rows if r['role']==role and r['scenario']==sc and r.get('status','ok')=='ok']
            if not subset:continue
            labels=sorted({x['model_id'] for x in subset})
            for metric,ylabel,stem in [('macro_eer','Macro EER','eer'),
                                        ('macro_auc','Macro AUC','auc')]:
                fig,ax=plt.subplots(figsize=(6.4,3.9))
                for name in labels:
                    data=sorted((x for x in subset if x['model_id']==name),key=lambda x:int(x['k']))
                    ax.plot([int(x['k']) for x in data],[float(x[metric]) for x in data],
                            marker='o',label=name)
                ax.set_xlabel('Number of consecutive gestures');ax.set_ylabel(ylabel)
                ax.set_title(f'{sc}: {role}')
                ax.grid(alpha=.25);ax.legend(fontsize=8)
                fig.tight_layout();fig.savefig(out/f'{role}_{sc}_{stem}_vs_k.svg');plt.close(fig)
            fig,ax=plt.subplots(figsize=(6.4,3.9))
            for name in labels:
                data=sorted((x for x in subset if x['model_id']==name),key=lambda x:int(x['k']))
                ax.plot([float(x['macro_latency_mean_s']) for x in data],
                        [float(x['macro_eer']) for x in data],marker='o',label=name)
                for x in data:ax.annotate(str(x['k']),(float(x['macro_latency_mean_s']),float(x['macro_eer'])),fontsize=7)
            ax.set_xlabel('Mean behavioral collection latency (s)');ax.set_ylabel('Macro EER')
            ax.set_title(f'{sc}: {role}');ax.grid(alpha=.25);ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(out/f'{role}_{sc}_eer_vs_latency.svg');plt.close(fig)
            # Macro ROC at k=1 and k=5; each line equally weights identities.
            for k in (1,5):
                fig,ax=plt.subplots(figsize=(5.5,5.1));has_any=False
                for name in labels:
                    npz=base/name/'evaluation'/role/f'{sc}_k{k:02d}_scenario_roc.npz'
                    if not npz.exists():continue
                    with np.load(npz,allow_pickle=False) as d:
                        ax.plot(d['fpr_grid'],d['macro_tpr'],label=name)
                    has_any=True
                if has_any:
                    ax.plot([0,1],[0,1],ls=':',color='gray')
                    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{sc}: {role}, k={k}')
                    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
                    fig.savefig(out/f'{role}_{sc}_roc_k{k:02d}.svg')
                plt.close(fig)
    print(f'[DONE] SVG figures -> {out}')
if __name__=='__main__':main()
