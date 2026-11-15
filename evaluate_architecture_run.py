"""Safe, validation-only wrapper for the existing v2 authentication evaluator.

Writes exactly the legacy method_study layout under a NEW experiment root.
Refuses to replace prior evaluation output. This is not a new scoring protocol.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,help='Dedicated run root, e.g. experiments/architecture_study/tcn_supcon_seed20261112')
    p.add_argument('--model-id',default='global')
    p.add_argument('--splits',default='experiments/splits/hmog_feta_splits_seed_20261112.json')
    p.add_argument('--methods',default='centroid,top5')
    p.add_argument('--windows',default=','.join(str(i) for i in range(1,21)))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--target-far',type=float,default=.05)
    p.add_argument('--impostor-per-source',type=int,default=100)
    args=p.parse_args()
    out=Path(args.results)/args.model_id
    if out.resolve()==Path('experiments/results').resolve()/args.model_id:
        raise ValueError('Refusing original results directory')
    if not (out/'model_best.pt').is_file() or not (out/'embeddings'/'validation.npz').is_file():
        raise FileNotFoundError(f'Missing model_best.pt or validation.npz under {out}')
    if (out/'evaluation').exists():
        raise FileExistsError(f'Existing evaluation folder protected: {out / "evaluation"}')
    evaluator=Path(__file__).resolve().parent/'evaluate_authentication.py'
    if not evaluator.is_file():
        raise FileNotFoundError(f'Existing v2 evaluator missing next to this script: {evaluator}')
    cmd=[sys.executable,'-u',str(evaluator),
         '--model-id',args.model_id,'--role','validation',
         '--results',args.results,'--splits',args.splits,
         '--methods',args.methods,'--windows',args.windows,
         '--workers',str(args.workers),'--bootstrap',str(args.bootstrap),
         '--target-far',str(args.target_far),
         '--impostor-per-source',str(args.impostor_per_source)]
    print('[EVALUATE] '+ ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
