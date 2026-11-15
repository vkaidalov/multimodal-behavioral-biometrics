"""Run a saved encoder once on validation or test; preserve timestamps and IDs."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from gesture_architectures import make_encoder
from gesture_pipeline_data import GestureCacheDataset,collate,file_hash
from gesture_evaluation_core import save_embedding_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--role',choices=('validation','test'),required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--stats',default='output_data/preprocessing_v2_first_four/normalizers_train.json')
    p.add_argument('--splits',default='experiments/splits/hmog_feta_splits_seed_20261112.json')
    p.add_argument('--results',default='experiments/results')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=p.parse_args()
    root=Path(args.results)/args.model_id
    if root.resolve()==Path('experiments/results').resolve()/args.model_id:
        raise ValueError('Use new run --results root, not legacy experiments/results.')
    dst=root/'embeddings'/f'{args.role}.npz'
    if dst.exists():
        raise FileExistsError(f'Refusing to overwrite embeddings: {dst}')
    ck=torch.load(root/'model_best.pt',map_location='cpu',weights_only=True)
    config=ck['config']
    if config['normalizer_sha256']!=file_hash(args.stats):raise ValueError('Wrong normalizer file')
    if config['splits_sha256']!=file_hash(args.splits):raise ValueError('Wrong split file')
    scenario=None if args.model_id=='global' else args.model_id
    scope='global' if scenario is None else 'scenario'
    train=config['training']
    ds=GestureCacheDataset(args.cache,args.stats,expected_role=args.role,scenario=scenario,scope=scope,
             max_touch=train['max_touch'],max_motion=train['max_motion'],clip_z=train['clip_z'],splits=args.splits)
    device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    model=make_encoder(ck['architecture']).to(device)
    model.load_state_dict(ck['model']);model.eval()
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate,
                      num_workers=args.workers,pin_memory=(device.type=='cuda'))
    vectors=[];meta=[];start=time.perf_counter()
    with torch.inference_mode():
        for count,b in enumerate(loader,1):
            meta.extend(b['meta'])
            b={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
            vectors.append(model(b).float().cpu().numpy())
            if count%20==0 or count==len(loader):
                done=len(meta);time_used=time.perf_counter()-start
                eta=time_used/max(done,1)*(len(ds)-done)
                print(f'[EMBED] {args.model_id}/{args.role} {done}/{len(ds)} ETA={eta/60:.1f}m',flush=True)
    if device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    save_embedding_table(dst,np.concatenate(vectors),meta,extra={
        'architecture':config['architecture'], 'training_objective':config['training'].get('objective','supcon'),
        'role':args.role,'model_id':args.model_id,'checkpoint_epoch':ck['epoch'],
        'checkpoint_validation_auc':ck['validation_macro_auc'],
        'cache_index_sha256':file_hash(Path(args.cache)/'index.json'),
        'splits_sha256':file_hash(args.splits),'normalizer_sha256':file_hash(args.stats)})
    report={'model_id':args.model_id,'role':args.role,'gestures':len(ds),
            'inference_and_data_loading_seconds':seconds,'gestures_per_second':len(ds)/max(seconds,1e-6),
            'device':str(device),'source_cache':str(args.cache),'embedding_file':str(dst)}
    timing=root/'embeddings'/f'{args.role}_timing.json'
    timing.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'[DONE] {dst} | {seconds:.1f}s',flush=True)
if __name__=='__main__':main()
