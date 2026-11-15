"""Train baseline/TCN/attention/crossmodal gesture encoders; disjoint validation users.

Runs one model per invocation; checkpoints include optimizer/scaler/RNG states.
Validation selection is macro-AUC at k=1 on a fixed session-stratified subset.
Neither testing data nor testing labels are read by this program.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from hmog_feta_gesture_dataset_prototype_v1 import SCENARIOS
from gesture_pipeline_data import (GestureCacheDataset,IdentityScenarioPKSampler,
                                  select_validation_indices,collate,file_hash)
from gesture_model import supervised_contrastive_loss
from gesture_architectures import ARCHITECTURES,make_encoder,prototypical_cross_session_loss
from gesture_evaluation_core import checkpoint_macro_auc


def device_batch(b,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v)
            for k,v in b.items()}


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def dump_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)


def atomic_checkpoint(path,payload):
    path=Path(path);temp=path.with_suffix('.tmp')
    torch.save(payload,temp);os.replace(temp,path)


@torch.inference_mode()
def encode_subset(model,ds,ids,device,batch_size,workers=0):
    model.eval()
    loader=DataLoader(Subset(ds,ids),batch_size=batch_size,shuffle=False,
                      collate_fn=collate,num_workers=workers,pin_memory=(device.type=='cuda'))
    result=[];metadata=[]
    for b in loader:
        metadata.extend(b['meta'])
        emb=model(device_batch(b,device)).float().cpu().numpy()
        result.append(emb)
    return np.concatenate(result),metadata


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['global',*SCENARIOS],required=True)
    p.add_argument('--train-cache',default='output_data/gesture_cache_train_first_four')
    p.add_argument('--validation-cache',default='output_data/gesture_cache_validation_first_four')
    p.add_argument('--stats',default='output_data/preprocessing_v2_first_four/normalizers_train.json')
    p.add_argument('--splits',default='experiments/splits/hmog_feta_splits_seed_20261112.json')
    p.add_argument('--output-root',required=True,help='NEW dedicated run root; contains global/ or scenario/ subfolder')
    p.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    p.add_argument('--objective',choices=('supcon','prototypical'),default='supcon')
    p.add_argument('--grid-points',type=int,default=32)
    p.add_argument('--attention-heads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--steps-per-epoch',type=int,default=120)
    p.add_argument('--p',type=int,default=12)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--width',type=int,default=48)
    p.add_argument('--embedding-dim',type=int,default=64)
    p.add_argument('--temperature',type=float,default=0.10)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--validate-every',type=int,default=2)
    p.add_argument('--patience',type=int,default=4,help='Number of validation checks without improvement')
    p.add_argument('--val-gestures-per-session',type=int,default=32)
    p.add_argument('--val-impostors-per-source',type=int,default=30)
    p.add_argument('--inference-batch-size',type=int,default=128)
    p.add_argument('--max-touch',type=int,default=256)
    p.add_argument('--max-motion',type=int,default=512)
    p.add_argument('--clip-z',type=float,default=0.0)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--resume',action='store_true')
    return p


def main():
    args=args_parser().parse_args()
    if not (args.epochs>0 and args.steps_per_epoch>0 and args.validate_every>0 and args.p>1 and args.k>1):
        raise ValueError('Invalid epoch/batch/validation parameters')
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda' if (args.device=='cuda' or args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(torch.get_num_threads(),8))
    scenario=None if args.scenario=='global' else args.scenario
    scope='global' if scenario is None else 'scenario'
    train=GestureCacheDataset(args.train_cache,args.stats,expected_role='train',
            scenario=scenario,scope=scope,max_touch=args.max_touch,max_motion=args.max_motion,
            clip_z=args.clip_z,splits=args.splits)
    val=GestureCacheDataset(args.validation_cache,args.stats,expected_role='validation',
            scenario=scenario,scope=scope,max_touch=args.max_touch,max_motion=args.max_motion,
            clip_z=args.clip_z,splits=args.splits)
    splits=json.loads(Path(args.splits).read_text(encoding='utf-8'))
    scenarios=sorted({m['scenario'] for m in val.metadata})
    if scenarios!=(sorted(SCENARIOS) if scenario is None else [scenario]):
        raise ValueError(f'Unexpected validation scenarios: {scenarios}')
    if set(train.by_identity).intersection(val.by_identity):
        raise ValueError('CRITICAL: overlap of train/validation identities')
    ids=select_validation_indices(val,args.val_gestures_per_session,args.seed)
    sampler=IdentityScenarioPKSampler(train,args.p,args.k,args.steps_per_epoch,args.seed)
    loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate,
                      num_workers=args.workers,pin_memory=(device.type=='cuda'),
                      persistent_workers=(args.workers>0))
    arch={'name':args.architecture,'width':args.width,'embedding_dim':args.embedding_dim}
    if args.architecture=='crossmodal':
        arch.update(grid_points=args.grid_points,attention_heads=args.attention_heads)
    model=make_encoder(arch).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # Stable LR schedule without extra tuning; cosine for full fixed epoch budget.
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*0.1)
    amp=(device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    out=Path(args.output_root)/args.scenario
    if out.resolve()==Path('experiments/results').resolve()/args.scenario:
        raise ValueError('Refusing to overwrite original experiments/results. Choose dedicated --output-root.')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'{out} is not empty. Choose unique --output-root or explicitly --resume.')
    if args.resume and not out.is_dir():
        raise FileNotFoundError(f'Cannot resume missing run: {out}')
    out.mkdir(parents=True,exist_ok=True)
    config={'experiment_version':'gesture-architecture-study-v1.0','model_id':args.scenario,
            'scenario':args.scenario,'scope':scope,'architecture':arch,
            'training':{k:getattr(args,k) for k in ('epochs','steps_per_epoch','p','k','temperature','lr',
             'weight_decay','validate_every','patience','val_gestures_per_session','val_impostors_per_source',
             'inference_batch_size','max_touch','max_motion','clip_z','seed','objective')},
            'normalizer_path':str(Path(args.stats).resolve()),'normalizer_sha256':file_hash(args.stats),
            'splits_sha256':file_hash(args.splits),'train_index_sha256':file_hash(Path(args.train_cache)/'index.json'),
            'validation_index_sha256':file_hash(Path(args.validation_cache)/'index.json'),
            'scenarios':scenarios,'selection_metric':'equal-scenario macro of per-user validation AUC k=1',
            'validation_subset_only_for_checkpoint':True,'train_users':len(train.by_identity),
            'validation_users':len(val.by_identity),'device':str(device),'torch_version':str(torch.__version__)}
    if (out/'model_best.pt').exists() and not args.resume:
        raise FileExistsError(f'{out}/model_best.pt already exists; refuse to overwrite (use --resume)')
    if not args.resume:dump_json(out/'config.json',config)
    elif not (out/'config.json').exists():raise FileNotFoundError('Missing config for resume')
    else:
        old=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ('architecture','training','normalizer_sha256','splits_sha256',
                    'train_index_sha256','validation_index_sha256'):
            if old[key]!=config[key]:raise ValueError(f'Resume config mismatch: {key}')
    start_epoch=1;best=-math.inf;no_improve=0;history=[];train_seconds=0.;val_seconds=0.
    last=out/'checkpoint_last.pt'
    if args.resume:
        if not last.exists():raise FileNotFoundError(last)
        ck=torch.load(last,map_location=device,weights_only=True)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1;best=ck['best'];no_improve=ck['no_improve']
        history=ck['history'];train_seconds=ck['train_seconds'];val_seconds=ck['val_seconds']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        print(f'[RESUME] epoch {start_epoch}, best AUC={best:.5f}',flush=True)
    print(f'[START] {args.scenario} on {device} | train={len(train)} val={len(val)} '
          f'| arch={args.architecture} objective={args.objective} seed={args.seed} '
          f'| P={args.p} K={args.k} | steps/epoch={args.steps_per_epoch}',flush=True)
    t0=time.perf_counter();epoch_times=[]
    try:
        for epoch in range(start_epoch,args.epochs+1):
            sampler.set_epoch(epoch);model.train()
            losses=[];epoch_start=time.perf_counter()
            for i,b in enumerate(loader,1):
                b=device_batch(b,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    embedding=model(b)
                    loss=(supervised_contrastive_loss(embedding,b['labels'],args.temperature)
                          if args.objective=='supcon' else
                          prototypical_cross_session_loss(embedding,b['labels'],b['meta'],args.temperature))
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}/{i}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update()
                losses.append(float(loss.item()))
                if i==1 or i%args.log_every==0 or i==len(loader):
                    done=time.perf_counter()-epoch_start
                    eta=(done/max(i,1))*(len(loader)-i)
                    print(f'[TRAIN] {args.scenario} e{epoch}/{args.epochs} step {i}/{len(loader)} '
                          f'loss={np.mean(losses[-min(20,len(losses)):]):.4f} '
                          f'epoch ETA={eta/60:.1f}m',flush=True)
            sync(device);elapsed=time.perf_counter()-epoch_start;train_seconds+=elapsed
            epoch_times.append(elapsed)
            scheduler.step()
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),
                 'train_seconds':elapsed,'validation_macro_auc':None,'validation_seconds':0.,
                 'lr':float(optimizer.param_groups[0]['lr'])}
            if epoch%args.validate_every==0 or epoch==args.epochs:
                t1=time.perf_counter()
                valemb,valmeta=encode_subset(model,val,ids,device,args.inference_batch_size,args.workers)
                valauc,by_scenario=checkpoint_macro_auc(valemb,valmeta,splits,'validation',scenarios,
                                                         args.val_impostors_per_source)
                sync(device);ve=time.perf_counter()-t1;val_seconds+=ve
                row['validation_macro_auc']=valauc;row['validation_seconds']=ve
                improved=valauc>best+1e-5
                if improved:
                    best=valauc;no_improve=0
                    atomic_checkpoint(out/'model_best.pt',{
                        'model':model.state_dict(),'architecture':config['architecture'],
                        'epoch':epoch,'validation_macro_auc':valauc,
                        'validation_scenario_auc':by_scenario,'config':config})
                else:no_improve+=1
                print(f'[VALID] e{epoch} macro AUC={valauc:.5f} best={best:.5f} '
                      f'checks_without_improvement={no_improve}/{args.patience} '
                      f'elapsed={ve:.1f}s',flush=True)
            history.append(row)
            atomic_checkpoint(last,{'epoch':epoch,'model':model.state_dict(),
                 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'scaler':scaler.state_dict(),'best':best,'no_improve':no_improve,
                 'history':history,'train_seconds':train_seconds,'val_seconds':val_seconds,
                 'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None})
            with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
            mean_epoch=float(np.mean(epoch_times[-5:]))
            print(f'[EPOCH] {epoch} time={elapsed:.1f}s '
                  f'planned remaining ~{mean_epoch*(args.epochs-epoch)/60:.1f}m '
                  f'(upper bound; early stopping may shorten)',flush=True)
            if no_improve>=args.patience:
                print(f'[STOP] Validation early stopping at epoch {epoch}',flush=True)
                break
    except KeyboardInterrupt:
        print('[INTERRUPTED] Last completed epoch is safely checkpointed.',flush=True)
        raise
    finally:
        duration=time.perf_counter()-t0
        dump_json(out/'timing.json',{'training_active_seconds':train_seconds,
                 'checkpoint_validation_seconds':val_seconds,
                 'this_invocation_wall_seconds':duration,'epochs_completed':len(history),
                 'device':str(device),'note':'Training_active includes data loading; '
                 'wall time additionally includes checkpoint I/O.'})
    dump_json(out/'training_finished.json',{'model_id':args.scenario,'epochs_completed':len(history),
        'best_validation_macro_auc':best,'finished':True})
    print(f'[DONE] {args.scenario} best macro AUC={best:.5f} -> {out}/model_best.pt',flush=True)


if __name__=='__main__':main()
