#!/usr/bin/env python3
"""HMOG/FETA gesture-level preprocessing + PyTorch Dataset/DataLoader prototype.

Uses the existing frozen 60/20/20 split JSON, but processes TRAIN users only.
Preserves native per-stream timestamps; no interpolation, no fixed time windows.
Gesture validity matches HMOG/FETA v2.0 (and adds finite XYZ checks).

Put next to analyze_hmog_feta_feta_criterion_v2_0.py.  Python 3.14 supported.

Subcommands:
    prepare: read raw data, form gesture records, serialize safe ragged NPZ shards
    inspect: check masks, lengths, timestamps and sample shapes with DataLoader
"""
from __future__ import annotations
import argparse
import csv
import functools
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import analyze_hmog_feta_feta_criterion_v2_0 as v2

VERSION = 'gesture-dataloader-prototype-1.0.0'
TOUCH_FEATURES = ('x_screen', 'y_screen', 'dx_screen', 'dy_screen', 'elapsed_s', 'dt_s')
MOTION_FEATURES = ('x_raw', 'y_raw', 'z_raw', 'elapsed_s', 'dt_s')
PHONE_LOGICAL = {
    'Samsung Galaxy S4': (1080, 1920),
    'iPhone 6s': (375, 667), 'iPhone 6s Plus': (414, 736),
    'iPhone 7': (375, 667), 'iPhone 7 Plus': (414, 736),
    'iPhone 8': (375, 667), 'iPhone 8 Plus': (414, 736),
    'iPhone X': (375, 812), 'iPhone XS': (375, 812),
    'iPhone XS Max': (414, 896),
}
PHONE_PIXELS = {
    'iPhone 6s': (750, 1334), 'iPhone 6s Plus': (1080, 1920),
    'iPhone 7': (750, 1334), 'iPhone 7 Plus': (1080, 1920),
    'iPhone 8': (750, 1334), 'iPhone 8 Plus': (1080, 1920),
    'iPhone X': (1125, 2436), 'iPhone XS': (1125, 2436),
    'iPhone XS Max': (1242, 2688),
}
SCENARIOS = {
    'feta_6s_scroll': ('FETA', 'iPhone 6s', 'scroll'),
    'feta_6s_swipe': ('FETA', 'iPhone 6s', 'swipe'),
    'feta_7_scroll': ('FETA', 'iPhone 7', 'scroll'),
    'feta_7_swipe': ('FETA', 'iPhone 7', 'swipe'),
    'feta_x_scroll': ('FETA', 'iPhone X', 'scroll'),
    'feta_x_swipe': ('FETA', 'iPhone X', 'swipe'),
    'hmog_sitting': ('HMOG', 'Samsung Galaxy S4', 'reading_sitting'),
    'hmog_walking': ('HMOG', 'Samsung Galaxy S4', 'reading_walking'),
}
MIN_SAMPLES = 3
MAX_EDGE_GAP_MS = 40.0
MAX_INTERNAL_GAP_MS = 120.0
PAIR_TOLERANCE_MS = 25.0


class Gesture(NamedTuple):
    start_ms: float
    end_ms: float
    touch_t: np.ndarray
    touch_x: np.ndarray
    touch_y: np.ndarray
    portrait: bool
    multitouch: bool
    activity: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def parse_feta_touch(path: Path) -> list[Gesture]:
    df = v2.feta_csv(path)
    if df.empty:
        return []
    times = v2.feta_time_ms(df)
    xs, ys = v2.numeric(df, 'x'), v2.numeric(df, 'y')
    phase_cols = [c for c in ('type','touchtype','phase','action','actiontype',
                              'touchphase','eventtype','state') if c in df]
    if not phase_cols:
        raise ValueError(f'Missing FETA touch phase in {path}')
    phcol = phase_cols[0]
    valid = np.isfinite(times) & np.isfinite(xs) & np.isfinite(ys)
    order = sorted(np.flatnonzero(valid), key=lambda i: (times[i], int(i)))
    output, active = [], None
    for i in order:
        state = v2.parse_feta_phase(df[phcol].iloc[i], phcol)
        tm, x, y = float(times[i]), float(xs[i]), float(ys[i])
        if state == 'down':
            active = ([tm], [x], [y])
        elif state in ('move','up') and active is not None:
            active[0].append(tm); active[1].append(x); active[2].append(y)
            if state == 'up':
                t = np.asarray(active[0], dtype=np.float64)
                output.append(Gesture(float(t[0]), float(t[-1]), t,
                    np.asarray(active[1], dtype=np.float64),
                    np.asarray(active[2], dtype=np.float64), True, False, ''))
                active = None
        elif state == 'cancel':
            active = None
    return output


def parse_hmog_touch(df: pd.DataFrame, activity_to_task: dict[str,str]) -> list[Gesture]:
    if df.empty: return []
    # IMPORTANT: same native touch clock as v2.0; do not divide touch.EventTime by 1e6.
    times = v2.numeric(df,'eventtime')
    xs, ys = v2.numeric(df,'x'), v2.numeric(df,'y')
    actions = v2.numeric(df,'actionid')
    pcounts = v2.numeric(df,'pointercount')
    pids = v2.numeric(df,'pointerid')
    orientations = v2.numeric(df,'phoneorientation')
    aids = df.get('activityid',pd.Series(['']*len(df))).astype(str).str.strip().to_numpy()
    valid = np.isfinite(times)&np.isfinite(xs)&np.isfinite(ys)&np.isfinite(actions)
    order = sorted((int(i) for i in np.flatnonzero(valid) if aids[i] in activity_to_task),
                   key=lambda i:(aids[i],times[i],i))
    output=[]; active=None
    for i in order:
        a=str(aids[i]);tm=float(times[i]); x=float(xs[i]);y=float(ys[i]); act=int(actions[i])&255
        multi = act in (5,6) or (np.isfinite(pcounts[i]) and pcounts[i]>1) or (
            np.isfinite(pids[i]) and pids[i]>0)
        portrait = bool(np.isfinite(orientations[i]) and int(orientations[i])==0)
        if act==0:
            active = {'activity':a, 't':[tm], 'x':[x], 'y':[y],
                      'portrait':portrait, 'multi':bool(multi)}
        elif act in (5,6):
            if active is not None: active['multi']=True
        elif act==3:active=None
        elif act in (1,2) and active is not None:
            if active['activity']!=a:
                active=None;continue
            active['t'].append(tm);active['x'].append(x);active['y'].append(y)
            active['portrait'] &= portrait
            active['multi'] |= bool(multi)
            if act==1:
                t=np.asarray(active['t'],dtype=np.float64)
                output.append(Gesture(float(t[0]),float(t[-1]),t,
                    np.asarray(active['x'],dtype=np.float64),
                    np.asarray(active['y'],dtype=np.float64),
                    bool(active['portrait']),bool(active['multi']),a))
                active=None
    return output


def load_motion_feta(rep:Path, channel:str) -> dict[str,np.ndarray]:
    sub={'accel':'accelerometer_data','gyro':'gyroscope_data'}[channel]
    pieces=[]
    for f in sorted((rep/sub).glob('*.csv')):
        d=v2.feta_csv(f)
        if d.empty:continue
        t=v2.feta_time_ms(d)
        xyz=np.column_stack([v2.numeric(d,axis) for axis in ('x','y','z')])
        pieces.append((t,xyz))
    if not pieces:
        return {'t':np.empty(0,dtype=np.float64), 'xyz':np.empty((0,3),dtype=np.float64),
                'portrait':np.empty(0,dtype=bool)}
    t=np.concatenate([p[0] for p in pieces])
    xyz=np.concatenate([p[1] for p in pieces])
    valid=np.isfinite(t)
    order=np.argsort(t[valid],kind='stable')
    return {'t':t[valid][order], 'xyz':xyz[valid][order],
            'portrait':np.ones(int(np.count_nonzero(valid)),dtype=bool)}


def load_motion_hmog(df:pd.DataFrame, aid:str) -> dict[str,np.ndarray]:
    if df.empty:
        return {'t':np.empty(0,dtype=np.float64), 'xyz':np.empty((0,3),dtype=np.float64),
                'portrait':np.empty(0,dtype=bool)}
    ids=df.get('activityid',pd.Series(['']*len(df))).astype(str).str.strip().to_numpy()
    keep=ids==str(aid)
    t=v2.numeric(df,'eventtime')[keep]/1e6
    xyz=np.column_stack([v2.numeric(df,axis)[keep] for axis in ('x','y','z')])
    ori=v2.numeric(df,'phoneorientation')[keep]
    valid=np.isfinite(t)
    order=np.argsort(t[valid],kind='stable')
    return {'t':t[valid][order], 'xyz':xyz[valid][order],
            'portrait':(np.isfinite(ori[valid])&(ori[valid]==0))[order]}


def valid_geometry(g:Gesture)->bool:
    if g.multitouch or not g.portrait or len(g.touch_t)<3 or not g.end_ms>g.start_ms:
        return False
    maximum=float(np.hypot(g.touch_x-g.touch_x[0],g.touch_y-g.touch_y[0]).max())
    return maximum > 5.0


def sensor_segment(motion:dict[str,np.ndarray],start:float,end:float):
    times=motion['t']
    lo=int(np.searchsorted(times,start,side='left'))
    hi=int(np.searchsorted(times,end,side='right'))
    t=times[lo:hi];xyz=motion['xyz'][lo:hi];portrait=motion['portrait'][lo:hi]
    # Same strict temporal checks as v2.0; also require finite sensor axes for actual tensors.
    if len(t)<MIN_SAMPLES:return None
    if (t[0]-start)>MAX_EDGE_GAP_MS or (end-t[-1])>MAX_EDGE_GAP_MS:return None
    if np.any(np.diff(t)>MAX_INTERNAL_GAP_MS):return None
    if not np.all(portrait) or not np.isfinite(xyz).all():return None
    return t,xyz


def temporal(t_ms:np.ndarray,origin_ms:float)->tuple[np.ndarray,np.ndarray]:
    elapsed=(t_ms-origin_ms)/1000.0
    dt=np.zeros_like(elapsed)
    if len(dt)>1:dt[1:]=np.diff(t_ms)/1000.0
    # A repeated timestamp is represented by dt=0, not invented time.
    return elapsed,dt


def prepare_gesture(g:Gesture, accel:dict, gyro:dict, screen:tuple[int,int],
                    dataset:str,model:str,user:str,session:str,task:str,rep:str):
    if not valid_geometry(g):return None
    a=sensor_segment(accel,g.start_ms,g.end_ms)
    b=sensor_segment(gyro,g.start_ms,g.end_ms)
    if a is None or b is None:return None
    if v2.count_pairs(a[0],b[0],PAIR_TOLERANCE_MS)<1:return None
    # Separate real sequence lengths for each channel; no synthetic 10 ms sampling.
    w,h=screen
    tt,tdt=temporal(g.touch_t,g.start_ms)
    at,adt=temporal(a[0],g.start_ms)
    gt,gdt=temporal(b[0],g.start_ms)
    touch=np.column_stack((g.touch_x/w, g.touch_y/h,
         (g.touch_x-g.touch_x[0])/w,(g.touch_y-g.touch_y[0])/h,tt,tdt)).astype(np.float32)
    accel_x=np.column_stack((a[1],at,adt)).astype(np.float32)
    gyro_x=np.column_stack((b[1],gt,gdt)).astype(np.float32)
    if not (np.isfinite(touch).all() and np.isfinite(accel_x).all() and np.isfinite(gyro_x).all()):
        return None
    meta={'dataset':dataset,'model':model,'user_id':user,'session_id':session,
          'task':task,'repetition':rep,'start_ms':g.start_ms,'end_ms':g.end_ms,
          'duration_ms':g.end_ms-g.start_ms,
          'coord_outside_screen':bool(np.any(g.touch_x<-0.25*w) or np.any(g.touch_x>1.25*w) or
                                      np.any(g.touch_y<-0.25*h) or np.any(g.touch_y>1.25*h))}
    return {'touch':touch,'accel':accel_x,'gyro':gyro_x,'meta':meta}


def hmog_folder_session_id(folder:Path,act:pd.DataFrame):
    if 'sessionnumber' in act:
        codes=act['sessionnumber'].dropna().astype(str).unique()
        if len(codes)==1:return str(codes[0])
    return folder.name


def extract_hmog_user(root:Path,user:str,task:str):
    home=root/user
    if not home.is_dir():raise FileNotFoundError(home)
    records=[];stats=Counter();issues=[]
    screen=PHONE_LOGICAL['Samsung Galaxy S4']
    for folder in sorted(p for p in home.iterdir() if p.is_dir()):
        ap=v2.first_existing(folder,v2.H_FILENAMES['activity'])
        if ap is None:continue
        act=v2.read_hmog(ap,'activity')
        if act.empty:continue
        aid=act.get('id',pd.Series(['']*len(act))).astype(str).str.strip().to_numpy()
        taskids=v2.numeric(act,'taskid')
        mapping={aid[i]:v2.HMOG_TASKS[int(taskids[i])] for i in range(len(act))
                 if np.isfinite(taskids[i]) and int(taskids[i]) in v2.HMOG_TASKS
                 and v2.HMOG_TASKS[int(taskids[i])]==task}
        if not mapping:continue
        sid=hmog_folder_session_id(folder,act)
        tp=v2.first_existing(folder,v2.H_FILENAMES['touch'])
        if tp is None:issues.append(f'{user}/{sid}: no touch file');continue
        touch_df=v2.read_hmog(tp,'touch')
        gestures=parse_hmog_touch(touch_df,mapping)
        stats['contacts_seen']+=len(gestures)
        ad=v2.first_existing(folder,v2.H_FILENAMES['accel'])
        gd=v2.first_existing(folder,v2.H_FILENAMES['gyro'])
        a_df=v2.read_hmog(ad,'accel') if ad else pd.DataFrame()
        g_df=v2.read_hmog(gd,'gyro') if gd else pd.DataFrame()
        sensors={actid:(load_motion_hmog(a_df,actid),load_motion_hmog(g_df,actid)) for actid in mapping}
        for gesture in gestures:
            if not valid_geometry(gesture):continue
            stats['geometry_usable']+=1
            a,g=sensors[gesture.activity]
            sample=prepare_gesture(gesture,a,g,screen,'HMOG','Samsung Galaxy S4',user,sid,task,'')
            if sample is not None:
                records.append(sample);stats['strict_usable']+=1
    return records,stats,issues


def extract_feta_user(root:Path,user:str,model:str,task:str,feta_coordinates:str):
    home=root/user
    if not home.is_dir():raise FileNotFoundError(home)
    screen=PHONE_LOGICAL[model] if feta_coordinates=='points' else PHONE_PIXELS[model]
    records=[];stats=Counter();issues=[]
    for session in sorted(p for p in home.iterdir() if p.is_dir()):
        tdir=session/task
        if not tdir.is_dir():continue
        for rep in sorted(p for p in tdir.iterdir() if p.is_dir()):
            touch=[]
            for f in sorted((rep/'touch_data').glob('*.csv')):
                try:touch.extend(parse_feta_touch(f))
                except Exception as exc:issues.append(f'{f}: {type(exc).__name__}: {exc}')
            stats['contacts_seen']+=len(touch)
            try:
                accel=load_motion_feta(rep,'accel');gyro=load_motion_feta(rep,'gyro')
            except Exception as exc:
                issues.append(f'{rep}: {type(exc).__name__}: {exc}');continue
            for gesture in touch:
                if not valid_geometry(gesture):continue
                stats['geometry_usable']+=1
                sample=prepare_gesture(gesture,accel,gyro,screen,'FETA',model,user,
                                        session.name,task,rep.name)
                if sample is not None:
                    records.append(sample);stats['strict_usable']+=1
    return records,stats,issues


def subsample(records:list,limit:int,seed:int):
    if limit<=0 or len(records)<=limit:return records
    # Include one shortest and one longest gesture plus random examples per user.
    rg=random.Random(seed)
    chosen=[]
    if limit>=2:
        by_len=sorted(range(len(records)),key=lambda i:records[i]['meta']['duration_ms'])
        chosen=[by_len[0],by_len[-1]]
    remaining=[i for i in range(len(records)) if i not in set(chosen)]
    chosen+=rg.sample(remaining,min(limit-len(chosen),len(remaining)))
    return [records[i] for i in sorted(chosen,key=lambda i:(records[i]['meta']['session_id'],
                                                      records[i]['meta']['start_ms']))]


def save_shard(path:Path,records:list):
    path.parent.mkdir(parents=True,exist_ok=True)
    pack={}
    for key,cols in (('touch',len(TOUCH_FEATURES)),('accel',len(MOTION_FEATURES)),('gyro',len(MOTION_FEATURES))):
        lengths=np.asarray([len(r[key]) for r in records],dtype=np.int64)
        offset=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
        arrays=[r[key] for r in records]
        data=np.concatenate(arrays,axis=0) if arrays else np.empty((0,cols),dtype=np.float32)
        pack[f'{key}_data']=data
        pack[f'{key}_offsets']=offset
    # Store plain unicode metadata, never object arrays/pickle.
    meta=[json.dumps(r['meta'],ensure_ascii=False,separators=(',',':')) for r in records]
    pack['meta_json']=np.asarray(meta,dtype=np.str_)
    np.savez_compressed(path,**pack)
    return len(records)


def read_splits(path:Path):
    doc=json.loads(path.read_text(encoding='utf-8'))
    if doc.get('status')!='FROZEN_VERIFIED_AGAINST_V2_AUDIT':
        raise ValueError('Refusing non-frozen splits JSON')
    prot=doc.get('protocol',{})
    if prot.get('count_mode')!='strict' or prot.get('threshold_gestures_per_each_first_four_sessions')!=20:
        raise ValueError('Frozen protocol must specify strict/20')
    # A user may appear in multiple scenarios, but their global role must never change.
    assigned={}
    for ds, groups in doc['splits'].items():
        for group, parts in groups.items():
            if not isinstance(parts,dict):continue
            for role in ('train','validation','test'):
                for uid in parts.get(role,[]):
                    key=(ds,str(uid))
                    if key in assigned and assigned[key]!=role:
                        raise ValueError(f'Subject leakage across scenarios: {key}: '
                                         f'{assigned[key]} and {role}')
                    assigned[key]=role
    return doc


def get_users(doc,ds,model,task):
    if ds=='FETA':return list(doc['splits']['FETA'][model]['train'])
    return list(doc['splits']['HMOG'][task]['train'])


def selected_scenarios(text):
    if not text or text.lower()=='all':return list(SCENARIOS)
    keys=[x.strip() for x in text.split(',') if x.strip()]
    invalid=set(keys)-set(SCENARIOS)
    if invalid:raise ValueError(f'Unknown scenarios: {sorted(invalid)}. Available: {list(SCENARIOS)}')
    return list(dict.fromkeys(keys))


def prepare(args):
    splits_path=Path(args.splits)
    doc=read_splits(splits_path)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    modelmap=v2.feta_user_mapping(Path(args.userdata)) if any(
        SCENARIOS[s][0]=='FETA' for s in selected_scenarios(args.scenarios)) else {}
    rng=random.Random(args.seed)
    entries=[];counts={};issues=[]
    for slug in selected_scenarios(args.scenarios):
        ds,model,task=SCENARIOS[slug]
        users=sorted(get_users(doc,ds,model,task))
        rng.shuffle(users)
        if args.max_users_per_scenario>0:users=users[:args.max_users_per_scenario]
        counts[slug]={'users_requested':len(users),'users_with_samples':0,
                      'gestures':0,'geometry_usable':0,'strict_usable':0,'contacts_seen':0}
        print(f'[{slug}] {len(users)} TRAIN users',flush=True)
        for n,user in enumerate(users,1):
            if ds=='FETA' and modelmap.get(user)!=model:
                raise ValueError(f'FETA userdata.csv model mismatch: {user}: {modelmap.get(user)!r} vs {model!r}')
            try:
                if ds=='HMOG': records,stat,errs=extract_hmog_user(Path(args.hmog_root),user,task)
                else:records,stat,errs=extract_feta_user(Path(args.feta_root),user,model,
                                                           task,args.feta_coordinates)
            except Exception as exc:
                issues.append({'scenario':slug,'user':user,'error':f'{type(exc).__name__}: {exc}'})
                if args.fail_on_errors:raise
                continue
            for k,v in stat.items():counts[slug][k]+=int(v)
            for msg in errs[:20]:issues.append({'scenario':slug,'user':user,'warning':msg})
            if args.fail_on_errors and errs:raise ValueError(f'{slug}/{user}: {errs[0]}')
            chosen=subsample(records,args.max_gestures_per_user,
                             args.seed+int(hashlib.sha256((slug+user).encode()).hexdigest()[:8],16))
            if not chosen:
                issues.append({'scenario':slug,'user':user,'warning':'zero strict usable gestures'})
                continue
            fname=f'{slug}/{user}.npz'
            save_shard(out/'shards'/fname,chosen)
            entries.append({'scenario':slug,'dataset':ds,'device_model':model,'task':task,
                            'role':'train','user_id':user,'file':str(Path('shards')/fname),
                            'count':len(chosen)})
            counts[slug]['gestures']+=len(chosen)
            counts[slug]['users_with_samples']+=1
            if n%5==0 or n==len(users):
                print(f'[{slug}] users {n}/{len(users)}; samples={counts[slug]["gestures"]}',flush=True)
    if not entries:raise RuntimeError('No samples collected. Check input paths and issues.')
    idx={'version':VERSION,'role':'train','source_splits_sha256':sha256(splits_path),
         'parser_version':v2.VERSION,
         'parser_file_sha256':sha256(Path(v2.__file__)),
         'preprocessor_file_sha256':sha256(Path(__file__)),
         'seed':args.seed,'settings':{
         'feta_coordinates':args.feta_coordinates,
         'max_users_per_scenario':args.max_users_per_scenario,
         'max_gestures_per_user':args.max_gestures_per_user,
         'no_resampling':True,'filter':'FETA >=3, radial >5, portrait HMOG, strict paired IMU'},
         'feature_names':{'touch':TOUCH_FEATURES,'accel':MOTION_FEATURES,'gyro':MOTION_FEATURES},
         'scenarios':counts,'shards':entries,'warnings_count':len(issues)}
    (out/'index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'issues.json').write_text(json.dumps(issues[:2000],ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'DONE: {sum(e["count"] for e in entries)} gestures; {len(entries)} user/scenario shards -> {out}')
    print('Run inspect --cache',out)


class GestureDataset:
    """Ragged, lazy, cached NPZ-backed dataset. Reads only frozen TRAIN shards.

    Implements PyTorch Dataset protocol. Arrays loaded with allow_pickle=False.
    """
    def __init__(self,cache:str|Path,scenario:str|None=None):
        self.root=Path(cache)
        index=json.loads((self.root/'index.json').read_text(encoding='utf-8'))
        if index.get('role')!='train':raise ValueError('Refusing to load non-train cache')
        self.shards=[e for e in index['shards'] if scenario is None or e['scenario']==scenario]
        self.spans=[]
        for n,e in enumerate(self.shards):
            self.spans.extend((n,k) for k in range(e['count']))
        identities=sorted({f"{e['dataset']}/{e['user_id']}" for e in self.shards})
        self.identity_to_label={name:i for i,name in enumerate(identities)}
        self.feature_names=index['feature_names']

    def __len__(self):return len(self.spans)

    @functools.lru_cache(maxsize=4)
    def _read(self,shard_idx:int):
        path=self.root/self.shards[shard_idx]['file']
        with np.load(path,allow_pickle=False) as packed:
            return {k:packed[k] for k in packed.files}

    def __getitem__(self,i):
        import torch
        sid,local=self.spans[i]
        shard=self.shards[sid]; data=self._read(sid)
        result={}
        for kind in ('touch','accel','gyro'):
            a=int(data[f'{kind}_offsets'][local]);b=int(data[f'{kind}_offsets'][local+1])
            values=data[f'{kind}_data'][a:b]
            result[kind]=torch.from_numpy(values.copy())
        meta=json.loads(str(data['meta_json'][local]))
        result['meta']=meta
        result['label']=self.identity_to_label[f"{shard['dataset']}/{shard['user_id']}"]
        return result


def collate_gestures(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    result={}
    for kind in ('touch','accel','gyro'):
        xs=[row[kind] for row in batch]
        lens=torch.tensor([len(x) for x in xs],dtype=torch.long)
        padded=pad_sequence(xs,batch_first=True,padding_value=0.)
        mask=torch.arange(padded.shape[1]).unsqueeze(0)<lens.unsqueeze(1)
        result[kind]=padded
        result[kind+'_mask']=mask
        result[kind+'_lengths']=lens
    result['labels']=torch.tensor([x['label'] for x in batch],dtype=torch.long)
    result['meta']=[x['meta'] for x in batch]
    return result


def inspect(args):
    import torch
    from torch.utils.data import DataLoader
    ds=GestureDataset(args.cache,args.scenario)
    if not len(ds):raise ValueError('No matching samples in cache')
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,
                      collate_fn=collate_gestures)
    output=Path(args.cache)
    rows=[];count=0
    for batch in loader:
        bs=len(batch['meta'])
        for kind in ('touch','accel','gyro'):
            tensor=batch[kind]; mask=batch[kind+'_mask']; lens=batch[kind+'_lengths']
            assert tensor.shape[0]==bs and mask.shape==tensor.shape[:2]
            assert torch.equal(mask.sum(dim=1).cpu(),lens.cpu())
            assert torch.isfinite(tensor).all(),f'Invalid {kind} values'
            # No synthetic/padded data in unmasked positions.
            assert torch.count_nonzero(tensor[~mask])==0
            assert bool(torch.all(tensor[mask][:,-2]>=-1e-5)),f'{kind} negative elapsed time'
            assert bool(torch.all(tensor[mask][:,-1]>=-1e-5)),f'{kind} negative delta time'
        for i,meta in enumerate(batch['meta']):
            rec={'dataset':meta['dataset'],'model':meta['model'],'task':meta['task'],
                 'user_id':meta['user_id'],'session_id':meta['session_id'],
                 'repetition':meta['repetition'],'duration_ms':meta['duration_ms'],
                 'coord_outside_screen':int(meta['coord_outside_screen'])}
            for kind in ('touch','accel','gyro'):
                arr=batch[kind][i,batch[kind+'_mask'][i]]
                rec[kind+'_samples']=len(arr)
                rec[kind+'_max_dt_ms']=float(arr[:,-1].max()*1000) if len(arr) else 0.
                rec[kind+'_last_elapsed_ms']=float(arr[:,-2].max()*1000) if len(arr) else 0.
            rows.append(rec)
        count+=bs
    fieldnames=list(rows[0])
    with (output/'sample_inspection.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=fieldnames);wr.writeheader();wr.writerows(rows)
    summary=[]
    by=defaultdict(list)
    for r in rows:by[(r['dataset'],r['model'],r['task'])].append(r)
    for (dsname,model,task),part in sorted(by.items()):
        item={'dataset':dsname,'model':model,'task':task,'gestures':len(part)}
        for feature in ('duration_ms','touch_samples','accel_samples','gyro_samples','accel_max_dt_ms','gyro_max_dt_ms'):
            vals=np.asarray([r[feature] for r in part],dtype=np.float64)
            item[feature+'_median']=round(float(np.median(vals)),3)
            item[feature+'_p95']=round(float(np.percentile(vals,95)),3)
            item[feature+'_max']=round(float(vals.max()),3)
        item['coordinate_warnings']=sum(int(r['coord_outside_screen']) for r in part)
        summary.append(item)
    (output/'inspection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASSED variable-length DataLoader + mask + finite/time validation for',count,'gestures')
    for r in summary:
        print(f"  {r['dataset']:4s} {r['model']:18s} {r['task']:17s} n={r['gestures']:4d} "
              f"duration p50/p95={r['duration_ms_median']:.1f}/{r['duration_ms_p95']:.1f}ms "
              f"touch p95={r['touch_samples_p95']:.0f} accel p95={r['accel_samples_p95']:.0f} "
              f"gyro p95={r['gyro_samples_p95']:.0f} coordinate warnings={r['coordinate_warnings']}")
    print('Wrote',output/'sample_inspection.csv','and',output/'inspection_summary.json')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Read raw TRAIN-only gestures and cache ragged sequences')
    prep.add_argument('--splits',type=Path,default=Path('experiments/splits/hmog_feta_splits_seed_20261112.json'))
    prep.add_argument('--hmog-root',type=Path,default=Path('input_data/hmog_unarchived'))
    prep.add_argument('--feta-root',type=Path,default=Path('input_data/data_files'))
    prep.add_argument('--userdata',type=Path,default=Path('input_data/tables/tables/userdata.csv'))
    prep.add_argument('--output',type=Path,default=Path('output_data/gesture_dataloader_prototype_v1'))
    prep.add_argument('--scenarios',type=str,default='all',help='Comma-separated scenario IDs or all')
    prep.add_argument('--max-users-per-scenario',type=int,default=4,help='0 means all TRAIN users')
    prep.add_argument('--max-gestures-per-user',type=int,default=10,help='0 means all strict valid gestures')
    prep.add_argument('--feta-coordinates',choices=('points','pixels'),default='points',
                      help='FETA iOS coordinate-system assumption: default UIKit points')
    prep.add_argument('--seed',type=int,default=20261112)
    prep.add_argument('--fail-on-errors',action='store_true')
    ins=sub.add_parser('inspect',help='Check saved tensors, padding masks, timing and samples')
    ins.add_argument('--cache',type=Path,default=Path('output_data/gesture_dataloader_prototype_v1'))
    ins.add_argument('--scenario',choices=list(SCENARIOS),default=None)
    ins.add_argument('--batch-size',type=int,default=8)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    try:
        if args.command=='prepare':prepare(args)
        else:inspect(args)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
