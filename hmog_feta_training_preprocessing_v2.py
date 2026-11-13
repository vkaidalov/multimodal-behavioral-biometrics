#!/usr/bin/env python3
"""Training-only physical-unit audit, normalizers and length-bucketed HMOG/FETA loader.

Consumes cache produced by hmog_feta_gesture_dataset_prototype_v1.py.
Does NOT re-read the raw datasets or touch frozen splits. Does NOT train any network.

Commands:
  audit-units : per-scenario numerical unit diagnostics, no conversion, no torch
  fit         : explicit unit conversion and training-only, user-balanced scaler fit
  inspect     : inspect normalized/capped/padded tensors, masks and bucket batches

The original v1 cache remains unchanged, preserving raw sensor values and native times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import hmog_feta_gesture_dataset_prototype_v1 as v1

VERSION = 'gesture-training-preprocessing-2.0.0-20261113'
G0 = 9.80665
MODALITIES = ('accel', 'gyro')
KINDS = ('touch', 'accel', 'gyro')


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def read_index(cache: Path):
    path = cache/'index.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    if index.get('role') != 'train':
        raise ValueError('Only training-only v1 caches are supported. Refusing validation/test data.')
    if index.get('version') != v1.VERSION:
        raise ValueError('Unexpected source cache format/version.')
    expected = {'touch': list(v1.TOUCH_FEATURES),
                'accel': list(v1.MOTION_FEATURES), 'gyro': list(v1.MOTION_FEATURES)}
    if {k: list(index['feature_names'][k]) for k in KINDS} != expected:
        raise ValueError('Source feature columns differ from v1. Do not process silently.')
    return index, checksum(path)


def load_shard(cache: Path, entry: dict):
    target = (cache / entry['file']).resolve()
    if cache.resolve() not in target.parents:
        raise ValueError('Invalid path outside the cache folder')
    with np.load(target, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_indices(n: int, k: int):
    """Evenly spaced source indices; retain physical endpoints."""
    if k <= 0 or n <= k:
        return np.arange(n, dtype=np.int64)
    if k == 1:
        raise ValueError('A maximum length of 1 cannot preserve both endpoints')
    # Integer rounding gives unique indices whenever n >= k.
    return np.rint(np.linspace(0, n - 1, k)).astype(np.int64)


def take_gesture(packed: dict, row: int, kind: str):
    offsets = packed[kind + '_offsets']
    s, e = int(offsets[row]), int(offsets[row+1])
    return np.asarray(packed[kind+'_data'][s:e], dtype=np.float32)


def balanced_probe(cache: Path, index: dict, max_users: int, gestures_per_user: int,
                   points_per_gesture: int, seed: int):
    """Streaming sample; only a small number of gestures per user/scenario.

    Each scenario is reported separately. No assumptions about source units.
    """
    groups = defaultdict(list)
    for entry in index['shards']:
        groups[entry['scenario']].append(entry)
    rng = np.random.default_rng(seed)
    out = {}
    for scenario, entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda x: x['user_id'])
        if max_users > 0 and len(entries) > max_users:
            pick = rng.choice(len(entries), size=max_users, replace=False)
            entries = [entries[int(j)] for j in sorted(pick)]
        xyz_values = {kind: [] for kind in MODALITIES}
        selected = 0
        for entry in entries:
            packed = load_shard(cache, entry)
            n = entry['count']
            if gestures_per_user > 0 and n > gestures_per_user:
                ids = np.sort(rng.choice(n, gestures_per_user, replace=False))
            else:
                ids = np.arange(n)
            for j in ids:
                for kind in MODALITIES:
                    a = take_gesture(packed, int(j), kind)
                    ix = sample_indices(len(a), points_per_gesture)
                    xyz_values[kind].append(a[ix, :3].astype(np.float64))
                selected += 1
        summary = {'users': len(entries), 'gestures_sampled': selected}
        for kind in MODALITIES:
            v = np.concatenate(xyz_values[kind]) if xyz_values[kind] else np.empty((0,3))
            if not len(v):
                raise ValueError(f'No {kind} readings for {scenario}')
            norms = np.linalg.norm(v, axis=1)
            abs_xyz = np.abs(v).reshape(-1)
            summary[kind] = {
                'readings': int(len(v)),
                'norm_p10': round(float(np.percentile(norms, 10)),6),
                'norm_p50': round(float(np.percentile(norms, 50)),6),
                'norm_p90': round(float(np.percentile(norms, 90)),6),
                'norm_p99': round(float(np.percentile(norms, 99)),6),
                'norm_mean': round(float(norms.mean()),6),
                'abs_axis_p99': round(float(np.percentile(abs_xyz, 99)),6),
            }
        out[scenario] = summary
    return out


def audit_units(args):
    cache = Path(args.cache)
    index, index_sha = read_index(cache)
    report = {
        'version': VERSION,
        'source_index_sha256': index_sha,
        'splits_sha256': index['source_splits_sha256'],
        'seed': args.seed,
        'sample_policy': {'max_users_per_scenario':args.max_users_per_scenario,
                          'gestures_per_user':args.gestures_per_user,
                          'points_per_gesture':args.points_per_gesture},
        'note': 'Raw values only. No units inferred automatically; no conversion performed.',
        'scenarios': balanced_probe(cache, index, args.max_users_per_scenario,
                                    args.gestures_per_user, args.points_per_gesture, args.seed)
    }
    atomic_json(Path(args.output), report)
    print('Wrote', args.output)
    print('Raw accelerometer vector-norm medians (check gravity vs userAcceleration):')
    for s,r in sorted(report['scenarios'].items()):
        print(f'  {s:20s} accel p50={r["accel"]["norm_p50"]:.4f} '
              f'gyro p50={r["gyro"]["norm_p50"]:.4f} '
              f'(n={r["users"]} users, {r["gestures_sampled"]} gestures)')
    print('IMPORTANT: do not run fit until FETA units/sensor semantics are confirmed.')


def unit_config(args):
    if args.feta_accel_unit is None:
        raise ValueError('--feta-accel-unit is required: g or mps2. Verify with audit first.')
    return {
        'HMOG': {'accel': args.hmog_accel_unit, 'gyro': args.hmog_gyro_unit},
        'FETA': {'accel': args.feta_accel_unit, 'gyro': args.feta_gyro_unit},
    }


def unit_scale(dataset: str, kind: str, units: dict) -> float:
    unit = units[dataset][kind]
    if kind == 'accel':
        return G0 if unit == 'g' else 1.0
    return math.pi / 180.0 if unit == 'degps' else 1.0


def convert_xyz(arr: np.ndarray, dataset: str, kind: str, units: dict) -> np.ndarray:
    a = arr.copy()
    a[:, :3] *= unit_scale(dataset, kind, units)
    return a


def weighted_moments(a: np.ndarray):
    # Each GESTURE contributes equal weight, regardless of sample count/length.
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite sensor array in source cache')
    f = np.asarray(a, dtype=np.float64)
    return f.mean(axis=0), np.square(f).mean(axis=0)


def combine_moments(moment_list):
    means = np.stack([p[0] for p in moment_list])
    sqmeans = np.stack([p[1] for p in moment_list])
    mean = means.mean(axis=0)
    variance = np.maximum(sqmeans.mean(axis=0) - np.square(mean), 1e-10)
    return {'mean': [float(x) for x in mean],
            'std': [float(x) for x in np.sqrt(variance)]}


def fit(args):
    cache = Path(args.cache)
    index, index_sha = read_index(cache)
    units = unit_config(args)
    rng = np.random.default_rng(args.seed)
    by_scenario = defaultdict(list)
    for e in index['shards']:
        by_scenario[e['scenario']].append(e)
    scenario_moments = {}
    coverage = {}
    for scenario, entries in sorted(by_scenario.items()):
        user_moments = {kind: [] for kind in MODALITIES}
        total_gestures = 0
        for entry in sorted(entries, key=lambda x: x['user_id']):
            packed = load_shard(cache, entry)
            n = entry['count']
            if args.gestures_per_user > 0 and n > args.gestures_per_user:
                ids = np.sort(rng.choice(n,args.gestures_per_user,replace=False))
            else:
                ids = np.arange(n)
            per_user={kind:[] for kind in MODALITIES}
            for j in ids:
                for kind in MODALITIES:
                    arr = take_gesture(packed,int(j),kind)
                    ix = sample_indices(len(arr), args.points_per_gesture)
                    xyz = convert_xyz(arr[ix], entry['dataset'], kind, units)[:,:3]
                    per_user[kind].append(weighted_moments(xyz))
                total_gestures += 1
            for kind in MODALITIES:
                user_moments[kind].append(combine_moments(per_user[kind]))
        scenario_moments[scenario] = {}
        for kind in MODALITIES:
            user_ms = user_moments[kind]
            user_tuple=[(np.asarray(u['mean']), np.square(u['std'])+np.square(u['mean']))
                        for u in user_ms]
            scenario_moments[scenario][kind] = combine_moments(user_tuple)
        coverage[scenario] = {'users':len(entries), 'gestures_sampled':total_gestures}
    # Global stats give each SCENARIO equal weight (not each raw gesture).
    global_norm = {}
    for kind in MODALITIES:
        scenario_tuple = [(np.asarray(m[kind]['mean']),
                           np.square(m[kind]['std'])+np.square(m[kind]['mean']))
                          for m in scenario_moments.values()]
        global_norm[kind] = combine_moments(scenario_tuple)
    result={
        'version': VERSION,
        'source_index_sha256': index_sha,
        'splits_sha256': index['source_splits_sha256'],
        'training_only': True,
        'sensor_units_input': units,
        'sensor_units_output': {'accel':'m/s^2','gyro':'rad/s'},
        'gravity_constant_m_s2': G0,
        'source_motion_feature_names': list(v1.MOTION_FEATURES),
        'sampling': {'seed':args.seed,'gestures_per_user':args.gestures_per_user,
                     'points_per_gesture':args.points_per_gesture,
                     'aggregation':'equal gesture within user; equal user within scenario; equal scenario for global'},
        'coverage':coverage,
        'global':global_norm,
        'by_scenario':scenario_moments,
        'note': ('Only sensor XYZ are z-scored. Touch coordinates and real elapsed/dt '
                 'remain unchanged; original cache is unmodified.')
    }
    atomic_json(Path(args.output),result)
    print('Wrote TRAIN-only scaling stats',args.output)
    for kind in MODALITIES:
        print(kind, 'global mean',result['global'][kind]['mean'],
              'std',result['global'][kind]['std'])


def time_subsample(a: np.ndarray, max_len: int):
    """Retain actually measured rows, including first/last, never interpolate.

    NOTE: dt must be recalculated after dropping samples; elapsed is unchanged.
    """
    n=len(a)
    if n<2:raise ValueError('Expected at least 2 samples per accepted gesture')
    if max_len<=0 or n<=max_len:
        return a.copy(), False
    if max_len<3:raise ValueError('Maximum allowed length must be >=3')
    elapsed = np.asarray(a[:,-2],dtype=np.float64)
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed)<-1e-6):
        raise ValueError('Bad/nonmonotonic elapsed time')
    targets = np.linspace(elapsed[0], elapsed[-1], max_len)
    pos = np.searchsorted(elapsed, targets, side='left')
    pos = np.clip(pos,1,n-1)
    left=pos-1
    pos=np.where(abs(elapsed[left]-targets)<=abs(elapsed[pos]-targets),left,pos)
    pos[0]=0;pos[-1]=n-1
    indices=np.unique(pos)
    if len(indices)<max_len:
        remaining=np.setdiff1d(np.arange(n),indices,assume_unique=True)
        add=sample_indices(len(remaining), min(max_len-len(indices),len(remaining)))
        indices=np.sort(np.concatenate((indices,remaining[add])))
    if len(indices)>max_len:
        indices=indices[sample_indices(len(indices),max_len)]
    out=a[indices].copy()
    out[0,-1]=0.0
    out[1:,-1]=np.maximum(0.0,np.diff(out[:,-2]))
    return out, True


class TrainingGestureDataset:
    """Read-only training set with unit conversion, optional global/scenario z-score and caps."""
    def __init__(self,cache,stats,scenario=None,scope='global',
                 max_touch=256,max_motion=512,clip_z=0.0):
        self.base=v1.GestureDataset(cache,scenario=scenario)
        self.root=Path(cache)
        self.index, index_sha=read_index(self.root)
        if isinstance(stats,(str,Path)):
            self.stats=json.loads(Path(stats).read_text(encoding='utf-8'))
        else:self.stats=stats
        if self.stats.get('source_index_sha256')!=index_sha:
            raise ValueError('Scaler belongs to another source cache!')
        if self.stats.get('splits_sha256')!=self.index['source_splits_sha256']:
            raise ValueError('Scaler belongs to another frozen split!')
        if not self.stats.get('training_only'):
            raise ValueError('Scaler was not fitted on training data')
        if scope not in ('global','scenario'):
            raise ValueError('scope must be global or scenario')
        self.scope=scope
        self.max_len={'touch':max_touch,'accel':max_motion,'gyro':max_motion}
        for kind, cap in self.max_len.items():
            if cap!=0 and cap<3:raise ValueError(f'{kind} cap must be 0 or >=3')
        self.clip_z=float(clip_z)
        self.identity_to_label=self.base.identity_to_label
        # Cheap length indices for a length-aware batch sampler.
        self.lengths=[]
        for shard in self.base.shards:
            with np.load(self.root/shard['file'],allow_pickle=False) as p:
                offsets=[np.asarray(p[f'{key}_offsets']) for key in KINDS]
                self.lengths.extend(tuple(int(off[i+1]-off[i]) for off in offsets)
                                    for i in range(shard['count']))
        if len(self.lengths)!=len(self.base):raise ValueError('Lengths do not match dataset')

    def __len__(self):return len(self.base)

    def __getitem__(self,i):
        import torch
        sample=self.base[i]
        metadata=sample['meta']
        shard_idx, _ = self.base.spans[i]
        scenario = self.base.shards[shard_idx]['scenario']
        stat=self.stats['global'] if self.scope=='global' else self.stats['by_scenario'][scenario]
        out={'meta':{**metadata,'scenario':scenario},'label':sample['label']}
        lengths_before={}
        downsampled={}
        for kind in KINDS:
            a=sample[kind].numpy()
            lengths_before[kind]=len(a)
            a,changed=time_subsample(a,self.max_len[kind])
            downsampled[kind]=changed
            if kind!='touch':
                a=convert_xyz(a,metadata['dataset'],kind,self.stats['sensor_units_input'])
                m=np.asarray(stat[kind]['mean'],dtype=np.float32)
                std=np.asarray(stat[kind]['std'],dtype=np.float32)
                a[:,:3]=(a[:,:3]-m)/std
                if self.clip_z>0:a[:,:3]=np.clip(a[:,:3],-self.clip_z,self.clip_z)
            if not np.isfinite(a).all():raise ValueError(f'Nonfinite {kind} tensor')
            out[kind]=torch.from_numpy(a)
        out['lengths_before']=lengths_before
        out['downsampled']=downsampled
        return out


class LengthBucketBatchSampler:
    """Buckets similar lengths, shuffles between epochs; NOT a metric-learning PK sampler."""
    def __init__(self,dataset:TrainingGestureDataset,batch_size=64,
                 bucket_multiplier=50,seed=20261112,drop_last=False):
        if batch_size<1:raise ValueError('batch_size must be positive')
        self.ds=dataset;self.batch_size=batch_size
        self.bucket_multiplier=max(1,bucket_multiplier)
        self.seed=seed;self.epoch=0;self.drop_last=drop_last

    def set_epoch(self,epoch:int):self.epoch=int(epoch)

    def __len__(self):
        n=len(self.ds)
        return n//self.batch_size if self.drop_last else math.ceil(n/self.batch_size)

    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        ids=list(range(len(self.ds)));rng.shuffle(ids)
        span=self.batch_size*self.bucket_multiplier
        all_batches=[]
        def cost(i):
            lens=self.ds.lengths[i]
            caps=self.ds.max_len
            effective=[min(lens[j],caps[KINDS[j]]) if caps[KINDS[j]] else lens[j]
                       for j in range(3)]
            return max(effective[0],effective[1],effective[2])
        for start in range(0,len(ids),span):
            chunk=ids[start:start+span]
            chunk.sort(key=cost)
            for k in range(0,len(chunk),self.batch_size):
                batch=chunk[k:k+self.batch_size]
                if len(batch)==self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        rng.shuffle(all_batches)
        yield from all_batches


def inspect(args):
    import torch
    from torch.utils.data import DataLoader
    cache=Path(args.cache)
    dataset=TrainingGestureDataset(cache,args.stats,args.scenario,args.scope,
                                   args.max_touch,args.max_motion,args.clip_z)
    sampler=LengthBucketBatchSampler(dataset,batch_size=args.batch_size,
                    bucket_multiplier=args.bucket_multiplier,seed=args.seed)
    loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=v1.collate_gestures,
                      num_workers=0)
    report={'version':VERSION,'scenario':args.scenario or 'all','scope':args.scope,
            'dataset_size':len(dataset),'batches_inspected':0,'gestures_inspected':0,
            'capped_count':Counter(),'largest_padded_shapes':{},
            'negative_dt':0,'nonfinite':0,'mask_errors':0,
            'max_abs_normalized':{'accel':0.,'gyro':0.},
            'note':'Only smoke-check batches, not full training.'}
    for k,b in enumerate(loader):
        if k>=args.inspect_batches:break
        report['batches_inspected']+=1
        report['gestures_inspected']+=len(b['meta'])
        for kind in KINDS:
            x=b[kind]; mask=b[kind+'_mask'];lens=b[kind+'_lengths']
            if not torch.equal(mask.sum(1),lens):report['mask_errors']+=1
            if not torch.isfinite(x).all():report['nonfinite']+=1
            if torch.any(x[mask][:,-1]<-1e-6):report['negative_dt']+=1
            if torch.count_nonzero(x[~mask])!=0:report['mask_errors']+=1
            report['largest_padded_shapes'][kind]=max(report['largest_padded_shapes'].get(kind,0),int(x.shape[1]))
            if kind!='touch':
                report['max_abs_normalized'][kind]=max(report['max_abs_normalized'][kind],
                       float(x[mask][:,:3].abs().max()))
        for meta in b['meta']:
            if meta['dataset'] not in ('FETA','HMOG'):raise ValueError('Bad dataset source')
    if report['mask_errors'] or report['nonfinite'] or report['negative_dt']:
        raise AssertionError('Batch validation failed: '+str(report))
    # Count true sample capping using fixed dataset lengths, independent of inspected batches.
    cap_counts={kind:sum(lens[j]>dataset.max_len[kind] for lens in dataset.lengths)
                if dataset.max_len[kind] else 0 for j,kind in enumerate(KINDS)}
    report['capped_count']=cap_counts
    atomic_json(Path(args.output),report)
    print('PASSED:',report['gestures_inspected'],'gestures in',report['batches_inspected'],'batches')
    print('Dataset size:',len(dataset),'Capped counts:',cap_counts)
    print('Max padding time axis:',report['largest_padded_shapes'])
    print('Wrote',args.output)


def make_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    def common(c):
        c.add_argument('--cache',type=Path,default=Path('output_data/gesture_cache_train_full'))
        c.add_argument('--seed',type=int,default=20261112)
    au=sub.add_parser('audit-units',help='Check raw sensor scale per dataset and scenario')
    common(au)
    au.add_argument('--max-users-per-scenario',type=int,default=10)
    au.add_argument('--gestures-per-user',type=int,default=30)
    au.add_argument('--points-per-gesture',type=int,default=128)
    au.add_argument('--output',type=Path,default=Path('output_data/preprocessing_v2/unit_audit.json'))
    ft=sub.add_parser('fit',help='Fit motion scaling statistics on training users ONLY')
    common(ft)
    ft.add_argument('--feta-accel-unit',choices=('g','mps2'),required=True,
                    help='Mandatory explicit choice after auditing FETA source data')
    ft.add_argument('--hmog-accel-unit',choices=('g','mps2'),default='mps2')
    ft.add_argument('--feta-gyro-unit',choices=('radps','degps'),default='radps')
    ft.add_argument('--hmog-gyro-unit',choices=('radps','degps'),default='radps')
    ft.add_argument('--gestures-per-user',type=int,default=300)
    ft.add_argument('--points-per-gesture',type=int,default=64)
    ft.add_argument('--output',type=Path,default=Path('output_data/preprocessing_v2/normalizers_train.json'))
    ins=sub.add_parser('inspect',help='Smoke-check unit-converted, normalized and bucketed batches')
    common(ins)
    ins.add_argument('--stats',type=Path,default=Path('output_data/preprocessing_v2/normalizers_train.json'))
    ins.add_argument('--scenario',choices=list(v1.SCENARIOS),default=None)
    ins.add_argument('--scope',choices=('global','scenario'),default='global')
    ins.add_argument('--max-touch',type=int,default=256)
    ins.add_argument('--max-motion',type=int,default=512)
    ins.add_argument('--clip-z',type=float,default=0.0)
    ins.add_argument('--batch-size',type=int,default=16)
    ins.add_argument('--bucket-multiplier',type=int,default=20)
    ins.add_argument('--inspect-batches',type=int,default=20)
    ins.add_argument('--output',type=Path,default=Path('output_data/preprocessing_v2/loader_inspection.json'))
    return p


def main():
    args=make_parser().parse_args()
    if args.command=='audit-units':audit_units(args)
    elif args.command=='fit':fit(args)
    elif args.command=='inspect':inspect(args)


if __name__=='__main__':main()
