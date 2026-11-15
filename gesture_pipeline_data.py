"""Role-aware, read-only HMOG/FETA cache loader and cross-session PK sampler.

Expected cache: gesture-dataloader-prototype-1.1.0 NPZ files. Never fits on
validation/test. Uses the existing train-only physical-unit scaler unmodified.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import hmog_feta_gesture_dataset_prototype_v1 as raw
import hmog_feta_training_preprocessing_v2 as pre


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class GestureCacheDataset(Dataset):
    def __init__(self, cache, stats, *, expected_role, scenario=None, scope='global',
                 max_touch=256, max_motion=512, clip_z=0.0, splits=None):
        self.root=Path(cache)
        self.index=json.loads((self.root/'index.json').read_text(encoding='utf-8'))
        if self.index['role'] != expected_role:
            raise ValueError(f"Expected {expected_role} cache, got {self.index['role']}")
        if self.index['version'] != raw.VERSION:
            raise ValueError('Unsupported NPZ cache version')
        if self.index.get('settings',{}).get('session_scope') != 'first-four':
            raise ValueError('This experiment requires first-four cache')
        expected_features={'touch':list(raw.TOUCH_FEATURES),'accel':list(raw.MOTION_FEATURES),
                           'gyro':list(raw.MOTION_FEATURES)}
        if {k:list(v) for k,v in self.index['feature_names'].items()} != expected_features:
            raise ValueError('Unexpected feature schema')
        self.stats_path=Path(stats)
        self.stats=json.loads(self.stats_path.read_text(encoding='utf-8'))
        if not self.stats.get('training_only'):
            raise ValueError('Scaler must originate from training only')
        if self.stats['splits_sha256'] != self.index['source_splits_sha256']:
            raise ValueError('Train scaler and cache use different splits')
        if expected_role=='train' and self.stats['source_index_sha256']!=file_hash(self.root/'index.json'):
            raise ValueError('Train scaler belongs to another training cache')
        if splits is not None and file_hash(splits)!=self.index['source_splits_sha256']:
            raise ValueError('Cache does not match requested frozen splits')
        if scope not in ('global','scenario'):
            raise ValueError('Normalizer scope must be global/scenario')
        self.role=expected_role;self.scope=scope
        self.max_len={'touch':int(max_touch),'accel':int(max_motion),'gyro':int(max_motion)}
        if any(0<c<3 for c in self.max_len.values()):raise ValueError('Caps must be 0 or >=3')
        self.clip_z=float(clip_z)
        self.shards=[x for x in self.index['shards'] if scenario is None or x['scenario']==scenario]
        self.spans=[]; self.metadata=[]; self.lengths=[]
        self.by_scenario_user=defaultdict(list)
        self.by_identity=defaultdict(list)
        self.by_identity_session=defaultdict(lambda:defaultdict(list))
        for shard_idx,entry in enumerate(self.shards):
            path=self.root/entry['file']
            with np.load(path,allow_pickle=False) as z:
                meta=z['meta_json']; offsets={k:z[k+'_offsets'] for k in pre.KINDS}
                if len(meta)!=entry['count']:raise ValueError(f'Metadata count mismatch: {path}')
                for j in range(len(meta)):
                    m=json.loads(str(meta[j])); n=len(self.spans)
                    if m['user_id']!=entry['user_id'] or m['dataset']!=entry['dataset']:
                        raise ValueError('Shard metadata identity mismatch')
                    m['scenario']=entry['scenario']
                    identity=f"{entry['dataset']}/{m['user_id']}"
                    self.spans.append((shard_idx,j));self.metadata.append(m)
                    self.lengths.append(tuple(int(offsets[k][j+1]-offsets[k][j]) for k in pre.KINDS))
                    self.by_scenario_user[(entry['scenario'],identity)].append(n)
                    self.by_identity[identity].append(n)
                    self.by_identity_session[identity][str(m['session_id'])].append(n)
        if not self.spans:raise ValueError(f'Empty cache for scenario {scenario}')
        # Independently check that every cached user/session is authorized by the
        # frozen split for this exact role; prevents mislabeled validation/test.
        if splits is not None:
            frozen=json.loads(Path(splits).read_text(encoding='utf-8'))
            for (sc,identity),indices in self.by_scenario_user.items():
                ds,device,task=raw.SCENARIOS[sc]
                uid=identity.split('/',1)[1]
                expected_users=(frozen['splits']['FETA'][device][expected_role]
                                if ds=='FETA' else frozen['splits']['HMOG'][task][expected_role])
                if uid not in set(map(str,expected_users)):
                    raise ValueError(f'Forbidden {expected_role} identity in {sc}: {uid}')
                allowed=set(map(str,frozen['session_index'][ds][device][uid][task]))
                if len(allowed)!=4 or any(str(self.metadata[i]['session_id']) not in allowed for i in indices):
                    raise ValueError(f'Incorrect frozen first-four sessions for {sc}/{uid}')
        identities=sorted(self.by_identity)
        self.identity_to_label={k:i for i,k in enumerate(identities)}
        self.labels=[self.identity_to_label[f"{m['dataset']}/{m['user_id']}"] for m in self.metadata]

    @functools.lru_cache(maxsize=32)
    def _shard(self, shard_idx):
        path=self.root/self.shards[shard_idx]['file']
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files if k!='meta_json'}

    def __len__(self):return len(self.spans)

    def __getitem__(self, i):
        shard_idx,j=self.spans[i]; z=self._shard(shard_idx)
        meta=self.metadata[i]
        norm=self.stats['global'] if self.scope=='global' else self.stats['by_scenario'][meta['scenario']]
        result={'meta':meta,'label':self.labels[i]}
        for k in pre.KINDS:
            offsets=z[k+'_offsets'];s,e=int(offsets[j]),int(offsets[j+1])
            a=np.asarray(z[k+'_data'][s:e],dtype=np.float32)
            a,_=pre.time_subsample(a,self.max_len[k])
            if k!='touch':
                a=pre.convert_xyz(a,meta['dataset'],k,self.stats['sensor_units_input'])
                a[:,:3]=(a[:,:3]-np.asarray(norm[k]['mean'],np.float32))/np.asarray(norm[k]['std'],np.float32)
                if self.clip_z>0:np.clip(a[:,:3],-self.clip_z,self.clip_z,out=a[:,:3])
            if not np.isfinite(a).all():raise ValueError(f'Nonfinite values in {k}, row {i}')
            result[k]=torch.from_numpy(a)
        return result


def collate(batch):return raw.collate_gestures(batch)


class IdentityScenarioPKSampler(Sampler):
    """P distinct identities, K samples/identity, from >=2 sessions when possible.

    All eight scenario identities are independently balanced in global mode.
    Fixed batches/epoch keep specialized/global GPU budgets comparable.
    """
    def __init__(self, ds, p=12, k=4, steps=120, seed=20261112):
        if k<2 or p<2:raise ValueError('P and K must each be >=2')
        self.ds=ds;self.p=p;self.k=k;self.steps=steps;self.seed=seed;self.epoch=0
        self.groups=defaultdict(dict)
        for (sc,identity),indices in ds.by_scenario_user.items():
            sessions=defaultdict(list)
            for i in indices:sessions[str(ds.metadata[i]['session_id'])].append(i)
            if len(sessions)<2:
                raise ValueError(f'Need >=2 sessions for PK positives: {sc}/{identity}')
            self.groups[sc][identity]=sessions
        if len({i for g in self.groups.values() for i in g})<p:
            raise ValueError('Not enough distinct training identities for P')

    def set_epoch(self,e):self.epoch=int(e)
    def __len__(self):return self.steps

    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        scenarios=sorted(self.groups)
        for step in range(self.steps):
            selected=[];used=set()
            # Round-robin across scenarios: avoid overrepresenting numerous FETA gestures.
            seq=[scenarios[(step*self.p+j)%len(scenarios)] for j in range(self.p)]
            for scenario in seq:
                ids=[x for x in self.groups[scenario] if x not in used]
                if not ids:
                    ids=[x for sc in scenarios for x in self.groups[sc] if x not in used]
                    if not ids:raise ValueError('Cannot fill batch with distinct subjects')
                    identity=rng.choice(ids)
                    scenario=rng.choice([sc for sc in scenarios if identity in self.groups[sc]])
                else:identity=rng.choice(ids)
                used.add(identity)
                sessions=self.groups[scenario][identity]
                keys=rng.sample(list(sessions),2)
                picks=[rng.choice(sessions[keys[0]]),rng.choice(sessions[keys[1]])]
                for _ in range(self.k-2):
                    key=rng.choice(list(sessions));picks.append(rng.choice(sessions[key]))
                rng.shuffle(picks);selected.extend(picks)
            # Keep per-identity samples adjacent for compressed-NPZ cache locality.
            yield selected


def select_validation_indices(ds, max_per_session=32, seed=20261112):
    """Deterministic subset per identity/session used only for checkpoint selection."""
    rng=np.random.default_rng(seed)
    chosen=[]
    for identity,sessions in sorted(ds.by_identity_session.items()):
        for session,idx in sorted(sessions.items()):
            # Preserve both task types; bound training-time validation cost.
            if max_per_session and len(idx)>max_per_session:
                sample=np.sort(rng.choice(idx,max_per_session,replace=False))
                chosen.extend(int(x) for x in sample)
            else:chosen.extend(idx)
    return sorted(chosen)
