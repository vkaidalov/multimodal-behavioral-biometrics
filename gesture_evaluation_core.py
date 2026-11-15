"""Authentication protocol: first two sessions enroll, last two verify.

Within scenario: other role-matched subjects are zero-effort impostors.
Windows do not cross either session or FETA repetition boundaries. The
reference is an equally weighted two-session embedding centroid. Window score
is mean cosine similarity of its individual gestures to the reference.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

WINDOWS=(1,2,3,5,10,20)


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/max(float(np.linalg.norm(x)),1e-12)


def save_embedding_table(path, embeddings, metadata, extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    attrs=('scenario','dataset','user_id','session_id','repetition')
    data={'embeddings':np.asarray(embeddings,dtype=np.float32)}
    for key in attrs:data[key]=np.array([str(m.get(key,'')) for m in metadata],dtype=np.str_)
    for key in ('start_ms','end_ms'):
        data[key]=np.array([float(m[key]) for m in metadata],dtype=np.float64)
    data['extra_json']=np.asarray(json.dumps(extra or {},sort_keys=True),dtype=np.str_)
    np.savez_compressed(path,**data)


def load_embedding_table(path):
    # IMPORTANT: NpzFile.__getitem__ decompresses the entire ZIP member
    # on EVERY access.  `z[key][i]` inside a per-gesture loop repeatedly
    # decompressed seven columns N times (O(N^2) decompression work), which
    # could make a small evaluation archive appear frozen for many minutes.
    # Read each column exactly once, then index the already-loaded arrays.
    with np.load(path, allow_pickle=False) as z:
        emb = z['embeddings'].copy()
        keys = ['scenario', 'dataset', 'user_id', 'session_id',
                'repetition', 'start_ms', 'end_ms']
        columns = {key: z[key] for key in keys}
        extra = json.loads(str(z['extra_json']))
    metadata = [
        {key: columns[key][i].item() for key in keys}
        for i in range(len(emb))
    ]
    if len(emb)==0 or not np.isfinite(emb).all():raise ValueError('Empty/nonfinite embeddings')
    return emb,metadata,extra


def get_session_ids(splits, scenario, user):
    from hmog_feta_gesture_dataset_prototype_v1 import SCENARIOS
    ds,model,task=SCENARIOS[scenario]
    return [str(s) for s in splits['session_index'][ds][model][str(user)][task]]


def build_role_scenarios(embeddings,meta, splits,role,scenario_filter=None):
    """Verify identity role and 4-session membership BEFORE running an evaluation."""
    from hmog_feta_gesture_dataset_prototype_v1 import SCENARIOS
    groups=defaultdict(lambda:defaultdict(list))
    for i,m in enumerate(meta):
        sc=m['scenario']
        if scenario_filter is not None and sc not in scenario_filter:continue
        ds,model,task=SCENARIOS[sc]
        candidates=splits['splits']['FETA'][model][role] if ds=='FETA' else splits['splits']['HMOG'][task][role]
        if str(m['user_id']) not in set(map(str,candidates)):
            raise ValueError(f'Role leakage: {sc}/{m["user_id"]} is not a {role} user')
        ids=get_session_ids(splits,sc,m['user_id'])
        if len(ids)!=4 or str(m['session_id']) not in ids:
            raise ValueError(f'Unexpected session: {sc}/{m["user_id"]}/{m["session_id"]}')
        groups[sc][str(m['user_id'])].append(i)
    if not groups:raise ValueError('No embeddings for requested scenarios')
    return groups


def user_reference(embeddings,meta,indices,session_ids):
    vectors=[]
    for s in session_ids[:2]:
        rows=[i for i in indices if str(meta[i]['session_id'])==s]
        if not rows:raise ValueError(f'Missing enrollment session {s}')
        vectors.append(unit(embeddings[rows].mean(axis=0)))
    return unit(np.stack(vectors).mean(axis=0))


def windows_for_user(embeddings,meta,indices,session_ids,k):
    groups=defaultdict(list)
    for i in indices:
        m=meta[i]
        if str(m['session_id']) not in session_ids[2:]:continue
        groups[(str(m['session_id']),str(m.get('repetition','')))].append(i)
    out=[]
    for (session,rep),idx in sorted(groups.items()):
        idx.sort(key=lambda i:(float(meta[i]['start_ms']),float(meta[i]['end_ms']),i))
        for end in range(k-1,len(idx)):
            window=idx[end-k+1:end+1]
            start_ms=float(meta[window[0]]['start_ms']);end_ms=float(meta[window[-1]]['end_ms'])
            if end_ms < start_ms:raise ValueError('Nonchronological gesture window')
            out.append((unit(embeddings[window].mean(axis=0)),
                        # Do NOT normalize the mean when scoring: use average
                        # of individual normalized cosine scores instead.
                        embeddings[window].mean(axis=0).astype(np.float64),
                        start_ms,end_ms,session,rep))
    return out


def biometric_scores_for_scenario(emb,meta,users,split_doc,scenario,k,
                                   impostor_per_source=100,seed=20261112):
    """Returns rows {target,source,label,score,start,end,session,rep,latency}."""
    if len(users)<2:raise ValueError(f'Need >=2 distinct users for ROC: {scenario}')
    refs={}; candidates={}
    for u,indices in sorted(users.items()):
        sessions=get_session_ids(split_doc,scenario,u)
        present={str(meta[i]['session_id']) for i in indices}
        if set(sessions)!=present:
            raise ValueError(f'Not all first four sessions exist for {scenario}/{u}: {present}')
        refs[u]=user_reference(emb,meta,indices,sessions)
        candidates[u]=windows_for_user(emb,meta,indices,sessions,k)
    active=[user for user in sorted(users) if candidates[user]]
    if len(active)<2:
        return []  # Insufficient k-window coverage for meaningful ROC.
    rng=np.random.default_rng(seed+k)
    rows=[]
    for target in active:
        genuine=candidates[target]
        for source in active:
            windows=candidates[source]
            if source!=target and impostor_per_source>0 and len(windows)>impostor_per_source:
                choices=np.sort(rng.choice(len(windows),size=impostor_per_source,replace=False))
                chosen=(windows[int(j)] for j in choices)
            else:chosen=iter(windows)
            for _,avg_embed,start,end,session,rep in chosen:
                score=float(np.dot(avg_embed,refs[target]))
                rows.append((target,source,1 if target==source else 0,score,start,end,
                             session,rep,(end-start)/1000.0))
    return rows


def roc_metrics(labels,scores):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(set(labels))!=2:raise ValueError('Cannot calculate ROC with only one class')
    fpr,tpr,thr=roc_curve(labels,scores,drop_intermediate=False)
    fnr=1-tpr
    gap=fpr-fnr
    crossing=np.flatnonzero(gap>=0)
    if not len(crossing):eer=float((fpr[-1]+fnr[-1])/2)
    elif crossing[0]==0:eer=float((fpr[0]+fnr[0])/2)
    else:
        j=int(crossing[0]);i=j-1
        alpha=float(-gap[i]/(gap[j]-gap[i])) if gap[j]!=gap[i] else 0.0
        eer=float((fpr[i]+alpha*(fpr[j]-fpr[i])+fnr[i]+alpha*(fnr[j]-fnr[i]))/2)
    auc=float(roc_auc_score(labels,scores))
    return {'eer':eer,'auc':auc,'fpr':fpr,'tpr':tpr,'thresholds':thr}


def operational_metrics(labels,scores,threshold):
    labels=np.asarray(labels,dtype=np.int8);scores=np.asarray(scores,dtype=float)
    genuine=labels==1;impostor=labels==0
    # Equality is an acceptance (>= threshold). Calibration uses a percentile
    # strictly above typical impostor scores where possible.
    accepted=scores>=threshold
    far=float(accepted[impostor].mean())
    frr=float((~accepted[genuine]).mean())
    return {'far':far,'frr':frr,'balanced_accuracy':float(1-(far+frr)/2),
            'genuine_accuracy':float(1-frr),'impostor_accuracy':float(1-far)}


def checkpoint_macro_auc(emb,meta,splits,role,scenarios,impostor_per_source=30):
    groups=build_role_scenarios(emb,meta,splits,role,set(scenarios))
    scenario_aucs=[]
    for sc in scenarios:
        users=groups[sc]
        rows=biometric_scores_for_scenario(emb,meta,users,splits,sc,1,impostor_per_source)
        per=[]
        for u in users:
            cur=[x for x in rows if x[0]==u]
            per.append(roc_metrics([x[2] for x in cur],[x[3] for x in cur])['auc'])
        scenario_aucs.append(float(np.mean(per)))
    return float(np.mean(scenario_aucs)),dict(zip(scenarios,scenario_aucs))
