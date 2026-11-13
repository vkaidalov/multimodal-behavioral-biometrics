#!/usr/bin/env python3
"""HMOG raw TouchEvent audit, with FETA-compatible physical contacts and keypress reconciliation.

Requires Python >=3.11, pandas, numpy. Run from repository root:
    python analyze_hmog_raw_feta_comparable.py --workers 4
Input defaults to input_data/hmog_unarchived; output to output_data/hmog_raw_feta_audit.

Design constraints:
* No gestures are synthesized from OneFingerTouchEvent / ScrollEvent / StrokeEvent.
* Map tasks 5/6/11/12/17/18/23/24 are ignored, even within a mixed session.
* Each physical DOWN ... UP (single pointer) is counted once; classification uses
  the same geometric 30 px displacement / 50 px path as FETA v1.1.
* No assumption that a keyboard press is present in the raw TouchEvent logger.
* EventTime and PressTime relative clocks are audited separately from Systime;
  no unverified clock shift is silently introduced.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'hmog-raw-feta-comparable-1.0.0-20261108'
TASKS = {
    1: 'reading_sitting', 7: 'reading_sitting', 13: 'reading_sitting', 19: 'reading_sitting',
    2: 'reading_walking', 8: 'reading_walking', 14: 'reading_walking', 20: 'reading_walking',
    3: 'writing_sitting', 9: 'writing_sitting', 15: 'writing_sitting', 21: 'writing_sitting',
    4: 'writing_walking', 10: 'writing_walking', 16: 'writing_walking', 22: 'writing_walking',
}
TASK_ORDER = ('reading_sitting','reading_walking','writing_sitting','writing_walking')
SCHEMAS = {
    'activity': ['ID','SubjectID','Session_number','Start_time','End_time','Relative_Start_time','Relative_End_time','Gesture_scenario','TaskID','ContentID'],
    'touch': ['Systime','EventTime','ActivityID','Pointer_count','PointerID','ActionID','X','Y','Pressure','Contact_size','Phone_orientation'],
    'key': ['Systime','PressTime','PressType','ActivityID','KeyID','Phone_orientation'],
    'accel': ['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation'],
    'gyro': ['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation'],
}
FILE_NAMES = {
    'activity':('Activity.csv',), 'touch':('TouchEvent.csv',),
    'key':('KeyPressEvent.csv',), 'accel':('Accelerometer.csv',),
    'gyro':('Gyroscope.csv','Gyroscrope.csv'),
}
KEY_MATCH_FIELDS = ('key_down','key_up')
SOURCES = ('accel','gyro','touch','key')
NO = float('nan')


def canonical(x):
    return re.sub('[^a-z0-9]', '', str(x).lower().lstrip('\ufeff'))


def read_table(path: Path | None, source: str, use: tuple[str,...] | None = None) -> pd.DataFrame:
    """Accept HMOG headers and its documented headerless positional CSV format."""
    if path is None or not path.exists():
        return pd.DataFrame()
    schema = SCHEMAS[source]
    with path.open(encoding='utf-8-sig', errors='replace') as h:
        head = h.readline().strip()
    if not head:
        return pd.DataFrame()
    separator = max((',', '\t', ';'), key=head.count)
    first = [canonical(v.strip(' "')) for v in head.split(separator)]
    expected = set(map(canonical, schema))
    has_header = len(expected.intersection(first)) >= 2
    kwargs = {'sep':separator, 'dtype':str, 'keep_default_na':True,
              'encoding':'utf-8-sig', 'on_bad_lines':'warn', 'low_memory':False}
    if has_header:
        if use:
            wanted = set(map(canonical,use))
            kwargs['usecols'] = lambda x: canonical(x) in wanted
        frame = pd.read_csv(path, **kwargs)
        frame.columns = [canonical(v) for v in frame.columns]
    else:
        kwargs.update(header=None, names=schema)
        if use:
            kwargs['usecols'] = list(use)
        frame = pd.read_csv(path, **kwargs)
        frame.columns = [canonical(v) for v in frame.columns]
    return frame


def numeric(df, key):
    if key not in df:
        return np.full(len(df), np.nan, dtype=np.float64)
    return pd.to_numeric(df[key], errors='coerce').to_numpy(dtype=np.float64, na_value=np.nan)


def strings(df,key):
    if key not in df:
        return np.full(len(df), '', dtype=object)
    return df[key].fillna('').astype(str).str.strip().str.replace(r'\.0$', '',regex=True).to_numpy(dtype=object)


def epoch_ms(values):
    """Only for Systime / Activity absolute timestamps; numeric epoch unit detector."""
    a = np.asarray(values, dtype=np.float64)
    finite = np.abs(a[np.isfinite(a)])
    if not len(finite):
        return a
    median = float(np.median(finite))
    if median > 1e17: return a / 1e6
    if median > 1e14: return a / 1e3
    if median > 1e11: return a
    if median > 1e8: return a * 1e3
    return a  # relative-looking Systime is flagged at session level


def clock_fit(sys_ms, relative):
    """Infer relative time scale and residual; do not presume SensorEvent and MotionEvent clocks coincide."""
    m = np.isfinite(sys_ms) & np.isfinite(relative)
    a, r = sys_ms[m], relative[m]
    if len(a) < 4:
        return {'valid':False,'scale':None,'offset_ms':None,'residual_p90_ms':None,'sample_count':int(len(a))}
    if len(a)>5000:
        inds=np.linspace(0,len(a)-1,5000,dtype=int)
        a,r=a[inds],r[inds]
    candidates=[]
    for scale in (1e-6, 1e-3, 1., 1e3, 1e6):
        d = a-r*scale
        off=float(np.median(d))
        residual=float(np.percentile(np.abs(d-off),90))
        # Require monotonic 3-second span to reduce accidental near-ties.
        candidates.append((residual, scale, off, float(np.percentile(r,90)-np.percentile(r,10))*scale))
    # Avoid unrealistic tiny scales that make relative data almost constant.
    plausible=[v for v in candidates if v[3]>=1000]
    error,scale,off,span = min(plausible or candidates, key=lambda q:(q[0],abs(math.log10(q[1]))))
    return {'valid':bool(error <= 250 and span>=1000), 'scale':float(scale),
            'offset_ms':float(off),'residual_p90_ms':round(error,3),'sample_count':int(len(a))}


def event_clock(df, field):
    sysms=epoch_ms(numeric(df,'systime'))
    rel=numeric(df,field)
    fit=clock_fit(sysms,rel)
    # Prefer recorded event-time when calibration is sensible; use Systime otherwise.
    if fit['valid']:
        event=rel*fit['scale']+fit['offset_ms']
        event[~np.isfinite(event)]=sysms[~np.isfinite(event)]
        return event,sysms,rel,fit,'relative_calibrated'
    return sysms,sysms,rel,fit,'systime_fallback'


def orientation_counts(values):
    a=np.asarray(values,dtype=np.float64)
    p=int(np.count_nonzero(a==0))
    l=int(np.count_nonzero(np.isin(a,[1,3])))
    u=len(a)-p-l
    return {'portrait':p,'landscape':l,'unknown':u,'rows':len(a),
            'portrait_pct_known':round(100*p/(p+l),4) if p+l else None,
            'portrait_pct_all':round(100*p/len(a),4) if len(a) else None}


def signature(files, config):
    h=hashlib.sha256()
    h.update(VERSION.encode()); h.update(json.dumps(config,sort_keys=True,default=str).encode())
    for source in sorted(FILE_NAMES):
        path=files.get(source)
        if path and path.exists():
            st=path.stat()
            h.update(f'{source}\0{path.resolve()}\0{st.st_size}\0{st.st_mtime_ns}'.encode())
        else: h.update(f'{source}\0ABSENT'.encode())
    return h.hexdigest()


def detect_files(folder):
    found={}
    for kind,names in FILE_NAMES.items():
        for name in names:
            p=folder/name
            if p.exists():
                found[kind]=p; break
    return found


def discover(root):
    found=defaultdict(list)
    for base,dirs,files in os.walk(root):
        if 'Activity.csv' not in files and 'TouchEvent.csv' not in files:
            continue
        p=Path(base)
        rel=p.relative_to(root).parts
        uid=next((v for v in rel if re.fullmatch(r'\d{6}',v)),None)
        if uid is None:
            # Support nested structure with folder 123456_session_7 but no parent.
            uid=next((m.group(1) for part in rel if (m:=re.match(r'(\d{6})[_-]session',part))),None)
        if uid is None: continue
        found[uid].append(str(p))
    return {uid:sorted(set(paths)) for uid,paths in sorted(found.items())}


def absolute_series_ms(df, field):
    vals=numeric(df,field)
    times=epoch_ms(vals)
    if field in df and np.any(~np.isfinite(times)):
        missing=np.flatnonzero(~np.isfinite(times))
        # HMOG documents numeric absolute timestamps. ISO strings are a cautious fallback.
        for i in missing:
            raw=str(df.iloc[i][field]).strip()
            if raw and raw.lower()!='nan':
                try:
                    stamp=pd.to_datetime(raw,utc=True,errors='raise')
                    times[i]=stamp.timestamp()*1000.
                except (ValueError,TypeError,OverflowError):pass
    return times

def activity_info(frame):
    if frame.empty: return {},[],None
    ids=strings(frame,'id')
    tasks=numeric(frame,'taskid')
    activity_map={}
    ignored=Counter()
    starts=absolute_series_ms(frame,'starttime')
    for i,(aid,tid) in enumerate(zip(ids,tasks)):
        if not aid or not np.isfinite(tid): continue
        task=TASKS.get(int(tid))
        if task:
            activity_map[aid]=task
        else:
            ignored[str(int(tid))]+=1
    labels=set(activity_map.values())
    dates=starts[np.isfinite(starts)]
    start=float(np.min(dates)) if dates.size else None
    return activity_map,list(sorted(labels)),start


def resolve_session_number(frame, folder):
    vals=set(str(v) for v in strings(frame,'sessionnumber') if str(v))
    if len(vals)==1: return next(iter(vals))
    match=re.search(r'(?:session[_-]?)(\d+)',folder.name,re.I)
    if match: return match.group(1)
    return folder.name


def reconstruct_contacts(df, allowed, cfg):
    counts=Counter(); contacts=[]
    events,system_time,rel,fit,basis=event_clock(df,'eventtime')
    acid=strings(df,'activityid')
    code=numeric(df,'actionid'); x=numeric(df,'x'); y=numeric(df,'y')
    pc=numeric(df,'pointercount'); pid=numeric(df,'pointerid'); orient=numeric(df,'phoneorientation')
    valid=np.isfinite(events)&np.isfinite(code)&np.isfinite(x)&np.isfinite(y)
    indices=np.flatnonzero(valid)
    indices=indices[np.argsort(events[indices],kind='stable')]
    active=None
    raw_event_ids=np.full(len(df),-1,dtype=np.int32)
    event_meta={0:[],1:[]}
    for j in indices:
        a=acid[j]
        if a not in allowed:
            counts['ignored_other_activity_touch_rows']+=1;continue
        act=int(code[j])&255
        multi=act in (5,6) or (np.isfinite(pc[j]) and pc[j]>1) or (np.isfinite(pid[j]) and pid[j]>0)
        if act==0:
            if active is not None: counts['interrupted_down']+=1
            active={'activity_id':a,'start_ms':float(events[j]),'start_x':float(x[j]),'start_y':float(y[j]),
                    'last_x':float(x[j]),'last_y':float(y[j]),'path_px':0.,'points':1,
                    'orientation_codes':set(), 'has_unknown_orientation':False,'multi':bool(multi),
                    'down_row':int(j)}
            if np.isfinite(orient[j]):active['orientation_codes'].add(int(orient[j]))
            else:active['has_unknown_orientation']=True
            event_meta[0].append((a,float(events[j]),float(system_time[j]),float(rel[j]),int(j)))
            continue
        if act==5:
            counts['pointer_down_events']+=1
            if active: active['multi']=True
            continue
        if act==6:
            counts['pointer_up_events']+=1
            if active: active['multi']=True
            continue
        if act==3:
            counts['cancel_events']+=1
            if active: counts['cancelled_contacts']+=1;active=None
            continue
        if act not in (1,2):
            counts['other_actions']+=1
            continue
        if act==1:
            event_meta[1].append((a,float(events[j]),float(system_time[j]),float(rel[j]),int(j)))
        if active is None:
            counts['orphan_'+('up' if act==1 else 'move')]+=1
            continue
        if a != active['activity_id']:
            counts['activity_changed_while_touch_active']+=1
            active=None
            continue
        active['path_px']+=math.hypot(float(x[j])-active['last_x'],float(y[j])-active['last_y'])
        active['last_x'],active['last_y']=float(x[j]),float(y[j])
        active['points']+=1
        active['multi'] |= bool(multi)
        if np.isfinite(orient[j]):active['orientation_codes'].add(int(orient[j]))
        else:active['has_unknown_orientation']=True
        if act==2: continue
        duration=float(events[j]-active['start_ms'])
        disp=math.hypot(float(x[j])-active['start_x'],float(y[j])-active['start_y'])
        kind='swipe' if (disp>=cfg['swipe_distance_px'] or active['path_px']>=cfg['swipe_path_px']) else 'tap'
        eligible_timing=np.isfinite(duration) and duration>0
        is_portrait=(active['orientation_codes']=={0} and not active['has_unknown_orientation'])
        p={'activity_id':a,'task':allowed[a],'type':kind,
           'start_ms':round(active['start_ms'],3),'end_ms':round(float(events[j]),3),
           'duration_ms':round(duration,3),'start_x':active['start_x'],'start_y':active['start_y'],
           'end_x':float(x[j]),'end_y':float(y[j]),'displacement_px':round(disp,3),
           'path_px':round(active['path_px'],3),'touch_points':active['points'],
           'portrait_only':int(is_portrait),'multitouch':int(active['multi']),
           'valid_duration':int(bool(eligible_timing)),
           'duration_exceeds_max_contact':int(bool(np.isfinite(duration) and duration>cfg['max_contact_ms'])),
           'orientation_codes':','.join(map(str,sorted(active['orientation_codes']))),
           'has_unknown_orientation':int(active['has_unknown_orientation']),
           'down_row':active['down_row'],'up_row':int(j)}
        contacts.append(p)
        contactidx=len(contacts)-1
        raw_event_ids[active['down_row']]=contactidx
        raw_event_ids[j]=contactidx
        counts['complete_contacts']+=1
        active=None
    if active is not None: counts['unclosed_down']+=1
    counts['invalid_touch_rows']=int(len(df)-len(indices))
    counts['touch_rows_in_scope']=int(np.count_nonzero(np.isin(acid,list(allowed))))
    counts['touch_rows_all']=len(df)
    counts['touch_actions_0']=int(np.count_nonzero(code==0))
    counts['touch_actions_1']=int(np.count_nonzero(code==1))
    counts['touch_actions_2']=int(np.count_nonzero(code==2))
    return contacts,counts,event_meta,fit,basis,raw_event_ids


def sorted_sensor(df,allowed):
    if df.empty: return np.empty(0),np.empty(0),np.empty(0,dtype=object),orientation_counts([]),0
    activity=strings(df,'activityid')
    mask=np.isin(activity,list(allowed))
    time=epoch_ms(numeric(df,'systime'))[mask]
    orientation=numeric(df,'phoneorientation')[mask]
    aids=activity[mask]
    good=np.isfinite(time)
    time=time[good];orientation=orientation[good];aids=aids[good]
    order=np.argsort(time,kind='stable')
    return time[order],orientation[order],aids[order],orientation_counts(orientation),int(len(df)-mask.sum())


def sensor_interval(times, orientations, start, end, cfg):
    lo=int(np.searchsorted(times,start,'left'));hi=int(np.searchsorted(times,end,'right'))
    t=times[lo:hi];o=orientations[lo:hi]
    n=len(t)
    if n:
        edge=max(float(t[0])-start,end-float(t[-1]))
        gap=float(np.max(np.diff(t))) if n>1 else None
    else: edge=None;gap=None
    strict=bool(n>=cfg['min_sensor_samples'] and edge is not None and edge<=cfg['max_edge_gap_ms'] and
                (gap is None or gap<=cfg['max_sensor_gap_ms']))
    return t,n,edge,gap,strict,bool(n and np.all(o==0))


def count_pairs(a,b,tolerance):
    i=j=matched=0
    while i<len(a) and j<len(b):
        delta=float(a[i])-float(b[j])
        if abs(delta)<=tolerance:matched+=1;i+=1;j+=1
        elif delta<0:i+=1
        else:j+=1
    return matched


def classify_sensor_samples(times,contacts):
    """Partition samples by union of all complete contact time intervals in selected tasks."""
    if len(times)==0: return {'inside':0,'between':0,'before':0,'after':0,'unclassified':0,'between_seconds':0.}
    intervals=sorted((c['start_ms'],c['end_ms']) for c in contacts if c['valid_duration'])
    if not intervals:
        return {'inside':0,'between':0,'before':0,'after':0,'unclassified':len(times),'between_seconds':0.}
    merged=[]
    for s,e in intervals:
        if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
        else:merged.append([s,e])
    inside=0;gap_seconds=0.
    for s,e in merged:
        inside+=int(np.searchsorted(times,e,'right')-np.searchsorted(times,s,'left'))
    before=int(np.searchsorted(times,merged[0][0],'left'))
    after=int(len(times)-np.searchsorted(times,merged[-1][1],'right'))
    for (_,end),(next_start,_) in zip(merged,merged[1:]):
        gap_seconds+=(next_start-end)/1000.
    between=len(times)-inside-before-after
    if between<0:raise ValueError('Negative between samples: union intervals inconsistent')
    return {'inside':inside,'between':between,'before':before,'after':after,
            'unclassified':0,'between_seconds':round(gap_seconds,3)}


def nearest_event_stats(key_rows, touch_rows, tol):
    """One-to-one nearest-neighbor matching per ActivityID, with diagnostic nearest-distance distribution."""
    # row shape: (activity, event_clock_ms, recorded_sys_ms, relative_timestamp, unique_index)
    by_group=defaultdict(list)
    for a,t,s,r,j in touch_rows:
        if np.isfinite(t):by_group[a].append((float(t),int(j)))
    for v in by_group.values():v.sort()
    match={};used_touch=set();distances=[];near_ranges=Counter()
    # Closest first greedily; chronology deterministic with increasing key time.
    events=sorted((k for k in key_rows if np.isfinite(k[1])),key=lambda v:(v[0],v[1],v[4]))
    for a,t,s,r,kidx in events:
        seq=by_group.get(a,[])
        if not seq:continue
        # bisect over tuple avoids building a second list each time.
        from bisect import bisect_left
        pos=bisect_left(seq,(t,-1))
        close=[]
        for off in range(-4,5):
            p=pos+off
            if 0<=p<len(seq):
                d=abs(seq[p][0]-t)
                close.append((d,seq[p][1]))
        if not close:continue
        dmin=min(q[0] for q in close)
        distances.append(dmin)
        for lim in (5,20,50,80,150,250,500):
            if dmin<=lim:near_ranges[str(lim)]+=1
        # Reserve one touch action for only one key action.
        for d,j in sorted(close):
            if d<=tol and j not in used_touch:
                used_touch.add(j)
                match[int(kidx)]={'touch_row':int(j),'delta_ms':round(float(d),4)}
                break
    return match,{'events_with_near_touch':len(distances),
                  'nearest_delta_median_ms':round(float(np.median(distances)),3) if distances else None,
                  'nearest_delta_p90_ms':round(float(np.percentile(distances,90)),3) if distances else None,
                  **{f'nearest_at_most_{n}ms':int(near_ranges[str(n)]) for n in (5,20,50,80,150,250,500)}}


def attempt_matching(key_rows, touch_rows, tol, clock='event', key_scale=1., touch_scale=1.):
    if clock=='event': return nearest_event_stats(key_rows,touch_rows,tol)
    if clock=='relative_direct':
        return nearest_event_stats(
            [(a,r*key_scale,s,r,j) for a,t,s,r,j in key_rows],
            [(a,r*touch_scale,s,r,j) for a,t,s,r,j in touch_rows],tol)
    def trans(rows):
        return [(a,s,s,r,j) for a,t,s,r,j in rows]
    return nearest_event_stats(trans(key_rows),trans(touch_rows),tol)


def key_analysis(df,allowed,touch_events,contact_ids,cfg,touch_clock_fit):
    results=Counter()
    if df.empty:return dict(results),[],{'valid':False},'missing_key_file'
    keystime,keysystime,rel,fit,basis=event_clock(df,'presstime')
    act=strings(df,'activityid'); presstype=numeric(df,'presstype'); codes=strings(df,'keyid')
    valid_activity=np.isin(act,list(allowed)); valid=np.isfinite(keystime)&valid_activity
    results['key_rows_all']=len(df)
    results['key_rows_in_scope']=int(valid_activity.sum())
    types=Counter(str(x) for x in df.loc[valid_activity,'presstype'].fillna('NULL')) if 'presstype' in df else Counter()
    # Including unknown values is essential: a previous audit reported zero KEY DOWN despite 1.6M key rows.
    rows_by_kind={0:[],1:[]}
    for i in np.flatnonzero(valid):
        ty=int(presstype[i]) if np.isfinite(presstype[i]) else -1
        if ty in rows_by_kind:
            rows_by_kind[ty].append((act[i],float(keystime[i]),float(keysystime[i]),float(rel[i]),int(i)))
        else:results['keypress_unrecognized_types']+=1
    results['keypress_type_values']=dict(types)
    result_events={}; key_details=[]
    for typ,label in ((0,'down'),(1,'up')):
        events=rows_by_kind[typ]; targets=touch_events[typ]
        # Report both independent clock bases: event and system.
        for clock in ('event','system','relative_direct'):
            if clock=='relative_direct' and not (fit.get('valid') and touch_clock_fit.get('valid') and
                                                  abs(float(fit['offset_ms'])-float(touch_clock_fit['offset_ms']))<=1000):
                results[f'key_{label}_relative_direct_skipped']='relative_clocks_not_proven_compatible'
                continue
            matches,diag=attempt_matching(events,targets,cfg['key_match_tolerance_ms'],clock,
                                          float(fit.get('scale') or 1.),float(touch_clock_fit.get('scale') or 1.))
            results[f'key_{label}_matched_{clock}']=len(matches)
            for k,v in diag.items():results[f'key_{label}_{clock}_{k}']=v
            if clock=='event':result_events[typ]=matches
        results[f'key_{label}_rows']=len(events)
    # Attempt to pair keyboard DOWN+UP of the same KeyID in same activity.
    pending=defaultdict(list)
    pairs=[]
    for i in sorted(np.flatnonzero(valid),key=lambda j:(act[j],keystime[j],j)):
        if not np.isfinite(presstype[i]):continue
        typ=int(presstype[i]);key=(act[i],codes[i])
        if typ==0:pending[key].append(int(i))
        elif typ==1:
            if not pending[key]:results['key_up_without_down']+=1;continue
            beg=pending[key].pop(0)
            delta=float(keystime[i]-keystime[beg])
            if 0<=delta<=cfg['max_key_hold_ms']:
                pairs.append((beg,int(i),delta))
            else:results['key_bad_hold_duration']+=1
    results['key_down_unclosed']=sum(len(p) for p in pending.values())
    results['key_completed_pairs']=len(pairs)
    raw_to_contact={j:int(v) for j,v in enumerate(contact_ids) if v>=0}
    pair_shared=0; pair_both=0
    for down,up,delta in pairs:
        d=result_events[0].get(down);u=result_events[1].get(up)
        if d and u:
            pair_both+=1
            cd=raw_to_contact.get(d['touch_row']);cu=raw_to_contact.get(u['touch_row'])
            if cd is not None and cd==cu:pair_shared+=1
    results['key_pairs_both_endpoint_events_matched']=pair_both
    results['key_pairs_same_physical_touch_contact']=pair_shared
    # Save a limited sample of unmatched key actions, not plaintext KeyID sequences.
    for typ,label in ((0,'down'),(1,'up')):
        missing=[x for x in rows_by_kind[typ] if x[4] not in result_events[typ]]
        for x in missing[:cfg['key_diagnostic_examples']]:
            key_details.append({'key_action':label,'activity_id':x[0],
                'event_time_ms':round(x[1],3),'system_time_ms':round(x[2],3),
                'note':'no_timestamp_match_with_same_action_in_touch'})
    return dict(results),key_details,fit,basis


def csv_write(path,rows,fields=None,delimiter=','):
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if fields is None:fields=sorted(set().union(*(x.keys() for x in rows))) if rows else []
    open_fn=gzip.open if path.name.endswith('.gz') else open
    kwargs={'mode':'wt','newline':'','encoding':'utf-8'}
    with open_fn(path,**kwargs) as h:
        writer=csv.DictWriter(h,fieldnames=fields,delimiter=delimiter,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)


def analyze_session(uid,folder,out,config,no_cache=False):
    folder=Path(folder);out=Path(out); files=detect_files(folder)
    token=hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:13]
    cache=out/'cache'/'sessions'/f'{uid}__{token}.json'
    sig=signature(files,config)
    if cache.exists() and not no_cache:
        try:
            data=json.loads(cache.read_text(encoding='utf-8'))
            if data['signature']==sig:return data['result'],True
        except (ValueError,KeyError,OSError):pass
    base={'user_id':uid,'session_dir':str(folder),'session_number':folder.name,
          'status':'processed','task':'','session_start_ms':None,'session_start_iso':'',
          'warnings':[]}
    if 'activity' not in files:
        base.update(status='missing_activity',warnings=['Activity.csv missing'])
        json_write(cache,{'signature':sig,'result':base});return base,False
    activity=read_table(files['activity'],'activity')
    activities,labels,start=activity_info(activity)
    base['session_number']=resolve_session_number(activity,folder)
    base['activity_rows']=len(activity)
    base['session_start_ms']=start
    if start is None:base['warnings'].append('Activity.Start_time_missing_no_proven_chronology')
    base['session_start_iso']=(datetime.fromtimestamp(start/1000,timezone.utc).isoformat() if start and start>1e11 else '')
    base['task']=';'.join(labels)
    base['task_ids_in_scope']=';'.join(sorted(activities))
    base['map_or_unknown_activity_rows']=len(activity)-len(activities)
    if not activities:
        base.update(status='map_or_other_out_of_scope')
        json_write(cache,{'signature':sig,'result':base});return base,False
    if len(labels)>1:base['warnings'].append('mixed_task_categories_in_session')
    if not {'touch','accel','gyro'}.issubset(files):
        base['warnings'].append('missing_core_files:'+','.join(k for k in ('touch','accel','gyro') if k not in files))
    df_touch=read_table(files.get('touch'),'touch')
    df_key=read_table(files.get('key'),'key')
    sensor={};orients={}
    for short in ('accel','gyro'):
        frame=read_table(files.get(short),short,use=('Systime','ActivityID','Phone_orientation'))
        times,orientation,sensor_activities,stats,excluded=sorted_sensor(frame,activities)
        sensor[short]=(times,orientation)
        sensor[short+"_by_activity"]={aid:(times[sensor_activities==aid],orientation[sensor_activities==aid]) for aid in activities}
        orients[short]=stats
        base[short+'_excluded_other_activity_rows']=excluded
        del frame
    touch_activity=strings(df_touch,'activityid')
    key_activity=strings(df_key,'activityid')
    orients['touch']=orientation_counts(numeric(df_touch,'phoneorientation')[np.isin(touch_activity,list(activities))])
    orients['key']=orientation_counts(numeric(df_key,'phoneorientation')[np.isin(key_activity,list(activities))])
    for source in SOURCES:
        for k,v in orients[source].items():base[f'{source}_{k}']=v
    p=sum(orients[s]['portrait'] for s in SOURCES)
    l=sum(orients[s]['landscape'] for s in SOURCES)
    unknown=sum(orients[s]['unknown'] for s in SOURCES)
    base['all_sources_rows']=p+l+unknown
    base['all_sources_portrait_pct_known']=round(100*p/(p+l),4) if p+l else None
    base['all_sources_portrait_pct_all']=round(100*p/(p+l+unknown),4) if p+l+unknown else None
    base['all_sources_unknown_pct']=round(100*unknown/(p+l+unknown),4) if p+l+unknown else None
    contacts,diag,touch_events,touch_cal,touch_basis,raw_contact_ids=reconstruct_contacts(df_touch,activities,config)
    base.update({f'raw_{k}':v for k,v in diag.items()})
    base['touch_clock_basis']=touch_basis;base['touch_clock_fit']=touch_cal
    key_stats,key_examples,key_cal,key_basis=key_analysis(df_key,activities,touch_events,raw_contact_ids,config,touch_cal)
    base.update(key_stats);base['key_clock_basis']=key_basis;base['key_clock_fit']=key_cal
    # Summaries must distinguish the physical contact count from FETA-like eligible single finger contacts.
    sums=Counter()
    for idx,g in enumerate(contacts):
        g.update(user_id=uid,session_number=base['session_number'],gesture_id=f'{uid}:{base["session_number"]}:{idx}')
        valid=bool(g['valid_duration']) and not g['multitouch']
        portrait=valid and bool(g['portrait_only'])
        g['single_valid']=int(valid)
        g['portrait_single_valid']=int(portrait)
        sums['all_completed_contacts']+=1
        if valid:sums['single_valid_contacts']+=1
        if portrait:sums['portrait_contacts']+=1;sums['portrait_'+g['type']+'s']+=1
        # Use all gesture orientations as a second report; primary is all-portrait only.
        if valid:sums['single_'+g['type']+'s']+=1
        if valid and not portrait:sums['nonportrait_or_unknown_contacts']+=1
        if g['multitouch']:sums['multitouch_contacts']+=1
        if not g['valid_duration']:sums['bad_duration_contacts']+=1
        if g.get('duration_exceeds_max_contact'):sums['long_duration_contacts']+=1
        if not g['portrait_only']:sums['not_all_portrait_contacts']+=1
        if g['valid_duration']:
            readings={}
            for short in ('accel','gyro'):
                t,o=sensor[short+'_by_activity'][g['activity_id']]
                parts,n,edge,gap,strict,allp=sensor_interval(t,o,g['start_ms'],g['end_ms'],config)
                g[short+'_samples']=n
                g[short+'_edge_max_ms']=round(edge,3) if edge is not None else ''
                g[short+'_max_gap_ms']=round(gap,3) if gap is not None else ''
                g[short+'_strict']=int(strict)
                g[short+'_all_portrait']=int(allp)
                readings[short]=parts
            pairs=count_pairs(readings['accel'],readings['gyro'],config['pair_tolerance_ms'])
            g['accel_gyro_pairs']=pairs
            g['both_any']=int(bool(g['accel_samples'] and g['gyro_samples']))
            g['both_strict_paired']=int(bool(g['accel_strict'] and g['gyro_strict'] and pairs>=config['min_sensor_pairs']))
            g['both_strict_paired_portrait_motion']=int(bool(g['both_strict_paired'] and g['accel_all_portrait'] and g['gyro_all_portrait']))
            if portrait:
                sums['portrait_both_any']+=int(g['both_any'])
                sums['portrait_both_strict_paired']+=int(g['both_strict_paired'])
                sums['portrait_both_strict_paired_portrait_motion']+=int(g['both_strict_paired_portrait_motion'])
                sums['portrait_'+g['type']+'s_both_strict_paired']+=int(g['both_strict_paired'])
                sums['portrait_'+g['type']+'s_both_strict_paired_portrait_motion']+=int(g['both_strict_paired_portrait_motion'])
        else:
            for v in ('accel_samples','gyro_samples','accel_strict','gyro_strict','accel_gyro_pairs','both_any','both_strict_paired','both_strict_paired_portrait_motion'):
                g[v]=0
    base.update(sums)
    for kind in ('tap','swipe'):
        gds=[g['duration_ms'] for g in contacts if g['type']==kind and g['portrait_single_valid']]
        base[f'portrait_{kind}_median_duration_ms']=round(float(np.median(gds)),3) if gds else None
        base[f'portrait_{kind}_p90_duration_ms']=round(float(np.percentile(gds,90)),3) if gds else None
    pgs=sorted([g for g in contacts if g['portrait_single_valid']],key=lambda g:g['start_ms'])
    gaps=[max(0.,b['start_ms']-a['end_ms']) for a,b in zip(pgs,pgs[1:]) if a['activity_id']==b['activity_id']]
    base['median_intergesture_gap_ms']=round(float(np.median(gaps)),3) if gaps else None
    for label in ('portrait_taps','portrait_swipes','portrait_taps_both_strict_paired','portrait_swipes_both_strict_paired',
                  'portrait_taps_both_strict_paired_portrait_motion','portrait_swipes_both_strict_paired_portrait_motion',
                  'portrait_both_any','portrait_both_strict_paired','portrait_both_strict_paired_portrait_motion',
                  'all_completed_contacts','single_valid_contacts','portrait_contacts','multitouch_contacts'):
        base.setdefault(label,0)
    for short in ('accel','gyro'):
        stats=[classify_sensor_samples(v[0],[g for g in contacts if g['activity_id']==aid])
               for aid,v in sensor[short+'_by_activity'].items()]
        pos={k:sum(d[k] for d in stats) for k in ('inside','between','before','after','unclassified','between_seconds')}
        for k,v in pos.items():base[f'{short}_{k}']=v
    # Clock-compatibility is checked per ActivityID rather than across unrelated tasks.
    incompatible=[]
    for aid in activities:
        gg=[g for g in contacts if g['activity_id']==aid and g['valid_duration']]
        if not gg:continue
        gmin=min(g['start_ms'] for g in gg);gmax=max(g['end_ms'] for g in gg)
        for short in ('accel','gyro'):
            tt=sensor[short+'_by_activity'][aid][0]
            if not len(tt) or tt[-1]<gmin or tt[0]>gmax:incompatible.append(f'{aid}:{short}')
    base['activity_sensor_clock_incompatibilities']=';'.join(incompatible)
    if incompatible:base['warnings'].append('activity_sensor_clock_disjoint')
    base['recovered_gestures_file']=str(out/'cache'/'gestures'/f'{uid}__{token}.csv.gz')
    csv_write(Path(base['recovered_gestures_file']),contacts)
    if key_examples:
        csv_write(out/'cache'/'key_unmatched_examples'/f'{uid}__{token}.csv',key_examples)
    if not touch_cal['valid']:base['warnings'].append('touch_relative_clock_unreliable_uses_systime')
    if not key_cal['valid'] and len(df_key):base['warnings'].append('key_relative_clock_unreliable_uses_systime')
    if base['raw_invalid_touch_rows']>0:base['warnings'].append('invalid_touch_rows_present')
    if base.get('raw_interrupted_down',0)>0:base['warnings'].append('interrupted_touch_sequences')
    if base['key_down_rows']+base['key_up_rows'] < base['key_rows_in_scope']:
        base['warnings'].append('unrecognized_key_press_types_or_timestamps')
    json_write(cache,{'signature':sig,'result':base})
    return base,False


def worker(job):
    uid,folders,out,config,no_cache=job
    rows=[];hits=0
    for path in folders:
        row,hit=analyze_session(uid,path,out,config,no_cache)
        rows.append(row);hits+=int(hit)
    return uid,rows,hits


def matrix(out,filename,users,rows,metric):
    columns=[f'{name}_{n}' for name in TASK_ORDER for n in range(1,5)]
    by_user=defaultdict(lambda:defaultdict(list))
    overflow=[]
    for r in rows:
        if r.get('status')!='processed':continue
        for label in r['task'].split(';'):
            if label in TASK_ORDER:by_user[r['user_id']][label].append(r)
    cell={};metadata=[]
    for uid in users:
        rec={'user_id':uid}
        for task in TASK_ORDER:
            ordered=sorted(by_user[uid][task],key=lambda r:(r['session_start_ms'] is None,
                        r['session_start_ms'] if r['session_start_ms'] is not None else float('inf'),
                        str(r['session_number'])))
            if len(ordered)>4:
                overflow.append({'user_id':uid,'task':task,'session_count':len(ordered),'excess_sessions':len(ordered)-4})
            for idx,r in enumerate(ordered[:4],1):
                label=f'{task}_{idx}'
                value=r.get(metric)
                rec[label]='' if value is None else value
                metadata.append({'user_id':uid,'column':label,'session_number':r['session_number'],
                    'session_start_iso':r.get('session_start_iso'),'session_start_ms':r.get('session_start_ms'),
                    'session_dir':r['session_dir'],'task':task,
                    'portrait_pct_all_sources':r.get('all_sources_portrait_pct_known'),
                    'portrait_pct_touch':r.get('touch_portrait_pct_known'),
                    'portrait_taps':r.get('portrait_taps',0),'portrait_swipes':r.get('portrait_swipes',0)})
        cell[uid]=rec
    csv_write(out/filename,[cell[u] for u in users],['user_id']+columns,delimiter='\t')
    return metadata,overflow


def summarize(rows,users,out,config,failed):
    processed=[r for r in rows if r.get('status')=='processed']
    totals=Counter()
    by_category=defaultdict(Counter)
    for r in processed:
        for key in ('portrait_taps','portrait_swipes','portrait_taps_both_strict_paired',
            'portrait_swipes_both_strict_paired','portrait_taps_both_strict_paired_portrait_motion',
            'portrait_swipes_both_strict_paired_portrait_motion','all_completed_contacts',
            'portrait_contacts','raw_touch_rows_in_scope','key_rows_in_scope','key_down_rows','key_up_rows',
            'key_down_matched_event','key_up_matched_event','key_down_matched_system','key_up_matched_system',
            'key_down_matched_relative_direct','key_up_matched_relative_direct',
            'key_completed_pairs','key_pairs_both_endpoint_events_matched','key_pairs_same_physical_touch_contact',
            'accel_inside','accel_between','gyro_inside','gyro_between'):
            val=int(r.get(key,0) or 0)
            totals[key]+=val
            for label in str(r.get('task','')).split(';'):
                if label in TASK_ORDER:by_category[label][key]+=val
    csv_write(out/'sessions.csv',rows)
    percent=['all_sources_portrait_pct_known','all_sources_portrait_pct_all','touch_portrait_pct_known',
             'key_portrait_pct_known','accel_portrait_pct_known','gyro_portrait_pct_known']
    gestures=['portrait_taps','portrait_swipes','portrait_taps_both_strict_paired',
              'portrait_swipes_both_strict_paired','portrait_taps_both_strict_paired_portrait_motion',
              'portrait_swipes_both_strict_paired_portrait_motion']
    all_metadata=[];all_overflow=[]
    for metric in percent+gestures:
        filename=('portrait_percent' if metric=='all_sources_portrait_pct_known' else metric)+ '_matrix.tsv'
        metadata,overflow=matrix(out,filename,users,rows,metric)
        if not all_metadata:all_metadata=metadata;all_overflow=overflow
    csv_write(out/'session_slot_metadata.tsv',all_metadata,delimiter='\t')
    csv_write(out/'session_slot_overflow.csv',all_overflow,fields=['user_id','task','session_count','excess_sessions'])
    user_rows=[]
    byuser=defaultdict(list)
    for r in processed:byuser[r['user_id']].append(r)
    for uid in users:
        group=byuser[uid]
        record={'user_id':uid,'sessions_processed':len(group)}
        for task in TASK_ORDER:
            filtered=[x for x in group if task in str(x['task']).split(';')]
            record[task+'_sessions']=len(filtered)
            for key in gestures:record[task+'_'+key]=sum(int(r.get(key,0) or 0) for r in filtered)
        user_rows.append(record)
    csv_write(out/'users.csv',user_rows)
    eligibility={}
    for task in TASK_ORDER:
        primary=('portrait_swipes_both_strict_paired_portrait_motion' if task.startswith('reading')
                 else 'portrait_taps_both_strict_paired_portrait_motion')
        eligibility[task]={}
        for threshold in (20,50,100):
            eligibility[task][str(threshold)]={str(n):sum(
                sum(int(x.get(primary,0) or 0)>=threshold for x in byuser[uid]
                    if task in str(x.get('task','')).split(';'))>=n
                for uid in users) for n in (2,3,4)}
    report={'version':VERSION,'input':config.get('root'), 'users_discovered':len(users),
            'sessions_discovered':len(rows),'sessions_processed':len(processed),
            'sessions_ignored_map_or_other':sum(r.get('status')=='map_or_other_out_of_scope' for r in rows),
            'sessions_failed_or_missing':sum(r.get('status') not in ('processed','map_or_other_out_of_scope') for r in rows),
            'failed_users':failed,'processing_config':{k:v for k,v in config.items() if k!='root'},
            'totals':dict(totals),'by_activity':{k:dict(by_category[k]) for k in TASK_ORDER},
            'users_with_at_least_N_sessions_of_min_M_primary_gestures':eligibility,
            'orientation_definition':'raw row-weighted portrait % among recognized 0/1/3 orientations, all four sources combined; per-source matrices also provided',
            'gesture_definition':'physical raw TouchEvent DOWN -> UP, one contact, single pointer, 0 portrait for EVERY row; 30px displacement or 50px path -> swipe otherwise tap; duration >12000ms flagged but NOT automatically discarded, like FETA v1.1',
            'matching_definition':'KeyPressEvent PressType 0 DOWN, 1 UP; greedy 1-to-1 nearest TouchEvent of same action and ActivityID; compare calibrated event times, independently Systime and (when clock offsets compatible) unshifted relative event times; pair same contact only when both endpoints map to same reconstructed raw contact',
            'strict_coverage_definition':'FETA v1.1: >=3 per sensor, max endpoint gap 40ms, max internal sensor gap 120ms, >=1 accel/gyro pair <=25ms; use ONLY in-gesture sensor samples',
            'notes':['No map-navigation session is included.','Matrix task order uses earliest Activity.Start_time within each user and task, NOT global session numbering.',
                     'Portrait fraction is NOT the same as percent of time in portrait: high frequency streams dominate row-weighted totals.',
                     'If KeyPressEvent press types differ from 0/1 in raw data, per-session keypress_type_values diagnose the mismatch.',
                     'TouchEvent and KeyPressEvent event-clock fit statistics are diagnostics; source system times are also checked.']}
    json_write(out/'summary.json',report)
    return report


def cli():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--input',type=Path,default=Path('input_data/hmog_unarchived'))
    a.add_argument('--output',type=Path,default=Path('output_data/hmog_raw_feta_audit'))
    a.add_argument('--workers',type=int,default=max(1,min(4,(os.cpu_count() or 2)//2)))
    a.add_argument('--limit-users',type=int,default=0)
    a.add_argument('--no-cache',action='store_true')
    a.add_argument('--swipe-distance-px',type=float,default=30.)
    a.add_argument('--swipe-path-px',type=float,default=50.)
    a.add_argument('--max-contact-ms',type=float,default=12000.)
    a.add_argument('--min-sensor-samples',type=int,default=3)
    a.add_argument('--max-edge-gap-ms',type=float,default=40.)
    a.add_argument('--max-sensor-gap-ms',type=float,default=120.)
    a.add_argument('--pair-tolerance-ms',type=float,default=25.)
    a.add_argument('--min-sensor-pairs',type=int,default=1)
    a.add_argument('--key-match-tolerance-ms',type=float,default=80.)
    a.add_argument('--max-key-hold-ms',type=float,default=3000.)
    a.add_argument('--key-diagnostic-examples',type=int,default=20)
    return a.parse_args()


def main():
    args=cli();root=args.input.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger('HMOG-raw');log.setLevel(logging.INFO);log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),logging.FileHandler(out/'analysis.log',mode='w',encoding='utf-8')):
        h.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'));log.addHandler(h)
    if not root.is_dir():log.error('Input folder absent: %s',root);return 2
    users=discover(root)
    if args.limit_users>0:users=dict(list(users.items())[:args.limit_users])
    if not users:log.error('No six-digit HMOG user directories with Activity/Touch CSVs found');return 2
    config={k:v for k,v in vars(args).items() if k not in ('input','output','workers','limit_users','no_cache')}
    config['root']=str(root)
    log.info('Input=%s; users=%d; session dirs=%d; workers=%d',root,len(users),sum(len(x) for x in users.values()),args.workers)
    results=[];failed=[];hits=0
    jobs=[(uid,folders,str(out),config,args.no_cache) for uid,folders in users.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut={pool.submit(worker,job):job[0] for job in jobs}
        for n,done in enumerate(as_completed(fut),1):
            uid=fut[done]
            try:
                user,rows,cached=done.result();results.extend(rows);hits+=cached
                active=[r for r in rows if r.get('status')=='processed']
                log.info('[%d/%d] user=%s sessions=%d map_skipped=%d cache=%d portrait_taps=%d portrait_swipes=%d key-down match=%d/%d',
                    n,len(fut),uid,len(active),sum(x.get('status')=='map_or_other_out_of_scope' for x in rows),cached,
                    sum(x.get('portrait_taps',0) for x in active),sum(x.get('portrait_swipes',0) for x in active),
                    sum(x.get('key_down_matched_event',0) for x in active),sum(x.get('key_down_rows',0) for x in active))
            except Exception:
                log.exception('FAILED user=%s',uid);failed.append(uid)
    report=summarize(results,list(users),out,config,failed)
    log.info('DONE users=%d; processed=%d; map_skipped=%d; cached=%d',len(users),report['sessions_processed'],report['sessions_ignored_map_or_other'],hits)
    print('\n=== SHARE BACK THESE RESULTS ===')
    print(json.dumps({'sessions_processed':report['sessions_processed'],
                      'sessions_ignored_map_or_other':report['sessions_ignored_map_or_other'],
                      'totals':report['totals'],'by_activity':report['by_activity'],
                      'failed_users':report['failed_users']},ensure_ascii=False,indent=2))
    print('=== END ===\n')
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
