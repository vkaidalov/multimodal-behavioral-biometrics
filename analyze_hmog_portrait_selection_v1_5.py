#!/usr/bin/env python3
"""HMOG per-gesture portrait audit and independent user-by-activity eligibility selection.

A gesture is individually portrait if EVERY touch row in its physical DOWN->UP
contact has Phone_orientation=0. Nonportrait gestures are skipped, not used to
interrupt an otherwise valid sequence. Four sessions for each user + activity
are evaluated independently of that user's other activities.

Requires Python >=3.11, pandas, numpy. Run from repository root:
    python analyze_hmog_portrait_selection_v1_5.py --workers 4
Input defaults to input_data/hmog_unarchived; output to output_data/hmog_portrait_selection_v1_5.

Selection defaults: strict portrait touch + paired accelerometer/gyroscope,
reading >=20 and writing >=100 selected gestures IN EACH of four sessions.
All thresholds and the selection mode are configurable; the thresholds are
illustrative, not prescribed by the HMOG paper.

Design constraints:
* Prefer TouchEvent_im.csv when present; otherwise use TouchEvent.csv. Never merge the two.
* No gestures are synthesized from OneFingerTouchEvent / ScrollEvent / StrokeEvent.
* Map tasks 5/6/11/12/17/18/23/24 are ignored, even within a mixed session.
* Each physical DOWN ... UP (single pointer) is counted once; classification uses
  the same geometric 30 px displacement / 50 px path as FETA v1.1.
* No assumption that a keyboard press is present in the raw TouchEvent logger.
* EventTime and PressTime relative clocks are audited separately from Systime;
  no unverified clock shift is silently introduced.
* TouchEvent.EventTime and KeyPressEvent.PressTime are milliseconds; sensor EventTime
  is nanoseconds, divided by 1,000,000. Systime is diagnostic only.
* Android MotionEvent uses uptimeMillis and SensorEvent elapsedRealtimeNanos.
  These origins can differ after device deep sleep: estimate offsets for warnings,
  never silently align them. Keep original FETA strict quality flags AND
  separate relaxed/sensitivity indicators and gesture-level failure diagnostics.
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

VERSION = 'hmog-raw-feta-comparable-1.5.0-per-gesture-portrait-selection-20261111'
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
    'key': ['Systime','PressTime','ActivityID','PressType','KeyID','Phone_orientation'],
    'accel': ['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation'],
    'gyro': ['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation'],
}
FILE_NAMES = {
    'activity':('Activity.csv',), 'touch':('TouchEvent_im.csv','TouchEvent.csv'),
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
        if not {'Activity.csv','TouchEvent_im.csv','TouchEvent.csv'}.intersection(files):
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
    system_time=epoch_ms(numeric(df,'systime'))
    rel=numeric(df,'eventtime')  # milliseconds (MotionEvent uptimeMillis).
    events=rel.copy()
    fit=clock_fit(system_time,rel)  # diagnostic only; NEVER applied to gestures.
    basis='touch_eventtime_ms'
    sequence_time=rel
    acid=strings(df,'activityid')
    code=numeric(df,'actionid'); x=numeric(df,'x'); y=numeric(df,'y')
    pc=numeric(df,'pointercount'); pid=numeric(df,'pointerid'); orient=numeric(df,'phoneorientation')
    valid=np.isfinite(sequence_time)&np.isfinite(code)&np.isfinite(x)&np.isfinite(y)
    indices=np.flatnonzero(valid)
    # ActivityID boundaries must not intermingle even after Android uptime reset.
    indices=np.array(sorted(indices,key=lambda j:(acid[j],sequence_time[j],int(j))),dtype=int)
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
            active={'activity_id':a,'start_ms':float(events[j]),'start_rel_ms':float(rel[j]),'start_x':float(x[j]),'start_y':float(y[j]),
                    'last_x':float(x[j]),'last_y':float(y[j]),'path_px':0.,'points':1,
                    'orientation_codes':set(), 'has_unknown_orientation':False,'multi':bool(multi),
                    'down_row':int(j),'start_systime_ms':float(system_time[j])}
            if np.isfinite(orient[j]):active['orientation_codes'].add(int(orient[j]))
            else:active['has_unknown_orientation']=True
            event_meta[0].append((a,float(rel[j]),float(system_time[j]),float(rel[j]),int(j)))
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
            event_meta[1].append((a,float(rel[j]),float(system_time[j]),float(rel[j]),int(j)))
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
        duration=float(rel[j]-active['start_rel_ms'])
        disp=math.hypot(float(x[j])-active['start_x'],float(y[j])-active['start_y'])
        kind='swipe' if (disp>=cfg['swipe_distance_px'] or active['path_px']>=cfg['swipe_path_px']) else 'tap'
        eligible_timing=np.isfinite(duration) and duration>0
        is_portrait=(active['orientation_codes']=={0} and not active['has_unknown_orientation'])
        p={'activity_id':a,'task':allowed[a],'type':kind,
           'start_rel_ms':round(active['start_rel_ms'],3),'end_rel_ms':round(float(rel[j]),3),
           'start_ms':round(active['start_ms'],3),'end_ms':round(float(events[j]),3),
           'start_touch_systime_ms':round(active['start_systime_ms'],3),
           'end_touch_systime_ms':round(float(system_time[j]),3),
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


def sensor_clock_diagnostics(df, short, allowed):
    """Report sensor EventTime sanity without silently aligning time origins.

    SensorEvent.timestamp is nanoseconds since elapsedRealtime; Systime is the
    logging wall clock, and the difference is only a diagnostic estimate.
    """
    aid=strings(df,'activityid')
    m=np.isin(aid,list(allowed))
    raw=numeric(df,'eventtime')[m]
    clock=raw/1e6
    sys_time=epoch_ms(numeric(df,'systime'))[m]
    valid=np.isfinite(clock)
    recorded=np.asarray(clock[valid],dtype=np.float64)
    sys_valid=np.isfinite(sys_time)&valid
    if np.any(sys_valid):
        offsets=sys_time[sys_valid]-clock[sys_valid]
        median_offset=float(np.median(offsets))
        offset_p10=float(np.percentile(offsets,10))
        offset_p90=float(np.percentile(offsets,90))
    else:
        median_offset=offset_p10=offset_p90=None
    valid_raw=raw[np.isfinite(raw)]
    n=int(valid.sum())
    return dict(source=short,unit='ns / 1e6 = ms',rows=int(m.sum()),
        valid_event_rows=n,invalid_event_rows=int(m.sum())-n,
        event_clock_min_ms=float(np.min(recorded)) if n else None,
        event_clock_max_ms=float(np.max(recorded)) if n else None,
        event_raw_median=float(np.median(valid_raw)) if len(valid_raw) else None,
        eventtime_inversions_file_order=int(np.count_nonzero(np.diff(recorded)<0)),
        eventtime_duplicates_file_order=int(np.count_nonzero(np.diff(recorded)==0)),
        systime_minus_eventtime_median_ms=median_offset,
        systime_minus_eventtime_p10_ms=offset_p10,
        systime_minus_eventtime_p90_ms=offset_p90,
        systime_minus_eventtime_spread_p10_p90_ms=(offset_p90-offset_p10 if median_offset is not None else None))


def sorted_sensor(df,allowed,short='sensor'):
    if df.empty:
        return (np.empty(0),np.empty(0),np.empty(0,dtype=object),
                orientation_counts([]),0,
                dict(source=short,unit='ns / 1e6 = ms',rows=0,valid_event_rows=0,invalid_event_rows=0))
    activity=strings(df,'activityid')
    mask=np.isin(activity,list(allowed))
    # SENSOR EVENT timestamps are NANOSECONDS (Android SensorEvent.timestamp).
    # Do not calibrate or fall back to Systime: otherwise a different, sometimes
    # delayed logging time silently contaminates synchronization statistics.
    time=numeric(df,'eventtime')[mask]/1_000_000.0
    orientation=numeric(df,'phoneorientation')[mask]
    aids=activity[mask]
    good=np.isfinite(time)
    diag=sensor_clock_diagnostics(df,short,allowed)
    time=time[good];orientation=orientation[good];aids=aids[good]
    order=np.argsort(time,kind='stable')
    return time[order],orientation[order],aids[order],orientation_counts(orientation),int(len(df)-mask.sum()),diag


def sensor_interval(times, orientations, start, end, cfg, detailed=True):
    """Metrics refer ONLY to events within the touch EventTime interval."""
    lo=int(np.searchsorted(times,start,'left'));hi=int(np.searchsorted(times,end,'right'))
    t=times[lo:hi];o=orientations[lo:hi]
    n=len(t)
    edge_start=float(t[0]-start) if n else None
    edge_end=float(end-t[-1]) if n else None
    edge=max(edge_start,edge_end) if n else None
    d=np.diff(t) if n>1 else np.empty(0,dtype=np.float64)
    gap=float(np.max(d)) if len(d) else None
    gaps_gt_threshold=int(np.count_nonzero(d>cfg['max_sensor_gap_ms']))
    positive_d=d[d>0] if detailed else np.empty(0,dtype=np.float64)
    # Effective average density is descriptive; does not guarantee low jitter.
    duration=max(float(end-start),0.)
    info={'edge_start_ms':edge_start,'edge_end_ms':edge_end,
          'gap_median_ms':float(np.median(d)) if detailed and len(d) else None,
          'gap_p95_ms':float(np.percentile(d,95)) if detailed and len(d) else None,
          'gap_p99_ms':float(np.percentile(d,99)) if detailed and len(d) else None,
          'gaps_gt_threshold':gaps_gt_threshold,
          'gaps_gt_50ms':int(np.count_nonzero(d>50.)),
          'gaps_gt_100ms':int(np.count_nonzero(d>100.)),
          'gaps_gt_150ms':int(np.count_nonzero(d>150.)),
          'gaps_gt_200ms':int(np.count_nonzero(d>200.)),
          'gaps_gt_threshold_pct':float(100*gaps_gt_threshold/len(d)) if len(d) else None,
          'excess_gap_over_threshold_ms':float(np.maximum(d-cfg['max_sensor_gap_ms'],0.).sum()),
          'mean_sample_rate_hz':float(n*1000./duration) if duration else None,
          'median_instantaneous_hz':float(1000./np.median(positive_d)) if detailed and len(positive_d) else None,
          'samples_portrait':int(np.count_nonzero(o==0)),
          'samples_landscape':int(np.count_nonzero(np.isin(o,[1,3]))),
          'samples_unknown':int(np.count_nonzero(~np.isin(o,[0,1,3])))}
    strict=bool(n>=cfg['min_sensor_samples'] and edge is not None and
                edge<=cfg['max_edge_gap_ms'] and
                (gap is None or gap<=cfg['max_sensor_gap_ms']))
    relaxed=bool(n>=cfg['min_sensor_samples'] and edge is not None and
                 edge<=cfg['diagnostic_edge_gap_ms'] and
                 (gap is None or gap<=cfg['diagnostic_sensor_gap_ms']))
    return t,n,edge,gap,strict,bool(n and np.all(o==0)),relaxed,info


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
        for lim in (0,1,2,5,20,50,80,150,250,500):
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
                  **{f'nearest_at_most_{n}ms':int(near_ranges[str(n)]) for n in (0,1,2,5,20,50,80,150,250,500)}}


def attempt_matching(key_rows, touch_rows, tol, clock='event', key_scale=1., touch_scale=1.):
    if clock=='event': return nearest_event_stats(key_rows,touch_rows,tol)
    if clock=='relative_direct':
        return nearest_event_stats(
            [(a,r*key_scale,s,r,j) for a,t,s,r,j in key_rows],
            [(a,r*touch_scale,s,r,j) for a,t,s,r,j in touch_rows],tol)
    def trans(rows):
        return [(a,s,s,r,j) for a,t,s,r,j in rows]
    return nearest_event_stats(trans(key_rows),trans(touch_rows),tol)


def match_interval_pairs(key_pairs, keytimes, keyactivities, contacts, shift_ms=0., threshold='any'):
    """One-to-one assignment of overlapping key-held and raw physical touch intervals.

    Uses only PressTime/EventTime (same Android uptime clock), never Systime.
    `shift_ms` is a negative-control clock shift, NOT an alignment correction.
    Valid, single-pointer contacts only; same ActivityID; no cross-activity matching.
    """
    from bisect import bisect_left, bisect_right
    touch_by_act=defaultdict(list)
    for tid,c in enumerate(contacts):
        t0=c.get('start_rel_ms');t1=c.get('end_rel_ms')
        if (not c.get('multitouch') and c.get('valid_duration') and
            np.isfinite(t0) and np.isfinite(t1) and 0<t1-t0<=12000):
            touch_by_act[c['activity_id']].append((float(t0),float(t1),tid))
    idx={}
    for aid,items in touch_by_act.items():
        items.sort()
        idx[aid]=(items,[v[0] for v in items])
    candidates=[];key_candidates=Counter();touch_candidates=Counter()
    for ki,(down,up,duration) in enumerate(key_pairs):
        if not (np.isfinite(keytimes[down]) and np.isfinite(keytimes[up])):continue
        a=keyactivities[down]
        if a not in idx:continue
        start=float(keytimes[down])+shift_ms
        end=float(keytimes[up])+shift_ms
        if end<=start:continue
        items,starts=idx[a]
        # Bound search by the maximum accepted physical touch duration.
        lo=bisect_left(starts,start-12000.)
        hi=bisect_right(starts,end)
        for t0,t1,tid in items[lo:hi]:
            overlap=max(0.,min(end,t1)-max(start,t0))
            if overlap<=0:continue
            proportion=overlap/min(end-start,t1-t0)
            allowed=(threshold=='any' or
                (threshold=='10ms' and overlap>=10) or
                (threshold=='25pct' and proportion>=.25) or
                (threshold=='50pct' and proportion>=.50))
            if allowed:
                # Prefer similar start/end timestamps after overlap ratio.
                penalty=abs(start-t0)+abs(end-t1)
                candidates.append((float(proportion),float(overlap),float(penalty),ki,tid,
                                   float(t0-start),float(t1-end)))
                key_candidates[ki]+=1;touch_candidates[tid]+=1
    candidates.sort(key=lambda x:(-x[0],-x[1],x[2],x[3],x[4]))
    used_k=set();used_t=set();picked=[]
    for c in candidates:
        if c[3] in used_k or c[4] in used_t:continue
        used_k.add(c[3]);used_t.add(c[4]);picked.append(c)
    def med(v):return round(float(np.median(v)),3) if v else None
    return {'matched':len(picked), 'pairs_with_any_candidate':len(key_candidates),
            'pairs_with_multiple_candidates':sum(n>1 for n in key_candidates.values()),
            'touches_with_multiple_candidate_keys':sum(n>1 for n in touch_candidates.values()),
            'candidate_edges':len(candidates),
            'matched_overlap_ratio_median':med([v[0] for v in picked]),
            'matched_overlap_ms_median':med([v[1] for v in picked]),
            'touch_minus_key_down_median_ms':med([v[5] for v in picked]),
            'touch_minus_key_up_median_ms':med([v[6] for v in picked]),
            'matched_key_pair_indices':[int(v[3]) for v in picked]}


def key_analysis(df,allowed,touch_events,contact_ids,contacts,cfg,touch_clock_fit):
    # A valid reading session can contain zero KeyPressEvent rows. A missing
    # KeyPressEvent.csv is also possible. Keep all numeric counters explicit:
    # Counter() returns zero for absent keys only until cast to dict.
    results=Counter({
        'key_rows_all': 0,
        'key_rows_in_scope': 0,
        'key_down_rows': 0,
        'key_up_rows': 0,
        'key_down_matched_event': 0,
        'key_up_matched_event': 0,
        'key_down_matched_system': 0,
        'key_up_matched_system': 0,
        'key_down_matched_relative_direct': 0,
        'key_up_matched_relative_direct': 0,
        'key_completed_pairs': 0,
        'key_pairs_both_endpoint_events_matched': 0,
        'key_pairs_same_physical_touch_contact': 0,
        'keypress_unrecognized_types': 0,
    })
    if df.empty:
        results['keypress_type_values'] = {}
        return dict(results),[],{'valid':False},'empty_key_data'
    # Crucial HMOG quirk: key rows are often logged UP then DOWN by Systime.
    # PressTime is the physical event timestamp and matches TouchEvent.EventTime.
    keysystime=epoch_ms(numeric(df,'systime'))
    rel=numeric(df,'presstime')
    keystime=rel.copy()
    fit=clock_fit(keysystime,rel)  # diagnostic only; NEVER used for matching.
    basis='press_time_raw'
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
        # 'event' and 'relative_direct' now both compare RAW millisecond timestamps.
        # 'system' is only a negative-control diagnostic; it is not used to form pairs.
        matches,diag=attempt_matching(events,targets,cfg['key_match_tolerance_ms'],'event')
        result_events[typ]=matches
        results[f'key_{label}_matched_event']=len(matches)
        results[f'key_{label}_matched_relative_direct']=len(matches)
        results[f'key_{label}_exact_match']=len(attempt_matching(events,targets,0.,'event')[0])
        for k,v in diag.items():results[f'key_{label}_event_{k}']=v
        sysmatch,_=attempt_matching(events,targets,cfg['key_match_tolerance_ms'],'system')
        results[f'key_{label}_matched_system']=len(sysmatch)
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
    results['key_pairs_up_record_appears_before_down_record']=sum(int(up<down) for down,up,_ in pairs)
    results['key_pairs_up_systime_before_down_systime']=sum(
        int(np.isfinite(keysystime[down]) and np.isfinite(keysystime[up]) and
            keysystime[up]<keysystime[down]) for down,up,_ in pairs)
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
    # Pair-overlap audit: direct PressTime vs EventTime, same activity, one-to-one.
    overlap_summary={}
    for threshold in ('any','10ms','25pct','50pct'):
        diag=match_interval_pairs(pairs,keystime,act,contacts,threshold=threshold)
        overlap_summary[threshold]=diag
        tag={'any':'any','10ms':'at_least_10ms','25pct':'at_least_25pct','50pct':'at_least_50pct'}[threshold]
        results[f'key_pairs_interval_{tag}_matched']=diag['matched']
        if threshold=='any':
            for metric in ('pairs_with_any_candidate','pairs_with_multiple_candidates',
                           'touches_with_multiple_candidate_keys','candidate_edges',
                           'matched_overlap_ratio_median','matched_overlap_ms_median',
                           'touch_minus_key_down_median_ms','touch_minus_key_up_median_ms'):
                results['key_interval_'+metric]=diag[metric]
    # A deliberately time-shifted null check helps diagnose accidental overlap.
    shifted=match_interval_pairs(pairs,keystime,act,contacts,
                                 shift_ms=cfg['key_negative_control_shift_ms'],threshold='50pct')
    results['key_pairs_interval_50pct_shifted_negative_control']=shifted['matched']
    selected_pairs={pairs[ki][0] for ki in overlap_summary['50pct']['matched_key_pair_indices']}
    results['key_pairs_interval_50pct_matched_and_endpoints']=sum(
        pair[0] in selected_pairs for pair in pairs
        if (result_events[0].get(pair[0]) and result_events[1].get(pair[1])))
    # Record source-order inversions (not proof of global system clock semantics).
    results['key_systime_inversion_between_consecutive_rows']=int(np.count_nonzero(np.diff(keysystime[np.isfinite(keysystime)])<0))
    results['key_presstime_inversion_between_consecutive_rows']=int(np.count_nonzero(np.diff(rel[np.isfinite(rel)])<0))
    results['key_press_down_system_lag_median_minus_up_ms']=(
        round(float(np.median((keysystime-rel)[(presstype==0)&valid]))-
              float(np.median((keysystime-rel)[(presstype==1)&valid])),3)
        if np.any((presstype==0)&valid) and np.any((presstype==1)&valid) else None)
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
    if cache.exists() and (out/'cache'/'gestures'/f'{uid}__{token}.csv.gz').exists() and not no_cache:
        try:
            data=json.loads(cache.read_text(encoding='utf-8'))
            if data['signature']==sig:return data['result'],True
        except (ValueError,KeyError,OSError):pass
    base={'user_id':uid,'session_dir':str(folder),'session_number':folder.name,
          'status':'processed','task':'','session_start_ms':None,'session_start_iso':'',
          'warnings':[],
          'touch_source_file':files['touch'].name if 'touch' in files else '',
          'touch_im_used':int(files.get('touch',None) is not None and files['touch'].name=='TouchEvent_im.csv'),
          'touch_im_available':int((folder/'TouchEvent_im.csv').is_file()),
          'touch_standard_available':int((folder/'TouchEvent.csv').is_file())}
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
    if 'key' not in files:
        base['warnings'].append('missing_KeyPressEvent_file_completeness_not_testable')
    sensor={};orients={}
    for short in ('accel','gyro'):
        frame=read_table(files.get(short),short,use=('Systime','EventTime','ActivityID','Phone_orientation'))
        times,orientation,sensor_activities,stats,excluded,clock_diag=sorted_sensor(frame,activities,short)
        base[short+'_clock']=clock_diag
        base[short+'_clock_basis']='sensor_eventtime_ns_div_1e6'
        sensor[short]=(times,orientation)
        sa=strings(frame,'activityid')
        se=numeric(frame,'eventtime')/1e6
        ss=epoch_ms(numeric(frame,'systime'))
        sensor[short+'_wall_event_offsets_by_activity']={}
        for aid in activities:
            valid=(sa==aid)&np.isfinite(se)&np.isfinite(ss)
            sensor[short+'_wall_event_offsets_by_activity'][aid]=float(np.median(ss[valid]-se[valid])) if np.any(valid) else None
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
    key_stats,key_examples,key_cal,key_basis=key_analysis(df_key,activities,touch_events,raw_contact_ids,contacts,config,touch_cal)
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
                parts,n,edge,gap,strict,allp,relaxed,details=sensor_interval(t,o,g['start_ms'],g['end_ms'],config,detailed=(g['type']=='swipe'))
                g[short+'_samples']=n
                g[short+'_edge_max_ms']=round(edge,3) if edge is not None else ''
                g[short+'_max_gap_ms']=round(gap,3) if gap is not None else ''
                g[short+'_strict']=int(strict)
                g[short+'_relaxed']=int(relaxed)
                g[short+'_all_portrait']=int(allp)
                for name,value in details.items():
                    g[short+'_'+name]=round(value,4) if isinstance(value,float) else value
                readings[short]=parts
            pairs=count_pairs(readings['accel'],readings['gyro'],config['pair_tolerance_ms'])
            g['accel_gyro_pairs']=pairs
            g['accel_gyro_paired_pct_of_smaller_stream']=round(100*pairs/max(1,min(g['accel_samples'],g['gyro_samples'])),3)
            g['both_any']=int(bool(g['accel_samples'] and g['gyro_samples']))
            g['both_strict_paired']=int(bool(g['accel_strict'] and g['gyro_strict'] and pairs>=config['min_sensor_pairs']))
            g['both_relaxed_paired']=int(bool(g['accel_relaxed'] and g['gyro_relaxed'] and pairs>=config['min_sensor_pairs']))
            g['both_strict_paired_portrait_motion']=int(bool(g['both_strict_paired'] and g['accel_all_portrait'] and g['gyro_all_portrait']))
            g['both_relaxed_paired_portrait_motion']=int(bool(g['both_relaxed_paired'] and g['accel_all_portrait'] and g['gyro_all_portrait']))
            fail_reasons=[]
            if min(g['accel_samples'],g['gyro_samples'])<config['min_sensor_samples']:
                fail_reasons.append('few_samples')
            if any(g[q+'_samples']>0 and float(g[q+'_edge_max_ms'])>config['max_edge_gap_ms'] for q in ('accel','gyro')):
                fail_reasons.append('edge_gap')
            if any(g[q+'_samples']>1 and float(g[q+'_max_gap_ms'])>config['max_sensor_gap_ms'] for q in ('accel','gyro')):
                fail_reasons.append('internal_gap')
            if pairs<config['min_sensor_pairs']:
                fail_reasons.append('insufficient_paired_samples')
            g['strict_fail_reasons']=';'.join(fail_reasons)
            g['strict_fail_count']=len(fail_reasons)
            if portrait:
                sums['portrait_both_any']+=int(g['both_any'])
                sums['portrait_both_strict_paired']+=int(g['both_strict_paired'])
                sums['portrait_both_relaxed_paired']+=int(g['both_relaxed_paired'])
                sums['portrait_both_strict_paired_portrait_motion']+=int(g['both_strict_paired_portrait_motion'])
                sums['portrait_'+g['type']+'s_both_any']+=int(g['both_any'])
                sums['portrait_'+g['type']+'s_both_strict_paired']+=int(g['both_strict_paired'])
                sums['portrait_'+g['type']+'s_both_relaxed_paired']+=int(g['both_relaxed_paired'])
                sums['portrait_'+g['type']+'s_both_strict_paired_portrait_motion']+=int(g['both_strict_paired_portrait_motion'])
                sums['portrait_'+g['type']+'s_both_relaxed_paired_portrait_motion']+=int(g['both_relaxed_paired_portrait_motion'])
                if not g['both_strict_paired']:
                    sums['portrait_'+g['type']+'s_failed']+=1
                    for reason in fail_reasons:
                        sums['portrait_'+g['type']+'s_fail_'+reason]+=1
        else:
            for v in ('accel_samples','gyro_samples','accel_strict','gyro_strict','accel_gyro_pairs','both_any','both_strict_paired','both_strict_paired_portrait_motion'):
                g[v]=0
    # New v1.5 session-task counters. All taps/swipes are considered INDEPENDENTLY:
    # landscape/unknown contacts are omitted instead of splitting a time segment.
    # Counts stay separate by activity even for an unusual mixed-task session.
    by_task_counts = defaultdict(Counter)
    for g in contacts:
        task = g.get('task', '')
        if task not in TASK_ORDER:
            continue
        c = by_task_counts[task]
        c['all_completed_contacts'] += 1
        valid = bool(g.get('single_valid'))
        portrait = bool(g.get('portrait_single_valid'))
        strict = bool(portrait and g.get('both_strict_paired_portrait_motion'))
        # Pairing and BOTH motion streams must be strictly covered AND portrait.
        g['portrait_multimodal_strict'] = int(strict)
        if valid:
            c['valid_gestures'] += 1
            c['valid_'+g['type']+'s'] += 1
            if portrait:
                c['portrait_gestures'] += 1
                c['portrait_'+g['type']+'s'] += 1
            else:
                c['nonportrait_or_unknown_gestures'] += 1
            if strict:
                c['strict_multimodal_gestures'] += 1
                c['strict_multimodal_'+g['type']+'s'] += 1
        else:
            c['invalid_or_multitouch_contacts'] += 1
    base['session_task_counts'] = {}
    for task in labels:
        c = by_task_counts[task]
        vals = {field: int(c[field]) for field in (
            'all_completed_contacts', 'valid_gestures', 'valid_taps', 'valid_swipes',
            'portrait_gestures', 'portrait_taps', 'portrait_swipes',
            'nonportrait_or_unknown_gestures', 'invalid_or_multitouch_contacts',
            'strict_multimodal_gestures', 'strict_multimodal_taps', 'strict_multimodal_swipes')}
        denom = vals['valid_gestures']
        vals['portrait_share_pct_valid'] = round(100 * vals['portrait_gestures'] / denom, 4) if denom else None
        vals['strict_multimodal_share_pct_valid'] = round(100 * vals['strict_multimodal_gestures'] / denom, 4) if denom else None
        base['session_task_counts'][task] = vals
    total_valid = sum(x['valid_gestures'] for x in base['session_task_counts'].values())
    total_portrait = sum(x['portrait_gestures'] for x in base['session_task_counts'].values())
    total_strict = sum(x['strict_multimodal_gestures'] for x in base['session_task_counts'].values())
    base['individual_portrait_gestures'] = total_portrait
    base['individual_portrait_share_pct_valid'] = round(100*total_portrait/total_valid,4) if total_valid else None
    base['individual_strict_multimodal_gestures'] = total_strict
    base['individual_strict_multimodal_share_pct_valid'] = round(100*total_strict/total_valid,4) if total_valid else None
    base.update(sums)
    for kind in ('tap','swipe'):
        total=int(base.get('portrait_'+kind+'s',0) or 0)
        strict=int(base.get('portrait_'+kind+'s_both_strict_paired',0) or 0)
        relaxed=int(base.get('portrait_'+kind+'s_both_relaxed_paired',0) or 0)
        base['portrait_'+kind+'s_strict_coverage_pct']=round(100*strict/total,3) if total else None
        base['portrait_'+kind+'s_relaxed_coverage_pct']=round(100*relaxed/total,3) if total else None
        base['portrait_'+kind+'s_uncovered_pct']=round(100*(total-strict)/total,3) if total else None
    for kind in ('tap','swipe'):
        gds=[g['duration_ms'] for g in contacts if g['type']==kind and g['portrait_single_valid']]
        base[f'portrait_{kind}_median_duration_ms']=round(float(np.median(gds)),3) if gds else None
        base[f'portrait_{kind}_p90_duration_ms']=round(float(np.percentile(gds,90)),3) if gds else None
    pgs=sorted([g for g in contacts if g['portrait_single_valid']],key=lambda g:g['start_ms'])
    gaps=[max(0.,b['start_ms']-a['end_ms']) for a,b in zip(pgs,pgs[1:]) if a['activity_id']==b['activity_id']]
    base['median_intergesture_gap_ms']=round(float(np.median(gaps)),3) if gaps else None
    for label in ('portrait_taps','portrait_swipes','portrait_taps_both_strict_paired','portrait_swipes_both_strict_paired',
                  'portrait_taps_both_strict_paired_portrait_motion','portrait_swipes_both_strict_paired_portrait_motion',
                  'portrait_both_any','portrait_both_strict_paired','portrait_both_relaxed_paired','portrait_both_strict_paired_portrait_motion',
                  'portrait_taps_both_any','portrait_swipes_both_any',
                  'portrait_taps_both_relaxed_paired','portrait_swipes_both_relaxed_paired',
                  'portrait_taps_failed','portrait_swipes_failed',
            'portrait_taps_both_relaxed_paired_portrait_motion','portrait_swipes_both_relaxed_paired_portrait_motion',
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
    if incompatible:base['warnings'].append('activity_sensor_eventtime_ranges_disjoint')
    # Compare clock origin estimates ONLY as diagnostic. No offset is applied.
    # Systime is wall time; event clocks are monotonic and not necessarily same origin.
    touch_sys=epoch_ms(numeric(df_touch,'systime'))
    touch_rel=numeric(df_touch,'eventtime')
    touch_aids=strings(df_touch,'activityid')
    clock_rows=[]
    for aid,task in activities.items():
        mask=(touch_aids==aid)&np.isfinite(touch_sys)&np.isfinite(touch_rel)
        offsets=touch_sys[mask]-touch_rel[mask]
        toff=float(np.median(offsets)) if len(offsets) else None
        touch_min=float(np.min(touch_rel[mask])) if len(offsets) else None
        touch_max=float(np.max(touch_rel[mask])) if len(offsets) else None
        for short in ('accel','gyro'):
            # For a more local estimate, Systime/EventTime offset is tracked
            # separately per activity (not just per sensor file).
            soff=sensor[short+'_wall_event_offsets_by_activity'][aid]
            # Difference is expected sensor clock minus touch clock at the same
            # physical instant, but recording delays make it approximate.
            apparent_offset=(toff-soff) if toff is not None and soff is not None else None
            st=sensor[short+'_by_activity'][aid][0]
            sd=np.diff(st) if len(st)>1 else np.empty(0,dtype=np.float64)
            sd_positive=sd[sd>0]
            overlap=(max(0.,min(touch_max,float(st[-1]))-max(touch_min,float(st[0])))
                     if touch_min is not None and len(st) else 0.)
            row=dict(user_id=uid,session_number=base['session_number'],task=task,
                     activity_id=aid,sensor=short,source_file=base['touch_source_file'],
                     touch_event_clock_min_ms=touch_min,touch_event_clock_max_ms=touch_max,
                     sensor_event_clock_min_ms=float(st[0]) if len(st) else None,
                     sensor_event_clock_max_ms=float(st[-1]) if len(st) else None,
                     direct_range_overlap_ms=round(overlap,3),
                     sensor_eventtime_samples=int(len(st)),
                     sensor_eventtime_median_gap_ms=round(float(np.median(sd)),4) if len(sd) else None,
                     sensor_eventtime_p95_gap_ms=round(float(np.percentile(sd,95)),4) if len(sd) else None,
                     sensor_eventtime_p99_gap_ms=round(float(np.percentile(sd,99)),4) if len(sd) else None,
                     sensor_eventtime_max_gap_ms=round(float(np.max(sd)),4) if len(sd) else None,
                     sensor_eventtime_gap_over_120ms_count=int(np.count_nonzero(sd>120)),
                     sensor_eventtime_global_density_hz=(round(1000*len(st)/max(st[-1]-st[0],0.001),3) if len(st)>1 else None),
                     sensor_eventtime_median_instantaneous_hz=(round(1000/np.median(sd_positive),3) if len(sd_positive) else None),
                     touch_systime_minus_eventtime_median_ms=toff,
                     sensor_systime_minus_eventtime_median_ms=soff,
                     inferred_sensor_minus_touch_origin_ms=round(apparent_offset,3) if apparent_offset is not None else None,
                     inferred_offset_is_diagnostic_not_applied=True)
            clock_rows.append(row)
            if apparent_offset is not None and abs(apparent_offset)>config['clock_offset_warning_ms']:
                base['warnings'].append('possible_sensor_touch_clock_origin_offset:'+short+':'+str(round(apparent_offset,1)))
    base['clock_diagnostics_file']=str(out/'cache'/'clock_diagnostics'/f'{uid}__{token}.csv')
    csv_write(Path(base['clock_diagnostics_file']),clock_rows)
    base['failed_gesture_diagnostics_file']=str(out/'cache'/'coverage_failures'/f'{uid}__{token}.csv.gz')
    failures=[g for g in contacts if g.get('portrait_single_valid') and not g.get('both_strict_paired')]
    base['failed_gestures_total_in_session']=len(failures)
    failures.sort(key=lambda g:(g['type']!='swipe',-g['duration_ms']))
    failures=failures[:config['diagnostic_max_failed_gestures_per_session']]
    base['failed_gestures_exported_for_diagnostics']=len(failures)
    csv_write(Path(base['failed_gesture_diagnostics_file']),failures)
    base['recovered_gestures_file']=str(out/'cache'/'gestures'/f'{uid}__{token}.csv.gz')
    csv_write(Path(base['recovered_gestures_file']),contacts)
    if key_examples:
        csv_write(out/'cache'/'key_unmatched_examples'/f'{uid}__{token}.csv',key_examples)
    if not touch_cal['valid']:base['warnings'].append('touch_event_to_systime_clock_fit_unreliable_diagnostic_only')
    if not key_cal['valid'] and len(df_key):base['warnings'].append('key_Systime_relative_fit_unreliable_but_PressTime_used_directly')
    if base['raw_invalid_touch_rows']>0:base['warnings'].append('invalid_touch_rows_present')
    if base.get('raw_interrupted_down',0)>0:base['warnings'].append('interrupted_touch_sequences')
    if (base.get('key_down_rows',0) + base.get('key_up_rows',0)
            < base.get('key_rows_in_scope',0)):
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
                    'touch_source_file':r.get('touch_source_file',''),
                    'touch_im_used':r.get('touch_im_used',0),
                    'portrait_pct_all_sources':r.get('all_sources_portrait_pct_known'),
                    'portrait_pct_touch':r.get('touch_portrait_pct_known'),
                    'portrait_taps':r.get('portrait_taps',0),'portrait_swipes':r.get('portrait_swipes',0)})
        cell[uid]=rec
    csv_write(out/filename,[cell[u] for u in users],['user_id']+columns,delimiter='\t')
    return metadata,overflow


def selected_count_for_session(row, selection):
    mode = selection['selection_mode']
    kind_mode = selection['gesture_count_mode']
    prefix = 'strict_multimodal' if mode == 'strict' else 'portrait'
    if kind_mode == 'primary':
        suffix = 'swipes' if row['task'].startswith('reading') else 'taps'
        return int(row.get(f'{prefix}_{suffix}', 0))
    return int(row.get(f'{prefix}_gestures', 0))


def build_selection_outputs(processed, users, out, selection):
    """Write every session/task and independent user/task eligibility decisions.

    Exactly four distinct session numbers are required by default. We never
    require a user to qualify for any OTHER activity type.
    """
    by_pair = defaultdict(list)
    task_rows = []
    for r in processed:
        uid = r['user_id']
        for task, stats in r.get('session_task_counts', {}).items():
            if task not in TASK_ORDER:
                continue
            row = {'user_id': uid, 'task': task,
                   'session_number': str(r['session_number']),
                   'session_dir': r['session_dir'],
                   'session_start_ms': r.get('session_start_ms'),
                   'session_start_iso': r.get('session_start_iso'),
                   'gesture_cache_file': r.get('recovered_gestures_file', ''),
                   'touch_source_file': r.get('touch_source_file', ''),
                   'clock_incompatibilities': r.get('activity_sensor_clock_incompatibilities',''),
                   **stats}
            row['selection_count'] = selected_count_for_session(row,selection)
            row['selection_share_pct_valid'] = (round(100*row['selection_count']/row['valid_gestures'],4)
                 if row['valid_gestures'] else None)
            row['threshold'] = (selection['min_reading_gestures'] if task.startswith('reading')
                                else selection['min_writing_gestures'])
            row['meets_threshold'] = int(row['selection_count'] >= row['threshold'])
            task_rows.append(row)
            by_pair[(uid,task)].append(row)
    pairs = []
    number_required = selection['required_sessions']
    for uid in users:
        for task in TASK_ORDER:
            rr = by_pair.get((uid,task),[])
            rr.sort(key=lambda r:(r['session_start_ms'] is None,
                  r['session_start_ms'] if r['session_start_ms'] is not None else float('inf'),
                  str(r['session_number']),str(r['session_dir'])))
            for index,r in enumerate(rr,1):
                r['chronological_slot'] = index
            n = len(rr)
            distinct = len(set(r['session_number'] for r in rr))
            threshold = selection['min_reading_gestures'] if task.startswith('reading') else selection['min_writing_gestures']
            if n != number_required:
                reason='session_count_'+str(n)+'_expected_'+str(number_required)
            elif distinct != number_required:
                reason='duplicate_session_numbers'
            elif any(not x['meets_threshold'] for x in rr):
                reason='below_threshold_in_'+str(sum(not x['meets_threshold'] for x in rr))+'_sessions'
            else:
                reason='eligible'
            eligible = int(reason=='eligible')
            for r in rr:
                r['selected_pair'] = eligible
                r['pair_exclusion_reason'] = '' if eligible else reason
            pair = {'user_id':uid, 'task':task, 'session_count':n,
                    'distinct_session_numbers':distinct, 'required_sessions':number_required,
                    'threshold_per_session':threshold,
                    'selection_mode':selection['selection_mode'],
                    'gesture_count_mode':selection['gesture_count_mode'],
                    'selected_pair':eligible, 'reason':reason,
                    'total_usable_gestures':sum(r['selection_count'] for r in rr),
                    'min_usable_in_session':min((r['selection_count'] for r in rr),default=0),
                    'max_usable_in_session':max((r['selection_count'] for r in rr),default=0),
                    'mean_usable_in_session':round(sum(r['selection_count'] for r in rr)/n,3) if n else None,
                    'session_numbers':';'.join(r['session_number'] for r in rr),
                    'per_session_usable_counts':';'.join(str(r['selection_count']) for r in rr),
                    'per_session_portrait_pct':';'.join(str(r['portrait_share_pct_valid']) for r in rr),
                    'per_session_strict_multimodal_pct':';'.join(str(r['strict_multimodal_share_pct_valid']) for r in rr)}
            pairs.append(pair)
    # CSV with real per-task measures; never duplicate whole-session totals for mixed-task sessions.
    task_rows.sort(key=lambda r:(r['user_id'],TASK_ORDER.index(r['task']),r['chronological_slot']))
    # Explicit headers even when no pair passes the threshold.
    task_fields = sorted(set().union(*(r.keys() for r in task_rows))) if task_rows else [
        'user_id','task','session_number','selection_count','selected_pair']
    pair_fields = sorted(set().union(*(r.keys() for r in pairs))) if pairs else [
        'user_id','task','session_count','selected_pair','reason']
    csv_write(out/'session_task_gestures.csv',task_rows,fields=task_fields)
    csv_write(out/'selected_session_tasks.csv',
              [r for r in task_rows if r.get('selected_pair')],fields=task_fields)
    csv_write(out/'user_task_pairs.csv',pairs,fields=pair_fields)
    csv_write(out/'selected_user_task_pairs.csv',
              [p for p in pairs if p['selected_pair']],fields=pair_fields)
    csv_write(out/'rejected_user_task_pairs.csv',
              [p for p in pairs if not p['selected_pair']],fields=pair_fields)
    count_mat=[];pct_mat=[]
    columns=[f'{task}_session{i}' for task in TASK_ORDER for i in range(1,number_required+1)]
    for uid in users:
        counts={'user_id':uid};pct={'user_id':uid}
        for task in TASK_ORDER:
            for r in by_pair.get((uid,task),[]):
                if r['chronological_slot']<=number_required:
                    col=f'{task}_session{r["chronological_slot"]}'
                    counts[col]=r['selection_count']
                    pct[col]=r['selection_share_pct_valid']
        count_mat.append(counts);pct_mat.append(pct)
    csv_write(out/'selection_counts_matrix.tsv',count_mat,fields=['user_id']+columns,delimiter='\t')
    csv_write(out/'selection_share_pct_matrix.tsv',pct_mat,fields=['user_id']+columns,delimiter='\t')
    # Exploratory eligibility sensitivity, no costly gesture reprocessing.
    sweep=[]
    for task in TASK_ORDER:
        for threshold in (10,20,30,40,50,75,100,150,200,300,500,1000):
            eligible_n=sum(len(rr:=by_pair.get((uid,task),[]))==number_required
                and len(set(x['session_number'] for x in rr))==number_required
                and all(x['selection_count']>=threshold for x in rr) for uid in users)
            sweep.append({'task':task,'threshold_per_session':threshold,
                          'eligible_user_task_pairs':eligible_n,
                          'required_sessions':number_required,
                          'selection_mode':selection['selection_mode'],
                          'gesture_count_mode':selection['gesture_count_mode']})
    csv_write(out/'eligibility_threshold_sweep.csv',sweep)
    stats={'selection_mode':selection['selection_mode'],
           'gesture_count_mode':selection['gesture_count_mode'],
           'required_sessions':number_required,
           'min_reading_gestures':selection['min_reading_gestures'],
           'min_writing_gestures':selection['min_writing_gestures'],
           'user_task_pairs_considered':len(pairs),
           'selected_user_task_pairs':sum(p['selected_pair'] for p in pairs),
           'selected_sessions':sum(p['session_count'] for p in pairs if p['selected_pair']),
           'selected_gestures':sum(p['total_usable_gestures'] for p in pairs if p['selected_pair']),
           'users_with_at_least_one_selected_task':len(set(p['user_id'] for p in pairs if p['selected_pair'])),
           'by_task':{task:{'pairs_selected':sum(p['selected_pair'] for p in pairs if p['task']==task),
                            'pairs_considered':sum(p['task']==task for p in pairs),
                            'selected_sessions':sum(p['session_count'] for p in pairs if p['task']==task and p['selected_pair']),
                            'selected_gestures':sum(p['total_usable_gestures'] for p in pairs if p['task']==task and p['selected_pair'])}
                      for task in TASK_ORDER},
           'definitions':{
              'valid_gesture':'Reconstructed single-pointer physical DOWN->UP contact, duration >0, tap or swipe.',
              'portrait_touch':'Every touch row within the contact has Phone_orientation=0; unknown and landscape reject only that contact.',
              'strict_multimodal':'Portrait touch AND both motion streams have strict within-contact coverage, paired samples and all portrait orientation.',
              'selection':'Exactly required_sessions per user+activity; EACH meets its threshold. The four activity types are independent.',
              'denominator':'All valid reconstructed touch gestures within same session+activity, INCLUDING nonportrait gestures.',
              'no_sequence_requirement':True,
              'thresholds_are_exploratory':True,
              'sensor_clock_warning':'Sensor elapsedRealtimeNanos and touch uptimeMillis may have different origins; no automatic shift.'}}
    return stats


def summarize(rows,users,out,config,failed,selection):
    processed=[r for r in rows if r.get('status')=='processed']
    totals=Counter()
    by_category=defaultdict(Counter)
    for r in processed:
        for key in ('portrait_taps','portrait_swipes','portrait_taps_both_strict_paired',
            'portrait_swipes_both_strict_paired','portrait_taps_both_strict_paired_portrait_motion',
            'portrait_swipes_both_strict_paired_portrait_motion','portrait_taps_both_relaxed_paired',
            'portrait_swipes_both_relaxed_paired','portrait_taps_both_any','portrait_swipes_both_any',
            'portrait_taps_failed','portrait_swipes_failed',
            'portrait_taps_both_relaxed_paired_portrait_motion','portrait_swipes_both_relaxed_paired_portrait_motion',
            'portrait_taps_fail_few_samples','portrait_swipes_fail_few_samples',
            'portrait_taps_fail_edge_gap','portrait_swipes_fail_edge_gap',
            'portrait_taps_fail_internal_gap','portrait_swipes_fail_internal_gap',
            'portrait_taps_fail_insufficient_paired_samples','portrait_swipes_fail_insufficient_paired_samples',
            'all_completed_contacts',
            'portrait_contacts','raw_touch_rows_in_scope','key_rows_in_scope','key_down_rows','key_up_rows',
            'key_down_matched_event','key_up_matched_event','key_down_matched_system','key_up_matched_system',
            'key_down_matched_relative_direct','key_up_matched_relative_direct',
            'key_completed_pairs','key_pairs_both_endpoint_events_matched','key_pairs_same_physical_touch_contact',
            'key_down_exact_match','key_up_exact_match',
            'key_pairs_up_record_appears_before_down_record',
            'key_pairs_up_systime_before_down_systime','key_pairs_interval_any_matched',
            'key_pairs_interval_at_least_10ms_matched','key_pairs_interval_at_least_25pct_matched',
            'key_pairs_interval_at_least_50pct_matched',
            'key_pairs_interval_50pct_shifted_negative_control',
            'accel_inside','accel_between','gyro_inside','gyro_between'):
            val=int(r.get(key,0) or 0)
            totals[key]+=val
            for label in str(r.get('task','')).split(';'):
                if label in TASK_ORDER:by_category[label][key]+=val
    csv_write(out/'sessions.csv',[{k:v for k,v in r.items() if k!='session_task_counts'} for r in rows])
    # Gather per-activity clock audit and per-gesture failures for convenient sharing.
    clocks=[];failures=[]
    for r in processed:
        p=r.get('clock_diagnostics_file')
        if p and Path(p).exists():
            with open(p,encoding='utf-8',newline='') as f:
                clocks.extend(csv.DictReader(f))
        p=r.get('failed_gesture_diagnostics_file')
        if p and Path(p).exists():
            with gzip.open(p,'rt',encoding='utf-8',newline='') as f:
                failures.extend(dict(x) for x in csv.DictReader(f))
    csv_write(out/'clock_diagnostics.csv',clocks)
    csv_write(out/'gesture_coverage_failures.csv',failures)
    quality_fields=['user_id','session_number','task','touch_source_file',
        'portrait_swipes','portrait_swipes_both_any','portrait_swipes_both_strict_paired',
        'portrait_swipes_both_relaxed_paired','portrait_swipes_failed',
        'portrait_swipes_strict_coverage_pct','portrait_swipes_relaxed_coverage_pct','portrait_swipes_uncovered_pct',
        'portrait_swipes_fail_few_samples','portrait_swipes_fail_edge_gap',
        'portrait_swipes_fail_internal_gap','portrait_swipes_fail_insufficient_paired_samples',
        'portrait_taps','portrait_taps_both_any','portrait_taps_both_strict_paired',
        'portrait_taps_both_relaxed_paired','portrait_taps_failed',
        'portrait_taps_strict_coverage_pct','portrait_taps_relaxed_coverage_pct','portrait_taps_uncovered_pct',
        'portrait_taps_fail_few_samples','portrait_taps_fail_edge_gap',
        'portrait_taps_fail_internal_gap','portrait_taps_fail_insufficient_paired_samples',
        'activity_sensor_clock_incompatibilities','warnings']
    csv_write(out/'sensor_coverage_diagnostics.csv',processed,fields=quality_fields)
    key_fields=['user_id','session_number','task','touch_source_file','touch_im_used',
                'key_clock_basis','touch_clock_basis',
                'key_rows_in_scope','key_down_rows','key_up_rows','key_completed_pairs',
                'key_pairs_up_record_appears_before_down_record',
                'key_pairs_up_systime_before_down_systime',
                'key_down_exact_match','key_up_exact_match',
                'key_down_matched_event','key_up_matched_event',
                'key_pairs_both_endpoint_events_matched','key_pairs_same_physical_touch_contact',
                'key_pairs_interval_any_matched','key_pairs_interval_at_least_10ms_matched',
                'key_pairs_interval_at_least_25pct_matched','key_pairs_interval_at_least_50pct_matched',
                'key_pairs_interval_50pct_shifted_negative_control',
                'key_interval_pairs_with_any_candidate','key_interval_pairs_with_multiple_candidates',
                'key_interval_touches_with_multiple_candidate_keys',
                'key_interval_touch_minus_key_down_median_ms','key_interval_touch_minus_key_up_median_ms',
                'key_press_down_system_lag_median_minus_up_ms',
                'raw_touch_actions_0','raw_touch_actions_1','raw_complete_contacts',
                'raw_interrupted_down','raw_orphan_up','raw_unclosed_down']
    csv_write(out/'key_matching_diagnostics.csv',
              [r for r in processed if int(r.get('key_rows_in_scope',0) or 0)>0],fields=key_fields)
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
            'touch_source_selection':{
                'sessions_using_TouchEvent_im':sum(r.get('touch_im_used',0)==1 for r in processed),
                'sessions_using_TouchEvent':sum(r.get('touch_source_file')=='TouchEvent.csv' for r in processed),
                'sessions_missing_both_touch_files':sum(not r.get('touch_source_file') for r in processed),
                'sessions_having_both_files':sum(r.get('touch_im_available',0) and r.get('touch_standard_available',0) for r in processed),
            },
            'failed_users':failed,'processing_config':{k:v for k,v in config.items() if k!='root'},
            'clock_quality_summary':{
                'sessions_with_eventtime_sensor_touch_disjoint':sum(bool(r.get('activity_sensor_clock_incompatibilities')) for r in processed),
                'sessions_with_possible_clock_origin_offset_warning':sum(any(str(x).startswith('possible_sensor_touch_clock_origin_offset:') for x in r.get('warnings',[])) for r in processed),
                'sensor_eventtime_unit':'nanoseconds / 1_000_000 = milliseconds',
                'touch_eventtime_unit':'milliseconds',
                'clock_alignment':'direct_eventtime_only_no_automatic_shift',
            },
            'totals':dict(totals),'by_activity':{k:dict(by_category[k]) for k in TASK_ORDER},
            'users_with_at_least_N_sessions_of_min_M_primary_gestures':eligibility,
            'orientation_definition':'raw row-weighted portrait % among recognized 0/1/3 orientations, all four sources combined; per-source matrices also provided',
            'gesture_definition':'physical raw touch DOWN -> UP, using TouchEvent_im.csv preferentially or TouchEvent.csv otherwise (never combined); one contact, single pointer, 0 portrait for EVERY row; 30px displacement or 50px path -> swipe otherwise tap; duration >12000ms flagged but NOT automatically discarded, like FETA v1.1',
            'matching_definition':'Physical key events use PressTime (ms); touch events use EventTime (ms); sensor events use EventTime (ns/1e6) DIRECTLY. Systime never used in gesture-to-sensor matching, only audit of clock origins and session chronology.',
            'strict_coverage_definition':'Same FETA v1.1 thresholds: >=3 per sensor, max endpoint gap 40ms, max internal sensor gap 120ms, >=1 accel/gyro pair <=25ms; ONLY in-gesture EVENT timestamps. Android touch uptimeMillis vs sensor elapsedRealtimeNanos clock-origin mismatch is possible; no implicit shift.',
            'relaxed_diagnostic_definition':'Alternative-only sensitivity: min sensor samples and pairs same; edge <= diagnostic_edge_gap_ms and internal gaps <= diagnostic_sensor_gap_ms. Does not replace strict counts.',
            'notes':['No map-navigation session is included.','TouchEvent_im.csv is preferred if present in the same session folder; alternative TouchEvent.csv is not merged or counted. This undocumented variant needs provenance auditing.', 'Matrix task order uses earliest Activity.Start_time within each user and task, NOT global session numbering.',
                     'Portrait fraction is NOT the same as percent of time in portrait: high frequency streams dominate row-weighted totals.',
                     'If KeyPressEvent press types differ from 0/1 in raw data, per-session keypress_type_values diagnose the mismatch.',
                     'Systime-to-EventTime offsets are only a clock-origin sanity check; direct sensor and touch timebases may differ after deep sleep. See clock_diagnostics.csv.', 'See sensor_coverage_diagnostics.csv and gesture_coverage_failures.csv for rejection reasons.']}
    report['per_gesture_portrait_selection'] = build_selection_outputs(processed,users,out,selection)
    json_write(out/'summary.json',report)
    return report


def cli():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--input',type=Path,default=Path('input_data/hmog_unarchived'))
    a.add_argument('--output',type=Path,default=Path('output_data/hmog_portrait_selection_v1_5'))
    a.add_argument('--workers',type=int,default=max(1,min(4,(os.cpu_count() or 2)//2)))
    a.add_argument('--limit-users',type=int,default=0)
    a.add_argument('--no-cache',action='store_true')
    a.add_argument('--selection-mode',choices=('strict','touch'),default='strict',
                   help='strict: portrait touch + strict paired portrait accelerometer & gyroscope (default); touch: portrait touch only')
    a.add_argument('--gesture-count-mode',choices=('all','primary'),default='all',
                   help='all counts taps+swipes; primary counts swipes in reading and taps in writing')
    a.add_argument('--min-reading-gestures',type=int,default=20,
                   help='Minimum qualifying gestures in EACH of 4 reading sessions (exploratory)')
    a.add_argument('--min-writing-gestures',type=int,default=100,
                   help='Minimum qualifying gestures in EACH of 4 writing sessions (exploratory)')
    a.add_argument('--required-sessions',type=int,default=4)
    a.add_argument('--swipe-distance-px',type=float,default=30.)
    a.add_argument('--swipe-path-px',type=float,default=50.)
    a.add_argument('--max-contact-ms',type=float,default=12000.)
    a.add_argument('--min-sensor-samples',type=int,default=3)
    a.add_argument('--max-edge-gap-ms',type=float,default=40.)
    a.add_argument('--max-sensor-gap-ms',type=float,default=120.)
    a.add_argument('--pair-tolerance-ms',type=float,default=25.)
    a.add_argument('--diagnostic-edge-gap-ms',type=float,default=80.)
    a.add_argument('--diagnostic-sensor-gap-ms',type=float,default=200.)
    a.add_argument('--clock-offset-warning-ms',type=float,default=250.)
    a.add_argument('--diagnostic-max-failed-gestures-per-session',type=int,default=200)
    a.add_argument('--min-sensor-pairs',type=int,default=1)
    a.add_argument('--key-match-tolerance-ms',type=float,default=80.)
    a.add_argument('--max-key-hold-ms',type=float,default=3000.)
    a.add_argument('--key-diagnostic-examples',type=int,default=20)
    a.add_argument('--key-negative-control-shift-ms',type=float,default=5000.)
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
    selection={k:getattr(args,k) for k in ('selection_mode','gesture_count_mode',
                  'min_reading_gestures','min_writing_gestures','required_sessions')}
    if min(selection['min_reading_gestures'],selection['min_writing_gestures'])<0 or selection['required_sessions']<1:
        log.error('Selection thresholds must be >=0 and required sessions >=1')
        return 2
    # Analysis cache is intentionally INDEPENDENT of selection thresholds/policies.
    # Changing a minimum or strict/touch mode only rebuilds reports from cached gestures.
    config={k:v for k,v in vars(args).items() if k not in (
        'input','output','workers','limit_users','no_cache',*selection)}
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
    report=summarize(results,list(users),out,config,failed,selection)
    (out/'HOW_TO_READ_SELECTION.md').write_text('''# HMOG individual portrait gestures and user-task selection (v1.5)

- `session_task_gestures.csv`: one row for each processed user+session+activity.
  `valid_gestures` counts all reconstructed single-pointer taps+swipes (portrait
  and landscape). `portrait_gestures`, `portrait_taps`, `portrait_swipes` count
  only contacts where EVERY touch row has orientation 0. `portrait_share_pct_valid`
  is 100 * `portrait_gestures` / `valid_gestures`.
  `strict_multimodal_gestures` additionally requires strict paired accel+gyro
  sampling and all portrait motion readings within each contact.
- `selection_count`: based on --selection-mode (strict|touch) and
  --gesture-count-mode (all|primary). `primary` = reading swipes or writing taps.
- `user_task_pairs.csv`: exactly four distinct sessions for each user+task,
  EACH with selection_count >= the task-specific minimum. Other tasks do not matter.
- `selected_user_task_pairs.csv`: all selected (user,activity) pairs; use this to
  build your authentication experiment cohorts.
- `selected_session_tasks.csv`: selected session manifests with path to
  `cache/gestures/...csv.gz`. Filter each file by matching `task` and
  `portrait_multimodal_strict==1` or `portrait_single_valid==1`, depending
  on the selected mode; apply primary-type filter if requested.
- `selection_counts_matrix.tsv`, `selection_share_pct_matrix.tsv`:
  per-user chronological session counts / proportions, four slots per task.
- `eligibility_threshold_sweep.csv`: eligibility counts under exploratory
  minimums without recomputing gesture reconstruction.
- `summary.json`: full legacy statistics and `per_gesture_portrait_selection`.
- `cache/sessions`: per-session JSON with source-file fingerprint and per-task
  counts. If source files or analysis parameters change, cache is invalidated.

IMPORTANT: A portrait touch contact is NOT necessarily supported by sufficient
accelerometer+gyroscope data. Default selection `strict` requires all three
modalities (and motion orientation 0); choose --selection-mode touch to study
orientation only. No uninterrupted portrait run is required. Percentages in
`session_task_gestures.csv` are gesture-weighted, not sensor-row weighted.

Thresholds 20 (reading) and 100 (writing) are exploratory, NOT a validated
training/validation/testing protocol. Establish minimum sample sizes for each
split BEFORE finalizing the cohort. If gesture types need separate classifiers,
consider --gesture-count-mode primary.
''',encoding='utf-8')
    log.info('Selection: %d eligible user-task pairs across %d users; mode=%s; gesture_types=%s',
             report['per_gesture_portrait_selection']['selected_user_task_pairs'],len(users),
             selection['selection_mode'],selection['gesture_count_mode'])
    log.info('DONE users=%d; processed=%d; map_skipped=%d; cached=%d',len(users),report['sessions_processed'],report['sessions_ignored_map_or_other'],hits)
    print('\n=== SHARE BACK THESE RESULTS ===')
    print(json.dumps({'sessions_processed':report['sessions_processed'],
                      'sessions_ignored_map_or_other':report['sessions_ignored_map_or_other'],
                      'totals':report['totals'],'by_activity':report['by_activity'],
                      'failed_users':report['failed_users'],
                      'per_gesture_portrait_selection':report['per_gesture_portrait_selection']},ensure_ascii=False,indent=2))
    print('=== END ===\n')
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
