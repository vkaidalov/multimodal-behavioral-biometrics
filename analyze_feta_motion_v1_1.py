#!/usr/bin/env python3
"""FETA/Oxford raw touch and motion coverage audit (Python 3.14 compatible).

Expected layout: data_files/<USER>/<SESSION>/{scroll,swipe}/<REPETITION>/
{touch_data,accelerometer_data,gyroscope_data}/*.csv

No interpolation, no fabricated sensor readings, no cross-user/session mixing.
All counts are measured within observed time ranges. Unknown schemas are reported
rather than silently treated as zero coverage.
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
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = 'feta-motion-audit-v1.1.0-20261108-explicit-type'
KINDS = {'touch_data': 'touch', 'accelerometer_data': 'accel', 'gyroscope_data': 'gyro'}
TOUCH_FIELDS = [
    'user_id','session','task','repetition','file','gesture_index','source','phase_mode',
    'start_ns','end_ns','duration_ms','touch_samples','gesture_kind','displacement_px','path_length_px',
    'accel_samples','gyro_samples','accel_edge_max_ms','gyro_edge_max_ms',
    'accel_max_gap_ms','gyro_max_gap_ms','accel_gyro_pairs',
    'accel_any','gyro_any','both_any','accel_strict','gyro_strict','both_strict',
    'both_strict_and_paired','time_compatible','notes'
]
REPETITION_FIELDS = [
    'user_id','session','task','repetition','touch_files','accel_files','gyro_files',
    'touch_rows','accel_rows','gyro_rows','touch_parse_errors','accel_parse_errors','gyro_parse_errors',
    'touch_time_columns','accel_time_columns','gyro_time_columns',
    'touch_time_units','accel_time_units','gyro_time_units','touch_phase_columns','touch_type_counts',
    'gestures','gestures_swipe','gestures_tap','gestures_unknown_geometry','gestures_both_any','gestures_both_strict_paired',
    'swipes_both_any','swipes_both_strict_paired','taps_both_any','taps_both_strict_paired',
    'within_rep_accel_inside','within_rep_accel_between','within_rep_accel_before','within_rep_accel_after',
    'within_rep_gyro_inside','within_rep_gyro_between','within_rep_gyro_before','within_rep_gyro_after',
    'time_compatible','time_warning','gesture_warning'
]
TASK_FIELDS = [
    'user_id','session','task','repetitions','gestures','gestures_swipe','gestures_tap','gestures_unknown_geometry',
    'swipes_both_any','swipes_both_strict_paired','taps_both_any','taps_both_strict_paired',
    'accel_samples','gyro_samples','gestures_both_any','gestures_both_strict_paired',
    'accel_inside','accel_between','accel_before','accel_after','accel_unclassified',
    'gyro_inside','gyro_between','gyro_before','gyro_after','gyro_unclassified',
    'sensor_between_gap_seconds','repetitions_clock_overlap','clock_compatible','warnings'
]


def normalize(name):
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def is_number(value):
    try:
        v = float(str(value).strip().strip('\ufeff'))
        return math.isfinite(v)
    except (ValueError, TypeError):
        return False


def time_field(header, forced=None):
    if forced:
        for idx, h in enumerate(header):
            if normalize(h) == normalize(forced):
                return idx
        raise ValueError(f'Requested timestamp column {forced!r} not in {header}')
    candidates = {'timestamp', 'timestamps','time','ts','timeepoch','unixtime','eventtime',
                  'sampletime','touchstarttime','touchendtime','touchesbegan','datetime',
                  'createdat','recordedat','sensorclocktime','date','t','timestampms',
                  'timeinms','timestampns','timeinseconds','elapsedtime','seconds',
                  'timeinterval','eventtimestamp','touchdatetime'}
    for i,h in enumerate(header):
        if normalize(h) in candidates:
            return i
    for i,h in enumerate(header):
        if 'timestamp' in normalize(h) or normalize(h).endswith('time'):
            return i
    raise ValueError(f'No recognizable timestamp field: {header}')


def phase_field(header):
    for i,h in enumerate(header):
        if normalize(h) in ('phase','touchphase','eventtype','touchstate','state','touchstatus',
                            'action','actiontype','touchaction','eventphase','type','touchtype'):
            return i, normalize(h)
    return None, ''


def raw_numeric_scale(values, kind):
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return 1.0, 'none'
    value = float(np.median(np.abs(finite)))
    # Unix epoch magnitudes (seconds / milliseconds / microseconds / nanoseconds).
    if value > 1e17: return 1., 'ns_epoch'
    if value > 1e14: return 1e3, 'us_epoch'
    if value > 1e11: return 1e6, 'ms_epoch'
    if value > 1e8: return 1e9, 's_epoch'
    diffs = np.diff(finite)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs):
        d = float(np.median(diffs))
        # For touch, a file may contain only 2 samples and span 0.2 s.
        if d >= 1e6: return 1., 'ns_relative'
        if d >= 1e3: return 1e3, 'us_relative'
        if d >= 1.: return 1e6, 'ms_relative'
        return 1e9, 's_relative'
    # Numeric relative times with no detectable time differences: cannot infer.
    return 1e9, 's_guess_no_deltas'


def parse_iso_ns(value):
    s = str(value).strip().strip('"').strip("'")
    if not s: return None
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Handle a colon before milliseconds occasionally found in legacy datasets.
        s2 = re.sub(r'(\d{2}:\d{2}:\d{2}):(\d{1,6})$', r'\1.\2', s)
        try: dt = datetime.fromisoformat(s2)
        except ValueError: return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1e9))


def read_times(path, kind, args, sample_schemas):
    """Read a single CSV, retaining only timestamp and (touch) phase columns."""
    p = Path(path)
    result = {'rows':0,'invalid':0,'timestamp_col':'','timestamp_unit':'',
              'phase_col':'','time_ns':np.empty(0,dtype=np.int64), 'phases':[], 'coords':[],
              'error':'','header':[]}
    try:
        with p.open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try: dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error: dialect = csv.excel
            reader = csv.reader(handle, dialect)
            first = next(reader, None)
            if first is None:
                result['error']='empty_file'
                return result
            first = [str(c).strip().strip('\ufeff') for c in first]
            header_like = (sum(not is_number(c) for c in first) >= 2)
            if header_like:
                header = first
            else:
                # The repository documentation does not specify headerless data. Stop
                # rather than silently invent a timestamp-column position.
                result['error']='headerless_csv_columns_unknown'
                result['header']=first[:30]
                return result
            result['header'] = header[:30]
            tcol = time_field(header, getattr(args, f'{kind}_time_col'))
            pcol, pfield = phase_field(header) if kind == 'touch' else (None,'')
            normalized_header = [normalize(h) for h in header]
            xcol = normalized_header.index('x') if 'x' in normalized_header and kind == 'touch' else None
            ycol = normalized_header.index('y') if 'y' in normalized_header and kind == 'touch' else None
            result['timestamp_col'] = header[tcol]
            result['phase_col'] = header[pcol] if pcol is not None else ''
            raw = []; phases=[]; coords=[]
            for row in reader:
                if not row or all(not x.strip() for x in row): continue
                result['rows'] += 1
                if len(row) <= tcol:
                    result['invalid'] += 1; continue
                value = row[tcol].strip()
                if not value:
                    result['invalid'] += 1; continue
                raw.append(value)
                phases.append(row[pcol].strip() if pcol is not None and len(row)>pcol else '')
                if xcol is not None and ycol is not None and len(row) > max(xcol,ycol):
                    try: coords.append((float(row[xcol]),float(row[ycol])))
                    except ValueError: coords.append((float('nan'),float('nan')))
                else:
                    coords.append((float('nan'),float('nan')))
        if not raw:
            result['error']='no_timestamp_values'
            return result
        sample_val = raw[0]
        numeric = is_number(sample_val)
        if numeric:
            ns_data = []; valid_phases=[]; valid_coords=[]; numeric_vals=[]
            for val, ph, xy in zip(raw,phases,coords):
                if is_number(val):
                    numeric_vals.append(float(val)); valid_phases.append(ph); valid_coords.append(xy)
                else: result['invalid'] += 1
            if not numeric_vals:
                result['error']='unreadable_numeric_times'; return result
            forced = args.time_unit
            if forced=='auto': scale,unit=raw_numeric_scale(numeric_vals,kind)
            else:
                scale = {'s':1e9,'ms':1e6,'us':1e3,'ns':1.0}[forced]; unit=forced+'_forced'
            ns_data = [round(v*scale) for v in numeric_vals]
            result['timestamp_unit']=unit
            phases=valid_phases; coords=valid_coords
        else:
            ns_data = []; valid_phases=[]; valid_coords=[]
            for val, ph, xy in zip(raw, phases, coords):
                n = parse_iso_ns(val)
                if n is not None:
                    ns_data.append(n); valid_phases.append(ph); valid_coords.append(xy)
                else: result['invalid'] += 1
            if not ns_data:
                result['error']='unreadable_datetime_values'; return result
            phases=valid_phases; coords=valid_coords
            result['timestamp_unit']='datetime_iso'
        result['time_ns']=np.asarray(ns_data,dtype=np.int64)
        result['phases']=phases
        result['coords']=coords
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {str(exc)[:180]}'
    return result


def interpret_phase(value, mode, col):
    """Interpret FETA's documented 0=DOWN, 1=MOVE, 2=UP.

    Do not mistake this for the Android MotionEvent action codes or UIKit's
    general-purpose UITouchPhase (where 2 means stationary).
    """
    s = str(value).strip().lower()
    if s in ('began','begin','down','start','touchbegan','touchdown','pressed'):
        return 'begin'
    if s in ('ended','end','up','release','touchended','touchup','released'):
        return 'end'
    if s in ('moved','move','changed','drag','stationary'):
        return 'move'
    if s in ('cancelled','canceled','cancel'):
        return 'cancel'
    if s in ('','nan','none','null'):
        return 'unknown'
    try:
        v=int(float(s))
    except (ValueError,TypeError):
        return 'unknown'
    if mode=='feta' or (mode=='auto' and col in ('type','touchtype')):
        return {0:'begin',1:'move',2:'end'}.get(v,'unknown')
    if mode=='android' or (mode=='auto' and ('action' in col)):
        masked = v & 255
        return {0:'begin',1:'end',2:'move',3:'cancel',5:'multitouch',6:'multitouch'}.get(masked,'unknown')
    # UIKit UITouchPhase: 0 began, 1 moved, 2 stationary, 3 ended, 4 cancelled.
    return {0:'begin',1:'move',2:'move',3:'end',4:'cancel'}.get(v,'unknown')


def touch_file_gestures(parsed, path, args):
    """Segment real contacts, not a 250-ms pause heuristic.

    Only DOWN->UP with valid timestamps counts as a verified contact. No
    sampling time or touch point is invented. Each contact has an optional
    *geometric* tap/swipe label with configurable thresholds; it is not the
    research authors' original swipe-classification algorithm.
    """
    t=np.asarray(parsed['time_ns'],dtype=np.int64)
    if not len(t): return [], 'no_timestamps'
    phases=parsed['phases']
    coords=parsed['coords']
    pcol=normalize(parsed['phase_col'])
    if not parsed['phase_col']:
        return [], 'touch_type_column_missing_NO_HEURISTIC_SEGMENTATION'
    if all(not str(x).strip() for x in phases):
        return [], 'touch_type_values_missing_NO_HEURISTIC_SEGMENTATION'
    events=sorted(zip(t,phases,coords), key=lambda x:x[0])
    gestures=[]; active=None
    counts=Counter()
    for at,raw,xy in events:
        ph=interpret_phase(raw,args.phase_mode,pcol)
        counts[ph]+=1
        if ph=='begin':
            if active is not None: counts['interrupted_by_down']+=1
            active=dict(start=int(at),samples=1,startxy=xy,lastxy=xy,path=0.0)
        elif ph=='move':
            if active is not None:
                active['samples']+=1
                previous=active['lastxy']
                if all(math.isfinite(q) for q in (*previous,*xy)):
                    active['path']+=math.hypot(xy[0]-previous[0],xy[1]-previous[1])
                active['lastxy']=xy
        elif ph=='end':
            if active is not None and int(at)>active['start']:
                previous=active['lastxy']
                if all(math.isfinite(q) for q in (*previous,*xy)):
                    active['path']+=math.hypot(xy[0]-previous[0],xy[1]-previous[1])
                startxy=active['startxy']
                displacement=(math.hypot(xy[0]-startxy[0],xy[1]-startxy[1])
                              if all(math.isfinite(q) for q in (*startxy,*xy)) else float('nan'))
                path_length=active['path']
                geometric=(math.isfinite(displacement) and math.isfinite(path_length))
                kind=('swipe' if (displacement>=args.swipe_distance_px or
                       path_length>=args.swipe_path_px) else 'tap') if geometric else 'unknown_geometry'
                duration_ms=(int(at)-active['start'])/1e6
                notes='duration_exceeds_max_contact' if duration_ms>args.max_contact_ms else ''
                gestures.append((active['start'],int(at),active['samples']+1,
                                 'feta_down_move_up',notes,kind,displacement,path_length))
            else: counts['orphan_up']+=1
            active=None
        elif ph in ('cancel','multitouch'):
            active=None
        else:
            counts['unrecognized_or_blank']+=1
    if active is not None: counts['unfinished_down']+=1
    warnings=[]
    for kind in ('orphan_up','unfinished_down','interrupted_by_down','unknown','unrecognized_or_blank','cancel','multitouch'):
        if counts[kind]: warnings.append(f'{kind}={counts[kind]}')
    if not gestures:
        warnings.append('no_completed_down_up_contacts')
    return gestures, ';'.join(warnings)


def files_of(rep, foldername):
    folder=rep / foldername
    if not folder.is_dir(): return []
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower()=='.csv')


def detect_tasks(user_dir):
    for session in sorted(p for p in user_dir.iterdir() if p.is_dir()):
        for task in ('scroll','swipe'):
            task_dir=session / task
            if task_dir.is_dir():
                reps=[p for p in task_dir.iterdir() if p.is_dir()]
                if reps: yield session.name, task, sorted(reps,key=lambda p:p.name)


def load_repetition(user,session,task,rep,args):
    row = {k:'' for k in REPETITION_FIELDS}
    row.update(user_id=user, session=session,task=task,repetition=rep.name)
    gestures=[]; sensors={}; times={}; bad=[]; intervals=[]
    for kind in ('touch','accel','gyro'):
        files=files_of(rep, {'touch':'touch_data','accel':'accelerometer_data','gyro':'gyroscope_data'}[kind])
        row[kind+'_files']=len(files)
        times_of_kind=[]
        for path in files:
            result=read_times(path,kind,args,None)
            row[kind+'_rows'] = int(row[kind+'_rows'] or 0) + result['rows']
            row[kind+'_parse_errors'] = int(row[kind+'_parse_errors'] or 0)+result['invalid']
            if result['error']:
                bad.append(f'{kind}:{path.name}:{result["error"]}')
                continue
            for col, value in ((kind+'_time_columns',result['timestamp_col']),
                               (kind+'_time_units',result['timestamp_unit'])):
                values=set(filter(None,str(row[col]).split('|')))
                values.add(value)
                row[col]='|'.join(sorted(values))
            if kind=='touch' and result['phase_col']:
                values=set(filter(None,str(row['touch_phase_columns']).split('|')))
                values.add(result['phase_col'])
                row['touch_phase_columns']='|'.join(sorted(values))
            if kind=='touch':
                type_counter=Counter(str(v).strip() for v in result['phases'])
                previous=Counter()
                for part in str(row['touch_type_counts']).split('|'):
                    if ':' in part:
                        k,v=part.rsplit(':',1)
                        try: previous[k]+=int(v)
                        except ValueError: pass
                previous.update(type_counter)
                row['touch_type_counts']='|'.join(f'{k}:{previous[k]}' for k in sorted(previous))
                g,warning=touch_file_gestures(result,path,args)
                for j,(a,b,n,src,note,kind_name,displacement,path_length) in enumerate(g):
                    gestures.append(dict(user_id=user,session=session,task=task,
                        repetition=rep.name,file=path.name,gesture_index=j,
                        source=src,phase_mode=args.phase_mode if result['phase_col'] else 'none',
                        start_ns=a,end_ns=b,duration_ms=(b-a)/1e6,touch_samples=n,
                        gesture_kind=kind_name,displacement_px=displacement,
                        path_length_px=path_length,notes=note))
                if warning: bad.append(f'touch:{path.name}:{warning}')
            times_of_kind.append(result['time_ns'])
        sensors[kind]=np.sort(np.concatenate(times_of_kind)) if times_of_kind else np.empty(0,dtype=np.int64)
    for kind in ('touch','accel','gyro'):
        if '|' in str(row[kind+'_time_units']):
            bad.append(f'{kind}:mixed_detected_timestamp_units')
    row['gesture_warning']=';'.join(bad[:8])
    row['gestures']=len(gestures)
    for category in ('swipe','tap','unknown_geometry'):
        row['gestures_'+category]=sum(g['gesture_kind']==category for g in gestures)
    return row,gestures,sensors


def overlap_and_compat(gestures,accel,gyro):
    if len(gestures)==0 or not len(accel) or not len(gyro):
        return False, 'no_gesture_or_sensor_data'
    gmin=min(g['start_ns'] for g in gestures)
    gmax=max(g['end_ns'] for g in gestures)
    # Bounds not necessarily overlapping when sensors stop mid-task; mark issue.
    intersects_a = accel[0]<=gmax and accel[-1]>=gmin
    intersects_g = gyro[0]<=gmax and gyro[-1]>=gmin
    if not intersects_a or not intersects_g:
        return False,'sensor_touch_time_ranges_disjoint'
    return True,''


def sensor_for_gesture(sensor,start,end,args):
    if sensor.size==0: return 0,float('nan'),float('nan'),False,np.empty(0,dtype=np.int64)
    a=int(np.searchsorted(sensor,start,side='left'))
    b=int(np.searchsorted(sensor,end,side='right'))
    part=sensor[a:b]; n=len(part)
    if not n: return 0,float('nan'),float('nan'),False,part
    edge_max=max((int(part[0])-start)/1e6,(end-int(part[-1]))/1e6)
    max_gap=float(np.max(np.diff(part))/1e6) if n>=2 else float('nan')
    strict=(n>=args.min_sensor_samples and edge_max<=args.max_edge_gap_ms
            and (n<2 or max_gap<=args.max_sensor_gap_ms))
    return n,edge_max,max_gap,strict,part


def paired_samples(a,b,tolerance_ns):
    i=j=matched=0
    while i<len(a) and j<len(b):
        d=int(a[i])-int(b[j])
        if abs(d)<=tolerance_ns:
            matched+=1; i+=1; j+=1
        elif d<0: i+=1
        else: j+=1
    return matched


def tag_gestures(gestures,accel,gyro,compatible,args):
    for g in gestures:
        a=g['start_ns']; b=g['end_ns']
        an,ae,ag,astr,aa=sensor_for_gesture(accel,a,b,args)
        gn,ge,gg,gstr,ga=sensor_for_gesture(gyro,a,b,args)
        pairs=paired_samples(aa,ga,int(args.pair_tolerance_ms*1e6))
        g.update(accel_samples=an,gyro_samples=gn,accel_edge_max_ms=ae,
                 gyro_edge_max_ms=ge,accel_max_gap_ms=ag,gyro_max_gap_ms=gg,
                 accel_gyro_pairs=pairs,accel_any=int(an>0),gyro_any=int(gn>0),
                 both_any=int(an>0 and gn>0),accel_strict=int(astr),
                 gyro_strict=int(gstr),both_strict=int(astr and gstr),
                 both_strict_and_paired=int(astr and gstr and pairs>=args.min_sensor_pairs),
                 time_compatible=int(compatible))
        if not compatible:
            g['notes']=(g.get('notes','')+';sensor_touch_ranges_disjoint').strip(';')
    return gestures


def classify_sensor_positions(sensor,gestures):
    """No gesture => unclassified, rather than falsely claiming 'between'."""
    n=len(sensor)
    if not gestures:
        return dict(inside=0,between=0,before=0,after=0,unclassified=n,gap_seconds=0.)
    intervals=sorted((g['start_ns'],g['end_ns']) for g in gestures)
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]: merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else: merged.append((start,end))
    inside=0; gaps=0.; between=0
    for start,end in merged:
        inside+=int(np.searchsorted(sensor,end,'right')-np.searchsorted(sensor,start,'left'))
    for (_,previous_end),(next_start,_) in zip(merged,merged[1:]):
        if next_start>previous_end:
            gaps+=(next_start-previous_end)/1e9
            between+=int(np.searchsorted(sensor,next_start,'left')-
                         np.searchsorted(sensor,previous_end,'right'))
    before=int(np.searchsorted(sensor,merged[0][0],'left'))
    after=n-int(np.searchsorted(sensor,merged[-1][1],'right'))
    unclassified=n-inside-between-before-after
    return dict(inside=inside,between=between,before=before,after=after,
                unclassified=unclassified,gap_seconds=gaps)


def analyze_task(user,session,task,reps,args):
    rep_rows=[]; gesture_rows=[]; sensor_reps=[]; warnings=[]
    for rep in reps:
        rr,gg,ss=load_repetition(user,session,task,rep,args)
        compatible,reason=overlap_and_compat(gg,ss['accel'],ss['gyro'])
        if reason: warnings.append(f'{rep.name}:{reason}')
        rr['time_compatible']=int(compatible)
        rr['time_warning']=reason
        tag_gestures(gg,ss['accel'],ss['gyro'],compatible,args)
        rr['gestures_both_any']=sum(g['both_any'] for g in gg)
        rr['gestures_both_strict_paired']=sum(g['both_strict_and_paired'] for g in gg)
        for group,label in (('swipe','swipes'),('tap','taps')):
            rr[label+'_both_any']=sum(g['both_any'] for g in gg if g['gesture_kind']==group)
            rr[label+'_both_strict_paired']=sum(g['both_strict_and_paired'] for g in gg if g['gesture_kind']==group)
        for name in ('accel','gyro'):
            cl=classify_sensor_positions(ss[name],gg)
            for label in ('inside','between','before','after'):
                rr[f'within_rep_{name}_{label}']=cl[label]
        rep_rows.append(rr)
        gesture_rows.extend(gg)
        sensor_reps.append((rep.name,ss))
    # Safe cross-repetition pooling only when non-overlapping clocks are observed.
    def task_clock_safe():
        spans=[]
        for name,ss in sensor_reps:
            vals=[v for key in ('touch','accel','gyro') for v in [ss[key]] if len(v)]
            if not vals: continue
            spans.append((int(min(v[0] for v in vals)),int(max(v[-1] for v in vals)),name))
        spans.sort()
        return all(right[0]>=left[1]-int(args.clock_overlap_tolerance_ms*1e6)
                   for left,right in zip(spans,spans[1:]))
    cross_clock=task_clock_safe()
    if cross_clock:
        merged_sensors={k:np.sort(np.concatenate([v[k] for _,v in sensor_reps]))
                        if sensor_reps else np.empty(0,dtype=np.int64)
                        for k in ('accel','gyro')}
        classif={k:classify_sensor_positions(merged_sensors[k],gesture_rows)
                 for k in ('accel','gyro')}
    else:
        warnings.append('repetitions_time_ranges_overlap;between_classification_within_repetitions_only')
        classif={}
        for kind in ('accel','gyro'):
            records=[classify_sensor_positions(s[kind],
                         [g for g in gesture_rows if g['repetition']==name])
                     for name,s in sensor_reps]
            classif[kind]={k:sum(r[k] for r in records) for k in
                           ('inside','between','before','after','unclassified','gap_seconds')}
    trow={k:'' for k in TASK_FIELDS}
    trow.update(user_id=user,session=session,task=task,repetitions=len(reps),
                gestures=len(gesture_rows),
                gestures_swipe=sum(g['gesture_kind']=='swipe' for g in gesture_rows),
                gestures_tap=sum(g['gesture_kind']=='tap' for g in gesture_rows),
                gestures_unknown_geometry=sum(g['gesture_kind']=='unknown_geometry' for g in gesture_rows),
                swipes_both_any=sum(g['both_any'] for g in gesture_rows if g['gesture_kind']=='swipe'),
                swipes_both_strict_paired=sum(g['both_strict_and_paired'] for g in gesture_rows if g['gesture_kind']=='swipe'),
                taps_both_any=sum(g['both_any'] for g in gesture_rows if g['gesture_kind']=='tap'),
                taps_both_strict_paired=sum(g['both_strict_and_paired'] for g in gesture_rows if g['gesture_kind']=='tap'),
                accel_samples=sum(len(ss['accel']) for _,ss in sensor_reps),
                gyro_samples=sum(len(ss['gyro']) for _,ss in sensor_reps),
                gestures_both_any=sum(g['both_any'] for g in gesture_rows),
                gestures_both_strict_paired=sum(g['both_strict_and_paired'] for g in gesture_rows),
                sensor_between_gap_seconds=classif['accel']['gap_seconds'],
                repetitions_clock_overlap=int(not cross_clock),
                clock_compatible=int(all(r['time_compatible'] for r in rep_rows)),
                warnings=';'.join(warnings[:12]))
    for kind in ('accel','gyro'):
        for label in ('inside','between','before','after','unclassified'):
            trow[kind+'_'+label]=classif[kind][label]
    return trow,rep_rows,gesture_rows


def fingerprint(files, args):
    h=hashlib.sha256()
    config={k:v for k,v in vars(args).items() if k not in ('workers','no_cache','verbose','limit_users')}
    h.update(json.dumps(config,sort_keys=True,default=str).encode())
    h.update(VERSION.encode())
    for path in sorted(files,key=lambda p:str(p)):
        try:
            st=path.stat()
            h.update(f'{path}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
        except OSError: h.update(f'MISSING:{path}'.encode())
    return h.hexdigest()


def save_csv(path,records,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


def user_worker(arg):
    user_path,args_dict,cache_dir=arg
    args=argparse.Namespace(**args_dict)
    user_path=Path(user_path); cache_dir=Path(cache_dir)
    all_files=[p for p in user_path.rglob('*.csv') if p.is_file()]
    fp=fingerprint(all_files,args)
    safe=hashlib.sha1(str(user_path).encode()).hexdigest()[:16]
    base=cache_dir/safe
    meta_path=base.with_suffix('.json')
    csv_path=base.with_suffix('.gestures.csv.gz')
    if not args.no_cache and meta_path.exists() and csv_path.exists():
        try:
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('fingerprint')==fp:
                return {'user':user_path.name,'cached':True,'meta':meta,'csv':str(csv_path)}
        except (ValueError,OSError): pass
    task_rows=[];rep_rows=[];gestures=[];issues=[]
    for sess,task,reps in detect_tasks(user_path):
        try:
            trow,rrow,grow=analyze_task(user_path.name,sess,task,reps,args)
            task_rows.append(trow);rep_rows.extend(rrow);gestures.extend(grow)
        except Exception as e:
            issues.append(f'{sess}/{task}: {type(e).__name__}: {str(e)[:200]}')
    with gzip.open(csv_path,'wt',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=TOUCH_FIELDS,extrasaction='ignore')
        writer.writeheader()
        for row in gestures:
            writer.writerow(row)
    meta=dict(user_id=user_path.name,version=VERSION,fingerprint=fp,task_rows=task_rows,
              repetition_rows=rep_rows,issues=issues,gestures=len(gestures))
    meta_path.write_text(json.dumps(meta,ensure_ascii=False),encoding='utf-8')
    return {'user':user_path.name,'cached':False,'meta':meta,'csv':str(csv_path)}


def setup_log(output,verbose=False):
    logger=logging.getLogger('feta');logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for old in logger.handlers[:]:
        old.close()
        logger.removeHandler(old)
    formatter=logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh=logging.FileHandler(output/'analysis.log',encoding='utf-8');fh.setFormatter(formatter)
    sh=logging.StreamHandler(sys.stdout);sh.setFormatter(formatter)
    logger.addHandler(fh);logger.addHandler(sh)
    return logger


def write_summary(output,user_metas,task_rows,rep_rows,issues):
    sums=defaultdict(int)
    for row in task_rows:
        for k in ('gestures','gestures_swipe','gestures_tap','gestures_unknown_geometry','gestures_both_any','gestures_both_strict_paired',
    'swipes_both_any','swipes_both_strict_paired','taps_both_any','taps_both_strict_paired',
                  'accel_samples','gyro_samples','accel_inside','accel_between','accel_before','accel_after',
                  'accel_unclassified','gyro_inside','gyro_between','gyro_before','gyro_after','gyro_unclassified'):
            sums[k]+=int(row[k] or 0)
    by_task={}
    for task in ('scroll','swipe'):
        rows=[r for r in task_rows if r['task']==task]
        by_task[task]={k:sum(int(r[k] or 0) for r in rows) for k in
                       ('gestures','gestures_swipe','gestures_tap','gestures_unknown_geometry','gestures_both_any','gestures_both_strict_paired',
    'swipes_both_any','swipes_both_strict_paired','taps_both_any','taps_both_strict_paired',
                        'accel_inside','accel_between','gyro_inside','gyro_between')}
        by_task[task].update(users=len({r['user_id'] for r in rows}),sessions=len(rows))
    schema_errors=Counter()
    for row in rep_rows:
        s=str(row.get('gesture_warning',''))
        if s:
            for err in s.split(';'):
                if 'No recognizable timestamp' in err: schema_errors['unrecognized_timestamp_column']+=1
                if 'headerless_csv' in err: schema_errors['headerless']+=1
    summary=dict(version=VERSION,users=len(user_metas),user_ids_with_data=len({r['user_id'] for r in task_rows}),
                 sessions_tasks=len(task_rows),repetitions=len(rep_rows),totals=dict(sums),
                 by_task=by_task,issues=issues[:100],schema_diagnostics=dict(schema_errors),
                 definitions={
                   'gesture_kind':'illustrative geometry label; exact physical contacts are DOWN->UP',
                   'inside':'readings whose timestamps lie within any observed gesture [start,end]',
                   'between':'readings strictly between two consecutive gesture intervals in same session/task, only with compatible repetition clocks',
                   'before_after':'readings before first / after last gesture of the analyzed scope; not necessarily continuous data collection outside the task',
                   'unclassified':'sensor readings without reliable gesture intervals',
                   'both_any':'each sensor has >=1 reading within gesture interval',
                   'both_strict_paired':'each sensor meets min count, edge and internal gap thresholds AND >= min time-matched accel/gyro pairs',
                   'clock_overlap':'repetition time ranges overlap; inter-repetition gaps excluded from between analysis'
                 })
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Audit FETA raw touch gesture coverage and inter-gesture motion streams.')
    p.add_argument('--input',type=Path,default=Path('input_data/data_files'))
    p.add_argument('--output',type=Path,default=Path('output_data/feta_motion_audit_v1_1'))
    p.add_argument('--workers',type=int,default=max(1,min(6,(os.cpu_count() or 2)-1)))
    p.add_argument('--limit-users',type=int,default=0,help='Fast diagnostic mode: first N users.')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--time-unit',choices=['auto','s','ms','us','ns'],default='auto')
    p.add_argument('--touch-time-col',default=None)
    p.add_argument('--accel-time-col',default=None)
    p.add_argument('--gyro-time-col',default=None)
    p.add_argument('--phase-mode',choices=['auto','feta','ios','android'],default='auto')
    p.add_argument('--swipe-distance-px',type=float,default=30.,help='Illustrative swipe displacement threshold')
    p.add_argument('--swipe-path-px',type=float,default=50.,help='Illustrative swipe path threshold')
    p.add_argument('--max-contact-ms',type=float,default=12000.,help='Warn for long physical contacts; do not silently discard')
    p.add_argument('--min-sensor-samples',type=int,default=3)
    p.add_argument('--max-edge-gap-ms',type=float,default=40.)
    p.add_argument('--max-sensor-gap-ms',type=float,default=120.)
    p.add_argument('--pair-tolerance-ms',type=float,default=25.)
    p.add_argument('--min-sensor-pairs',type=int,default=1)
    p.add_argument('--clock-overlap-tolerance-ms',type=float,default=0.)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if (args.input/'data_files').is_dir() and not any((args.input/u).is_dir() for u in ('scroll','swipe')):
        args.input=args.input/'data_files'
    if not args.input.is_dir():
        raise SystemExit(f'Input directory not found: {args.input.resolve()}')
    args.output.mkdir(parents=True,exist_ok=True)
    logger=setup_log(args.output,args.verbose)
    users=[p for p in args.input.iterdir() if p.is_dir()]
    users=sorted(users,key=lambda p:p.name)
    if args.limit_users: users=users[:args.limit_users]
    logger.info('FETA audit %s | input=%s | users=%d | workers=%d', VERSION,args.input,len(users),args.workers)
    logger.info('FETA type=0 DOWN, 1 MOVE, 2 UP; no heuristic splitting of missing-type CSVs. Check diagnostics.csv.')
    cache=args.output/'cache'/'users';cache.mkdir(parents=True,exist_ok=True)
    args_dict=vars(args).copy()
    tasks=[(str(user),args_dict,str(cache)) for user in users]
    entries=[];start=time.monotonic()
    if args.workers<=1:
        for n,task in enumerate(tasks,1):
            res=user_worker(task);entries.append(res)
            if n%10==0 or n==len(tasks): logger.info('Progress %d/%d users (cached=%s)',n,len(tasks),res['cached'])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(user_worker,task) for task in tasks]
            for n,f in enumerate(as_completed(futures),1):
                try: res=f.result();entries.append(res)
                except Exception as exc:
                    logger.exception('Worker failed: %s',exc);continue
                if n%10==0 or n==len(tasks):
                    logger.info('Progress %d/%d users (last=%s cached=%s)',n,len(tasks),res['user'],res['cached'])
    entries.sort(key=lambda x:x['user'])
    meta=[e['meta'] for e in entries]
    task_rows=[row for m in meta for row in m['task_rows']]
    rep_rows=[row for m in meta for row in m['repetition_rows']]
    issues=[{'user':m['user_id'],'message':msg} for m in meta for msg in m['issues']]
    save_csv(args.output/'sessions_tasks.csv',task_rows,TASK_FIELDS)
    save_csv(args.output/'repetitions.csv',rep_rows,REPETITION_FIELDS)
    schema_rows=[]
    for r in rep_rows:
        if r['gesture_warning'] or r['time_warning']:
            schema_rows.append({'user_id':r['user_id'],'session':r['session'],'task':r['task'],
                 'repetition':r['repetition'],'warning':r['gesture_warning'],
                 'clock_warning':r['time_warning']})
    save_csv(args.output/'diagnostics.csv',schema_rows,['user_id','session','task','repetition','warning','clock_warning'])
    logger.info('Writing gesture-level CSV from per-user caches...')
    with gzip.open(args.output/'gestures.csv.gz','wt',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=TOUCH_FIELDS);writer.writeheader()
        for entry in entries:
            with gzip.open(entry['csv'],'rt',newline='',encoding='utf-8') as inp:
                writer.writerows(csv.DictReader(inp))
    summary=write_summary(args.output,meta,task_rows,rep_rows,issues)
    text='\n'.join([
        '# FETA motion coverage audit',
        '',
        'Data hierarchy: user / session / (scroll|swipe) / repetition / modality files.',
        'FETA touch_data/type is interpreted as 0=DOWN, 1=MOVE, 2=UP.',
        'Each verified physical contact is segmented strictly from DOWN to UP.',
        'No arbitrary 250-ms gesture separation is used; missing type results in a warning.',
        'Geometric tap/swipe labels use configurable illustrative pixel thresholds, not FETA authors\' classifier.',
        'No readings outside a gesture are used as that gesture\'s measurements.',
        'No motion or touch samples are interpolated.',
        'Coverage is reported separately for geometrically classified swipes and taps.',
        'Each strict coverage result requires both sensors in the gesture interval, enough'
        'readings, small edge and internal gaps, and temporally paired samples.',
        'The sensor readings classified as between must be in a verified gap between',
        'consecutive gestures; before/after is separate, and missing touch data is not',
        'evidence of a gesture-free period.',
        'Timestamps are auto-detected and must be checked using diagnostics.csv,',
        'especially if sensors and touch have incompatible timestamp units or origins.',
        'If repetition clocks reset, classification is limited to individual repetitions.',
        'iPhone model metadata is NOT reliably present in this raw folder structure.',
        '',
        'Files:',
        '- summary.json: overall counts and definitions',
        '- sessions_tasks.csv: per user/session/task, including between-gesture sensor counts',
        '- repetitions.csv: repetition-level coverage and diagnostics',
        '- gestures.csv.gz: one row per provisional/confirmed gesture and motion coverage',
        '- diagnostics.csv: parsing and clock warnings',
        '- cache/users/: user-scoped cache invalidated by code/config/source timestamps',
    ])
    (args.output/'README_RESULTS.md').write_text(text,encoding='utf-8')
    logger.info('DONE in %.1fs | gestures=%d | both_any=%d | both_strict_paired=%d',
        time.monotonic()-start,summary['totals'].get('gestures',0),
        summary['totals'].get('gestures_both_any',0),summary['totals'].get('gestures_both_strict_paired',0))
    logger.info('Accel between gestures=%d; Gyro between gestures=%d',
        summary['totals'].get('accel_between',0),summary['totals'].get('gyro_between',0))
    logger.info('===== COPY BACK TO CHAT =====\n%s\n===== END =====',
                json.dumps({'users':summary['users'],'sessions_tasks':summary['sessions_tasks'],
                            'repetitions':summary['repetitions'],'totals':summary['totals'],
                            'by_task':summary['by_task']},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
