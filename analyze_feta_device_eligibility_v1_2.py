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

VERSION = 'feta-device-eligibility-v1.2.0-20261112'
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
    config={k:v for k,v in vars(args).items() if k not in ('workers','no_cache','verbose','limit_users','tables','min_gestures','thresholds','min_cohort_users','from_audit_dir','gesture_count_mode','user_meta_id_col','device_col','bridge_file','bridge_folder_col','bridge_meta_col')}
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
    p.add_argument('--output',type=Path,default=Path('output_data/feta_device_eligibility_v1_2'))
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
    p.add_argument('--tables', type=Path, default=Path('input_data/tables/tables'),
                   help='Metadata folder containing userdata.csv and optional linking tables')
    p.add_argument('--min-gestures', type=int, default=40,
                   help='Minimum multimodal gestures per task in each of four sessions; planning value')
    p.add_argument('--thresholds', type=str, default='5,10,15,20,25,30,40,50,60,75,100,125,150,200,250,300,400,500',
                   help='Comma-separated sweep of minimum gestures per session/task')
    p.add_argument('--gesture-count-mode', choices=['geometric_swipes','all_contacts'],
                   default='geometric_swipes',
                   help='Default excludes tap-like contacts in swipe/scroll tasks; use all_contacts as sensitivity analysis')
    p.add_argument('--min-cohort-users', type=int, default=30,
                   help='Descriptive cohort size target, NOT an optimum learned from EER')
    p.add_argument('--from-audit-dir', type=Path, default=None,
                   help='Reuse sessions_tasks.csv from v1.1 instead of parsing raw 8.6GB data')
    p.add_argument('--user-meta-id-col', default='', help='Override userdata.csv ID column matching raw user directory or bridge')
    p.add_argument('--device-col', default='', help='Override userdata.csv iPhone model column')
    p.add_argument('--bridge-file', default='', help='Override linking table CSV filename if indirect metadata join is needed')
    p.add_argument('--bridge-folder-col', default='', help='Column in bridge table containing raw user directory ID')
    p.add_argument('--bridge-meta-col', default='', help='Column in bridge table containing userdata ID')
    return p.parse_args(argv)


def audit_main(argv=None):
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





# ==========================================================================
# FETA user-to-device linkage and four-session eligibility audit.
# The raw gesture counter above is retained from the user's v1.1 program.
# ==========================================================================
import statistics

MODEL_CODES = {
    'iphone8,1':'iPhone 6s', 'iphone8,2':'iPhone 6s Plus',
    'iphone9,1':'iPhone 7','iphone9,3':'iPhone 7',
    'iphone9,2':'iPhone 7 Plus','iphone9,4':'iPhone 7 Plus',
    'iphone10,1':'iPhone 8','iphone10,4':'iPhone 8',
    'iphone10,2':'iPhone 8 Plus','iphone10,5':'iPhone 8 Plus',
    'iphone10,3':'iPhone X','iphone10,6':'iPhone X',
    'iphone11,2':'iPhone XS','iphone11,4':'iPhone XS Max','iphone11,6':'iPhone XS Max',
}
EXPECTED_MODELS = ('iPhone 6s','iPhone 6s Plus','iPhone 7','iPhone 7 Plus',
                   'iPhone 8','iPhone 8 Plus','iPhone X','iPhone XS','iPhone XS Max')


def cid(value):
    """Compare UUIDs with/without braces and hyphens; preserve all other IDs."""
    v = str(value or '').strip().strip('"').strip("'").strip()
    if not v or v.lower() in ('nan','none','null','na'): return ''
    # Full path in a metadata column may point to data_files/USER/SESSION/...
    if ('/' in v or '\\' in v) and not re.fullmatch(r'[\w.:-]+',v):
        parts = re.split(r'[\\/]+',v)
        # Avoid matching arbitrary path parts; favor raw UUIDs in path.
        for part in parts:
            if re.fullmatch(r'[0-9a-f]{8}-[0-9a-f-]{20,}',part,re.I):
                return re.sub(r'[-{}]','',part.lower())
        return ''
    return re.sub(r'[-{}]','',v.lower())


def classify_model(value, allow_bare=False):
    raw=str(value or '').strip()
    if not raw: return ''
    code=re.sub(r'\s+','',raw.lower())
    if code in MODEL_CODES:return MODEL_CODES[code]
    s=re.sub(r'[^a-z0-9+]+',' ',raw.lower()).strip()
    s=re.sub(r'\s+',' ',s)
    if s.startswith('apple '):s=s[6:]
    if s.startswith('iphone '):s=s[7:]
    elif s.startswith('iphone'):s=s[6:]
    elif not allow_bare:return ''
    s=s.strip().replace(' plus',' plus').replace('6s+','6s plus').replace('7+','7 plus').replace('8+','8 plus')
    opts={'6s':'iPhone 6s','6s plus':'iPhone 6s Plus',
          '7':'iPhone 7','7 plus':'iPhone 7 Plus',
          '8':'iPhone 8','8 plus':'iPhone 8 Plus',
          'x':'iPhone X','xs':'iPhone XS','xs max':'iPhone XS Max'}
    return opts.get(s,'')


def metadata_csv(path):
    """Read entire small CSV metadata table; no reliance on undocumented fixed schema."""
    with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
        first=f.read(4096);f.seek(0)
        try: dialect=csv.Sniffer().sniff(first,delimiters=',;\t')
        except csv.Error:dialect=csv.excel
        reader=csv.DictReader(f,dialect=dialect)
        if not reader.fieldnames:return [],[]
        fields=[str(h).strip().strip('\ufeff') for h in reader.fieldnames]
        rows=[]
        for row in reader:
            if row is None:continue
            rows.append({str(k).strip().strip('\ufeff'):str(v).strip() if v is not None else ''
                         for k,v in row.items() if k is not None})
    return fields,rows


def find_tables(root):
    root=Path(root)
    if not root.is_dir():raise ValueError(f'Metadata directory not found: {root}')
    found={}
    for file in sorted(root.rglob('*.csv')):
        # user-data CSV is mandatory; all other tables serve as optional bridge/order info
        try:
            if file.stat().st_size > 100_000_000: continue
            cols,rows=metadata_csv(file)
            if cols:found[str(file)]={'path':file,'columns':cols,'rows':rows}
        except (OSError,UnicodeError,csv.Error) as ex:
            print('WARNING cannot read table',file,type(ex).__name__,ex)
    return found


def choose_userdata(tables):
    cand=[t for t in tables.values() if t['path'].stem.lower()=='userdata']
    if len(cand)!=1:
        raise ValueError(f'Expected exactly one userdata.csv, found {len(cand)}. '
                         f'CSV files: {[str(t["path"]) for t in tables.values()]}')
    return cand[0]


def column_lookup(cols,requested):
    matches=[c for c in cols if normalize(c)==normalize(requested)]
    if len(matches)!=1:raise ValueError(f'Column {requested!r} does not uniquely exist: {cols}')
    return matches[0]


def detect_model_column(meta, requested=''):
    cols=meta['columns'];rows=meta['rows']
    if requested:
        col=column_lookup(cols,requested)
        values={classify_model(r.get(col,''),True) for r in rows}
        if not values.intersection(EXPECTED_MODELS):
            print('WARNING: forced model column has no recognizable iPhone model labels')
        return col
    candidates=[]
    for col in cols:
        norm=normalize(col)
        context=any(t in norm for t in ('model','device','phone','hardware','product'))
        yes=sum(bool(classify_model(r.get(col,''),context)) for r in rows)
        unique={classify_model(r.get(col,''),context) for r in rows}-set([''])
        if yes:
            candidates.append((yes+(len(rows)//3 if context else 0),yes, len(unique),col))
    candidates.sort(reverse=True)
    if not candidates:raise ValueError('No iPhone model column identified in userdata.csv; '
                                       'inspect metadata_columns.csv and use --device-col')
    return candidates[0][-1]


def unique_index(rows, col):
    ix=defaultdict(list)
    for i,row in enumerate(rows):
        key=cid(row.get(col,''))
        if key:ix[key].append(i)
    return {k:ids[0] for k,ids in ix.items() if len(ids)==1}


def candidate_id_cols(meta):
    return sorted(meta['columns'],key=lambda col:(0 if normalize(col) in ('id','uuid','userid','useruuid','identifier','userdataid','subjectid')
                                                    else 1, col))


def link_users(raw_users,metadata,tables,args):
    """Prefer direct UUID matching. Fallback: data_files UUID -> bridge CSV -> userdata key."""
    raw={cid(u):u for u in raw_users if cid(u)}
    if len(raw)!=len(raw_users):
        raise ValueError('Conflicting normalized raw user IDs: inspect data_files folder names')
    cand=[]
    for col in ([column_lookup(metadata['columns'],args.user_meta_id_col)] if args.user_meta_id_col
                else candidate_id_cols(metadata)):
        ix=unique_index(metadata['rows'],col)
        matches={uid:ix[uid] for uid in raw if uid in ix}
        cand.append(dict(method='direct',meta_col=col,mapping=matches,coverage=len(matches),
                         bridge_file='',bridge_folder_col='',bridge_meta_col=''))
    for table in tables.values():
        if table is metadata:continue
        if args.bridge_file and table['path'].name.lower()!=args.bridge_file.lower():continue
        for bcol in table['columns']:
            if args.bridge_folder_col and normalize(bcol)!=normalize(args.bridge_folder_col):continue
            # Data directory UUID -> bridge row. One user can have many rows, so allow duplicates.
            touched=defaultdict(list)
            for i,r in enumerate(table['rows']):
                key=cid(r.get(bcol,''))
                if key in raw:touched[key].append(i)
            if not touched:continue
            for mcol in candidate_id_cols(metadata):
                if args.user_meta_id_col and normalize(mcol)!=normalize(args.user_meta_id_col):continue
                mix=unique_index(metadata['rows'],mcol)
                for bmid in table['columns']:
                    if bmid==bcol:continue
                    if args.bridge_meta_col and normalize(bmid)!=normalize(args.bridge_meta_col):continue
                    if not args.bridge_meta_col and not any(x in normalize(bmid) for x in ('user','uuid','id','owner','subject')):
                        continue
                    mapping={}; conflicts=set()
                    for uid,indices in touched.items():
                        ids={mix[cid(table['rows'][i].get(bmid,''))]
                             for i in indices if cid(table['rows'][i].get(bmid,'')) in mix}
                        if len(ids)==1:mapping[uid]=next(iter(ids))
                        elif len(ids)>1:conflicts.add(uid)
                    if mapping:
                        cand.append(dict(method='bridge',meta_col=mcol,mapping=mapping,
                                         coverage=len(mapping),bridge_file=str(table['path']),
                                         bridge_folder_col=bcol,bridge_meta_col=bmid,
                                         conflicts=len(conflicts)))
    if not cand:raise ValueError('No ID join candidates found')
    # Prefer direct if tied. Detect disagreements between high-coverage candidates.
    cand.sort(key=lambda a:(a['coverage'],a['method']=='direct',
                             normalize(a['meta_col']) in ('uuid','userid','useruuid','id')),
              reverse=True)
    winner=cand[0]
    diagnostics=[]
    for x in cand[:70]:
        diagnostics.append(dict(method=x['method'],metadata_id_col=x['meta_col'],
                                bridge_file=x['bridge_file'],bridge_folder_col=x['bridge_folder_col'],
                                bridge_meta_col=x['bridge_meta_col'],users_matched=x['coverage'],
                                coverage_pct=round(100*x['coverage']/max(1,len(raw)),2),
                                conflicts=x.get('conflicts',0)))
    # Guard: if top two join candidates share broad coverage but map users differently,
    # their identity mapping cannot be trusted; raise an actionable error.
    for challenger in cand[1:]:
        if challenger['coverage']<max(2,0.8*winner['coverage']):break
        shared=set(winner['mapping'])&set(challenger['mapping'])
        inconsistent=[u for u in shared if winner['mapping'][u]!=challenger['mapping'][u]]
        if len(inconsistent)>=max(3,0.05*len(shared)):
            raise ValueError(f'Conflicting metadata mappings, {len(inconsistent)} users: '
                             f'{winner["method"]}:{winner["meta_col"]} vs '
                             f'{challenger["method"]}:{challenger["meta_col"]}. '
                             f'Force --user-meta-id-col and/or --bridge-* args.')
    return {raw[uid]:rowidx for uid,rowidx in winner['mapping'].items()},diagnostics,winner


def parse_date_maybe(value):
    s=str(value or '').strip()
    if not s:return None
    try:
        f=float(s)
        if abs(f)>=1e17: return f/1e9
        if abs(f)>=1e14: return f/1e6
        if abs(f)>=1e11: return f/1e3
        if 900_000_000<=f<=2_600_000_000: return f
    except ValueError:pass
    if re.fullmatch(r'\d{8}',s):
        try:return datetime.strptime(s,'%Y%m%d').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:pass
    try:
        dt=datetime.fromisoformat(s.replace('Z','+00:00'))
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError: return None


def session_order(user,session_names,tables,raw_user_root=None):
    """Verify chronology via metadata table or date/ordinal encoded in directory name.

    Does NOT use filesystem mtime from extracted zip; that's not session time.
    """
    if not session_names:return {},'no_sessions',False
    rows=[]
    # Metadata date: match session identifier; if the same folder name occurs in many users
    # it must be disambiguated by a user UUID in that table.
    for table in tables.values():
        if table['path'].stem.lower()=='userdata':continue
        datecols=[c for c in table['columns'] if any(x in normalize(c) for x in
                  ('createdat','startedat','starttime','timestamp','sessiondate','date','created','started','completed'))]
        if not datecols:continue
        usercols=[c for c in table['columns'] if any(x in normalize(c) for x in ('user','subject','owner','uuid'))]
        for scol in table['columns']:
            if not any(x in normalize(scol) for x in ('session','uuid','id','folder','filename','path')):continue
            for datecol in datecols:
                mapping=defaultdict(list)
                for row in table['rows']:
                    sv=cid(row.get(scol,''))
                    if not sv:continue
                    t=parse_date_maybe(row.get(datecol,''))
                    if t is None:continue
                    if usercols and not any(cid(row.get(c,''))==cid(user) for c in usercols):
                        # May be user IDs referring to metadata rather than directory;
                        # allow session-only when unique throughout the entire table.
                        pass
                    mapping[sv].append((t,row))
                found={}
                for name in session_names:
                    rec=mapping.get(cid(name),[])
                    if len(rec)==1:found[name]=rec[0][0]
                    elif len(rec)>1:
                        mine=[t for t,r in rec if any(cid(r.get(c,''))==cid(user) for c in usercols)]
                        if len(mine)==1:found[name]=mine[0]
                if len(found)>=4 and len(found)==len(session_names):
                    rows.append((found,f'metadata:{table["path"].name}:{scol}/{datecol}'))
    if rows:
        rows.sort(key=lambda p:p[1])
        return rows[0][0],rows[0][1],True
    # Encoded dates are safe for chronology.
    stamps={x:parse_date_maybe(x) for x in session_names}
    if all(v is not None for v in stamps.values()) and len(set(stamps.values()))==len(stamps):
        return stamps,'timestamp_in_session_directory',True
    # Strict small ordinal series only; arbitrary numeric directory names are not dates.
    if all(re.fullmatch(r'\d{1,3}',x) for x in session_names):
        nums=[int(x) for x in session_names]
        if len(set(nums))==len(nums) and min(nums) in (0,1) and max(nums)<100:
            return {x:float(int(x)) for x in session_names},'session_ordinal_directory',True
    # Date prefix YYYY-MM-DD or YYYYMMDD in directory names.
    stamps={}
    for x in session_names:
        m=re.search(r'(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)',x)
        if not m:break
        d=''.join(m.groups())
        stamps[x]=parse_date_maybe(d)
    if len(stamps)==len(session_names) and all(v is not None for v in stamps.values()) and len(set(stamps.values()))==len(stamps):
        return stamps,'date_in_session_directory',True
    # Last resort: inspect absolute timestamp of first raw touch sample per session.
    # This is safer than ZIP extraction mtimes or lexical UUID ordering; reject
    # session-relative clocks because they cannot order different days.
    if raw_user_root is not None:
        from_touch={}
        for session in session_names:
            root=Path(raw_user_root)/session
            values=[]
            for task in ('scroll','swipe'):
                folder=root/task
                if not folder.is_dir():continue
                candidates=sorted(folder.glob('*/touch_data/*.csv'))
                for csv_path in candidates[:3]:
                    try:
                        with csv_path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
                            sample=f.read(2048);f.seek(0)
                            try:dialect=csv.Sniffer().sniff(sample,delimiters=',;\t')
                            except csv.Error:dialect=csv.excel
                            reader=csv.reader(f,dialect)
                            header=next(reader,[])
                            j=time_field(header)
                            for _ in range(4):
                                rec=next(reader,None)
                                if rec is None:break
                                if len(rec)>j:
                                    dt=parse_date_maybe(rec[j])
                                    if dt is not None:values.append(dt);break
                    except (ValueError,OSError,csv.Error,IndexError):pass
            if values:from_touch[session]=min(values)
        if len(from_touch)==len(session_names) and len(set(from_touch.values()))==len(session_names):
            return from_touch,'first_raw_touch_epoch_timestamp',True
    return {x:None for x in session_names},'unverified_opaque_session_identifiers',False


def number(value):
    try:return int(float(str(value)))
    except (TypeError,ValueError):return 0


def safe_write(path,rows,columns):
    save_csv(path,rows,columns)


def get_audit_task_rows(args):
    if args.from_audit_dir:
        path=args.from_audit_dir/'sessions_tasks.csv'
        if not path.is_file():raise ValueError(f'Missing {path}')
    else:
        path=args.output/'sessions_tasks.csv'
        if not path.is_file():raise ValueError(f'Missing {path}. Run raw audit first.')
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def model_selection(args):
    tables=find_tables(args.tables)
    metadata=choose_userdata(tables)
    raw_users=sorted(p.name for p in args.input.iterdir() if p.is_dir())
    if args.limit_users: raw_users=raw_users[:args.limit_users]
    col_info=[{'table':str(t['path']), 'rows':len(t['rows']), 'columns':' | '.join(t['columns'])}
              for t in tables.values()]
    safe_write(args.output/'metadata_columns.csv',col_info,['table','rows','columns'])
    devcol=detect_model_column(metadata,args.device_col)
    mapping,joins,chosen=link_users(raw_users,metadata,tables,args)
    safe_write(args.output/'metadata_join_candidates.csv',joins,
               ['method','metadata_id_col','bridge_file','bridge_folder_col',
                'bridge_meta_col','users_matched','coverage_pct','conflicts'])
    if not mapping:
        raise ValueError('No match between userdata.csv and raw user directories; '
                         'inspect metadata_columns.csv and metadata_join_candidates.csv')
    recognized=0
    identity_rows=[];model_of={}
    for user in raw_users:
        ix=mapping.get(user)
        rawval=metadata['rows'][ix].get(devcol,'') if ix is not None else ''
        model=classify_model(rawval,allow_bare=True) if ix is not None else ''
        status=('unmapped' if ix is None else 'unknown_device' if not model else 'matched')
        if model: recognized+=1;model_of[user]=model
        identity_rows.append(dict(raw_user_id=user,userdata_row_index=ix if ix is not None else '',
                                  userdata_identifier=metadata['rows'][ix].get(chosen['meta_col'],'') if ix is not None else '',
                                  raw_device_value=rawval,iphone_model=model,join_status=status,
                                  userdata_file=str(metadata['path']),device_column=devcol))
    safe_write(args.output/'user_device_mapping.csv',identity_rows,
               ['raw_user_id','userdata_row_index','userdata_identifier','raw_device_value',
                'iphone_model','join_status','userdata_file','device_column'])
    task_rows=get_audit_task_rows(args)
    # Build calendar for all user directories. Repetitions do not count as sessions.
    counts={}
    dupes=[]
    for r in task_rows:
        user=str(r.get('user_id',''));ses=str(r.get('session',''));task=str(r.get('task',''))
        if task not in ('scroll','swipe'):continue
        k=(user,ses,task)
        if k in counts:
            dupes.append(k);continue
        counts[k]=r
    if dupes:raise ValueError(f'Duplicate user/session/task rows in audit: {dupes[:5]}')
    session_rows=[];user_sess=defaultdict(set)
    for user,ses,task in counts:user_sess[user].add(ses)
    # Also collect raw folders with missing task records.
    for user in raw_users:
        user_dir=args.input/user
        for sespath in user_dir.iterdir():
            if sespath.is_dir():user_sess[user].add(sespath.name)
    by_user_task={}
    order_problems=[]
    for user in raw_users:
        sessions=sorted(user_sess.get(user,set()))
        tm,source,verified=session_order(user,sessions,tables,args.input/user)
        if not verified and len(sessions)>=4:order_problems.append(user)
        for task in ('scroll','swipe'):
            result=[]
            for ses in sessions:
                entry=counts.get((user,ses,task),{})
                # Our primary count is all real physical contacts from the task
                # with synchronized strict accelerometer + gyro, NOT geometric swipe tags.
                complete=number(entry.get('gestures',0))
                geom=number(entry.get('gestures_swipe',0))
                strict_all=number(entry.get('gestures_both_strict_paired',0))
                strict_swipe=number(entry.get('swipes_both_strict_paired',0))
                usable=strict_swipe if args.gesture_count_mode=='geometric_swipes' else strict_all
                anymotion=number(entry.get('gestures_both_any',0))
                parse_note=str(entry.get('warnings',''))
                item=dict(raw_user_id=user,iphone_model=model_of.get(user,''),
                          session=ses,task=task,session_timestamp=tm.get(ses,''),
                          session_order_source=source,session_order_verified=int(verified),
                          repetitions=number(entry.get('repetitions',0)),
                          gestures_total=complete,gestures_both_any=anymotion,
                          gestures_strict_multimodal=usable,
                          strict_all_contacts=strict_all,strict_geometric_swipes=strict_swipe,
                          geometrically_swipe=geom,coverage_pct=round(100*usable/complete,3) if complete else 0,
                          audit_record_present=int(bool(entry)),
                          warnings=parse_note)
                result.append(item);session_rows.append(item)
            by_user_task[(user,task)]=(result,verified)
    safe_write(args.output/'session_task_counts.csv',session_rows,
               ['raw_user_id','iphone_model','session','task','session_timestamp',
                'session_order_source','session_order_verified','repetitions','gestures_total',
                'gestures_both_any','gestures_strict_multimodal','strict_all_contacts',
                'strict_geometric_swipes','geometrically_swipe',
                'coverage_pct','audit_record_present','warnings'])
    thresholds=sorted(set(int(t.strip()) for t in args.thresholds.split(',') if t.strip()))
    if not thresholds: raise ValueError('--thresholds must contain at least one value')
    if min(thresholds)<0:raise ValueError('--thresholds cannot contain negative values')
    if args.min_gestures not in thresholds:thresholds.append(args.min_gestures);thresholds.sort()
    candidate_rows=[];sweep=[];cohorts=[];selected_first_four=[]
    for user in raw_users:
        for task in ('scroll','swipe'):
            items,verified=by_user_task[(user,task)]
            ordered=sorted(items,key=lambda z:float(z['session_timestamp'])) if verified else []
            # In FETA the first four *calendar* sessions are selected, even when
            # one of them has no task data; it must not be silently replaced by later data.
            first=ordered[:4] if len(ordered)>=4 else []
            good_total=sum(number(x['gestures_strict_multimodal'])>=args.min_gestures for x in items)
            first_ok=(len(first)==4 and all(number(x['gestures_strict_multimodal'])>=args.min_gestures for x in first))
            if first_ok and model_of.get(user):
                for rank,rec in enumerate(first,1):
                    selected_first_four.append(dict(raw_user_id=user,iphone_model=model_of[user],
                        task=task,session=rec['session'],session_rank=rank,
                        role='enrollment' if rank<=2 else 'genuine_test',
                        gestures_strict_multimodal=rec['gestures_strict_multimodal'],
                        session_dir=str(args.input/user/rec['session'])))
            candidate_rows.append(dict(raw_user_id=user,iphone_model=model_of.get(user,''),task=task,
                total_sessions=len(items),sessions_at_threshold=good_total,
                at_least_4_qualifying=int(good_total>=4),first_four_order_verified=int(verified),
                first_four_meet_threshold=int(first_ok) if verified else '',
                first_four_min_usable=min((number(x['gestures_strict_multimodal']) for x in first),default=''),
                first_four_gestures='|'.join(str(x['gestures_strict_multimodal']) for x in first),
                first_four_session_ids='|'.join(x['session'] for x in first),
                four_session_threshold=args.min_gestures))
    safe_write(args.output/'user_task_eligibility.csv',candidate_rows,
               ['raw_user_id','iphone_model','task','total_sessions','sessions_at_threshold',
                'at_least_4_qualifying','first_four_order_verified','first_four_meet_threshold',
                'first_four_min_usable','first_four_gestures','first_four_session_ids','four_session_threshold'])
    safe_write(args.output/'eligible_first_four_sessions.csv',selected_first_four,
               ['raw_user_id','iphone_model','task','session','session_rank','role',
                'gestures_strict_multimodal','session_dir'])
    for model in EXPECTED_MODELS:
        for task in ('scroll','swipe'):
            us=[u for u in raw_users if model_of.get(u)==model]
            for threshold in thresholds:
                any_4=first_4=known=0
                for user in us:
                    items,verified=by_user_task[(user,task)]
                    if verified:known+=1
                    if sum(number(v['gestures_strict_multimodal'])>=threshold for v in items)>=4:any_4+=1
                    if verified and len(items)>=4:
                        earliest=sorted(items,key=lambda v:float(v['session_timestamp']))[:4]
                        if all(number(v['gestures_strict_multimodal'])>=threshold for v in earliest):first_4+=1
                sweep.append(dict(iphone_model=model,task=task,min_gestures_per_session=threshold,
                    total_mapped_users=len(us),users_4_qualifying_sessions=any_4,
                    users_with_verified_order=known,users_first_4_qualify=first_4,
                    first4_is_lower_bound_when_order_unknown=int(known<len(us))))
                if threshold==args.min_gestures:
                    cohorts.append(dict(iphone_model=model,task=task,total_users=len(us),
                        users_4_qualifying_sessions=any_4,users_first_4_qualify=first_4,
                        users_with_verified_order=known))
    safe_write(args.output/'threshold_sweep.csv',sweep,
               ['iphone_model','task','min_gestures_per_session','total_mapped_users',
                'users_4_qualifying_sessions','users_with_verified_order',
                'users_first_4_qualify','first4_is_lower_bound_when_order_unknown'])
    safe_write(args.output/'device_task_summary.csv',cohorts,
               ['iphone_model','task','total_users','users_4_qualifying_sessions',
                'users_first_4_qualify','users_with_verified_order'])
    # Planning-only threshold: strongest minimum retaining a target number of users.
    # NOT an optimum for EER or authentication accuracy.
    plan=[]
    for model in EXPECTED_MODELS:
        for task in ('scroll','swipe'):
            rr=[r for r in sweep if r['iphone_model']==model and r['task']==task]
            for metric in ('users_4_qualifying_sessions','users_first_4_qualify'):
                viable=[r for r in rr if r[metric]>=args.min_cohort_users]
                high=max(viable,key=lambda x:x['min_gestures_per_session']) if viable else None
                plan.append(dict(iphone_model=model,task=task,selection_metric=metric,
                    target_cohort_size=args.min_cohort_users,
                    max_swept_threshold_preserving_target=high['min_gestures_per_session'] if high else '',
                    users_at_that_threshold=high[metric] if high else '',
                    order_verified_users=rr[0]['users_with_verified_order'] if rr else 0,
                    is_heuristic_only=1))
    safe_write(args.output/'threshold_planning.csv',plan,
               ['iphone_model','task','selection_metric','target_cohort_size',
                'max_swept_threshold_preserving_target','users_at_that_threshold',
                'order_verified_users','is_heuristic_only'])
    # Independent top-three per task, and single common top-three on the smaller of two task counts.
    def rank(metric):
        return sorted(EXPECTED_MODELS,key=lambda m:(-metric(m),m))[:3]
    by_model_task={(c['iphone_model'],c['task']):c for c in cohorts}
    top_three={t:rank(lambda m:by_model_task[(m,t)]['users_4_qualifying_sessions']) for t in ('scroll','swipe')}
    joint=rank(lambda m:min(by_model_task[(m,'scroll')]['users_4_qualifying_sessions'],
                              by_model_task[(m,'swipe')]['users_4_qualifying_sessions']))
    ranking=[]
    for mode in ('any_four','first_four'):
        col='users_4_qualifying_sessions' if mode=='any_four' else 'users_first_4_qualify'
        for task in ('scroll','swipe','both_tasks_minimum'):
            ranked=rank(lambda m:by_model_task[(m,task)][col] if task!='both_tasks_minimum'
                        else min(by_model_task[(m,'scroll')][col],by_model_task[(m,'swipe')][col]))
            for i,m in enumerate(ranked,1):
                c1=by_model_task[(m,'scroll')];c2=by_model_task[(m,'swipe')]
                ranking.append(dict(ranking=f'{mode}:{task}',rank=i,iphone_model=m,
                    scroll_eligible=c1[col],swipe_eligible=c2[col],
                    minimum_of_two=min(c1[col],c2[col])))
    safe_write(args.output/'top_three_iphone_models.csv',ranking,
               ['ranking','rank','iphone_model','scroll_eligible','swipe_eligible','minimum_of_two'])
    # Exact overlap of users within each model for both tasks.
    overlaps=[]
    for model in EXPECTED_MODELS:
        us=[u for u in raw_users if model_of.get(u)==model]
        for metric in ('any_four','first_four'):
            selected={}
            for task in ('scroll','swipe'):
                selected[task]=set()
                for u in us:
                    items,verified=by_user_task[(u,task)]
                    if metric=='any_four':
                        ok=sum(number(x['gestures_strict_multimodal'])>=args.min_gestures for x in items)>=4
                    else:
                        ok=False
                        if verified and len(items)>=4:
                            f=sorted(items,key=lambda x:float(x['session_timestamp']))[:4]
                            ok=all(number(x['gestures_strict_multimodal'])>=args.min_gestures for x in f)
                    if ok:selected[task].add(u)
            overlaps.append(dict(iphone_model=model,metric=metric,
                scroll_users=len(selected['scroll']),swipe_users=len(selected['swipe']),
                both_task_users=len(selected['scroll']&selected['swipe']),
                both_task_user_ids='|'.join(sorted(selected['scroll']&selected['swipe']))))
    safe_write(args.output/'task_intersection.csv',overlaps,
               ['iphone_model','metric','scroll_users','swipe_users','both_task_users','both_task_user_ids'])
    summary={
        'version':VERSION,'input':str(args.input),'tables':str(args.tables),
        'metadata_userdata':str(metadata['path']),'device_model_column':devcol,
        'id_matching':{k:v for k,v in chosen.items() if k!='mapping'},
        'users_raw':len(raw_users),'users_metadata_joined':len(mapping),
        'users_with_recognized_iphone_model':recognized,
        'unknown_or_unmapped_users':len(raw_users)-recognized,
        'minimum_multimodal_gestures_per_task_session':args.min_gestures,
        'gesture_count_mode':args.gesture_count_mode,
        'threshold_is_exploratory_not_eer_optimum':True,
        'target_minimum_iphone_cohort_users':args.min_cohort_users,
        'users_with_unverified_session_chronology':len(order_problems),
        'top_three_by_scroll':top_three['scroll'],
        'top_three_by_swipe':top_three['swipe'],
        'top_three_by_minimum_across_both_tasks':joint,
        'top_three_first_four_by_scroll':rank(lambda m:by_model_task[(m,'scroll')]['users_first_4_qualify']),
        'top_three_first_four_by_swipe':rank(lambda m:by_model_task[(m,'swipe')]['users_first_4_qualify']),
        'device_task_summary':cohorts,
        'warnings':[
            'FETA tasks are separate: scroll/social-feed and swipe/image-gallery; repetitions are NOT sessions.',
            'Default counts use geometric swipe-like contacts with strict paired accel+gyro, for BOTH tasks; all_contacts is a sensitivity mode. The 30px/50px thresholds are analyst heuristics, not the authors original classification.',
            'At_least_four_qualifying can select non-contiguous sessions; first_four_meet_threshold tests only chronologically earliest four. Ordering may use metadata, encoded dates, ordinals, or absolute raw touch timestamps; otherwise remains unknown.',
            'The iOS collection app kept its layout portrait; no raw orientation filter is applied.',
            'Data-driven optimal gesture threshold requires actual model performance; threshold sweep is a cohort-size planning tool, not an EER optimization.',
            'Model metadata linkage is checked by column value matches; inspect user_device_mapping.csv for unmapped users and metadata_join_candidates.csv for alternate joins.',
            'The reused v1.1 time-unit autodetection must be validated against sample raw CSV headers and time scales before training classifiers.'
        ]}
    (args.output/'device_eligibility_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    readme="# FETA device mapping / four-session selection\n\n"
    readme+="""Selection uses `data_files/USER/SESSION/{scroll,swipe}/REPETITION`, not individual repetitions as sessions.\n
By default, the primary count is `swipes_both_strict_paired` from the v1.1 audit:
contacts must be DOWN->UP and reach >=30px displacement OR >=50px path length.
This is an explicit **analyst heuristic**, not the authors' original stroke definition;
change --gesture-count-mode all_contacts to count every valid physical DOWN-UP contact
regardless of whether it looks like a tap. In `scroll` tasks, scrolling also consists of
finger strokes, so geometric swipe-like contacts are counted there too.
Both counts are saved in `session_task_counts.csv`.
The strict multimodal criterion requires every chosen contact
with both accelerometer and gyroscope meeting >=3 samples, <=40ms endpoint gap, <=120ms inside gap,
and at least one accel-gyro pair <=25ms. These are configurable **audit conventions**, not
requirements asserted by the FETA authors. FETA task membership (`scroll`/`swipe`) comes from
the directory name, not geometric motion direction.

Counts distinguish (A) **any four** sessions that pass the threshold and (B) the **chronologically
first four** sessions that all pass. If session chronology cannot be verified from metadata,
recorded dates, recognized ordinal names or absolute touch-event timestamps, first-four eligibility is left unknown, not falsely
inferred from lexicographic ordering or extracted ZIP modification times.

The analysis outputs:
- `user_device_mapping.csv`: metadata joins and raw source model values
- `metadata_columns.csv`, `metadata_join_candidates.csv`: schema/join audit
- `session_task_counts.csv`: per raw session, per task, multimodal gesture counts
- `user_task_eligibility.csv`: per user and task at chosen threshold
- `eligible_first_four_sessions.csv`: exact first-four session roster and enrollment/test role
- `threshold_sweep.csv`: model/task cohort sizes across thresholds
- `threshold_planning.csv`: highest swept threshold retaining a target cohort (NOT EER-optimal)
- `device_task_summary.csv`: per model and task
- `top_three_iphone_models.csv`: separate scroll/swipe and common top-three rankings
- `task_intersection.csv`: users qualified for both tasks, per device
- `device_eligibility_summary.json`: summary and warnings
- Original v1.1 audit outputs are retained alongside these files unless `--from-audit-dir` is used.

Changing --min-gestures or --thresholds does not invalidate raw per-user cached gesture processing.
**An optimal threshold for EER cannot be inferred from counts alone**: it requires a pilot
authentication model and fixed genuine/impostor evaluation protocol.
"""
    (args.output/'HOW_TO_READ_DEVICE_SELECTION.md').write_text(readme,encoding='utf-8')
    print('\n========== FETA MODEL / SESSION SELECTION ==========')
    print(f'User directories: {len(raw_users)}; metadata joins: {len(mapping)}; recognized models: {recognized}')
    print(f'Unverified session chronology: {len(order_problems)} users')
    print(f'Min gestures: {args.min_gestures}, mode={args.gesture_count_mode}, strict touch+accel+gyro')
    for row in cohorts:
        print(f"{row['iphone_model']:>15} {row['task']:>6} | users {row['total_users']:>3} | "
              f"4 eligible: {row['users_4_qualifying_sessions']:>3} | "
              f"first 4: {row['users_first_4_qualify']:>3} "
              f"(verified order {row['users_with_verified_order']:>3})")
    print('Top 3 common:',', '.join(joint))
    print(f'Files in: {args.output}')
    if not mapping or recognized < 0.8*len(raw_users):
        print('WARNING: many unknown device mappings. Check metadata_columns.csv '
              'and user_device_mapping.csv; use CLI mapping overrides if required.')
    return 0


def main(argv=None):
    args=parse_args(argv)
    if (args.input/'data_files').is_dir() and not any((args.input/u).is_dir() for u in ('scroll','swipe')):
        args.input=args.input/'data_files'
    if not args.input.is_dir():
        raise SystemExit(f'Input directory not found: {args.input.resolve()}')
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.from_audit_dir:
        # Raw counts are computed using the full v1.1 logic included above.
        result=audit_main(argv)
        if result:return result
    try:return model_selection(args)
    except Exception as exc:
        # Always expose metadata inventory to diagnose the exact schema.
        print('ERROR during metadata linkage:',type(exc).__name__,str(exc))
        raise

if __name__=='__main__':
    raise SystemExit(main())
