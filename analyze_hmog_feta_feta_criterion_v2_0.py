#!/usr/bin/env python3
"""HMOG + FETA comparable motion-covered gestures and four-session cohorts.

Original FETA paper filtering: >=3 touch samples and MAX radial deviation
from DOWN > 5 native screen coordinate units (pixels as reported by authors).
HMOG additionally: every touch sample has Phone_orientation == 0.
Gesture counts are distinct physical DOWN->UP contacts; neither HMOG's
ScrollEvent rows nor its StrokeEvent rows are added to them.

Python >=3.10, numpy and pandas. No external network access needed.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='hmog-feta-gesture-audit-2.0.0-20261112'
HMOG_TASKS={1:'reading_sitting',7:'reading_sitting',13:'reading_sitting',19:'reading_sitting',
            2:'reading_walking',8:'reading_walking',14:'reading_walking',20:'reading_walking',
            3:'writing_sitting',9:'writing_sitting',15:'writing_sitting',21:'writing_sitting',
            4:'writing_walking',10:'writing_walking',16:'writing_walking',22:'writing_walking'}
FETA_TASKS=('scroll','swipe')
ALL_HMOG_TASKS=('reading_sitting','reading_walking','writing_sitting','writing_walking')
H_SCHEMA={'activity':['ID','SubjectID','Session_number','Start_time','End_time','Relative_Start_time','Relative_End_time','Gesture_scenario','TaskID','ContentID'],
          'touch':['Systime','EventTime','ActivityID','Pointer_count','PointerID','ActionID','X','Y','Pressure','Contact_size','Phone_orientation'],
          'accel':['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation'],
          'gyro':['Systime','EventTime','ActivityID','X','Y','Z','Phone_orientation']}
H_FILENAMES={'touch':('TouchEvent_im.csv','TouchEvent.csv'),'accel':('Accelerometer.csv',),
             'gyro':('Gyroscope.csv','Gyroscrope.csv'),'activity':('Activity.csv',)}
MIN_GESTURE_POINTS=3
MIN_RADIAL_PX=5.0
GESTURE_FIELDS=('dataset','device_model','user_id','session_id','task','activity_id','repetition',
                'start_ms','end_ms','duration_ms','touch_points','max_radial_px','net_displacement_px',
                'portrait_only','multitouch','complete','feta_geometry','feta_touch_usable','strict_multimodal',
                'accel_samples','gyro_samples','accel_gyro_pairs','reason')
SESSION_FIELDS=('dataset','device_model','user_id','session_id','task','session_time','order_source','order_verified',
                'repetitions','touch_files','contacts_complete','contacts_valid_single','gestures_3_points',
                'gestures_feta_geometry','gestures_feta_touch_usable','gestures_strict_multimodal',
                'excluded_nonportrait','excluded_multitouch','strict_coverage_failed','touch_rows','warning')

def norm(s):return re.sub('[^a-z0-9]','',str(s).lower().lstrip('\ufeff'))
def strid(s):return str(s).strip() if pd.notna(s) else ''
def num(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')
def safeint(x):
    try:return int(float(x))
    except (ValueError,TypeError,OverflowError):return 0

def read_hmog(path,kind):
    if path is None or not path.is_file() or path.stat().st_size==0:return pd.DataFrame()
    with path.open(encoding='utf-8-sig',errors='replace') as f:first=f.readline()
    sep=max((',', '\t',';'),key=first.count)
    parts=[norm(x.strip(' "')) for x in first.strip().split(sep)]
    has_header=len(set(parts).intersection(map(norm,H_SCHEMA[kind])))>=2
    kwargs={'sep':sep,'encoding':'utf-8-sig','dtype':str,'on_bad_lines':'warn','low_memory':False}
    if has_header:df=pd.read_csv(path,**kwargs)
    else:df=pd.read_csv(path,header=None,names=H_SCHEMA[kind],**kwargs)
    df.columns=[norm(x) for x in df.columns]
    return df

def numeric(df,col):
    if col not in df:return np.full(len(df),np.nan,dtype=float)
    return pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)

def key_time_from_string(s):
    x=str(s).strip()
    if not x:return None
    try:
        v=float(x)
        if not np.isfinite(v):return None
        if abs(v)>1e17:return v/1e9
        if abs(v)>1e14:return v/1e6
        if abs(v)>1e11:return v/1e3
        if 9e8<=v<3e9:return v
    except (ValueError,TypeError):pass
    if re.fullmatch(r'20\d{6}',x):
        try:return datetime.strptime(x,'%Y%m%d').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:pass
    try:
        dt=datetime.fromisoformat(x.replace('Z','+00:00'))
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:return None

def parse_hmog_contacts(touch,activity_map):
    if touch.empty:return [],0,[]
    aid=touch.get('activityid',pd.Series(['']*len(touch))).astype(str).str.strip().to_numpy()
    tm=numeric(touch,'eventtime');x=numeric(touch,'x');y=numeric(touch,'y')
    actions=numeric(touch,'actionid');pcount=numeric(touch,'pointercount')
    pid=numeric(touch,'pointerid');orientation=numeric(touch,'phoneorientation')
    valid=np.isfinite(tm)&np.isfinite(x)&np.isfinite(y)&np.isfinite(actions)
    indices=np.flatnonzero(valid & np.isin(aid,list(activity_map)))
    indices=sorted(indices,key=lambda j:(aid[j],tm[j],int(j)))
    gestures=[];issues=[];active=None
    for j in indices:
        a=aid[j];t=float(tm[j]);xx=float(x[j]);yy=float(y[j])
        action=int(actions[j])&255
        multi=bool(action in (5,6) or (np.isfinite(pcount[j]) and pcount[j]>1) or
                   (np.isfinite(pid[j]) and pid[j]>0))
        isportrait=bool(np.isfinite(orientation[j]) and int(orientation[j])==0)
        if action==0:
            if active is not None:issues.append('interrupted_down')
            active={'activity_id':a,'task':activity_map[a],'start_ms':t,
                    'x0':xx,'y0':yy,'xlast':xx,'ylast':yy,'max_radial_px':0.,
                    'touch_points':1,'portrait_only':isportrait,'multitouch':multi}
        elif action in (5,6):
            if active:active['multitouch']=True
        elif action==3:
            active=None
        elif action in (1,2):
            if active is None:continue
            if active['activity_id']!=a:active=None;continue
            active['touch_points']+=1
            active['max_radial_px']=max(active['max_radial_px'],math.hypot(xx-active['x0'],yy-active['y0']))
            active['xlast']=xx;active['ylast']=yy
            active['portrait_only']&=isportrait
            active['multitouch']|=multi
            if action==1:
                duration=t-active['start_ms']
                gestures.append({**active,'end_ms':t,'duration_ms':duration,
                    'net_displacement_px':math.hypot(xx-active['x0'],yy-active['y0'])})
                active=None
    return gestures,int(np.count_nonzero(np.isin(aid,list(activity_map)))),issues

def parse_feta_phase(raw,col):
    s=norm(raw)
    if s in ('fingerdown','down','began','begin','touchdown','pressed','start'):return 'down'
    if s in ('fingerup','up','ended','end','touchup','released','release'):return 'up'
    if s in ('move','moved','changed','stationary','drag'):return 'move'
    if s in ('cancel','cancelled','canceled'):return 'cancel'
    if s in ('0','1','2','3','4','5','6'):
        v=int(s)
        if col in ('type','touchtype','phase'):
            # Oxford FETA custom TouchEvent type: 0 DOWN, 1 MOVE, 2 UP
            return {0:'down',1:'move',2:'up'}.get(v,'cancel')
        if 'action' in col:
            # FETA's custom action field is usually strings. Numeric generic
            # actions use FETA convention by default (overridable below).
            return {0:'down',1:'move',2:'up'}.get(v,'cancel')
    return 'unknown'

def feta_csv(path):
    """Return normalized header CSV; reject headerless files rather than invent schema."""
    if not path.is_file() or path.stat().st_size==0:return pd.DataFrame()
    with path.open('r',encoding='utf-8-sig',errors='replace') as f:line=f.readline()
    sep=max((',', '\t',';'),key=line.count)
    cols=[x.strip(' "\n\r\ufeff') for x in line.split(sep)]
    if len(cols)<2 or sum(not re.fullmatch(r'[+-]?(\d+\.?\d*|\.\d+)',x) for x in cols)<2:
        raise ValueError(f'FETA CSV without recognizable header: {path}')
    df=pd.read_csv(path,sep=sep,encoding='utf-8-sig',dtype=str,on_bad_lines='warn',low_memory=False)
    df.columns=[norm(c) for c in df.columns]
    return df

def feta_time_ms(df):
    matches=[c for c in ('timestamp','eventtime','time','ts','recordedat','sampletime','timestampms',
                         'touchstarttime','createdat','datetime','t') if c in df]
    if not matches:matches=[c for c in df if ('timestamp' in c or c.endswith('time'))]
    if not matches:raise ValueError('FETA timestamp column not found, columns='+repr(list(df.columns)))
    col=matches[0]
    raw=df[col]
    a=pd.to_numeric(raw,errors='coerce').to_numpy(dtype=float)
    good=a[np.isfinite(a)]
    if len(good)>0 and np.count_nonzero(np.isfinite(a))>=0.9*len(a):
        mag=float(np.median(np.abs(good)))
        if mag>=1e17:factor=1e-6     # ns -> ms
        elif mag>=1e14:factor=1e-3   # us -> ms
        elif mag>=1e11:factor=1.0    # epoch ms
        elif mag>=1e8:factor=1000.0 # epoch s
        else:
            steps=np.diff(good)
            steps=steps[np.isfinite(steps)&(steps>0)]
            if len(steps)==0:raise ValueError('Cannot infer relative time unit from single timestamp')
            d=float(np.median(steps))
            # FETA local clocks should use a consistent scale across modalities;
            # disjoint ranges will be flagged by sensor coverage checks.
            if d>=1e6:factor=1e-6
            elif d>=1e3:factor=1e-3
            elif d>=1:factor=1.0
            else:factor=1000.0
        return a*factor
    dt=pd.to_datetime(raw,utc=True,errors='coerce')
    return dt.map(lambda v:v.timestamp()*1000 if pd.notna(v) else np.nan).to_numpy(dtype=float)

def parse_feta_contacts(path):
    df=feta_csv(path)
    if df.empty:return [],0,[]
    t=feta_time_ms(df);x=numeric(df,'x');y=numeric(df,'y')
    ph=[c for c in ('type','touchtype','phase','action','actiontype','touchphase','eventtype','state') if c in df]
    if not ph:raise ValueError('FETA touch phase/type column missing, columns='+str(list(df.columns)))
    pcol=ph[0]
    events=sorted(((t[i],x[i],y[i],df[pcol].iloc[i],i) for i in range(len(df))
                   if all(np.isfinite(z) for z in (t[i],x[i],y[i]))),key=lambda r:(r[0],r[4]))
    gestures=[];issues=[];a=None
    for tm,xx,yy,phase,_ in events:
        state=parse_feta_phase(phase,pcol)
        if state=='down':
            if a is not None:issues.append('interrupted_down')
            a={'start_ms':float(tm),'x0':float(xx),'y0':float(yy),'max_radial_px':0.,
               'touch_points':1,'portrait_only':True,'multitouch':False}
        elif state=='move':
            if a is None:continue
            a['touch_points']+=1
            a['max_radial_px']=max(a['max_radial_px'],math.hypot(xx-a['x0'],yy-a['y0']))
        elif state=='up':
            if a is None:issues.append('orphan_up');continue
            a['touch_points']+=1
            a['max_radial_px']=max(a['max_radial_px'],math.hypot(xx-a['x0'],yy-a['y0']))
            gestures.append({**a,'end_ms':float(tm),'duration_ms':float(tm)-a['start_ms'],
                             'net_displacement_px':math.hypot(xx-a['x0'],yy-a['y0'])})
            a=None
        elif state=='cancel':a=None
        else:issues.append('unknown_phase:'+str(phase)[:30])
    if a is not None:issues.append('unfinished_down')
    return gestures,len(df),issues

def sensor_data(df,time_values,portrait_required):
    valid=np.isfinite(time_values)
    if portrait_required:
        vals=numeric(df,'phoneorientation')
        oo=np.isfinite(vals)&(vals==0)
    else:oo=np.ones(len(df),dtype=bool)
    ix=np.argsort(time_values[valid],kind='stable')
    return time_values[valid][ix],oo[valid][ix]

def sensor_check(times,orientation,start,end,min_samples,edge_gap,gap):
    if len(times)==0:return False,0,np.empty(0,dtype=float),'no_sensor'
    lo=int(np.searchsorted(times,start,side='left'))
    hi=int(np.searchsorted(times,end,side='right'))
    selected=times[lo:hi];n=len(selected)
    if n<min_samples:return False,n,selected,'few_samples'
    if (selected[0]-start)>edge_gap or (end-selected[-1])>edge_gap:
        return False,n,selected,'edge_gap'
    if n>1 and float(np.diff(selected).max())>gap:
        return False,n,selected,'internal_gap'
    if not bool(np.all(orientation[lo:hi])):
        return False,n,selected,'nonportrait_motion'
    return True,n,selected,''

def count_pairs(t1,t2,tol):
    i=j=n=0
    while i<len(t1) and j<len(t2):
        d=t1[i]-t2[j]
        if abs(d)<=tol:n+=1;i+=1;j+=1
        elif d<0:i+=1
        else:j+=1
    return n

def assess(gestures,sensors,identity,params):
    counts=Counter();issues=[];gesture_records=[]
    for g in gestures:
        counts['contacts_complete']+=1
        structural=bool(g['duration_ms']>0 and not g.get('multitouch'))
        if structural:counts['contacts_valid_single']+=1
        samples=bool(structural and g['touch_points']>=MIN_GESTURE_POINTS)
        if samples:counts['gestures_3_points']+=1
        geometry=bool(samples and g['max_radial_px']>MIN_RADIAL_PX)
        if geometry:counts['gestures_feta_geometry']+=1
        portrait=bool(g['portrait_only'])
        if geometry and not portrait:counts['excluded_nonportrait']+=1
        if g.get('multitouch'):counts['excluded_multitouch']+=1
        usable=bool(geometry and portrait)
        if usable:counts['gestures_feta_touch_usable']+=1
        a=g.get('activity_id','')
        motion=sensors.get(a,sensors.get('',{}))
        an=gn=pair=0;strict=False;reason=''
        if usable:
            if not all(k in motion for k in ('accel','gyro')):
                reason='missing_motion'
            else:
                s1=motion['accel'];s2=motion['gyro']
                ok1,an,t1,r1=sensor_check(*s1,g['start_ms'],g['end_ms'],params['min_sensor_samples'],
                                           params['edge_gap_ms'],params['internal_gap_ms'])
                ok2,gn,t2,r2=sensor_check(*s2,g['start_ms'],g['end_ms'],params['min_sensor_samples'],
                                           params['edge_gap_ms'],params['internal_gap_ms'])
                pair=count_pairs(t1,t2,params['pair_tolerance_ms']) if ok1 and ok2 else 0
                strict=ok1 and ok2 and pair>=params['min_sensor_pairs']
                if not strict:reason=';'.join(filter(None,(r1,r2,'few_pairs' if ok1 and ok2 and pair<params['min_sensor_pairs'] else '')))
            if strict:counts['gestures_strict_multimodal']+=1
            else:counts['strict_coverage_failed']+=1
        elif not geometry:reason='invalid_gesture_geometry_or_contact'
        elif not portrait:reason='not_portrait'
        if params['save_gestures']:
            gesture_records.append({**identity,'activity_id':a,'repetition':g.get('repetition',''),
                'start_ms':round(g['start_ms'],3),'end_ms':round(g['end_ms'],3),'duration_ms':round(g['duration_ms'],3),
                'touch_points':g['touch_points'],'max_radial_px':round(g['max_radial_px'],4),
                'net_displacement_px':round(g['net_displacement_px'],4),'portrait_only':int(portrait),
                'multitouch':int(g.get('multitouch',False)),'complete':1,
                'feta_geometry':int(geometry),'feta_touch_usable':int(usable),
                'strict_multimodal':int(strict),'accel_samples':an,'gyro_samples':gn,
                'accel_gyro_pairs':pair,'reason':reason})
    return counts,gesture_records

def first_existing(folder,names):
    return next((folder/name for name in names if (folder/name).is_file()),None)

def hmog_session(folder,user,params):
    files={k:first_existing(folder,v) for k,v in H_FILENAMES.items()}
    if files['activity'] is None:
        return [],[],['missing Activity.csv']
    a=read_hmog(files['activity'],'activity')
    if a.empty:return [],[],['empty Activity.csv']
    aids=a.get('id',pd.Series(['']*len(a))).astype(str).str.strip().to_numpy()
    tks=numeric(a,'taskid')
    taskmap={aids[i]:HMOG_TASKS[int(tks[i])] for i in range(len(a))
             if np.isfinite(tks[i]) and int(tks[i]) in HMOG_TASKS}
    if not taskmap:return [],[],[]
    session_number=folder.name
    if 'sessionnumber' in a:
        codes=a['sessionnumber'].dropna().astype(str).unique()
        if len(codes)==1:session_number=codes[0]
    start=numeric(a,'starttime')
    candidates=start[np.isfinite(start)]
    session_start=float(candidates.min()) if len(candidates) else None
    # Start_time is Unix epoch milliseconds in HMOG.
    base={'dataset':'HMOG','device_model':'Samsung Galaxy S4','user_id':user,
          'session_id':session_number,'session_time':session_start,'order_source':'Activity.Start_time' if session_start else 'unknown',
          'order_verified':int(session_start is not None)}
    task_contacts={v:[] for v in set(taskmap.values())}
    warnings=[];touchrows=0
    if files['touch'] is not None:
        tf=read_hmog(files['touch'],'touch')
        contacts,touchrows,issues=parse_hmog_contacts(tf,taskmap)
        warnings.extend(issues[:5])
        for g in contacts:task_contacts[g['task']].append(g)
    else:warnings.append('missing_touch')
    sensors={aid:{} for aid in taskmap}
    for sensor_name in ('accel','gyro'):
        file=files[sensor_name]
        if file is None:warnings.append('missing_'+sensor_name);continue
        frame=read_hmog(file,sensor_name)
        if frame.empty:continue
        st=numeric(frame,'eventtime')/1e6  # HMOG sensor.EventTime nanoseconds to milliseconds
        ids=frame.get('activityid',pd.Series(['']*len(frame))).astype(str).str.strip().to_numpy()
        for aid in taskmap:
            mask=(ids==aid)
            times,portrait=sensor_data(frame.loc[mask],st[mask],True)
            sensors[aid][sensor_name]=(times,portrait)
    rows=[];gesture_rows=[]
    for task,gs in task_contacts.items():
        ident={k:base[k] for k in ('dataset','device_model','user_id','session_id')}
        ident['task']=task
        c,gg=assess(gs,sensors,ident,params)
        row={**base,'task':task,'repetitions':'','touch_files':int(files['touch'] is not None),
             'touch_rows':touchrows,'warning':';'.join(warnings[:6]),**c}
        rows.append(row);gesture_rows.extend(gg)
    return rows,gesture_rows,warnings

def feta_user_mapping(userdata):
    df=pd.read_csv(userdata,dtype=str,encoding='utf-8-sig')
    df.columns=[norm(c) for c in df.columns]
    if 'uuid' not in df or 'phonemodel' not in df:
        raise ValueError('userdata.csv must contain uuid and phone_model; got '+repr(df.columns.tolist()))
    if df['uuid'].duplicated().any():
        # Duplicate UUIDs with different model values are a dataset ambiguity.
        ambiguous=df.groupby('uuid')['phonemodel'].nunique()
        if (ambiguous>1).any():raise ValueError('userdata.csv contains UUIDs with conflicting phone models')
    return df.drop_duplicates('uuid').set_index('uuid')['phonemodel'].fillna('').to_dict()

def meta_session_order(user,session_names,previous_calendar,session_root):
    """Use validated v1.2 calendar when available; otherwise trust only actual clock dates."""
    found={}
    for name in session_names:
        v=previous_calendar.get((user,name))
        if v is not None and np.isfinite(v):found[name]=v
    if len(found)==len(session_names) and len(set(found.values()))==len(found):
        return found,'previous_v1_2_calendar',True
    # Fall back to documented session ordinal / encoded date, then sample touch epochs.
    vals={n:key_time_from_string(n) for n in session_names}
    if len(vals)==len(session_names) and all(v is not None for v in vals.values()) and len(set(vals.values()))==len(vals):
        return vals,'timestamp_session_name',True
    if session_names and all(re.fullmatch(r'\d{1,3}',n) for n in session_names):
        vals={n:int(n) for n in session_names}
        if min(vals.values()) in (0,1) and max(vals.values())<100 and len(set(vals.values()))==len(vals):
            return vals,'ordinal_session_names',True
    vals={}
    for name in session_names:
        dates=[]
        for task in FETA_TASKS:
            taskfolder=session_root/name/task
            if not taskfolder.exists():continue
            for f in sorted(taskfolder.glob('*/touch_data/*.csv'))[:5]:
                try:
                    d=feta_csv(f)
                    if not d.empty:
                        t=feta_time_ms(d)
                        if len(t) and np.any(np.isfinite(t)):
                            candidate=float(np.nanmin(t))/1000
                            # Unix epoch, not device uptime (which resets).
                            if 900_000_000<=candidate<3_000_000_000:dates.append(candidate)
                except Exception:continue
        if dates:vals[name]=min(dates)
    if len(vals)==len(session_names) and len(set(vals.values()))==len(vals):
        return vals,'first_touch_timestamp',True
    return {},'unverified',False

def feta_rep(folder,identity,params):
    sensors={};issues=[];all_g=[];rows_total=0;touchfiles=0
    for channel,sub in (('touch','touch_data'),('accel','accelerometer_data'),('gyro','gyroscope_data')):
        csvs=sorted((folder/sub).glob('*.csv')) if (folder/sub).is_dir() else []
        if channel=='touch':
            for f in csvs:
                touchfiles+=1
                try:
                    gs,n,warnings=parse_feta_contacts(f)
                    for g in gs:g['repetition']=folder.name
                    all_g.extend(gs);rows_total+=n;issues.extend(warnings[:3])
                except Exception as exc:issues.append('touch:'+f.name+':'+str(exc)[:120])
        else:
            tt=[]
            for f in csvs:
                try:
                    df=feta_csv(f)
                    if not df.empty:tt.append(feta_time_ms(df))
                except Exception as exc:issues.append(channel+':'+f.name+':'+str(exc)[:120])
            times=np.sort(np.concatenate(tt)) if tt else np.empty(0,dtype=float)
            times=times[np.isfinite(times)]
            sensors[channel]=(times,np.ones(len(times),dtype=bool))
    counts,gestures=assess(all_g,{'':sensors},identity,params)
    return counts,gestures,rows_total,touchfiles,issues

def feta_user(user_dir,device,calendar,params):
    user=user_dir.name
    sessions=sorted(p for p in user_dir.iterdir() if p.is_dir())
    stamps,order_source,verified=meta_session_order(user,[x.name for x in sessions],calendar,user_dir)
    output=[];all_g=[];issues=[]
    for folder in sessions:
        for task in FETA_TASKS:
            taskdir=folder/task
            repetitions=sorted(x for x in taskdir.iterdir() if x.is_dir()) if taskdir.is_dir() else []
            identity={'dataset':'FETA','device_model':device,'user_id':user,'session_id':folder.name,'task':task}
            c=Counter();g=[];rowcnt=touchfiles=0;w=[]
            for rep in repetitions:
                ci,gi,ni,fi,wi=feta_rep(rep,identity,params)
                c.update(ci);g.extend(gi);rowcnt+=ni;touchfiles+=fi;w.extend(wi[:3])
            output.append({**identity,'session_time':stamps.get(folder.name),
                           'order_source':order_source,'order_verified':int(verified),
                           'repetitions':len(repetitions),'touch_files':touchfiles,
                           'touch_rows':rowcnt,'warning':';'.join(w[:8]),**c})
            all_g.extend(g);issues.extend(w[:2])
    return output,all_g,issues

def paths_signature(paths,params):
    h=hashlib.sha256()
    config={k:v for k,v in params.items() if k not in ('thresholds','save_gestures')}
    h.update(VERSION.encode());h.update(json.dumps(config,sort_keys=True,default=str).encode())
    for p in sorted(paths,key=str):
        try:s=p.stat();h.update(f'{p}:{s.st_size}:{s.st_mtime_ns}\n'.encode())
        except OSError:pass
    return h.hexdigest()

def cache_worker(entry):
    dataset,user_dir,device,calendar,params,cache_root,disable_cache=entry
    user_dir=Path(user_dir);cache_root=Path(cache_root)
    files=sorted(user_dir.rglob('*.csv'))
    sig=paths_signature(files,{**params,'device_model':device,'own_session_calendar':{str(s):t for (uid,s),t in calendar.items() if uid==user_dir.name}})
    key=hashlib.sha1((dataset+':'+str(user_dir.resolve())).encode()).hexdigest()[:20]
    f=cache_root/(dataset.lower()+'_'+key+'.json')
    if not disable_cache and f.exists() and not params['save_gestures']:
        try:
            j=json.loads(f.read_text(encoding='utf-8'))
            if j.get('signature')==sig:return j['rows'],[],j['issues'],True
        except (OSError,ValueError,KeyError):pass
    if dataset=='FETA':rows,gs,issues=feta_user(user_dir,device,calendar,params)
    else:
        rows=[];gs=[];issues=[]
        for session in sorted(x for x in user_dir.iterdir() if x.is_dir()):
            try:
                rr,gg,ii=hmog_session(session,user_dir.name,params)
                rows.extend(rr);gs.extend(gg);issues.extend(ii[:2])
            except Exception as exc:
                issues.append('HMOG '+session.name+': '+type(exc).__name__+': '+str(exc)[:140])
    f.parent.mkdir(parents=True,exist_ok=True)
    if not params['save_gestures']:
        f.write_text(json.dumps({'signature':sig,'rows':rows,'issues':issues},default=str),encoding='utf-8')
    return rows,gs,issues,False

def savecsv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def last_four_sweeps(allrows,thresholds,chosen_mode,output):
    user_sess=defaultdict(dict)
    for r in allrows:
        key=(r['dataset'],r['device_model'],r['user_id'],r['task'])
        if r['session_id'] in user_sess[key]:raise ValueError('Duplicate session+task: '+repr(key))
        user_sess[key][r['session_id']]=r
    by_group=defaultdict(list)
    roster=[];session_manifest=[]
    for (dataset,model,user,task),by_ses in sorted(user_sess.items()):
        sessions=list(by_ses.values())
        verified=all(safeint(x['order_verified']) for x in sessions)
        stamps=[num(x.get('session_time')) for x in sessions]
        verified=bool(verified and all(np.isfinite(stamps)) and len(set(stamps))==len(stamps))
        ordered=sorted(sessions,key=lambda x:float(x['session_time'])) if verified else []
        first=ordered[:4] if len(ordered)>=4 else []
        for i,r in enumerate(first,1):
            session_manifest.append({**r,'session_rank':i,'role':'enrollment' if i<=2 else 'genuine_test'})
        by_group[(dataset,model,task)].append((user,sessions,first,verified))
        row={'dataset':dataset,'device_model':model,'user_id':user,'task':task,
             'all_sessions':len(sessions),'first_four_verified':int(verified and len(first)==4),
             'first_four_ids':'|'.join(str(r['session_id']) for r in first)}
        for mode,col in [('touch','gestures_feta_touch_usable'),('strict','gestures_strict_multimodal')]:
            row[f'first_four_min_{mode}']=min((safeint(r.get(col,0)) for r in first),default='')
            row[f'first_four_counts_{mode}']='|'.join(str(safeint(r.get(col,0))) for r in first)
            for threshold in thresholds:
                row[f'eligible_{mode}_{threshold}']=int(len(first)==4 and all(safeint(r.get(col,0))>=threshold for r in first))
        roster.append(row)
    metrics=[('touch','gestures_feta_touch_usable'),('strict','gestures_strict_multimodal')]
    sweep=[];eligible_roster=[]
    sets=defaultdict(set)
    for (dataset,model,task),members in sorted(by_group.items()):
        for mode,col in metrics:
            for thr in thresholds:
                four_any=sum(sum(safeint(r.get(col,0))>=thr for r in sessions)>=4
                             for _,sessions,_,_ in members)
                first_users=[user for user,sessions,first,verified in members
                             if verified and len(first)==4 and all(safeint(r.get(col,0))>=thr for r in first)]
                sets[(dataset,model,task,mode,thr)]=set(first_users)
                sweep.append({'dataset':dataset,'device_model':model,'task':task,'count_mode':mode,
                              'threshold':thr,'total_users':len(members),
                              'users_any_four_qualifying':four_any,
                              'users_first_four_qualifying':len(first_users),
                              'users_with_verified_order':sum(bool(ok) for _,_,_,ok in members),
                              'total_qualifying_gestures_first_four':sum(sum(safeint(r.get(col,0)) for r in first)
                                           for _,ses,first,verified in members if verified and len(first)==4 and
                                           all(safeint(r.get(col,0))>=thr for r in first))})
                for user in first_users:eligible_roster.append({'dataset':dataset,'device_model':model,
                     'task':task,'count_mode':mode,'threshold':thr,'user_id':user})
    intersection=[]
    for dataset,model in sorted({(r['dataset'],r['device_model']) for r in allrows}):
        if dataset=='FETA':a,b='scroll','swipe'
        else:a,b='reading_sitting','reading_walking'
        for mode,_ in metrics:
            for thr in thresholds:
                x=sets.get((dataset,model,a,mode,thr),set())
                y=sets.get((dataset,model,b,mode,thr),set())
                intersection.append({'dataset':dataset,'device_model':model,'task_a':a,'task_b':b,
                     'count_mode':mode,'threshold':thr,'task_a_users':len(x),'task_b_users':len(y),
                     'users_both':len(x&y),'both_user_ids':'|'.join(sorted(x&y))})
    return sweep,roster,eligible_roster,intersection,session_manifest

def args_parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feta-input',type=Path,default=Path('input_data/data_files'))
    p.add_argument('--hmog-input',type=Path,default=Path('input_data/hmog_unarchived'))
    p.add_argument('--userdata',type=Path,default=Path('input_data/tables/tables/userdata.csv'))
    p.add_argument('--previous-feta-calendar',type=Path,
                   default=Path('output_data/feta_device_eligibility_v1_2/session_task_counts.csv'))
    p.add_argument('--output',type=Path,default=Path('output_data/hmog_feta_feta_criterion_v2_0'))
    p.add_argument('--datasets',choices=('both','feta','hmog'),default='both')
    p.add_argument('--thresholds',default='5,10,15,20,25,30,40,50,60,75,100,150,200,300,500')
    p.add_argument('--workers',type=int,default=max(1,min(4,(os.cpu_count() or 2)-1)))
    p.add_argument('--limit-users',type=int,default=0)
    p.add_argument('--min-sensor-samples',type=int,default=3)
    p.add_argument('--edge-gap-ms',type=float,default=40.)
    p.add_argument('--internal-gap-ms',type=float,default=120.)
    p.add_argument('--pair-tolerance-ms',type=float,default=25.)
    p.add_argument('--min-sensor-pairs',type=int,default=1)
    p.add_argument('--save-gestures',action='store_true')
    p.add_argument('--no-cache',action='store_true')
    return p.parse_args(argv)

def main(argv=None):
    args=args_parse(argv);out=args.output;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(out/'analysis.log',encoding='utf8'),logging.StreamHandler(sys.stdout)])
    thresholds=sorted(set(int(x) for x in args.thresholds.split(',') if x.strip()))
    if not thresholds or min(thresholds)<1:raise SystemExit('Thresholds must be positive integers')
    params={k:getattr(args,k) for k in ('min_sensor_samples','edge_gap_ms','internal_gap_ms',
           'pair_tolerance_ms','min_sensor_pairs','save_gestures')}
    params['feta_criteria']='min3_max_radial_strict_gt5';params['hmog_portrait_all_touch_points']=True
    jobs=[];calendar={};model_map={}
    if args.datasets in ('both','feta'):
        if not args.feta_input.is_dir() or not args.userdata.is_file():
            raise SystemExit('Missing FETA input folder or userdata.csv: '+str(args.feta_input)+' '+str(args.userdata))
        model_map=feta_user_mapping(args.userdata)
        if args.previous_feta_calendar.is_file():
            prev=pd.read_csv(args.previous_feta_calendar,dtype=str,encoding='utf-8-sig')
            if {'raw_user_id','session','session_timestamp','session_order_verified'}.issubset(prev.columns):
                for r in prev.itertuples(index=False):
                    if safeint(r.session_order_verified) and np.isfinite(num(r.session_timestamp)):
                        calendar[(str(r.raw_user_id),str(r.session))]=float(r.session_timestamp)
        users=sorted(p for p in args.feta_input.iterdir() if p.is_dir())
        if args.limit_users:users=users[:args.limit_users]
        for u in users:jobs.append(('FETA',str(u),model_map.get(u.name,'UNKNOWN'),calendar,params,str(out/'cache'),args.no_cache))
    if args.datasets in ('both','hmog'):
        if not args.hmog_input.is_dir():raise SystemExit('Missing HMOG folder: '+str(args.hmog_input))
        users=sorted(p for p in args.hmog_input.iterdir() if p.is_dir())
        if args.limit_users:users=users[:args.limit_users]
        for u in users:jobs.append(('HMOG',str(u),'Samsung Galaxy S4',{},params,str(out/'cache'),args.no_cache))
    logging.info('Start %s on %d users; criteria: >=3 samples, max radial distance >5 px; HMOG touch orientation=0',VERSION,len(jobs))
    results=[];cached=0
    if args.workers<=1:
        for i,job in enumerate(jobs,1):
            res=cache_worker(job);results.append(res);cached+=int(res[3])
            if i%20==0 or i==len(jobs):logging.info('Processed %d/%d users; cached %d',i,len(jobs),cached)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i,res in enumerate(executor.map(cache_worker,jobs),1):
                results.append(res);cached+=int(res[3])
                if i%20==0 or i==len(jobs):logging.info('Processed %d/%d users; cached %d',i,len(jobs),cached)
    allrows=[r for rows,_,_,_ in results for r in rows]
    issues=[s for _,_,w,_ in results for s in w]
    for row in allrows:
        for key in SESSION_FIELDS:
            if key.startswith(('contacts_','gestures_','excluded_')) or key=='strict_coverage_failed':
                row.setdefault(key,0)
        row.setdefault('session_time','');row.setdefault('warning','')
    savecsv(out/'session_task_counts.csv',allrows,SESSION_FIELDS)
    sweep,roster,selected,overlap,manifest=last_four_sweeps(allrows,thresholds,'strict',out)
    savecsv(out/'threshold_sweep.csv',sweep,
            ['dataset','device_model','task','count_mode','threshold','total_users','users_any_four_qualifying',
             'users_first_four_qualifying','users_with_verified_order','total_qualifying_gestures_first_four'])
    selcols=['dataset','device_model','user_id','task','all_sessions','first_four_verified','first_four_ids',
             'first_four_min_touch','first_four_counts_touch','first_four_min_strict','first_four_counts_strict']
    selcols.extend(f'eligible_{mode}_{t}' for mode in ('touch','strict') for t in thresholds)
    savecsv(out/'user_task_eligibility.csv',roster,selcols)
    savecsv(out/'eligible_user_ids_by_threshold.csv',selected,
            ['dataset','device_model','task','count_mode','threshold','user_id'])
    savecsv(out/'task_intersection.csv',overlap,
            ['dataset','device_model','task_a','task_b','count_mode','threshold','task_a_users',
             'task_b_users','users_both','both_user_ids'])
    savecsv(out/'first_four_session_manifest.csv',manifest,list(SESSION_FIELDS)+['session_rank','role'])
    if args.save_gestures:
        import gzip
        with gzip.open(out/'gestures.csv.gz','wt',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=GESTURE_FIELDS,extrasaction='ignore');w.writeheader()
            for _,gr,_,_ in results:w.writerows(gr)
    device_counts=[]
    groups=defaultdict(list)
    for row in allrows:groups[(row['dataset'],row['device_model'],row['task'])].append(row)
    for (dataset,model,task),data in sorted(groups.items()):
        device_counts.append({'dataset':dataset,'device_model':model,'task':task,
            'users':len({x['user_id'] for x in data}),'session_tasks':len(data),
            **{q:sum(safeint(x.get(q,0)) for x in data) for q in
               ('contacts_complete','gestures_feta_geometry','gestures_feta_touch_usable','gestures_strict_multimodal')},
            'average_strict_per_session':round(sum(safeint(x.get('gestures_strict_multimodal',0)) for x in data)/max(1,len(data)),3)})
    savecsv(out/'device_task_totals.csv',device_counts,
            ['dataset','device_model','task','users','session_tasks','contacts_complete','gestures_feta_geometry',
             'gestures_feta_touch_usable','gestures_strict_multimodal','average_strict_per_session'])
    summary={'version':VERSION,'users_examined':len(jobs),'sessions_task_rows':len(allrows),'cached_users':cached,
        'thresholds':thresholds,'parameters':params,'feta_userdata_joined':sum(x[0]=='FETA' and x[2]!='UNKNOWN' for x in jobs),
        'issues_first_80':issues[:80],'totals':device_counts,
        'interpretation':{'gesture':'completed physical DOWN-UP, >=3 touch points and max radial deviation >5 native px',
            'HMOG_portrait':'all touch rows within contact must have Phone_orientation == 0',
            'strict_motion':'accel and gyro each >=3 in-gesture samples, endpoint gaps<=40ms, internal gap<=120ms, paired timestamps<=25ms; HMOG in-gesture sensor orientations all portrait',
            'first_four':'first 4 CHRONOLOGICAL whole sessions; missing task means zero, not replaced by later sessions',
            'FETA_task':'scroll and swipe are two distinct interaction tasks; their contacts use the same geometric filtering',
            'HMOG_event_names':'ScrollEvent.csv and StrokeEvent.csv not counted as extra physical gestures',
            'threshold':'cohort size sensitivity; not an EER-optimal threshold'}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    logging.info('DONE. Raw counts, threshold groups and intersections: %s',out)
    for r in sweep:
        if r['count_mode']=='strict' and r['threshold'] in (20,25,30) and r['task'] in ('scroll','swipe','reading_sitting','reading_walking'):
            logging.info('%s %-18s %-17s threshold=%2d first-four=%3d any-four=%3d',r['dataset'],r['device_model'],r['task'],r['threshold'],r['users_first_four_qualifying'],r['users_any_four_qualifying'])
    return 0

if __name__=='__main__':raise SystemExit(main())
