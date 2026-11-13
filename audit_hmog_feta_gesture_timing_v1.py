#!/usr/bin/env python3
"""Inspect real timestamp spacing WITHIN individual FETA/HMOG gestures.

Uses identical gesture validity, timestamp conversion, and motion-coverage rules as
analyze_hmog_feta_feta_criterion_v2_0.py. It never assumes a uniform 100 Hz rate.
It reads only users listed in the FROZEN split JSON; by default TRAIN users only.

Python 3.14, pandas, numpy. Windows-safe multiprocessing.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import analyze_hmog_feta_feta_criterion_v2_0 as base

VERSION = 'timing-audit-1.0-20261113'
GAP_THRESHOLDS_MS = (15, 20, 30, 40, 60, 100, 120, 250)
RAW_FIELDS = [
    'dataset','device_model','user_id','role','session_id','task','activity_id','repetition',
    'strict_multimodal','feta_touch_usable','reason','duration_ms','touch_points',
    'accel_samples','gyro_samples','accel_gyro_pairs','start_ms','end_ms',
    *[f'{kind}_{metric}' for kind in ('touch','accel','gyro') for metric in (
        'dt_n','dt_zero_or_negative','dt_below_10ms','dt_near_10ms','dt_above_10ms','dt_min_ms','dt_p50_ms','dt_p95_ms','dt_max_ms',
        *(f'dt_over_{t}ms' for t in GAP_THRESHOLDS_MS)
    )],
    *[f'{kind}_edge_{edge}_ms' for kind in ('accel','gyro') for edge in ('start','end')]
]
STREAM_FIELDS = ['dataset','device_model','user_id','role','session_id','task','activity_id','repetition',
                 'sensor','samples','raw_order_backwards','raw_order_duplicates','gap_n',
                 'gap_below_10ms','gap_near_10ms','gap_above_10ms','gap_p50_ms',
                 'gap_p90_ms','gap_p95_ms','gap_p99_ms','gap_max_ms',
                 *[f'gap_over_{t}ms' for t in GAP_THRESHOLDS_MS]]

def quant(values, q):
    a = np.asarray(values,dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a,q)) if len(a) else ''

def interval_metrics(times, name):
    a=np.asarray(times,dtype=float)
    dt=np.diff(a)
    finite=dt[np.isfinite(dt)]
    pos=finite[finite>0]
    values={f'{name}_dt_below_10ms':int(np.count_nonzero(pos<9.999)),
            f'{name}_dt_near_10ms':int(np.count_nonzero((pos>=9.999)&(pos<=10.001))),
            f'{name}_dt_above_10ms':int(np.count_nonzero(pos>10.001)),
            f'{name}_dt_n':len(finite), f'{name}_dt_zero_or_negative':int(np.count_nonzero(finite<=0)),
            f'{name}_dt_min_ms':float(pos.min()) if len(pos) else '',
            f'{name}_dt_p50_ms':quant(pos,50), f'{name}_dt_p95_ms':quant(pos,95),
            f'{name}_dt_max_ms':float(pos.max()) if len(pos) else ''}
    values.update({f'{name}_dt_over_{b}ms':int(np.count_nonzero(pos>b)) for b in GAP_THRESHOLDS_MS})
    return values

def stream_metrics(times, kind, ident):
    a=np.asarray(times,dtype=float)
    a=a[np.isfinite(a)]
    raw=np.diff(a)
    sorted_times=np.sort(a)
    gap=np.diff(sorted_times)
    good=gap[gap>0]
    return dict(**ident,sensor=kind,samples=len(a),raw_order_backwards=int(np.count_nonzero(raw<0)),
        raw_order_duplicates=int(np.count_nonzero(raw==0)), gap_n=len(good),
        gap_below_10ms=int(np.count_nonzero(good<9.999)),
        gap_near_10ms=int(np.count_nonzero((good>=9.999)&(good<=10.001))),
        gap_above_10ms=int(np.count_nonzero(good>10.001)),
        gap_p50_ms=quant(good,50),gap_p90_ms=quant(good,90),gap_p95_ms=quant(good,95),
        gap_p99_ms=quant(good,99),gap_max_ms=float(good.max()) if len(good) else '',
        **{f'gap_over_{b}ms':int(np.count_nonzero(good>b)) for b in GAP_THRESHOLDS_MS})

def hm_contacts_with_times(touch, taskmap):
    if touch.empty:return []
    aid=touch.get('activityid',pd.Series(['']*len(touch))).astype(str).str.strip().to_numpy()
    tm=base.numeric(touch,'eventtime');x=base.numeric(touch,'x');y=base.numeric(touch,'y')
    action=base.numeric(touch,'actionid');pcount=base.numeric(touch,'pointercount')
    pid=base.numeric(touch,'pointerid');orientation=base.numeric(touch,'phoneorientation')
    valid=np.isfinite(tm)&np.isfinite(x)&np.isfinite(y)&np.isfinite(action)
    ix=np.flatnonzero(valid & np.isin(aid,list(taskmap)))
    ix=sorted(ix,key=lambda j:(aid[j],tm[j],int(j)))
    contacts=[];active=None
    for j in ix:
        a=aid[j];t=float(tm[j]);xx=float(x[j]);yy=float(y[j]);act=int(action[j])&255
        multi=bool(act in (5,6) or (np.isfinite(pcount[j]) and pcount[j]>1) or
                   (np.isfinite(pid[j]) and pid[j]>0))
        portrait=bool(np.isfinite(orientation[j]) and int(orientation[j])==0)
        if act==0:
            active=dict(activity_id=a,task=taskmap[a],start_ms=t,x0=xx,y0=yy,
                xlast=xx,ylast=yy,max_radial_px=0.,touch_points=1,
                portrait_only=portrait,multitouch=multi,touch_times_ms=[t])
        elif act in (5,6):
            if active is not None:active['multitouch']=True
        elif act==3:active=None
        elif act in (1,2):
            if active is None:continue
            if active['activity_id']!=a:active=None;continue
            active['touch_points']+=1
            active['touch_times_ms'].append(t)
            active['max_radial_px']=max(active['max_radial_px'],math.hypot(xx-active['x0'],yy-active['y0']))
            active['xlast']=xx;active['ylast']=yy
            active['portrait_only']&=portrait
            active['multitouch']|=multi
            if act==1:
                contacts.append({**active,'end_ms':t,'duration_ms':t-active['start_ms'],
                    'net_displacement_px':math.hypot(xx-active['x0'],yy-active['y0'])})
                active=None
    return contacts

def feta_contacts_with_times(path):
    d=base.feta_csv(path)
    if d.empty:return []
    t=base.feta_time_ms(d);x=base.numeric(d,'x');y=base.numeric(d,'y')
    phases=[c for c in ('type','touchtype','phase','action','actiontype','touchphase','eventtype','state') if c in d]
    if not phases:raise ValueError(f'No FETA touch phase column: {path}')
    col=phases[0]
    events=sorted(((t[i],x[i],y[i],d[col].iloc[i],i) for i in range(len(d))
                    if all(np.isfinite(v) for v in (t[i],x[i],y[i]))),key=lambda a:(a[0],a[4]))
    contacts=[];active=None
    for tm,xx,yy,ph,_ in events:
        t0=float(tm);xx=float(xx);yy=float(yy)
        state=base.parse_feta_phase(ph,col)
        if state=='down':
            active=dict(start_ms=t0,x0=xx,y0=yy,max_radial_px=0.,touch_points=1,
                portrait_only=True,multitouch=False,touch_times_ms=[t0])
        elif state in ('move','up'):
            if active is None:continue
            active['touch_points']+=1
            active['touch_times_ms'].append(t0)
            active['max_radial_px']=max(active['max_radial_px'],math.hypot(xx-active['x0'],yy-active['y0']))
            if state=='up':
                contacts.append({**active,'end_ms':t0,'duration_ms':t0-active['start_ms'],
                    'net_displacement_px':math.hypot(xx-active['x0'],yy-active['y0'])})
                active=None
        elif state=='cancel':active=None
    return contacts

def sensor_slice(sensor,start,end):
    if sensor is None:return np.array([],dtype=float)
    arr=sensor[0]
    a=np.searchsorted(arr,start,'left');b=np.searchsorted(arr,end,'right')
    return arr[a:b]

def timing_rows(contacts,sensors,identity,params):
    if not contacts:return []
    _counts,records=base.assess(contacts,sensors,identity,{**params,'save_gestures':True})
    results=[]
    for original,r in zip(contacts,records):
        # Keep *all* geometrically eligible/portrait contacts, not just strict,
        # so >120ms gaps rejected by the strict criterion are still visible.
        if not r['feta_touch_usable']:continue
        motion=sensors.get(original.get('activity_id',''),sensors.get('',{}))
        row={k:identity.get(k,'') for k in ('dataset','device_model','user_id','role','session_id','task')}
        row.update(activity_id=original.get('activity_id',''),repetition=original.get('repetition',''),
                   strict_multimodal=r['strict_multimodal'],feta_touch_usable=r['feta_touch_usable'],
                   reason=r['reason'],duration_ms=r['duration_ms'],touch_points=r['touch_points'],
                   accel_samples=r['accel_samples'],gyro_samples=r['gyro_samples'],
                   accel_gyro_pairs=r['accel_gyro_pairs'],start_ms=r['start_ms'],end_ms=r['end_ms'])
        row.update(interval_metrics(original['touch_times_ms'],'touch'))
        for kind in ('accel','gyro'):
            interval=sensor_slice(motion.get(kind),original['start_ms'],original['end_ms'])
            row.update(interval_metrics(interval,kind))
            row[f'{kind}_edge_start_ms']=float(interval[0]-original['start_ms']) if len(interval) else ''
            row[f'{kind}_edge_end_ms']=float(original['end_ms']-interval[-1]) if len(interval) else ''
        results.append(row)
    return results

def hmog_user(path,model,role,tasks,params,first_four,session_limit):
    all_rows=[];streams=[];errors=[]
    for sess in sorted(p for p in path.iterdir() if p.is_dir()):
        folder=sess
        if session_limit and sess.name not in set().union(*first_four.values()):continue
        try:
            af=base.first_existing(folder,base.H_FILENAMES['activity'])
            if af is None:continue
            activity=base.read_hmog(af,'activity')
            ids=activity.get('id',pd.Series(['']*len(activity))).astype(str).str.strip().to_numpy()
            task_ids=base.numeric(activity,'taskid')
            taskmap={ids[i]:base.HMOG_TASKS[int(task_ids[i])] for i in range(len(ids))
                if np.isfinite(task_ids[i]) and int(task_ids[i]) in base.HMOG_TASKS and
                   base.HMOG_TASKS[int(task_ids[i])] in tasks and
                   (not session_limit or sess.name in first_four.get(base.HMOG_TASKS[int(task_ids[i])],set()))}
            if not taskmap:continue
            touchf=base.first_existing(folder,base.H_FILENAMES['touch'])
            tf=base.read_hmog(touchf,'touch') if touchf else pd.DataFrame()
            gestures=hm_contacts_with_times(tf,taskmap)
            motion={aid:{} for aid in taskmap}
            for kind in ('accel','gyro'):
                p=base.first_existing(folder,base.H_FILENAMES[kind])
                frame=base.read_hmog(p,kind) if p else pd.DataFrame()
                if frame.empty:continue
                time=base.numeric(frame,'eventtime')/1e6
                aids=frame.get('activityid',pd.Series(['']*len(frame))).astype(str).str.strip().to_numpy()
                for aid,task in taskmap.items():
                    mask=(aids==aid)
                    times=time[mask]
                    motion[aid][kind]=base.sensor_data(frame.loc[mask],times,True)
                    ident=dict(dataset='HMOG',device_model=model,user_id=path.name,role=role,
                               session_id=sess.name,task=task,activity_id=aid,repetition='')
                    streams.append(stream_metrics(times,kind,ident))
            for task in tasks:
                relevant=[x for x in gestures if x['task']==task]
                if not relevant:continue
                ident=dict(dataset='HMOG',device_model=model,user_id=path.name,role=role,
                           session_id=sess.name,task=task)
                all_rows.extend(timing_rows(relevant,motion,ident,params))
        except Exception as exc:
            errors.append(f'HMOG {path.name}/{folder.name}: {type(exc).__name__}: {exc}')
    return all_rows,streams,errors

def feta_user(path,model,role,tasks,params,first_four,session_limit):
    all_rows=[];streams=[];errors=[]
    for session in sorted(p for p in path.iterdir() if p.is_dir()):
        if session_limit and session.name not in first_four:continue
        for task in tasks:
            tdir=session/task
            if not tdir.is_dir():continue
            for rep in sorted(p for p in tdir.iterdir() if p.is_dir()):
                ident=dict(dataset='FETA',device_model=model,user_id=path.name,role=role,
                           session_id=session.name,task=task)
                contacts=[];motion={}
                for f in sorted((rep/'touch_data').glob('*.csv')):
                    try:
                        g=feta_contacts_with_times(f)
                        for x in g:x['repetition']=rep.name
                        contacts.extend(g)
                    except Exception as exc:errors.append(f'touch {f}: {type(exc).__name__}: {exc}')
                for kind,sub in [('accel','accelerometer_data'),('gyro','gyroscope_data')]:
                    alltime=[]
                    for f in sorted((rep/sub).glob('*.csv')):
                        try:
                            d=base.feta_csv(f)
                            if d.empty:continue
                            t=base.feta_time_ms(d)
                            alltime.append(t)
                            stream_ident=dict(**ident,activity_id='',repetition=rep.name)
                            streams.append(stream_metrics(t,kind,stream_ident))
                        except Exception as exc:errors.append(f'{kind} {f}: {type(exc).__name__}: {exc}')
                    v=np.concatenate(alltime) if alltime else np.array([],dtype=float)
                    v=np.sort(v[np.isfinite(v)])
                    motion[kind]=(v,np.ones(len(v),dtype=bool))
                if contacts:
                    all_rows.extend(timing_rows(contacts,{'':motion},ident,params))
    return all_rows,streams,errors

def save_chunk(filename,rows,fields):
    filename.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(filename,'wt',newline='',encoding='utf-8') as handle:
        wr=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        wr.writeheader();wr.writerows(rows)

def worker(job):
    dataset,model,user,role,tasks,first_four,home,cache,session_limit,params=job
    path=Path(home)/user
    if not path.is_dir():return None,None,[f'Missing user directory: {path}']
    if dataset=='HMOG':rows,streams,err=hmog_user(path,model,role,tasks,params,first_four,session_limit)
    else:rows,streams,err=feta_user(path,model,role,tasks,params,first_four,session_limit)
    hashid=hashlib.sha1(f'{dataset}/{user}'.encode()).hexdigest()[:16]
    a=Path(cache)/(hashid+'.gesture.csv.gz');b=Path(cache)/(hashid+'.streams.csv.gz')
    save_chunk(a,rows,RAW_FIELDS);save_chunk(b,streams,STREAM_FIELDS)
    return str(a),str(b),err

def make_jobs(args,splits):
    j=[]
    roles=('train','validation','test') if args.roles=='all' else tuple(args.roles.split(','))
    sess= splits['session_index']
    for model,parts in splits['splits']['FETA'].items():
        for role in roles:
            for uid in parts.get(role,[]):
                first_four=set(sess['FETA'][model][uid]['scroll'])
                first_four.update(sess['FETA'][model][uid]['swipe'])
                j.append(('FETA',model,uid,role,('scroll','swipe'),first_four,
                    str(args.feta_input),str(args.output/'chunks'),role!='train',args.params))
    hm_roles=Counter()
    hm_sessions={}
    for task in ('reading_sitting','reading_walking'):
        for role in roles:
            for uid in splits['splits']['HMOG'][task].get(role,[]):
                key=(uid,role)
                hm_roles.setdefault(key,set()).add(task)
                hm_sessions.setdefault(key,{})[task]=set(sess['HMOG']['Samsung Galaxy S4'][uid][task])
    for (uid,role),tasks in hm_roles.items():
        j.append(('HMOG','Samsung Galaxy S4',uid,role,tuple(sorted(tasks)),hm_sessions[(uid,role)],
            str(args.hmog_input),str(args.output/'chunks'),role!='train',args.params))
    return j

def summary_csv(chunks,out):
    # Streaming aggregation: no need to load the entire raw dataset into memory.
    groups={}
    interval_counts=defaultdict(Counter)
    for filepath in chunks:
        if not filepath:continue
        for part in pd.read_csv(filepath,compression='gzip',chunksize=20000,low_memory=False):
            part=part[(part['dataset']=='FETA')| (part['task'].isin(['reading_sitting','reading_walking']))]
            for (dataset,model,task,role,strict),group in part.groupby(
                ['dataset','device_model','task','role','strict_multimodal'],dropna=False):
                key=(dataset,model,task,role,int(strict))
                if key not in groups:groups[key]={k:[] for k in ('duration_ms','touch_points','accel_samples','gyro_samples',
                      'touch_dt_p50_ms','accel_dt_p50_ms','gyro_dt_p50_ms',
                      'touch_dt_p95_ms','accel_dt_p95_ms','gyro_dt_p95_ms',
                      'touch_dt_max_ms','accel_dt_max_ms','gyro_dt_max_ms','accel_edge_start_ms','accel_edge_end_ms',
                      'gyro_edge_start_ms','gyro_edge_end_ms')}
                for k in groups[key]:
                    groups[key][k].extend(pd.to_numeric(group[k],errors='coerce').dropna().to_numpy(dtype=float).tolist())
                for kind in ('touch','accel','gyro'):
                    for suffix in ('dt_n','dt_zero_or_negative','dt_below_10ms','dt_near_10ms','dt_above_10ms',
                                   'dt_over_20ms','dt_over_40ms','dt_over_100ms','dt_over_120ms'):
                        column=f'{kind}_{suffix}'
                        interval_counts[key][column] += int(pd.to_numeric(group[column],errors='coerce').fillna(0).sum())
    rows=[]
    for key,data in sorted(groups.items()):
        row=dict(zip(['dataset','device_model','task','role','strict_multimodal'],key))
        row['gestures']=len(data['duration_ms'])
        for k,values in data.items():
            row[k+'_p05']=quant(values,5);row[k+'_p25']=quant(values,25)
            row[k+'_p50']=quant(values,50);row[k+'_p75']=quant(values,75)
            row[k+'_p90']=quant(values,90);row[k+'_p95']=quant(values,95)
            row[k+'_p99']=quant(values,99);row[k+'_max']=float(max(values)) if values else ''
        for kind in ('touch','accel','gyro'):
            c=interval_counts[key]
            positive=max(0,c[f'{kind}_dt_n']-c[f'{kind}_dt_zero_or_negative'])
            row[f'{kind}_positive_intervals']=positive
            row[f'{kind}_nonpositive_intervals']=c[f'{kind}_dt_zero_or_negative']
            for suffix in ('dt_below_10ms','dt_near_10ms','dt_above_10ms',
                           'dt_over_20ms','dt_over_40ms','dt_over_100ms','dt_over_120ms'):
                val=c[f'{kind}_{suffix}']
                row[f'{kind}_{suffix}_pct']=round(100*val/positive,3) if positive else ''
            longest=np.asarray(data[f'{kind}_dt_max_ms'],dtype=float)
            for t in (20,40,100,120):
                row[f'{kind}_gestures_with_gap_over_{t}ms_pct']=round(
                    100*float(np.count_nonzero(longest>t))/max(1,row['gestures']),3)
        rows.append(row)
    if rows:pd.DataFrame(rows).to_csv(out/'timing_summary.csv',index=False,encoding='utf-8-sig')
    return rows

def stream_summary_csv(chunks,out):
    data=[]
    for filepath in chunks:
        if not filepath:continue
        data.append(pd.read_csv(filepath,compression='gzip',low_memory=False))
    if not data:return []
    df=pd.concat(data,ignore_index=True)
    if df.empty:return []
    for key in ('samples','gap_n','raw_order_backwards','raw_order_duplicates',
                'gap_below_10ms','gap_near_10ms','gap_above_10ms',
                *(f'gap_over_{t}ms' for t in GAP_THRESHOLDS_MS)):
        df[key]=pd.to_numeric(df[key],errors='coerce').fillna(0)
    rows=[]
    for (dataset,model,task,role,sensor),part in df.groupby(
            ['dataset','device_model','task','role','sensor']):
        n=float(part['gap_n'].sum())
        r=dict(dataset=dataset,device_model=model,task=task,role=role,sensor=sensor,
               streams=len(part),samples=int(part['samples'].sum()),
               positive_gap_count=int(n),raw_backwards=int(part['raw_order_backwards'].sum()),
               duplicate_timestamps=int(part['raw_order_duplicates'].sum()),
               median_of_stream_median_ms=quant(pd.to_numeric(part['gap_p50_ms'],errors='coerce'),50),
               p95_of_stream_max_gap_ms=quant(pd.to_numeric(part['gap_max_ms'],errors='coerce'),95),
               maximum_gap_ms=float(pd.to_numeric(part['gap_max_ms'],errors='coerce').max()))
        for side in ('below','near','above'):
            c=int(part[f'gap_{side}_10ms'].sum())
            r[f'gaps_{side}_10ms']=c
            r[f'gaps_{side}_10ms_pct']=round(c/n*100,3) if n else ''
        for t in GAP_THRESHOLDS_MS:
            c=int(part[f'gap_over_{t}ms'].sum())
            r[f'gaps_over_{t}ms']=c
            r[f'gaps_over_{t}ms_pct']=round(c/n*100,3) if n else ''
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/'stream_gap_summary.csv',index=False,encoding='utf-8-sig')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,default=Path('hmog_feta_splits_seed_20261112.json'))
    p.add_argument('--feta-input',type=Path,default=Path('input_data/data_files'))
    p.add_argument('--hmog-input',type=Path,default=Path('input_data/hmog_unarchived'))
    p.add_argument('--output',type=Path,default=Path('output_data/hmog_feta_gesture_timing_v1'))
    p.add_argument('--roles',default='train',help='train (default), validation, test, train,validation, or all')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit-users',type=int,default=0,help='Dry run: only first N jobs')
    args=p.parse_args()
    if not args.splits.is_file():p.error('Frozen JSON not found: '+str(args.splits))
    if not args.feta_input.is_dir() or not args.hmog_input.is_dir():p.error('Raw dataset directories missing')
    splits=json.loads(args.splits.read_text(encoding='utf-8'))
    if splits.get('status')!='FROZEN_VERIFIED_AGAINST_V2_AUDIT':p.error('Not a verified splits JSON')
    args.params=dict(min_sensor_samples=3,edge_gap_ms=40.,internal_gap_ms=120.,
                     pair_tolerance_ms=25.,min_sensor_pairs=1)
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=make_jobs(args,splits)
    if args.limit_users:jobs=jobs[:args.limit_users]
    print(f'Analyzing {len(jobs)} unique user-directory jobs; role={args.roles}',flush=True)
    gesture_chunks=[];stream_chunks=[];errors=[]
    if args.workers<=1:
        results=map(worker,jobs)
        for idx,(a,b,e) in enumerate(results,1):
            if a:gesture_chunks.append(a)
            if b:stream_chunks.append(b)
            errors.extend(e)
            if idx%10==0 or idx==len(jobs):print(f'{idx}/{len(jobs)} users',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,job) for job in jobs]
            for idx,f in enumerate(as_completed(futures),1):
                a,b,e=f.result()
                if a:gesture_chunks.append(a)
                if b:stream_chunks.append(b)
                errors.extend(e)
                if idx%10==0 or idx==len(jobs):print(f'{idx}/{len(jobs)} users',flush=True)
    print('Compiling per-gesture temporal summary...',flush=True)
    rows=summary_csv(gesture_chunks,args.output)
    stream_rows=stream_summary_csv(stream_chunks,args.output)
    (args.output/'issues.json').write_text(json.dumps(errors[:1000],ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'audit_metadata.json').write_text(json.dumps(dict(version=VERSION,
        splits_sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest(),roles=args.roles,
        users=len(jobs),summary_rows=len(rows),stream_summary_rows=len(stream_rows),errors_count=len(errors),gap_thresholds_ms=GAP_THRESHOLDS_MS,
        note='strict=1 includes existing max internal gap <=120ms; strict=0 exposes rejected long gaps',
        module=base.VERSION),indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'DONE. {args.output / "timing_summary.csv"}; per-user gesture CSV in chunks/; errors={len(errors)}')
    print('Raw sensor timing by task: stream_gap_summary.csv; per-gesture details: chunks/*.gesture.csv.gz')

if __name__=='__main__':main()
