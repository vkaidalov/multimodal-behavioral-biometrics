#!/usr/bin/env python3
"""HMOG/FETA Session Explorer — interactive replay of raw touchscreen and motion streams.

Python >= 3.10. Requires ONLY Python's standard library (tkinter included in
most Windows/macOS distributions; on some Linux distros install python3-tk).

Usage:
    python hmog_session_explorer.py --input input_data/hmog_unarchived
    python hmog_session_explorer.py --session input_data/hmog_unarchived/918136/19

Data sources and clocks:
- TouchEvent_im.csv is preferred over TouchEvent.csv, as in HMOG audit v1.5;
  switch in the toolbar to inspect either file, NEVER merge them.
- TouchEvent.EventTime and KeyPressEvent.PressTime are milliseconds.
- Motion EventTime is in nanoseconds and divided by 1,000,000.
- KeyPressEvent is OPTIONAL, DIAGNOSTIC ONLY; it NEVER creates touch gestures.
- Android touch and sensor relative-time clock origins may differ, so sensor
  graphs warn rather than silently imposing a time shift.
- Screen 1080x1920 is an EDITABLE INITIAL ASSUMPTION, not a verified per-file
  property. The phone aspect ratio is preserved, not estimated from touched area.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import queue
import re
import statistics
import sys
import threading
import time
import tkinter as tk
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

TASKS = {
    1:'Reading · sitting', 7:'Reading · sitting', 13:'Reading · sitting', 19:'Reading · sitting',
    2:'Reading · walking', 8:'Reading · walking', 14:'Reading · walking', 20:'Reading · walking',
    3:'Writing · sitting', 9:'Writing · sitting', 15:'Writing · sitting', 21:'Writing · sitting',
    4:'Writing · walking', 10:'Writing · walking', 16:'Writing · walking', 22:'Writing · walking',
    5:'Map · sitting', 11:'Map · sitting', 17:'Map · sitting', 23:'Map · sitting',
    6:'Map · walking', 12:'Map · walking', 18:'Map · walking', 24:'Map · walking',
}
SCHEMA = {
    'touch':'Systime EventTime ActivityID Pointer_count PointerID ActionID X Y Pressure Contact_size Phone_orientation'.split(),
    'key':'Systime PressTime ActivityID PressType KeyID Phone_orientation'.split(),
    'accel':'Systime EventTime ActivityID X Y Z Phone_orientation'.split(),
    'gyro':'Systime EventTime ActivityID X Y Z Phone_orientation'.split(),
    'activity':'ID SubjectID Session_number Start_time End_time Relative_Start_time Relative_End_time Gesture_scenario TaskID ContentID'.split(),
}
BG='#10151E'; PANEL='#18212E'; PANEL2='#202C3C'; FG='#E4ECF4'; MUTED='#9EACBD'
CYAN='#50D5F2'; GREEN='#69D5A6'; GOLD='#FFCD79'; RED='#FF8E91'; BLUE='#97A8FF'
ACT_NAMES={0:'DOWN',1:'UP',2:'MOVE',3:'CANCEL',5:'POINTER↓',6:'POINTER↑'}
SPECIAL_KEYS={32:'Space', 10:'Enter', 13:'Enter', -1:'Shift', -2:'Mode', -3:'Cancel', -4:'Done', -5:'Backspace', -6:'Alt'}


def norm(s):
    return re.sub('[^a-z0-9]','',str(s).lower().lstrip('\ufeff'))


def records(path:Path, kind):
    """Stream headerless or headered HMOG CSV; tolerate optional BOM and delimiters."""
    if not path or not path.is_file():
        return
    cols=[norm(c) for c in SCHEMA[kind]]
    with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
        first=f.readline()
        if not first.strip():return
        sep=max((',', '\t',';'),key=first.count)
        f.seek(0)
        rdr=csv.reader(f,delimiter=sep)
        try:head=next(rdr)
        except StopIteration:return
        have_header=len(set(norm(c) for c in head)&set(cols))>=2
        names=[norm(c) for c in head] if have_header else cols
        if not have_header:
            yield dict(zip(names,head))
        for vals in rdr:
            if vals and any(v.strip() for v in vals):
                yield dict(zip(names,vals))


def num(x,default=math.nan):
    try:return float(x)
    except (TypeError,ValueError):return default


def integer(x,default=-1):
    f=num(x)
    return int(f) if math.isfinite(f) else default


def epoch_ms(raw):
    v=num(raw)
    if not math.isfinite(v):return math.nan
    a=abs(v)
    if a>1e17:return v/1e6
    if a>1e14:return v/1e3
    if a>1e11:return v
    if a>1e8:return v*1000
    return v


def rgb(hex_color):return tuple(int(hex_color[i:i+2],16) for i in (1,3,5))


def blend(a,b,alpha):
    x,y=rgb(a),rgb(b)
    return '#'+''.join(f'{round(j*(1-alpha)+i*alpha):02x}' for i,j in zip(x,y))


def key_name(k):
    if k in SPECIAL_KEYS:return SPECIAL_KEYS[k]
    if 33 <= k <= 126:return chr(k)
    return str(k)


def action_name(code):return ACT_NAMES.get(code,f'#{code}')


@dataclass(slots=True)
class Touch:
    seq:int; t:float; act:str; action:int; x:float; y:float; pc:int; pid:int
    orient:int; pressure:float; size:float; system:float
    gesture:int=-1
    keys:list[int]=field(default_factory=list)

@dataclass(slots=True)
class Key:
    seq:int; t:float; act:str; typ:int; code:int; orient:int; system:float
    touch:int=-1; match_ms:float=math.nan

@dataclass(slots=True)
class Sensor:
    t:float; x:float; y:float; z:float; ori:int; act:str; system:float

@dataclass
class Gesture:
    number:int; act:str; indices:list[int]; start:float; end:float
    completed:bool=False; multitouch:bool=False; portrait:bool=True
    kind:str='incomplete'; path:float=0.; displacement:float=0.

@dataclass
class Session:
    folder:Path; source:str; touches:list[Touch]; keys:list[Key]
    accel:list[Sensor]; gyro:list[Sensor]; gestures:list[Gesture]
    activities:dict[str,str]; t0:float; t1:float
    touch_times:list[float]; accel_times:list[float]; gyro_times:list[float]
    key_times:list[float]; bounds:tuple[float,float]; warning:str=''


def choose_source(folder,option):
    names={'Auto':['TouchEvent_im.csv','TouchEvent.csv'],
           'TouchEvent_im.csv':['TouchEvent_im.csv'],
           'TouchEvent.csv':['TouchEvent.csv']}.get(option,[option])
    for n in names:
        p=folder/n
        if p.is_file():return p
    raise FileNotFoundError('Cannot find requested TouchEvent file in '+str(folder))


def load_session(folder:Path,source_option='Auto',progress=None)->Session:
    folder=Path(folder)
    src=choose_source(folder,source_option)
    report=lambda m: progress(m) if progress else None
    activities={}
    for r in records(folder/'Activity.csv','activity'):
        aid=str(r.get('id','')).strip()
        task=integer(r.get('taskid'))
        activities[aid]=TASKS.get(task,f'TaskID {task}')
    report('Reading '+src.name+' …')
    touches=[]
    for rowno,r in enumerate(records(src,'touch')):
        t=num(r.get('eventtime'));x=num(r.get('x'));y=num(r.get('y'))
        if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
            continue
        touches.append(Touch(rowno,t,str(r.get('activityid','')).strip(),
                             integer(r.get('actionid')) & 255,x,y,integer(r.get('pointercount')),
                             integer(r.get('pointerid')),integer(r.get('phoneorientation')),
                             num(r.get('pressure')),num(r.get('contactsize')),
                             epoch_ms(r.get('systime'))))
    if not touches:raise ValueError('No valid X/Y/EventTime touch samples in '+src.name)
    touches.sort(key=lambda p:(p.t,p.seq))
    report(f'Reconstructing touch contacts ({len(touches):,} events) …')
    gestures=[]; active={}
    for i,p in enumerate(touches):
        a=p.act;code=p.action
        multi=(p.pc>1 or p.pid>0 or code in (5,6))
        if code==0:
            if a in active:
                old=gestures[active[a]]
                old.end=p.t
            g=Gesture(len(gestures)+1,a,[i],p.t,p.t,multitouch=multi,portrait=(p.orient==0))
            gestures.append(g);active[a]=len(gestures)-1;p.gesture=g.number
        elif a in active:
            g=gestures[active[a]]
            g.indices.append(i);p.gesture=g.number;g.end=p.t
            g.multitouch|=multi
            g.portrait &= (p.orient==0)
            if code==1:
                g.completed=True;active.pop(a,None)
            elif code==3:
                active.pop(a,None)
            elif code in (5,6):
                g.multitouch=True
        # Orphan UP/MOVE still remains visible in full touch-event history.
    for g in gestures:
        points=[touches[i] for i in g.indices]
        g.path=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(points,points[1:]))
        g.displacement=math.hypot(points[-1].x-points[0].x,points[-1].y-points[0].y)
        if g.completed:
            g.kind='swipe' if g.displacement>=30 or g.path>=50 else 'tap'
        else:g.kind='incomplete'

    report('Reading optional KeyPressEvent.csv …')
    keys=[]
    for rowno,r in enumerate(records(folder/'KeyPressEvent.csv','key')):
        t=num(r.get('presstime'))
        if not math.isfinite(t):continue
        keys.append(Key(rowno,t,str(r.get('activityid','')).strip(),
                        integer(r.get('presstype')),integer(r.get('keyid')),
                        integer(r.get('phoneorientation')),epoch_ms(r.get('systime'))))
    keys.sort(key=lambda k:(k.t,k.seq))
    # Same ActivityID + correct DOWN/UP phase; nearest unmatched raw TouchEvent.
    # EXACT matches first, then ±10ms tolerance. Neither creates gestures.
    matching=defaultdict(list)
    for i,p in enumerate(touches):
        if p.action in (0,1):matching[(p.act,p.action)].append((p.t,i))
    assigned=set()
    for phase_tol in (0,10):
        for ki,k in enumerate(keys):
            if k.touch>=0 or k.typ not in (0,1):continue
            candidates=matching.get((k.act,k.typ),[])
            if not candidates:continue
            # Each candidate list already sorted by t, but expensive full time-array
            # construction once per Key is avoided by pre-computed phase times below.
            # Binary-search tuple list; the second tuple element breaks time ties.
            pos=bisect.bisect_left(candidates,(k.t,-1))
            best=None
            for j in range(max(0,pos-8),min(len(candidates),pos+9)):
                tt,ti=candidates[j]
                err=abs(tt-k.t)
                if ti not in assigned and err<=phase_tol and (best is None or err<best[0]):
                    best=(err,ti)
            if best:
                err,ti=best;k.touch=ti;k.match_ms=err
                assigned.add(ti);touches[ti].keys.append(ki)

    def read_sensor(name,filename):
        out=[]
        p=folder/filename
        if not p.exists() and name=='gyro':p=folder/'Gyroscrope.csv'
        if not p.exists():return out
        report('Reading '+p.name+' …')
        for r in records(p,name):
            raw=num(r.get('eventtime'))
            x=num(r.get('x'));y=num(r.get('y'));z=num(r.get('z'))
            if not all(map(math.isfinite,(raw,x,y,z))):continue
            out.append(Sensor(raw/1e6,x,y,z,integer(r.get('phoneorientation')),
                              str(r.get('activityid','')).strip(),epoch_ms(r.get('systime'))))
        out.sort(key=lambda v:v.t)
        return out
    accel=read_sensor('accel','Accelerometer.csv')
    gyro=read_sensor('gyro','Gyroscope.csv')
    clock_warnings=[]
    ts=[p.system-p.t for p in touches if math.isfinite(p.system)]
    if ts:
        tbase=statistics.median(ts)
        for label,ss in (('Accelerometer',accel),('Gyroscope',gyro)):
            offsets=[s.system-s.t for s in ss if math.isfinite(s.system)]
            if offsets:
                difference=statistics.median(offsets)-tbase
                if abs(difference)>250:
                    clock_warnings.append(f'{label} vs touch clock: {difference:+.0f} ms (not shifted)')
    mx=max(t.x for t in touches);my=max(t.y for t in touches)
    return Session(folder,src.name,touches,keys,accel,gyro,gestures,activities,
                   touches[0].t,touches[-1].t,[p.t for p in touches],
                   [s.t for s in accel],[s.t for s in gyro],
                   [k.t for k in keys],(mx,my),'; '.join(clock_warnings))


def discover_sessions(root:Path):
    byuser=defaultdict(list)
    if not root.exists():return {}
    for base,dirs,files in os.walk(root):
        if not ('TouchEvent.csv' in files or 'TouchEvent_im.csv' in files):continue
        folder=Path(base)
        rel=folder.relative_to(root).parts
        user=next((x for x in rel if re.fullmatch(r'\d{6}',x)),None)
        if user is None:
            m=next((re.match(r'(\d{6})[_-]session',x) for x in rel if re.match(r'(\d{6})[_-]session',x)),None)
            user=m.group(1) if m else 'Other'
        byuser[user].append(folder)
    return {u:sorted(fs,key=lambda p:(int(p.name) if p.name.isdigit() else 999999,p.name))
            for u,fs in sorted(byuser.items())}


class PagedTable(ttk.Frame):
    """Low-memory, responsive view of the COMPLETE history, 300 rows per page."""
    def __init__(self,parent,columns,widths,on_pick,page_size=300):
        super().__init__(parent)
        self.columns=columns;self.widths=widths;self.pick=on_pick;self.page_size=page_size
        self.data=[];self.formatter=None;self.page=-1;self.is_updating=False
        self._pending_select=None
        bar=ttk.Frame(self);bar.pack(side='top',fill='x',pady=(0,4))
        ttk.Button(bar,text='◀',width=3,command=lambda:self.change_page(-1)).pack(side='left')
        ttk.Button(bar,text='▶',width=3,command=lambda:self.change_page(1)).pack(side='left',padx=3)
        self.info=ttk.Label(bar,text='0 rows');self.info.pack(side='left',padx=8)
        ttk.Label(bar,text='Go to #').pack(side='right',padx=(4,0))
        self.seek_var=tk.StringVar(value='1')
        entry=ttk.Entry(bar,textvariable=self.seek_var,width=9)
        entry.pack(side='right',padx=(3,0))
        entry.bind('<Return>',self.goto)
        area=ttk.Frame(self);area.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(area,columns=columns,show='headings',selectmode='browse')
        for c,w in zip(columns,widths):
            self.tree.heading(c,text=c)
            self.tree.column(c,width=w,minwidth=min(40,w),stretch=c in ('Key events','Activity'))
        sb=ttk.Scrollbar(area,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left',fill='both',expand=True)
        sb.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self.selection)

    def set_rows(self,data,formatter):
        self.data=data;self.formatter=formatter;self.page=-1;self.render_page(0)

    def render_page(self,page,select_index=None):
        self.is_updating=True
        count=len(self.data)
        pages=max(1,math.ceil(count/self.page_size))
        page=max(0,min(page,pages-1))
        self.page=page
        self.tree.delete(*self.tree.get_children())
        start=page*self.page_size;end=min(count,start+self.page_size)
        if self.formatter:
            for i in range(start,end):
                self.tree.insert('','end',iid=str(i),values=self.formatter(i,self.data[i]))
        self.info.configure(text=f'{start+1 if count else 0}–{end} / {count:,}  ·  page {page+1}/{pages}')
        self.is_updating=False
        if select_index is not None:self.select(select_index)

    def change_page(self,step):self.render_page(self.page+step)
    def goto(self,event=None):
        try:n=int(self.seek_var.get())-1
        except ValueError:return
        if 0<=n<len(self.data):
            self.show_index(n,force=True)
            self.pick(n)

    def show_index(self,i,force=False):
        if not 0<=i<len(self.data):return
        page=i//self.page_size
        if page!=self.page:self.render_page(page)
        if force or self.tree.selection()!= (str(i),):self.select(i)

    def select(self,i):
        iid=str(i)
        if not self.tree.exists(iid):return
        if self.tree.selection() == (iid,):
            self.tree.see(iid)
            return
        self._pending_select=iid
        self.is_updating=True
        self.tree.selection_set(iid);self.tree.see(iid)
        self.is_updating=False

    def selection(self,event):
        if self.is_updating:return
        s=self.tree.selection()
        if not s:return
        if s[0]==self._pending_select:
            self._pending_select=None
            return
        self._pending_select=None
        self.pick(int(s[0]))


class MotionPlot(tk.Canvas):
    def __init__(self,parent,title):
        super().__init__(parent,bg=PANEL,highlightthickness=0,height=112)
        self.title=title;self.samples=[];self.times=[];self.tt=0
        self.bind('<Configure>',lambda e:self.draw())

    def set_samples(self,samples):
        self.samples=samples;self.times=[x.t for x in samples];self.draw()

    def set_time(self,t):self.tt=t;self.draw()

    def draw(self):
        self.delete('all')
        w=max(100,self.winfo_width());h=max(70,self.winfo_height())
        self.create_text(12,9,anchor='nw',text=self.title,fill=FG,font=('Segoe UI',9,'bold'))
        if not self.samples:
            self.create_text(w/2,h/2,text='No sensor data',fill=MUTED);return
        idx=bisect.bisect_right(self.times,self.tt)-1
        if idx<0 or self.tt-self.times[idx]>800:
            self.create_text(w/2,h/2,text='No synchronized samples at this time',fill=MUTED)
            return
        last=self.samples[idx]
        self.create_text(12,26,anchor='nw',text=f'X {last.x:+.2f}     Y {last.y:+.2f}     Z {last.z:+.2f}',fill=MUTED,font=('Consolas',8))
        start=bisect.bisect_left(self.times,self.tt-7000)
        seg=self.samples[start:idx+1]
        if not seg:return
        stride=max(1,math.ceil(len(seg)/220));seg=seg[::stride]
        left=12;right=w-12;top=48;bottom=h-14
        self.create_line(left,(top+bottom)/2,right,(top+bottom)/2,fill='#344153',dash=(3,5))
        values=[val for p in seg for val in (p.x,p.y,p.z)]
        span=max(1,max(abs(v) for v in values))
        if len(seg)<2:return
        for axis,col in (('x',GOLD),('y',GREEN),('z',BLUE)):
            coords=[]
            for p in seg:
                xx=left+(p.t-(self.tt-7000))/7000*(right-left)
                yy=(top+bottom)/2-getattr(p,axis)/span*(bottom-top)*.43
                coords.extend((xx,yy))
            if len(coords)>=4:self.create_line(*coords,fill=col,width=1.7,smooth=False)


class Explorer(tk.Tk):
    def __init__(self,input_dir=None,session=None):
        super().__init__()
        self.title('HMOG Session Explorer  ·  Touch / Motion / Keyboard audit')
        self.geometry('1550x930');self.minsize(1050,700)
        self.configure(bg=BG)
        self.style=ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('.',background=BG,foreground=FG,font=('Segoe UI',10))
        self.style.configure('TFrame',background=BG)
        self.style.configure('TLabel',background=BG,foreground=FG)
        self.style.configure('TButton',background=PANEL2,foreground=FG,borderwidth=0,padding=(9,5))
        self.style.map('TButton',background=[('active','#35465A')])
        self.style.configure('TCombobox',fieldbackground=PANEL2,background=PANEL2,foreground=FG)
        self.style.configure('TEntry',fieldbackground=PANEL2,foreground=FG)
        self.style.configure('Treeview',background=PANEL,fieldbackground=PANEL,foreground=FG,
                             rowheight=25,borderwidth=0)
        self.style.configure('Treeview.Heading',background=PANEL2,foreground=FG,padding=(3,8))
        self.style.map('Treeview',background=[('selected','#245678')],foreground=[('selected','#FFFFFF')])
        self.style.configure('TNotebook',background=BG,borderwidth=0)
        self.style.configure('TNotebook.Tab',background=PANEL2,foreground=FG,padding=(12,6))
        self.style.map('TNotebook.Tab',background=[('selected','#30465D')])
        self.root_dir=Path(input_dir or 'input_data/hmog_unarchived').expanduser().resolve()
        self.session=None;self.sessions={};self.current=0.;self.playing=False
        self.last_perf=None;self.loading=False;self.load_queue=queue.Queue()
        self.frame_ms=33;self.last_follow=-1;self.selected_gesture=-1
        self.screen_w=tk.StringVar(value='1080');self.screen_h=tk.StringVar(value='1920')
        self.fade_sec=tk.StringVar(value='25');self.speed_var=tk.StringVar(value='1×')
        self.source_var=tk.StringVar(value='Auto')
        self.user_var=tk.StringVar();self.session_var=tk.StringVar()
        self.show_key=tk.BooleanVar(value=True);self.show_sensor=tk.BooleanVar(value=True)
        self.show_grid=tk.BooleanVar(value=True);self.follow=tk.BooleanVar(value=True)
        self.show_arrow=tk.BooleanVar(value=True)
        self.info_var=tk.StringVar(value='Choose an HMOG root directory and a session')
        self.clock_var=tk.StringVar(value='')
        self.play_var=tk.StringVar(value='▶ Play')
        self.seek_var=tk.DoubleVar(value=0)
        self.build_ui()
        self.bind('<space>',lambda e:self.toggle_play())
        self.bind('<Left>',lambda e:self.step_event(-1))
        self.bind('<Right>',lambda e:self.step_event(1))
        self.bind('<Home>',lambda e:self.seek_time(self.session.t0 if self.session else 0))
        self.bind('<End>',lambda e:self.seek_time(self.session.t1 if self.session else 0))
        self.bind('<Control-o>',lambda e:self.open_folder())
        self.protocol('WM_DELETE_WINDOW',self.destroy)
        if session:
            self.load_folder(Path(session))
        else:self.index_root()
        self.after(self.frame_ms,self.tick)
        self.after(120,self.poll_load)

    def build_ui(self):
        header=ttk.Frame(self);header.pack(fill='x',padx=12,pady=(10,6))
        ttk.Button(header,text='📁 Dataset',command=self.choose_root).pack(side='left')
        ttk.Label(header,text='  User').pack(side='left')
        self.user_box=ttk.Combobox(header,textvariable=self.user_var,state='readonly',width=10)
        self.user_box.pack(side='left',padx=5)
        self.user_box.bind('<<ComboboxSelected>>',self.user_selected)
        ttk.Label(header,text='Session').pack(side='left')
        self.session_box=ttk.Combobox(header,textvariable=self.session_var,state='readonly',width=12)
        self.session_box.pack(side='left',padx=5)
        self.session_box.bind('<<ComboboxSelected>>',lambda e:self.selected_session())
        ttk.Button(header,text='Open',command=self.selected_session).pack(side='left')
        ttk.Button(header,text='📂 Folder',command=self.open_folder).pack(side='left',padx=(5,0))
        ttk.Label(header,text='  Touch source').pack(side='left')
        self.source_box=ttk.Combobox(header,textvariable=self.source_var,state='readonly',
                 values=['Auto','TouchEvent_im.csv','TouchEvent.csv'],width=20)
        self.source_box.pack(side='left',padx=5)
        self.source_box.bind('<<ComboboxSelected>>',lambda e:self.reload_session())
        ttk.Button(header,text='Export audit CSV',command=self.export_csv).pack(side='right')

        sub=ttk.Frame(self);sub.pack(fill='x',padx=15,pady=(0,6))
        ttk.Label(sub,textvariable=self.info_var,foreground=MUTED).pack(side='left')
        self.activity_var=tk.StringVar(value='')
        ttk.Label(sub,textvariable=self.activity_var,foreground=CYAN).pack(side='right')

        middle=ttk.PanedWindow(self,orient='horizontal');middle.pack(fill='both',expand=True,padx=12)
        left=ttk.Frame(middle);right=ttk.Frame(middle)
        middle.add(left,weight=6);middle.add(right,weight=5)
        canvas_bar=ttk.Frame(left);canvas_bar.pack(fill='x',pady=3)
        ttk.Label(canvas_bar,text='SCREEN  ·  physical touch trajectory',font=('Segoe UI',11,'bold')).pack(side='left')
        self.coords_label=tk.StringVar(value='X —  Y —')
        ttk.Label(canvas_bar,textvariable=self.coords_label,foreground=MUTED).pack(side='right',padx=12)
        self.canvas=tk.Canvas(left,bg=BG,highlightthickness=0,cursor='crosshair')
        self.canvas.pack(fill='both',expand=True,padx=(0,7))
        self.canvas.bind('<Configure>',lambda e:self.draw_screen())
        self.canvas.bind('<Button-1>',self.canvas_click)
        sensorframe=ttk.Frame(left);sensorframe.pack(fill='x',padx=(0,7),pady=(7,0))
        self.accel_plot=MotionPlot(sensorframe,'ACCELEROMETER   ·   7s, X/Y/Z')
        self.gyro_plot=MotionPlot(sensorframe,'GYROSCOPE   ·   7s, X/Y/Z')
        self.accel_plot.pack(side='left',fill='both',expand=True,padx=(0,3))
        self.gyro_plot.pack(side='left',fill='both',expand=True,padx=(3,0))
        settings=ttk.Frame(left);settings.pack(fill='x',padx=2,pady=5)
        ttk.Label(settings,text='Touch coordinates W × H').pack(side='left')
        for var in (self.screen_w,self.screen_h):
            e=ttk.Entry(settings,textvariable=var,width=6)
            e.pack(side='left',padx=3);e.bind('<Return>',lambda event:self.draw_screen())
            e.bind('<FocusOut>',lambda event:self.draw_screen())
        ttk.Label(settings,text='Fade (s)').pack(side='left',padx=(12,0))
        fade=ttk.Entry(settings,textvariable=self.fade_sec,width=5);fade.pack(side='left',padx=4)
        fade.bind('<Return>',lambda e:self.draw_screen())
        for text,var in [('Grid',self.show_grid),('Arrow',self.show_arrow),('Keys',self.show_key),('Follow',self.follow)]:
            ttk.Checkbutton(settings,text=text,variable=var,command=self.draw_screen).pack(side='left',padx=4)
        # RIGHT side: full raw touch history, completed gestures and key events.
        self.notebook=ttk.Notebook(right);self.notebook.pack(fill='both',expand=True,pady=(0,5))
        self.events_tab=ttk.Frame(self.notebook);self.gestures_tab=ttk.Frame(self.notebook)
        self.keys_tab=ttk.Frame(self.notebook)
        self.notebook.add(self.events_tab,text='Touch events')
        self.notebook.add(self.gestures_tab,text='Gestures')
        self.notebook.add(self.keys_tab,text='Key events')
        self.events_table=PagedTable(self.events_tab,
              ['#','sec','Δms','Activity','Action','X','Y','Ori','Contact','Key events'],
              [55,78,67,117,94,68,68,48,66,160],self.pick_touch)
        self.events_table.pack(fill='both',expand=True)
        self.gestures_table=PagedTable(self.gestures_tab,
              ['#','sec','Activity','Type','ms','Points','Portrait','Key↑ / Key↓'],
              [50,75,120,95,63,65,62,115],self.pick_gesture)
        self.gestures_table.pack(fill='both',expand=True)
        self.keys_table=PagedTable(self.keys_tab,
              ['#','sec','Activity','Type','KeyID','Key','Touch #','Δ ms'],
              [52,80,125,75,74,95,75,65],self.pick_key)
        self.keys_table.pack(fill='both',expand=True)
        ttk.Label(right,text='EVENT INSPECTOR',font=('Segoe UI',10,'bold')).pack(anchor='w',padx=5,pady=(3,2))
        self.details=tk.Text(right,height=9,wrap='word',bg=PANEL,fg=FG,
              font=('Consolas',10),relief='flat',padx=10,pady=8)
        self.details.pack(fill='x')
        self.details.insert('1.0','Choose a touch event, a gesture, or a key event.')
        self.details.configure(state='disabled')
        # Bottom transport panel.
        bottom=ttk.Frame(self);bottom.pack(fill='x',padx=12,pady=(7,10))
        top=ttk.Frame(bottom);top.pack(fill='x')
        self.playbutton=ttk.Button(top,textvariable=self.play_var,command=self.toggle_play,width=12)
        self.playbutton.pack(side='left')
        ttk.Button(top,text='◀ Event',command=lambda:self.step_event(-1)).pack(side='left',padx=4)
        ttk.Button(top,text='Event ▶',command=lambda:self.step_event(1)).pack(side='left')
        ttk.Button(top,text='−5s',command=lambda:self.jump(-5000)).pack(side='left',padx=(12,3))
        ttk.Button(top,text='+5s',command=lambda:self.jump(5000)).pack(side='left')
        ttk.Label(top,text='Speed').pack(side='left',padx=(18,4))
        self.speed_box=ttk.Combobox(top,textvariable=self.speed_var,state='readonly',width=7,
                   values=['0.25×','0.5×','1×','2×','4×','8×','16×','32×'])
        self.speed_box.pack(side='left')
        ttk.Label(top,text='  Colored: portrait · landscape · unknown').pack(side='left',padx=12)
        self.time_label=tk.StringVar(value='00:00.000 / 00:00.000')
        ttk.Label(top,textvariable=self.time_label,font=('Consolas',11,'bold')).pack(side='right')
        self.slider=tk.Scale(bottom,from_=0,to=10000,resolution=1,showvalue=0,
                     orient='horizontal',variable=self.seek_var,command=self.slider_moved,
                     background=BG,foreground=FG,troughcolor=PANEL2,highlightthickness=0,
                     activebackground=CYAN,length=500)
        self.slider.pack(fill='x',pady=(0,2))
        ttk.Label(bottom,textvariable=self.clock_var,foreground=GOLD).pack(anchor='w')

    def index_root(self):
        self.sessions=discover_sessions(self.root_dir)
        users=list(self.sessions)
        self.user_box.configure(values=users)
        if users:
            self.user_var.set(users[0]);self.user_selected()
            self.info_var.set(f'{self.root_dir}   ·   {len(users)} users indexed. Select a session.')
        else:self.info_var.set('No HMOG sessions found in '+str(self.root_dir))

    def choose_root(self):
        chosen=filedialog.askdirectory(title='Select HMOG dataset root',initialdir=str(self.root_dir))
        if chosen:self.root_dir=Path(chosen);self.index_root()

    def user_selected(self,event=None):
        user=self.user_var.get()
        folders=self.sessions.get(user,[])
        self.session_box.configure(values=[f.name for f in folders]);
        if folders:self.session_var.set(folders[0].name)

    def selected_session(self):
        user=self.user_var.get();num=self.session_var.get()
        folder=next((f for f in self.sessions.get(user,[]) if f.name==num),None)
        if folder:self.load_folder(folder)

    def open_folder(self):
        folder=filedialog.askdirectory(title='Select a single HMOG session folder',
                           initialdir=str(self.session.folder if self.session else self.root_dir))
        if folder:self.load_folder(Path(folder))

    def reload_session(self):
        if self.session:self.load_folder(self.session.folder)

    def load_folder(self,folder):
        if self.loading:return
        self.pause();self.loading=True
        source=self.source_var.get();self.info_var.set('Loading '+str(folder)+' …')
        def worker():
            try:
                obj=load_session(folder,source,lambda msg:self.load_queue.put(('status',msg)))
                self.load_queue.put(('done',obj))
            except Exception as e:
                self.load_queue.put(('error',f'{type(e).__name__}: {e}'))
        threading.Thread(target=worker,daemon=True).start()

    def poll_load(self):
        try:
            while True:
                status,value=self.load_queue.get_nowait()
                if status=='status':self.info_var.set(value)
                elif status=='done':
                    self.loading=False;self.attach_session(value)
                elif status=='error':
                    self.loading=False;self.info_var.set(value)
                    messagebox.showerror('Cannot load session',value)
        except queue.Empty:pass
        self.after(120,self.poll_load)

    def attach_session(self,s):
        self.session=s;self.current=s.t0;self.last_follow=-1;self.selected_gesture=-1
        nportrait=sum(g.portrait for g in s.gestures if g.completed and not g.multitouch)
        total=sum(g.completed and not g.multitouch for g in s.gestures)
        paired=sum(k.touch>=0 for k in s.keys)
        self.info_var.set(f'{s.folder.parent.name}/{s.folder.name}   ·   {s.source}   ·   '
                  f'{len(s.touches):,} events   ·   {len(s.gestures):,} contacts   ·   '
                  f'{nportrait}/{total} portrait single-touch contacts   ·   '
                  f'{paired}/{len(s.keys)} matched key events')
        self.clock_var.set(s.warning or f'Raw relative timestamps: touch ms; motion ns/1e6. '
              f'Screen 1080×1920 is an editable assumption. Observed maxima: X={s.bounds[0]:.0f}, Y={s.bounds[1]:.0f}.')
        self.events_table.set_rows(s.touches,self.format_touch)
        self.gestures_table.set_rows(s.gestures,self.format_gesture)
        self.keys_table.set_rows(s.keys,self.format_key)
        self.accel_plot.set_samples(s.accel);self.gyro_plot.set_samples(s.gyro)
        self.show_details(f'Session folder: {s.folder}\nTouch source: {s.source}\n'
            f'Touch samples: {len(s.touches):,}\nGestures: {len(s.gestures):,}\n'
            f'Accelerometer: {len(s.accel):,}; Gyroscope: {len(s.gyro):,}\n'
            f'KeyPressEvent (diagnostic only): {len(s.keys):,}\n'
            'Click an event or a gesture to inspect it and seek to its timestamp.')
        self.sync_time(True)

    def format_ms(self,t):
        t=max(0,t);mins,sec=divmod(t/1000,60)
        return f'{int(mins):02}:{sec:06.3f}'

    def format_touch(self,i,p):
        s=self.session
        annotations=[]
        for ki in p.keys:
            k=s.keys[ki]
            annotations.append(('↓' if k.typ==0 else '↑')+key_name(k.code))
        dt=p.t-s.touches[i-1].t if i else 0
        return (i+1,f'{(p.t-s.t0)/1000:.3f}',f'{dt:.1f}',
            s.activities.get(p.act,p.act),action_name(p.action),
            f'{p.x:.0f}',f'{p.y:.0f}',str(p.orient) if p.orient>=0 else '?',
            p.gesture if p.gesture>0 else '—',', '.join(annotations))

    def format_gesture(self,i,g):
        s=self.session
        ks=[s.keys[ki] for j in g.indices for ki in s.touches[j].keys]
        return (g.number,f'{(g.start-s.t0)/1000:.3f}',s.activities.get(g.act,g.act),
                g.kind, f'{g.end-g.start:.0f}',len(g.indices),
                'yes' if g.portrait else 'no',
                f'{sum(k.typ==1 for k in ks)} / {sum(k.typ==0 for k in ks)}')

    def format_key(self,i,k):
        s=self.session
        return (i+1,f'{(k.t-s.t0)/1000:.3f}',s.activities.get(k.act,k.act),
                'DOWN' if k.typ==0 else 'UP' if k.typ==1 else str(k.typ),
                k.code,key_name(k.code),k.touch+1 if k.touch>=0 else '—',
                f'{k.match_ms:.1f}' if k.touch>=0 else '—')

    def show_details(self,text):
        self.details.configure(state='normal');self.details.delete('1.0','end')
        self.details.insert('1.0',text);self.details.configure(state='disabled')

    def pick_touch(self,i):
        s=self.session
        if s is None:return
        p=s.touches[i]
        self.selected_gesture=p.gesture
        content=[f'TOUCH EVENT #{i+1}   source row #{p.seq+1}',
            f'Activity: {p.act} · {s.activities.get(p.act,"unknown")}',
            f'Time: {p.t:.3f} ms   (+{(p.t-s.t0)/1000:.3f} s)',
            f'Action: {action_name(p.action)}   Pointer: {p.pid} / {p.pc}',
            f'Position: ({p.x:.2f}, {p.y:.2f}) px',
            f'Pressure: {p.pressure:.4g}  Size: {p.size:.4g}  Orientation: {p.orient}',
            f'Contact: {p.gesture if p.gesture>0 else "unassigned / orphan"}']
        if p.keys:
            content.append('LINKED KEY EVENTS:')
            for ki in p.keys:
                k=s.keys[ki]
                content.append(f'  {"DOWN" if k.typ==0 else "UP"}  KeyID={k.code} ({key_name(k.code)})   Δt={k.match_ms:.1f}ms')
        else:content.append('Linked KeyPressEvent: none')
        self.show_details('\n'.join(content))
        self.seek_time(p.t,follow=False)
        self.events_table.show_index(i,force=True)

    def pick_gesture(self,i):
        s=self.session
        if s is None:return
        g=s.gestures[i];self.selected_gesture=g.number
        ks=[s.keys[ki] for j in g.indices for ki in s.touches[j].keys]
        start=s.touches[g.indices[0]];end=s.touches[g.indices[-1]]
        content=[f'CONTACT #{g.number}   {g.kind.upper()}',
            f'Activity: {g.act} · {s.activities.get(g.act,"unknown")}',
            f'{(g.start-s.t0)/1000:.3f}s → {(g.end-s.t0)/1000:.3f}s   ({g.end-g.start:.0f}ms)',
            f'Points: {len(g.indices)}   Path: {g.path:.1f}px  Displacement: {g.displacement:.1f}px',
            f'Position: ({start.x:.0f},{start.y:.0f}) → ({end.x:.0f},{end.y:.0f})',
            f'Completed: {g.completed}  Multitouch: {g.multitouch}  All portrait: {g.portrait}',
            f'Matched Key UP / DOWN: {sum(k.typ==1 for k in ks)} / {sum(k.typ==0 for k in ks)}']
        for k in ks[:12]:content.append(f'  {"↓" if k.typ==0 else "↑"}{key_name(k.code)} at {k.t:.1f}ms')
        self.show_details('\n'.join(content))
        self.seek_time(g.end if g.completed else g.start,follow=False)
        self.gestures_table.show_index(i,force=True)

    def pick_key(self,i):
        s=self.session
        if s is None:return
        k=s.keys[i]
        content=[f'KEY EVENT #{i+1}  (DIAGNOSTIC ONLY)',
          f'Type: {"DOWN" if k.typ==0 else "UP" if k.typ==1 else k.typ}',
          f'KeyID: {k.code} ({key_name(k.code)})',
          f'Activity: {k.act}  Time: {k.t:.3f}ms',
          f'Orientation: {k.orient}',
          f'Matched touch event: {k.touch+1 if k.touch>=0 else "none"}',
          f'Match error: {k.match_ms:.3f}ms' if k.touch>=0 else 'Unmatched key event']
        self.show_details('\n'.join(content))
        self.selected_gesture=s.touches[k.touch].gesture if k.touch>=0 else -1
        self.seek_time(k.t,follow=False)
        self.keys_table.show_index(i,force=True)

    def toggle_play(self):
        if not self.session:return
        if self.playing:self.pause()
        else:
            if self.current>=self.session.t1:self.current=self.session.t0
            self.playing=True;self.last_perf=time.perf_counter();self.play_var.set('⏸ Pause')

    def pause(self):
        self.playing=False;self.last_perf=None;self.play_var.set('▶ Play')

    def seek_time(self,t,follow=True):
        if not self.session:return
        self.current=min(max(t,self.session.t0),self.session.t1)
        self.last_perf=time.perf_counter();self.sync_time(follow)

    def jump(self,dt):
        if self.session:self.seek_time(self.current+dt)

    def step_event(self,step):
        if not self.session:return
        self.pause()
        times=self.session.touch_times
        if step>0:
            i=bisect.bisect_right(times,self.current+0.001)
        else:i=bisect.bisect_left(times,self.current-0.001)-1
        i=max(0,min(i,len(times)-1))
        self.seek_time(times[i]);self.events_table.show_index(i)
        self.pick_touch(i)

    def slider_moved(self,raw):
        if not self.session or getattr(self,'slider_updating',False):return
        pct=float(raw)/10000
        self.seek_time(self.session.t0+pct*(self.session.t1-self.session.t0))

    def tick(self):
        if self.playing and self.session:
            now=time.perf_counter();elapsed=max(0,min(0.5,now-(self.last_perf or now)))
            self.last_perf=now
            speed=num(self.speed_var.get().replace('×',''),1)
            self.current=min(self.current+1000*elapsed*speed,self.session.t1)
            if self.current>=self.session.t1:self.pause()
            self.sync_time()
        self.after(self.frame_ms,self.tick)

    def sync_time(self,follow=True):
        if not self.session:return
        s=self.session
        ratio=(self.current-s.t0)/max(1,s.t1-s.t0)
        self.slider_updating=True
        self.seek_var.set(round(ratio*10000))
        self.slider_updating=False
        self.time_label.set(f'{self.format_ms(self.current-s.t0)} / {self.format_ms(s.t1-s.t0)}')
        idx=bisect.bisect_right(s.touch_times,self.current)-1
        if idx>=0:
            p=s.touches[idx]
            self.coords_label.set(f'X {p.x:.1f}  Y {p.y:.1f}  ·  {action_name(p.action)}')
            self.activity_var.set(s.activities.get(p.act,p.act))
            if follow and self.follow.get() and idx!=self.last_follow:
                self.events_table.show_index(idx);self.last_follow=idx
        self.draw_screen()
        self.accel_plot.set_time(self.current);self.gyro_plot.set_time(self.current)

    def screen_params(self):
        cw=max(80,self.canvas.winfo_width());ch=max(80,self.canvas.winfo_height())
        sw=max(1,num(self.screen_w.get(),1080));sh=max(1,num(self.screen_h.get(),1920))
        scale=min((cw-46)/sw,(ch-28)/sh)
        width=sw*scale;height=sh*scale
        x0=(cw-width)/2;y0=(ch-height)/2
        return x0,y0,width,height,sw,sh,scale

    def to_px(self,p):
        x0,y0,w,h,sw,sh,_=self.screen_params()
        return x0+p.x/sw*w,y0+p.y/sh*h

    def draw_screen(self):
        c=self.canvas;c.delete('all')
        x0,y0,w,h,sw,sh,scale=self.screen_params()
        c.create_rectangle(x0,y0,x0+w,y0+h,fill='#0B111A',outline='#40526A',width=2)
        if self.show_grid.get():
            for j in range(1,4):
                xx=x0+w*j/4;yy=y0+h*j/4
                c.create_line(xx,y0,xx,y0+h,fill='#202E40',dash=(3,8))
                c.create_line(x0,yy,x0+w,yy,fill='#202E40',dash=(3,8))
        c.create_text(x0+10,y0+10,text=f'{sw:.0f} × {sh:.0f} px  ·  proportions preserved',
                      anchor='nw',font=('Consolas',9),fill='#667B91')
        if not self.session:return
        s=self.session
        fade_s=max(0.1,num(self.fade_sec.get(),25))
        # Up to TEN finished contacts. Both gesture count and elapsed wall time
        # (relative session clock) diminish their color/visibility.
        finished=[g for g in s.gestures if g.completed and g.end<=self.current]
        trail=finished[-10:]
        for rank,g in enumerate(trail):
            age=max(0,(self.current-g.end)/1000)
            alpha=max(0,1-age/fade_s)*(0.18+0.72*(rank+1)/max(1,len(trail)))
            if alpha<=0:continue
            self.draw_gesture(g,alpha,selected=(g.number==self.selected_gesture))
        # Active (not-yet-finished) contacts: show progressive trajectory.
        started=[g for g in s.gestures if g.start<=self.current<g.end]
        for g in started:
            self.draw_gesture(g,1,selected=(g.number==self.selected_gesture),progressive=True)
        # Highlight nearest currently selected raw point.
        if self.selected_gesture>0:
            g=s.gestures[self.selected_gesture-1]
            if g.start<=self.current<=g.end:
                ids=[i for i in g.indices if s.touches[i].t<=self.current]
                if ids:
                    x,y=self.to_px(s.touches[ids[-1]])
                    c.create_oval(x-7,y-7,x+7,y+7,outline='#FFFFFF',width=2)
        c.create_text(x0+8,y0+h-8,anchor='sw',text='Teal: portrait  ·  Orange: landscape  ·  Red: unknown',
                      fill=MUTED,font=('Segoe UI',9))

    def draw_gesture(self,g,alpha,selected=False,progressive=False):
        s=self.session;c=self.canvas
        pts=[s.touches[i] for i in g.indices if not progressive or s.touches[i].t<=self.current]
        if not pts:return
        if all(p.orient==0 for p in pts):base=CYAN
        elif any(p.orient in (1,3) for p in pts):base=GOLD
        else:base=RED
        col=blend(base,'#0B111A',min(1,alpha+(0.15 if selected else 0)))
        xy=[self.to_px(p) for p in pts]
        if len(xy)>1:
            coords=[v for xy0 in xy for v in xy0]
            c.create_line(*coords,fill=col,width=3 if selected else 2,
                          capstyle='round',joinstyle='round',smooth=False)
        if len(xy)==1 or (len(xy)>1 and g.kind=='tap'):
            x,y=xy[-1];r=5 if selected else 4
            c.create_oval(x-r,y-r,x+r,y+r,outline=col,width=2)
        x,y=xy[-1]
        c.create_oval(x-3,y-3,x+3,y+3,fill=col,outline='')
        if progressive and len(xy)>1 and self.show_arrow.get():
            a,b=xy[-2],xy[-1]
            if math.hypot(b[0]-a[0],b[1]-a[1])>2:
                c.create_line(a[0],a[1],b[0],b[1],fill='#FFFFFF',width=2,
                              arrow=tk.LAST,arrowshape=(8,10,4))
        if self.show_key.get() and alpha>.25:
            annotations=[]
            for p in pts:
                for ki in p.keys:
                    k=s.keys[ki]
                    annotations.append(('↓' if k.typ==0 else '↑')+key_name(k.code))
            if annotations:
                c.create_text(x+10,y-10,anchor='sw',text=' '.join(annotations[:4]),
                              fill=blend(GREEN,'#0B111A',alpha),font=('Segoe UI',9,'bold'))
        if selected:
            x,y=xy[0]
            c.create_text(x+7,y+8,anchor='nw',text=f'#{g.number} · {g.kind}',fill=FG,
                          font=('Segoe UI',9,'bold'))

    def canvas_click(self,event):
        if not self.session:return
        s=self.session;idx=bisect.bisect_right(s.touch_times,self.current)
        near=[]
        for i in range(max(0,idx-2000),idx):
            p=s.touches[i]
            if (self.current-p.t)>30000:continue
            x,y=self.to_px(p)
            d=(x-event.x)**2+(y-event.y)**2
            if d<15**2:near.append((d,i))
        if near:self.pick_touch(min(near)[1])

    def export_csv(self):
        if not self.session:return
        f=filedialog.asksaveasfilename(title='Export gesture/key audit',
                 defaultextension='.csv',filetypes=[('CSV','*.csv')],
                 initialfile=f'{getattr(self.session, "dataset", "HMOG")}_{self.session.folder.parent.name}_{self.session.folder.name}_audit.csv')
        if not f:return
        s=self.session
        with open(f,'w',newline='',encoding='utf-8-sig') as stream:
            fields=['gesture','activity','task','start_ms','end_ms','duration_ms',
                    'type','completed','multitouch','all_portrait','touch_rows',
                    'path_px','displacement_px','key_down','key_up','key_codes_up']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for g in s.gestures:
                ks=[s.keys[ki] for i in g.indices for ki in s.touches[i].keys]
                writer.writerow(dict(gesture=g.number,activity=g.act,
                    task=s.activities.get(g.act,''),start_ms=g.start,end_ms=g.end,
                    duration_ms=g.end-g.start,type=g.kind,completed=g.completed,
                    multitouch=g.multitouch,all_portrait=g.portrait,touch_rows=len(g.indices),
                    path_px=round(g.path,2),displacement_px=round(g.displacement,2),
                    key_down=sum(k.typ==0 for k in ks),key_up=sum(k.typ==1 for k in ks),
                    key_codes_up=';'.join(str(k.code) for k in ks if k.typ==1)))
        messagebox.showinfo('Exported',f'Saved {len(s.gestures):,} contacts to:\n{f}')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',default='input_data/hmog_unarchived',help='HMOG root with user/session folders')
    ap.add_argument('--session',default=None,help='Open a specific HMOG session folder')
    args=ap.parse_args()
    app=Explorer(args.input,args.session);app.mainloop()


# Embedded HMOG viewer: entry point replaced by combined HMOG/FETA entry point.


# ===== FETA adapter and combined UI =====
#!/usr/bin/env python3
"""HMOG + FETA Session Explorer (Python 3.14, Tkinter, no pip dependencies).

Requires sibling module hmog_session_explorer.py (the supplied original HMOG viewer).
FETA uses a virtual timeline across repetitions, preserving timing *within* each
repetition but NOT inventing inter-repetition durations. Each sensor remains at
its observed timestamp; no interpolation or synthetic samples.

Use --help for paths and options.
"""

import argparse
import csv
import datetime as dt
import math
import os
import queue
import re
import statistics
import threading
import tkinter as tk
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import sys
hm=sys.modules[__name__]

# Native portrait width, height; UIKit logical width, height; diagonal inches; ppi.
# Apple technical specifications and UIKit Display Reference; Samsung Galaxy S4.
DEVICES = {
    'Samsung Galaxy S4': (1080,1920,1080,1920,5.0,441),
    'iPhone 6s': (750,1334,375,667,4.7,326),
    'iPhone 6s Plus': (1080,1920,414,736,5.5,401),
    'iPhone 7': (750,1334,375,667,4.7,326),
    'iPhone 7 Plus': (1080,1920,414,736,5.5,401),
    'iPhone 8': (750,1334,375,667,4.7,326),
    'iPhone 8 Plus': (1080,1920,414,736,5.5,401),
    'iPhone X': (1125,2436,375,812,5.8,458),
    'iPhone XS': (1125,2436,375,812,5.8,458),
    'iPhone XS Max': (1242,2688,414,896,6.5,458),
}

FETA_TIME_FIELDS = ('timestamp','eventtime','time','ts','recordedat','sampletime',
                    'timestampms','touchstarttime','createdat','datetime','t')
FETA_PHASE_FIELDS = ('type','touchtype','phase','action','actiontype',
                     'touchphase','eventtype','state')
HMOG_TASK_OPTIONS = ['All activities', 'Reading · sitting', 'Reading · walking',
                      'Writing · sitting', 'Writing · walking',
                      'Map · sitting', 'Map · walking']


def find_feta_root(p: Path) -> Path:
    p=p.expanduser().resolve()
    if (p/'data_files').is_dir(): return p/'data_files'
    return p


def user_phone_map(path: Path) -> dict[str,str]:
    if path.is_dir():path=path/'userdata.csv'
    if not path.is_file():return {}
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:return {}
        cols={hm.norm(c):c for c in reader.fieldnames}
        if 'uuid' not in cols or 'phonemodel' not in cols:
            raise ValueError(f'userdata.csv must contain uuid and phone_model: {reader.fieldnames}')
        return {str(r.get(cols['uuid'],'')).strip():str(r.get(cols['phonemodel'],'')).strip()
                for r in reader if r.get(cols['uuid'])}


def inspect_feta_csv(path:Path) -> list[dict[str,str]]:
    with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
        lead=f.readline()
        if not lead.strip():return []
        delimiter=max((',', ';', '\t'),key=lead.count)
        f.seek(0)
        rdr=csv.DictReader(f,delimiter=delimiter)
        if not rdr.fieldnames or len(rdr.fieldnames)<2:raise ValueError(f'Invalid CSV header: {path}')
        names=[hm.norm(c) for c in rdr.fieldnames]
        if len(set(names))<2:raise ValueError(f'Unrecognized CSV header: {path}')
        return [dict(zip(names,(str(c).strip() if c is not None else '' for c in row.values())))
                for row in rdr if row and any(str(v).strip() for v in row.values() if v is not None)]


def time_field(rows):
    if not rows:return None
    cols=rows[0]
    return next((c for c in FETA_TIME_FIELDS if c in cols),None) or next(
        (c for c in cols if 'timestamp' in c or c.endswith('time')),None)


def feta_timestamps_ms(rows: list[dict], path:Path) -> list[float]:
    """Same unit-scale conventions as our v2.0 FETA audit; do not align by row number."""
    if not rows:return []
    key=time_field(rows)
    if key is None:raise ValueError(f'No timestamp column in {path}; got {list(rows[0])}')
    raw=[str(r.get(key,'')).strip() for r in rows]
    numeric=[hm.num(v) for v in raw]
    good=[v for v in numeric if math.isfinite(v)]
    if len(good)>=0.9*len(raw):
        median_abs=statistics.median(abs(v) for v in good)
        if median_abs>=1e17:mul=1e-6
        elif median_abs>=1e14:mul=1e-3
        elif median_abs>=1e11:mul=1.
        elif median_abs>=1e8:mul=1000.
        else:
            # Relative clock: follow the convention in FETA audit v2.0.
            differences=[b-a for a,b in zip(good,good[1:]) if b>a]
            if not differences:raise ValueError(f'Ambiguous relative timestamp unit in {path}')
            step=statistics.median(differences)
            if step>=1e6:mul=1e-6
            elif step>=1e3:mul=1e-3
            elif step>=1:mul=1.
            else:mul=1000.
        return [x*mul for x in numeric]
    def parse_iso(v):
        try:
            t=dt.datetime.fromisoformat(v.replace('Z','+00:00'))
            if t.tzinfo is None:t=t.replace(tzinfo=dt.timezone.utc)
            return t.timestamp()*1000
        except ValueError:return math.nan
    return [parse_iso(x) for x in raw]


def feta_ph(raw,col):
    s=hm.norm(raw)
    if s in ('began','begin','down','start','fingerdown','touchdown','pressed'):return 0
    if s in ('moved','move','changed','drag','stationary'):return 2
    if s in ('ended','end','up','release','touchup','fingerup','released'):return 1
    if s in ('cancel','cancelled','canceled'):return 3
    try:v=int(float(s))
    except (ValueError,TypeError):return -1
    # FETA iOS collection app's custom type codes: 0=DOWN,1=MOVE,2=UP.
    # Distinct from generic Android MotionEvent codes.
    return {0:0,1:2,2:1}.get(v,3)


def file_csvs(rep:Path, kind:str) -> list[Path]:
    folder=rep/kind
    return sorted(folder.rglob('*.csv')) if folder.is_dir() else []


def read_feta_repetition(rep:Path) -> tuple[list[hm.Touch],list[hm.Sensor],list[hm.Sensor],list[list[int]],list[str]]:
    """Parse each repetition independently. Contacts never cross CSV file boundaries."""
    touches=[];accel=[];gyro=[];gestures=[];warnings=[]
    for filename in file_csvs(rep,'touch_data'):
        rows=inspect_feta_csv(filename)
        if not rows:continue
        times=feta_timestamps_ms(rows,filename)
        pcol=next((c for c in FETA_PHASE_FIELDS if c in rows[0]),None)
        if pcol is None:raise ValueError(f'No FETA touch type/phase in {filename}; {list(rows[0])}')
        evs=[]
        for i,(r,t) in enumerate(zip(rows,times)):
            x=hm.num(r.get('x'));y=hm.num(r.get('y'))
            if not all(math.isfinite(v) for v in (t,x,y)):continue
            evs.append((float(t),i,x,y,feta_ph(r.get(pcol,''),pcol),
                        hm.num(r.get('pressure')),hm.num(r.get('touchmajor',r.get('size')))))
        evs.sort(key=lambda row:(row[0],row[1]))
        active=None
        for t,ri,x,y,phase,p,size in evs:
            idx=len(touches)
            touches.append(hm.Touch(idx,t,rep.name,phase,x=x,y=y,pc=1,pid=0,
                                    orient=0,pressure=p,size=size,system=math.nan))
            if phase==0:
                if active is not None:warnings.append(f'{filename.name}: interrupted DOWN')
                active=[idx]
            elif phase==2:
                if active is not None:active.append(idx)
            elif phase==1:
                if active is not None:
                    active.append(idx)
                    gestures.append(active)
                else:warnings.append(f'{filename.name}: orphan UP')
                active=None
            elif phase==3:active=None
        if active is not None:warnings.append(f'{filename.name}: unfinished DOWN')
    for kind,out in (('accelerometer_data',accel),('gyroscope_data',gyro)):
        for filename in file_csvs(rep,kind):
            rows=inspect_feta_csv(filename)
            if not rows:continue
            times=feta_timestamps_ms(rows,filename)
            for r,t in zip(rows,times):
                xyz=[hm.num(r.get(k)) for k in ('x','y','z')]
                if all(math.isfinite(v) for v in (t,*xyz)):
                    out.append(hm.Sensor(float(t),*xyz,0,rep.name,math.nan))
        out.sort(key=lambda row:row.t)
    return touches,accel,gyro,gestures,warnings


def make_feta_session(folder:Path,task='scroll',repetition='All repetitions',model='Unknown',progress=None)->hm.Session:
    if not folder.is_dir():raise FileNotFoundError(folder)
    task_dir=folder/task
    if not task_dir.is_dir():raise FileNotFoundError(f'Task directory not found: {task_dir}')
    reps=sorted((p for p in task_dir.iterdir() if p.is_dir()),key=lambda p:p.name)
    if repetition!='All repetitions':reps=[p for p in reps if p.name==repetition]
    if not reps:raise ValueError('No repetitions found')
    touches=[];accel=[];gyro=[];gestures=[];activity_names={};warnings=[]
    elapsed=0.;gap=800.  # Virtual gap between repetitions, NOT a measured pause.
    for k,rep in enumerate(reps,1):
        if progress:progress(f'FETA {task}, repetition {k}/{len(reps)}: {rep.name} …')
        try:r_touch,r_acc,r_gyro,r_gest,r_warn=read_feta_repetition(rep)
        except (OSError,ValueError) as e:
            warnings.append(f'{rep.name}: {type(e).__name__}: {e}')
            continue
        activity_names[rep.name]=f'{task} · repetition {rep.name}'
        all_t=[x.t for arr in (r_touch,r_acc,r_gyro) for x in arr]
        if not all_t:continue
        base=min(all_t);end=max(all_t)
        # Within repetition, all modalities keep their original relative time offsets.
        # Across repetitions only, gaps are artificial and labelled in UI.
        shift=elapsed-base
        start_index=len(touches)
        for i,p in enumerate(r_touch):
            p.seq=start_index+i;p.t+=shift;touches.append(p)
        for arr,target in ((r_acc,accel),(r_gyro,gyro)):
            for p in arr:p.t+=shift
            target.extend(arr)
        for ids in r_gest:
            indices=[start_index+i for i in ids]
            if not indices:continue
            pts=[touches[i] for i in indices]
            if pts[-1].t<=pts[0].t:continue
            mx=max(math.hypot(pt.x-pts[0].x,pt.y-pts[0].y) for pt in pts)
            ok=len(pts)>=3 and mx>5
            g=hm.Gesture(len(gestures)+1,rep.name,indices,pts[0].t,pts[-1].t,
                         completed=True,multitouch=False,portrait=True,
                         kind=task if ok else 'short/tap')
            g.path=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(pts,pts[1:]))
            g.displacement=math.hypot(pts[-1].x-pts[0].x,pts[-1].y-pts[0].y)
            for j in indices:touches[j].gesture=g.number
            gestures.append(g)
        warnings.extend(r_warn[:5])
        elapsed+=(end-base)+gap
    if not touches:raise ValueError(f'No valid FETA touch events in {folder}/{task}')
    touches.sort(key=lambda p:p.t)
    # Sorted position may change once repetitions are concatenated; remap indices
    # to preserve gesture -> point relationships and table navigation.
    indices=sorted(range(len(touches)),key=lambda i:(touches[i].t,i))
    mapping={old:new for new,old in enumerate(indices)}
    touches=[touches[i] for i in indices]
    for g in gestures:g.indices=sorted((mapping[i] for i in g.indices),key=lambda j:(touches[j].t,j))
    accel.sort(key=lambda p:p.t);gyro.sort(key=lambda p:p.t)
    t0=touches[0].t;t1=touches[-1].t
    session=hm.Session(folder,f'FETA {task} / {len(reps)} repetitions',touches,[],accel,gyro,gestures,
                       activity_names,t0,t1,[p.t for p in touches],[p.t for p in accel],
                       [p.t for p in gyro],[],
                       (max(p.x for p in touches),max(p.y for p in touches)),
                       ('; '.join(warnings[:5])+'; ' if warnings else '')+
                       'Repetitions are concatenated on a VIRTUAL timeline with 0.8s display gaps; '
                       'inter-repetition elapsed time and exact repetition order may be unknown.')
    session.dataset='FETA';session.device_model=model;session.task=task
    return session


def slice_hmog_session(session:hm.Session, label:str)->hm.Session:
    if label in ('All activities',''):
        for g in session.gestures:
            pts=[session.touches[i] for i in g.indices]
            if not pts:continue
            mx=max(math.hypot(p.x-pts[0].x,p.y-pts[0].y) for p in pts)
            g.kind=('FETA-valid' if g.completed and not g.multitouch and
                    len(pts)>=3 and mx>5 else
                    'tap/short' if g.completed else 'incomplete')
        return session
    aids={a for a,task in session.activities.items() if task==label}
    if not aids:raise ValueError(f'No activity {label!r} in session')
    kept=[i for i,p in enumerate(session.touches) if p.act in aids]
    if not kept:raise ValueError(f'No touch events for {label}')
    old_to_new={old:new for new,old in enumerate(kept)}
    touches=[replace(session.touches[i],keys=[]) for i in kept]
    gestures=[];old_id_to_new={}
    for g in session.gestures:
        if g.act not in aids:continue
        ids=[old_to_new[i] for i in g.indices if i in old_to_new]
        if not ids:continue
        new=replace(g,number=len(gestures)+1,indices=ids)
        mx=max(math.hypot(touches[i].x-touches[ids[0]].x,
                          touches[i].y-touches[ids[0]].y) for i in ids)
        new.kind='FETA-valid' if g.completed and not g.multitouch and len(ids)>=3 and mx>5 else ('tap/short' if g.completed else 'incomplete')
        old_id_to_new[g.number]=new.number
        gestures.append(new)
    for p in touches:p.gesture=old_id_to_new.get(p.gesture,-1)
    keys=[]
    for k in session.keys:
        if k.act not in aids:continue
        new=replace(k,touch=old_to_new.get(k.touch,-1))
        if new.touch>=0:
            touches[new.touch].keys.append(len(keys))
        keys.append(new)
    accel=[p for p in session.accel if p.act in aids]
    gyro=[p for p in session.gyro if p.act in aids]
    t0=touches[0].t;t1=touches[-1].t
    result=hm.Session(session.folder,session.source,touches,keys,accel,gyro,gestures,
                      {a:label for a in aids},t0,t1,[p.t for p in touches],
                      [p.t for p in accel],[p.t for p in gyro],[p.t for p in keys],
                      (max(p.x for p in touches),max(p.y for p in touches)),session.warning)
    result.dataset='HMOG';result.device_model='Samsung Galaxy S4';result.task=label
    return result


class CombinedExplorer(hm.Explorer):
    def __init__(self,hmog_root='input_data/hmog_unarchived',
                 feta_root='input_data/data_files',userdata='input_data/tables/tables/userdata.csv',
                 dataset='HMOG',session=None):
        self.dataset_kind=dataset.upper()
        self.dataset_roots={'HMOG':Path(hmog_root).expanduser().resolve(),
                            'FETA':find_feta_root(Path(feta_root))}
        self.userdata_path=Path(userdata).expanduser().resolve()
        self.phone_by_id=user_phone_map(self.userdata_path)
        self.active_model='Samsung Galaxy S4'
        self.active_task=''
        super().__init__(input_dir=self.dataset_roots[self.dataset_kind],session=session)
        self.title('HMOG + FETA Session Explorer · Touch / IMU / Gesture playback')
        self.style.map('TCombobox',fieldbackground=[('readonly',hm.PANEL2)],
                       foreground=[('readonly',hm.FG)],background=[('readonly',hm.PANEL2)])

    def build_ui(self):
        # Build the original HMOG replay + tables, then add dataset/task/device controls.
        super().build_ui()
        self.dataset_var=tk.StringVar(value=self.dataset_kind)
        self.task_var=tk.StringVar(value='All activities' if self.dataset_kind=='HMOG' else 'scroll')
        self.rep_var=tk.StringVar(value='All repetitions')
        self.coordinate_var=tk.StringVar(value='Auto (UIKit for FETA)')
        self.device_info_var=tk.StringVar(value='')
        bar=ttk.Frame(self)
        bar.pack(before=self.winfo_children()[0],fill='x',padx=12,pady=(6,2))
        ttk.Label(bar,text='DATASET').pack(side='left',padx=(0,4))
        ds=ttk.Combobox(bar,textvariable=self.dataset_var,values=['HMOG','FETA'],
                        width=7,state='readonly')
        ds.pack(side='left');ds.bind('<<ComboboxSelected>>',self.change_dataset)
        ttk.Label(bar,text='   Task').pack(side='left')
        self.task_box=ttk.Combobox(bar,textvariable=self.task_var,width=21,state='readonly')
        self.task_box.pack(side='left',padx=4)
        self.task_box.bind('<<ComboboxSelected>>',self.task_selected)
        ttk.Label(bar,text='   Repetition').pack(side='left')
        self.rep_box=ttk.Combobox(bar,textvariable=self.rep_var,width=21,state='readonly')
        self.rep_box.pack(side='left',padx=4)
        self.rep_box.bind('<<ComboboxSelected>>',lambda e:self.selected_session())
        ttk.Label(bar,text='   Touch coordinates').pack(side='left')
        self.coord_box=ttk.Combobox(bar,textvariable=self.coordinate_var,
             values=['Auto (UIKit for FETA)','UIKit points','Native pixels','Custom entries'],
             width=23,state='readonly')
        self.coord_box.pack(side='left',padx=4)
        self.coord_box.bind('<<ComboboxSelected>>',lambda e:self.update_device_screen())
        spec=ttk.Frame(self)
        spec.pack(before=self.winfo_children()[1],fill='x',padx=14,pady=(0,2))
        ttk.Label(spec,textvariable=self.device_info_var,foreground=hm.GREEN).pack(anchor='w')
        self.update_controls()

    def update_controls(self):
        is_feta=self.dataset_kind=='FETA'
        self.task_box.configure(values=['scroll','swipe'] if is_feta else HMOG_TASK_OPTIONS)
        if self.task_var.get() not in self.task_box.cget('values'):
            self.task_var.set('scroll' if is_feta else 'All activities')
        self.rep_box.configure(state='readonly' if is_feta else 'disabled')
        self.source_box.configure(state='disabled' if is_feta else 'readonly')
        self.notebook.tab(self.keys_tab,state='disabled' if is_feta else 'normal')
        if is_feta:
            self.show_key.set(False)
            if self.notebook.select()==str(self.keys_tab):self.notebook.select(self.gestures_tab)
        else:self.show_key.set(True)
        self.device_info_var.set('')

    def change_dataset(self,event=None):
        self.pause()
        self.dataset_kind=self.dataset_var.get()
        self.root_dir=self.dataset_roots[self.dataset_kind]
        self.task_var.set('scroll' if self.dataset_kind=='FETA' else 'All activities')
        self.rep_var.set('All repetitions')
        self.session=None
        self.update_controls();self.index_root()
        self.active_model='Samsung Galaxy S4' if self.dataset_kind=='HMOG' else 'Unknown'
        self.update_device_screen()

    def index_root(self):
        if self.dataset_kind=='HMOG':
            super().index_root()
            return
        root=find_feta_root(self.root_dir)
        self.root_dir=root
        self.sessions={}
        if root.is_dir():
            for u in sorted(p for p in root.iterdir() if p.is_dir()):
                ss=sorted((p for p in u.iterdir() if p.is_dir()),key=lambda p:p.name)
                if ss:self.sessions[u.name]=ss
        users=list(self.sessions)
        self.user_box.configure(values=users)
        if users:
            self.user_var.set(users[0]);self.user_selected()
            self.info_var.set(f'FETA · {root} · {len(users)} users indexed')
        else:self.info_var.set(f'No FETA users found at {root}')

    def choose_root(self):
        root=filedialog.askdirectory(title=f'Select {self.dataset_kind} dataset root',initialdir=str(self.root_dir))
        if root:
            self.root_dir=Path(root).expanduser().resolve()
            self.dataset_roots[self.dataset_kind]=self.root_dir
            self.index_root()

    def user_selected(self,event=None):
        super().user_selected(event)
        self.rep_selected()
        self.update_device_screen()

    def task_selected(self,event=None):
        self.rep_selected();self.selected_session()

    def rep_selected(self):
        if self.dataset_kind!='FETA':return
        u=self.user_var.get();sid=self.session_var.get();task=self.task_var.get()
        target=next((p for p in self.sessions.get(u,[]) if p.name==sid),None)
        folder=target/task if target else None
        values=['All repetitions']
        if folder and folder.is_dir():
            values.extend(sorted(p.name for p in folder.iterdir() if p.is_dir()))
        self.rep_box.configure(values=values)
        if self.rep_var.get() not in values:self.rep_var.set('All repetitions')

    def selected_session(self):
        self.rep_selected()
        super().selected_session()

    def open_folder(self):
        path=filedialog.askdirectory(title=f'Select {self.dataset_kind} session folder',
                                     initialdir=str(self.root_dir))
        if path:self.load_folder(Path(path))

    def load_folder(self,folder):
        if self.loading:return
        self.pause();self.loading=True
        dataset=self.dataset_kind
        task=self.task_var.get();rep=self.rep_var.get()
        option=self.source_var.get()
        model=self.phone_by_id.get(folder.parent.name,'Unknown') if dataset=='FETA' else 'Samsung Galaxy S4'
        self.info_var.set(f'Loading {dataset} {folder} …')
        def worker():
            try:
                if dataset=='FETA':
                    obj=make_feta_session(folder,task,rep,model,
                                          lambda m:self.load_queue.put(('status',m)))
                else:
                    obj=hm.load_session(folder,option,
                            lambda m:self.load_queue.put(('status',m)))
                    obj=slice_hmog_session(obj,task)
                    obj.dataset='HMOG';obj.device_model='Samsung Galaxy S4'
                self.load_queue.put(('done',obj))
            except Exception as e:
                self.load_queue.put(('error',f'{type(e).__name__}: {e}'))
        threading.Thread(target=worker,daemon=True).start()

    def attach_session(self,s):
        self.active_model=getattr(s,'device_model','Samsung Galaxy S4')
        self.active_task=getattr(s,'task','')
        self.update_device_screen()
        super().attach_session(s)
        dataset=getattr(s,'dataset','HMOG')
        if dataset=='FETA':
            self.show_details(
                f'FETA SESSION  ·  {s.folder.parent.name}/{s.folder.name}\n'
                f'Device: {self.active_model}\nTask: {s.task}\n'
                f'Touch events: {len(s.touches):,}\n'
                f'Completed gestures: {len(s.gestures):,}\n'
                f'Accelerometer: {len(s.accel):,} samples\n'
                f'Gyroscope: {len(s.gyro):,} samples\n'
                'FETA contains no KeyPressEvent.\n'
                'Open a touch event or gesture to inspect it.')
            self.info_var.set(f'FETA · {s.folder.parent.name}/{s.folder.name} · '
                 f'{self.active_model} · task {s.task} · '
                 f'{len(s.touches):,} touch points · {len(s.gestures):,} complete contacts · '
                 f'{len(s.accel):,} accel · {len(s.gyro):,} gyro readings')
            self.clock_var.set('FETA: each repetition keeps real intra-repetition timestamps; '
                 '0.8s virtual display gaps between repetitions, NOT recorded intervals. '+
                 (s.warning or ''))
        else:
            self.info_var.set('HMOG · '+self.info_var.get())
        self.update_device_screen()

    def update_device_screen(self):
        model=getattr(self,'active_model','Samsung Galaxy S4')
        if self.dataset_kind=='FETA':
            u=self.user_var.get()
            if self.session is None or getattr(self.session,'dataset',None)!='FETA':
                model=self.phone_by_id.get(u,'Unknown')
        d=DEVICES.get(model)
        if not d:
            self.device_info_var.set(f'Model: {model} · specification unavailable; enter coordinate dimensions manually')
            return
        pw,ph,uw,uh,diagonal,ppi=d
        choice=self.coordinate_var.get()
        mode='UIKit points' if choice=='Auto (UIKit for FETA)' and self.dataset_kind=='FETA' else choice
        if mode=='Auto (UIKit for FETA)':mode='Native pixels'
        if mode=='UIKit points':dw,dh=uw,uh
        elif mode=='Native pixels':dw,dh=pw,ph
        else:dw=dh=None
        if dw and dh:
            self.screen_w.set(str(dw));self.screen_h.set(str(dh))
        approximate_w=pw/ppi*25.4;approximate_h=ph/ppi*25.4
        logical=(f'UIKit {uw}×{uh} pt' if model.startswith('iPhone')
                 else f'Android touch coordinates ≈{uw}×{uh} px')
        assumed=(' (iOS coordinate assumption)' if mode=='UIKit points' and model.startswith('iPhone') else '')
        self.device_info_var.set(
          f'{model} · {diagonal:.1f}″ display · {pw}×{ph} native pixels · {ppi} ppi · '
          f'≈{approximate_w:.1f}×{approximate_h:.1f} mm display · '
          f'{logical} · input: {mode}{assumed}')
        if hasattr(self,'canvas'):self.draw_screen()

    def screen_params(self):
        cw=max(80,self.canvas.winfo_width());ch=max(80,self.canvas.winfo_height())
        spec=DEVICES.get(self.active_model)
        aspect=spec[0]/spec[1] if spec else 1080/1920
        width=min(cw-46,(ch-28)*aspect)
        height=width/aspect
        x0=(cw-width)/2;y0=(ch-height)/2
        sw=max(1,hm.num(self.screen_w.get(),1080))
        sh=max(1,hm.num(self.screen_h.get(),1920))
        return x0,y0,width,height,sw,sh,width/sw


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hmog-root',default='input_data/hmog_unarchived')
    ap.add_argument('--feta-root',default='input_data/data_files')
    ap.add_argument('--userdata',default='input_data/tables/tables/userdata.csv')
    ap.add_argument('--dataset',choices=['HMOG','FETA','hmog','feta'],default='HMOG')
    ap.add_argument('--session',type=Path,default=None,help='Open a specific session folder from selected dataset')
    args=ap.parse_args(argv)
    app=CombinedExplorer(args.hmog_root,args.feta_root,args.userdata,args.dataset.upper(),args.session)
    app.mainloop()

if __name__=='__main__':main()
