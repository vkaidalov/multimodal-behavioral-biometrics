#!/usr/bin/env python3
"""Audit HMOG reading/writing sessions for portrait, gesture and motion coverage.

Requires: Python >=3.14, numpy, pandas.
Run from repository root:
    python analyze_hmog_sessions.py --input input_data/hmog_unarchived --workers 6

The original HMOG CSV schema is documented in hmog-data_description.pdf.
TouchEvent.csv supplies raw one-finger strokes/taps. OneFingerTouchEvent.csv supplies
additional, often numerous, one-finger taps. KeyPressEvent.csv is diagnostic only:
it has timestamps but no touch coordinates. Cross-file duplicates are removed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_VERSION = "hmog-audit-v2.1-onefinger-taps"

# HMOG data description, pages 1-10. Note the typo 'Gyroscrope' in the PDF.
SCHEMAS = {
    "activity": ["ID", "SubjectID", "Session_number", "Start_time", "End_time",
                 "Relative_Start_time", "Relative_End_time", "Gesture_scenario", "TaskID", "ContentID"],
    "accelerometer": ["Systime", "EventTime", "ActivityID", "X", "Y", "Z", "Phone_orientation"],
    "gyroscope": ["Systime", "EventTime", "ActivityID", "X", "Y", "Z", "Phone_orientation"],
    "touchevent": ["Systime", "EventTime", "ActivityID", "Pointer_count", "PointerID",
                   "ActionID", "X", "Y", "Pressure", "Contact_size", "Phone_orientation"],
    "onefinger": ["Systime", "PressTime", "ActivityID", "TapID", "Tap_type", "Action_type",
                  "X", "Y", "Pressure", "Contact_size", "Phone_orientation"],
    "keypress": ["Systime", "PressTime", "PressType", "ActivityID", "KeyID", "Phone_orientation"],
    "scroll": ["Systime", "BeginTime", "CurrentTime", "ActivityID", "ScrollID",
               "Start_action_type", "Start_X", "Start_Y", "Start_pressure", "Start_size",
               "Current_action_type", "Current_X", "Current_Y", "Current_pressure", "Current_size",
               "Distance_X", "Distance_Y", "Phone_orientation"],
    "stroke": ["Systime", "Begin_time", "End_time", "ActivityID", "Start_action_type",
               "Start_X", "Start_Y", "Start_pressure", "Start_size", "End_action_type",
               "End_X", "End_Y", "End_pressure", "End_size", "Speed_X", "Speed_Y", "Phone_orientation"],
}

FILE_KEYS = {
    "activity.csv": "activity",
    "accelerometer.csv": "accelerometer",
    "gyroscope.csv": "gyroscope",
    "gyroscrope.csv": "gyroscope",  # typo in documentation
    "touchevent.csv": "touchevent",
    "onefingertouchevent.csv": "onefinger",
    "keypressevent.csv": "keypress",
    "scrollevent.csv": "scroll",
    "strokeevent.csv": "stroke",
}

READING = {1, 2, 7, 8, 13, 14, 19, 20}
WRITING = {3, 4, 9, 10, 15, 16, 21, 22}
MAP = {5, 6, 11, 12, 17, 18, 23, 24}
TASK_NAMES = {"reading", "writing", "map", "unknown"}
OUTPUT_GESTURE_COLUMNS = [
    "user_id", "session_id", "task", "condition", "activity_id", "gesture_type",
    "start_ms", "end_ms", "duration_ms", "start_x", "start_y", "movement_px", "path_px", "touch_points",
    "touch_portrait", "source", "onefinger_tapid", "onefinger_double_annotation",
    "onefinger_second_action",
    "double_tap_annotated", "accelerometer_samples", "gyroscope_samples",
    "accelerometer_max_gap_ms", "gyroscope_max_gap_ms",
    "accelerometer_portrait_fraction", "gyroscope_portrait_fraction",
    "motion_covered", "rejection_reason",
]


def canonical(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().replace("\ufeff", ""))


def task_for_id(value: int | None) -> str:
    if value in READING:
        return "reading"
    if value in WRITING:
        return "writing"
    if value in MAP:
        return "map"
    return "unknown"


def activity_condition(task_id: int | None) -> str:
    if task_id is None:
        return "unknown"
    return "sitting" if task_id % 2 else "walking"


def as_number(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def normalize_systime(values: np.ndarray) -> np.ndarray:
    """Convert epoch seconds/microseconds/nanoseconds to milliseconds; do not guess relative clocks."""
    if values.size == 0:
        return values
    good = values[np.isfinite(values)]
    if not good.size:
        return values
    median = float(np.median(np.abs(good)))
    if median > 1e17:  # epoch ns
        return values / 1e6
    if median > 1e14:  # epoch us
        return values / 1e3
    if 1e9 < median < 1e11:  # epoch seconds
        return values * 1000.0
    return values  # epoch ms (expected)


def parse_number(value) -> int | None:
    try:
        x = float(value)
        return int(x) if np.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def file_info(path: Path, schema: list[str]):
    """Recognize header/no-header; tolerate commas, tabs and semicolons."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        sample = stream.readline()
    if not sample.strip():
        return ",", False
    separator = max((",", "\t", ";"), key=lambda c: sample.count(c))
    first = [canonical(x.strip(' "')) for x in sample.strip().split(separator)]
    expected = {canonical(n) for n in schema}
    with_header = bool(first and (len(set(first) & expected) >= 2 or first[0] == canonical(schema[0])))
    return separator, with_header


def csv_chunks(path: Path, key: str, use: list[str] | None = None, size: int = 200_000):
    schema = SCHEMAS[key]
    separator, with_header = file_info(path, schema)
    if with_header:
        wanted = set(map(canonical, use or schema))
        kw = {"header": 0, "usecols": lambda n: canonical(n) in wanted}
    else:
        names = schema
        if use:
            kw = {"header": None, "names": names, "usecols": use}
        else:
            kw = {"header": None, "names": names}
    iterator = pd.read_csv(
        path, sep=separator, chunksize=size, encoding="utf-8-sig",
        on_bad_lines="skip", low_memory=False, **kw,
    )
    for chunk in iterator:
        chunk.columns = [canonical(x) for x in chunk.columns]
        if len(chunk):
            yield chunk


def read_activities(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = list(csv_chunks(path, "activity", size=20_000))
    if not frames:
        return pd.DataFrame(), {}
    df = pd.concat(frames, ignore_index=True)
    counters: dict[str, int] = defaultdict(int)
    # May contain repeated activity records with mismatched end timestamps.
    if "id" in df.columns:
        counters["duplicated_activity_ids"] = int(df.duplicated("id", keep=False).sum())
        df = df.drop_duplicates(subset=["id"], keep="first")
    return df, dict(counters)


def sensor_data(path: Path | None, key: str, chunk_size: int):
    if path is None:
        return {"times": np.array([], dtype=np.float64), "orientations": np.array([], dtype=np.int8),
                "rows": 0, "unknown": 0, "portrait": 0, "landscape": 0, "nonmonotonic": 0}
    time_parts, orientation_parts = [], []
    unknown = portrait = landscape = rows = nonmono = 0
    previous = None
    for frame in csv_chunks(path, key, ["Systime", "Phone_orientation"], chunk_size):
        if "systime" not in frame or "phoneorientation" not in frame:
            raise ValueError(f"Required sensor columns missing from {path}")
        t = normalize_systime(as_number(frame["systime"]))
        o = as_number(frame["phoneorientation"])
        ok = np.isfinite(t)
        t, o = t[ok], o[ok]
        rows += int(len(t))
        portrait += int(np.count_nonzero(o == 0))
        landscape += int(np.count_nonzero(np.isin(o, [1, 3])))
        unknown += int(np.count_nonzero(~np.isin(o, [0, 1, 3])))
        if len(t) > 1:
            nonmono += int(np.count_nonzero(np.diff(t) < 0))
        if previous is not None and len(t) and t[0] < previous:
            nonmono += 1
        if len(t):
            previous = float(t[-1])
        time_parts.append(t)
        orientation_parts.append(o.astype(np.int8))
    times = np.concatenate(time_parts) if time_parts else np.empty(0)
    orient = np.concatenate(orientation_parts) if orientation_parts else np.empty(0, dtype=np.int8)
    if len(times) > 1 and np.any(np.diff(times) < 0):
        order = np.argsort(times, kind="stable")
        times, orient = times[order], orient[order]
    return {"times": times, "orientations": orient, "rows": rows, "unknown": unknown,
            "portrait": portrait, "landscape": landscape, "nonmonotonic": nonmono}


def touch_stats_and_gestures(path: Path | None, chunk_size: int, config: dict):
    counts = Counter()
    gestures: list[dict] = []
    if path is None:
        return counts, gestures
    active: dict | None = None
    last_t = None

    def close_gesture(a: dict, end_time: float, end_x: float, end_y: float, orient: int | None):
        d = end_time - a["start"]
        if not np.isfinite(d) or d < 0 or d > config["max_gesture_duration_ms"]:
            counts["gesture_bad_duration"] += 1
            return
        displacement = float(np.hypot(end_x - a["x0"], end_y - a["y0"]))
        path_len = a["path"] + float(np.hypot(end_x - a["last_x"], end_y - a["last_y"]))
        is_swipe = displacement >= config["swipe_distance_px"] or path_len >= config["swipe_path_px"]
        is_tap = d <= config["max_tap_duration_ms"] and not is_swipe
        gesture_type = "swipe" if is_swipe else "tap" if is_tap else "other"
        counts[f"gesture_detected_{gesture_type}"] += 1
        if a["multi"]:
            counts["gesture_multitouch"] += 1
            return
        if gesture_type == "other":
            counts["gesture_other"] += 1
            return
        codes = a["orientation_codes"].copy()
        if orient is not None:
            codes.add(orient)
        portrait_only = bool(codes) and codes == {0}
        gestures.append({
            "activity_id": a["activity_id"], "gesture_type": gesture_type,
            "start_ms": float(a["start"]), "end_ms": float(end_time),
            "duration_ms": float(d), "start_x": a["x0"], "start_y": a["y0"],
            "movement_px": displacement, "path_px": path_len,
            "touch_points": int(a["points"] + 1), "touch_portrait": portrait_only,
            "source": "touchevent", "onefinger_double_annotation": False,
            "onefinger_tapid": "", "onefinger_second_action": False,
        })

    for frame in csv_chunks(path, "touchevent", [
        "Systime", "ActivityID", "Pointer_count", "PointerID", "ActionID", "X", "Y", "Phone_orientation"
    ], chunk_size):
        req = {"systime", "activityid", "pointercount", "pointerid", "actionid", "x", "y", "phoneorientation"}
        if not req.issubset(frame.columns):
            raise ValueError(f"TouchEvent missing {req - set(frame.columns)}: {path}")
        t = normalize_systime(as_number(frame["systime"]))
        activity = frame["activityid"].astype("string").to_numpy()
        pc = as_number(frame["pointercount"])
        pid = as_number(frame["pointerid"])
        action = as_number(frame["actionid"])
        x = as_number(frame["x"])
        y = as_number(frame["y"])
        orientation = as_number(frame["phoneorientation"])
        counts["touch_rows"] += len(frame)
        counts["touch_portrait_rows"] += int(np.count_nonzero(orientation == 0))
        counts["touch_landscape_rows"] += int(np.count_nonzero(np.isin(orientation, [1, 3])))
        counts["touch_unknown_rows"] += int(np.count_nonzero(~np.isin(orientation, [0, 1, 3])))
        for j in range(len(frame)):
            if not np.isfinite(t[j]) or not np.isfinite(action[j]) or not np.isfinite(x[j]) or not np.isfinite(y[j]):
                counts["touch_invalid_rows"] += 1
                continue
            tm = float(t[j]); act = int(action[j]); ac_id = str(activity[j])
            o = int(orientation[j]) if np.isfinite(orientation[j]) else None
            if last_t is not None and tm < last_t:
                counts["touch_out_of_order"] += 1
                # A backwards timestamp can invalidate active gesture reconstruction.
                active = None
            last_t = tm
            if act in (0, 5):
                if active is not None:
                    counts["gesture_unclosed"] += 1
                active = {"start": tm, "x0": float(x[j]), "y0": float(y[j]),
                          "last_x": float(x[j]), "last_y": float(y[j]), "path": 0.0,
                          "points": 1, "activity_id": ac_id,
                          "orientation_codes": {o} if o is not None else set(),
                          "multi": act == 5 or pc[j] != 1 or pid[j] != 0}
            elif act in (1, 6, 2):
                if active is None:
                    if act in (1, 6):
                        counts["touch_orphan_up"] += 1
                    continue
                if ac_id != active["activity_id"]:
                    counts["gesture_activity_mismatch"] += 1
                    active = None
                    continue
                if o is not None:
                    active["orientation_codes"].add(o)
                if pc[j] != 1 or pid[j] != 0 or act == 6:
                    active["multi"] = True
                if act == 2:
                    active["path"] += float(np.hypot(x[j] - active["last_x"], y[j] - active["last_y"]))
                    active["last_x"] = float(x[j]); active["last_y"] = float(y[j])
                    active["points"] += 1
                else:
                    close_gesture(active, tm, float(x[j]), float(y[j]), o)
                    active = None
            else:
                counts["touch_other_action"] += 1
                if active is not None and (pc[j] != 1 or pid[j] != 0):
                    active["multi"] = True
    if active is not None:
        counts["gesture_unclosed"] += 1
    return counts, gestures


def onefinger_stats_and_taps(path: Path | None, chunk_size: int, config: dict):
    """Reconstruct actual DOWN -> UP intervals, NOT a gesture per annotation row.

    OneFingerTouchEvent often includes taps missing from TouchEvent (notably during
    writing). TapID=-1/-2 are legitimate UI location codes and may repeat, so TapID
    must *not* be used as a unique gesture identifier. Match by contact sequence.
    """
    counts = Counter()
    taps = []
    if path is None:
        return counts, taps
    active = None
    previous = None

    def finish(a, tm, xx, yy, orient, kind):
        duration = tm - a["start"]
        if not np.isfinite(duration) or duration < 0 or duration > config["max_gesture_duration_ms"]:
            counts["onefinger_bad_duration"] += 1
            return
        delta = float(np.hypot(xx-a["x0"], yy-a["y0"]))
        path_len = a["path"] + float(np.hypot(xx-a["last_x"], yy-a["last_y"]))
        if delta >= config["swipe_distance_px"] or path_len >= config["swipe_path_px"]:
            counts["onefinger_moving_gesture_not_counted"] += 1
            return  # TouchEvent.csv is used for swipe reconstruction.
        if duration > config["max_tap_duration_ms"]:
            counts["onefinger_long_press_not_counted"] += 1
            return
        codes = set(a["orient"])
        if orient is not None:
            codes.add(orient)
        tapid = a["tapid"]
        has_mark = a["double_mark"] or kind in (2, 3)
        second = a["second_action"] or kind == 3
        taps.append({
            "activity_id": a["activity_id"], "gesture_type": "tap",
            "start_ms": float(a["start"]), "end_ms": float(tm),
            "duration_ms": float(duration), "start_x": a["x0"], "start_y": a["y0"],
            "movement_px": delta, "path_px": path_len, "touch_points": a["points"]+1,
            "touch_portrait": bool(codes) and codes == {0},
            "source": "onefinger", "onefinger_tapid": tapid,
            "onefinger_double_annotation": bool(has_mark),
            "onefinger_second_action": bool(second),
        })
        counts["onefinger_complete_taps"] += 1

    wanted = ["Systime", "ActivityID", "TapID", "Tap_type", "Action_type",
              "X", "Y", "Phone_orientation"]
    for frame in csv_chunks(path, "onefinger", wanted, chunk_size):
        required = {"systime", "activityid", "tapid", "taptype", "actiontype",
                    "x", "y", "phoneorientation"}
        if not required.issubset(frame.columns):
            raise ValueError(f"OneFingerTouchEvent missing columns {required-set(frame.columns)}: {path}")
        ts = normalize_systime(as_number(frame["systime"]))
        ac = frame["activityid"].astype("string").to_numpy()
        ids = frame["tapid"].astype("string").to_numpy()
        kinds = as_number(frame["taptype"])
        actions = as_number(frame["actiontype"])
        xs = as_number(frame["x"]); ys = as_number(frame["y"])
        orient = as_number(frame["phoneorientation"])
        counts["onefinger_rows"] += len(frame)
        counts["onefinger_portrait_rows"] += int(np.count_nonzero(orient == 0))
        counts["onefinger_landscape_rows"] += int(np.count_nonzero(np.isin(orient, [1, 3])))
        counts["onefinger_unknown_rows"] += int(np.count_nonzero(~np.isin(orient, [0, 1, 3])))
        counts["onefinger_annotation_rows_2_3"] += int(np.count_nonzero(np.isin(kinds, [2, 3])))
        for j in range(len(frame)):
            if not (np.isfinite(ts[j]) and np.isfinite(actions[j]) and
                    np.isfinite(xs[j]) and np.isfinite(ys[j])):
                counts["onefinger_invalid_rows"] += 1
                continue
            tm=float(ts[j]); act=int(actions[j]); acid=str(ac[j]); tid=str(ids[j])
            xx=float(xs[j]); yy=float(ys[j])
            orient_j=int(orient[j]) if np.isfinite(orient[j]) else None
            kind=int(kinds[j]) if np.isfinite(kinds[j]) else None
            current=(tm, act, acid, tid, xx, yy)
            if current == previous:
                counts["onefinger_duplicate_rows"] += 1
                continue
            previous=current
            if act == 0:  # DOWN
                if active is not None:
                    counts["onefinger_unclosed"] += 1
                active={"start":tm, "x0":xx, "y0":yy, "last_x":xx, "last_y":yy,
                        "path":0.0, "points":1, "activity_id":acid, "tapid":tid,
                        "double_mark":kind in (2,3), "second_action": kind == 3,
                        "orient":{orient_j} if orient_j is not None else set()}
            elif act in (1,2):
                if active is None:
                    counts["onefinger_orphan_rows"] += 1
                    continue
                if acid != active["activity_id"]:
                    counts["onefinger_activity_mismatch"] += 1
                    active=None
                    continue
                if orient_j is not None:
                    active["orient"].add(orient_j)
                active["double_mark"] |= kind in (2, 3)
                active["second_action"] |= kind == 3
                if act == 2:
                    active["path"] += float(np.hypot(xx-active["last_x"], yy-active["last_y"]))
                    active["last_x"]=xx; active["last_y"]=yy
                    active["points"]+=1
                else:
                    finish(active, tm, xx, yy, orient_j, kind)
                    active=None
            else:
                counts["onefinger_other_action"] += 1
    if active is not None:
        counts["onefinger_unclosed"] += 1
    return counts, taps


def merge_gestures(raw: list[dict], extra_taps: list[dict], config: dict):
    """Remove TouchEvent duplicates of OneFinger taps, preserving all swipes.

    Only near-coincident intervals *and* nearby positions are considered duplicates;
    unrelated contacts that happen close in time are not merged.
    """
    counts=Counter()
    if not extra_taps:
        return raw, counts
    raw_taps=sorted((x for x in raw if x["gesture_type"]=="tap"),
                    key=lambda g:g["start_ms"])
    raw_swipes=sorted((x for x in raw if x["gesture_type"]=="swipe"),
                      key=lambda g:g["start_ms"])
    tap_starts=np.array([g["start_ms"] for g in raw_taps],dtype=float)
    swipe_starts=np.array([g["start_ms"] for g in raw_swipes],dtype=float)
    duplicate_ids=set()
    kept=[]
    for g in extra_taps:
        lo=np.searchsorted(tap_starts,g["start_ms"]-config["dedup_time_ms"])
        hi=np.searchsorted(tap_starts,g["start_ms"]+config["dedup_time_ms"],side="right")
        found=None
        for candidate in raw_taps[lo:hi]:
            if (candidate["activity_id"] == g["activity_id"] and
                abs(candidate["end_ms"]-g["end_ms"])<=config["dedup_time_ms"] and
                float(np.hypot(candidate["start_x"]-g["start_x"],
                               candidate["start_y"]-g["start_y"]))<=config["dedup_distance_px"]):
                found=candidate
                break
        if found is not None:
            duplicate_ids.add(id(found))
            counts["onefinger_deduplicated_against_touch"] += 1
            kept.append(g)
            continue
        # A tap callback describing the start of a raw swipe is not a separate tap.
        lo=np.searchsorted(swipe_starts,g["start_ms"]-config["dedup_time_ms"])
        hi=np.searchsorted(swipe_starts,g["start_ms"]+config["dedup_time_ms"],side="right")
        overlaps_swipe=any(
            raw_s["activity_id"]==g["activity_id"] and
            np.hypot(raw_s["start_x"]-g["start_x"],raw_s["start_y"]-g["start_y"])
                <=config["dedup_distance_px"] and
            g["end_ms"]<=raw_s["end_ms"]+config["dedup_time_ms"]
            for raw_s in raw_swipes[lo:hi]
        )
        if overlaps_swipe:
            counts["onefinger_suppressed_overlapping_swipe"] += 1
            continue
        kept.append(g)
    out=[g for g in raw if id(g) not in duplicate_ids] + kept
    # Rare duplicate OneFinger DOWN/UP callbacks can produce repeated intervals.
    out.sort(key=lambda g: (g["start_ms"],g["end_ms"]))
    result=[]
    for g in out:
        if result and g["source"]=="onefinger" and result[-1]["source"]=="onefinger":
            prev=result[-1]
            if (g["activity_id"]==prev["activity_id"] and
                abs(g["start_ms"]-prev["start_ms"])<=1 and
                abs(g["end_ms"]-prev["end_ms"])<=1 and
                np.hypot(g["start_x"]-prev["start_x"],g["start_y"]-prev["start_y"])<=2):
                counts["onefinger_duplicate_intervals"] += 1
                continue
        result.append(g)
    return result, counts


def mark_double_taps(gestures: list[dict], config: dict):
    """Require near-identical tap positions, short delay, and type=3 evidence.

    On a fast virtual keyboard there can be thousands of rows tagged 2/3. Treating
    each such row as a distinct double-tap causes pathological mass exclusions.
    The default narrow spatial window deliberately favors avoiding false exclusions.
    """
    gestures.sort(key=lambda g:(g["start_ms"],g["end_ms"]))
    previous_by_activity={}
    paired_count=0
    for g in gestures:
        g["double_tap_annotated"]=False
        if g["gesture_type"]!="tap":
            continue
        prev=previous_by_activity.get(g["activity_id"])
        if prev is not None:
            dt=g["start_ms"]-prev["end_ms"]
            dist=float(np.hypot(g["start_x"]-prev["start_x"],g["start_y"]-prev["start_y"]))
            if (0<=dt<=config["double_tap_match_ms"] and
                dist<=config["double_tap_match_px"] and
                (g.get("onefinger_second_action") or prev.get("onefinger_second_action"))):
                prev["double_tap_annotated"]=True
                g["double_tap_annotated"]=True
                paired_count+=1
        previous_by_activity[g["activity_id"]]=g
    return paired_count


def keypress_stats(path: Path | None, chunk_size: int):
    counts=Counter()
    if path is None:
        return counts
    for frame in csv_chunks(path,"keypress",["Systime","PressType","Phone_orientation"],chunk_size):
        counts["keypress_rows"]+=len(frame)
        if "presstype" in frame:
            actions=as_number(frame["presstype"])
            counts["keypress_down_rows"]+=int(np.count_nonzero(actions==0))
            counts["keypress_up_rows"]+=int(np.count_nonzero(actions==1))
    return counts


def gesture_sensor_coverage(gestures: list[dict], acc: dict, gyro: dict, config: dict):
    if not gestures:
        return
    starts = np.array([g["start_ms"] for g in gestures], dtype=np.float64)
    ends = np.array([g["end_ms"] for g in gestures], dtype=np.float64)
    for source, data in (("accelerometer", acc), ("gyroscope", gyro)):
        vector = data["times"]
        orientations = data["orientations"]
        i0 = np.searchsorted(vector, starts, side="left")
        i1 = np.searchsorted(vector, ends, side="right")
        for j, g in enumerate(gestures):
            inside = vector[i0[j]:i1[j]]
            g[f"{source}_samples"] = int(len(inside))
            subset_orient = orientations[i0[j]:i1[j]]
            known = np.isin(subset_orient, [0, 1, 3])
            g[f"{source}_portrait_fraction"] = (float(np.mean(subset_orient[known] == 0))
                                                if np.any(known) else None)
            # Do not count samples from outside the gesture as motion coverage.
            g[f"{source}_max_gap_ms"] = (float(np.diff(inside).max()) if len(inside) > 1 else None)
    for j, g in enumerate(gestures):
        annotated_double=bool(g["double_tap_annotated"])
        rejection = []
        if not g["touch_portrait"]:
            rejection.append("gesture_not_portrait")
        if annotated_double:
            rejection.append("annotated_double_tap")
        for source in ("accelerometer", "gyroscope"):
            if g[f"{source}_samples"] < config["min_sensor_samples"]:
                rejection.append(f"low_{source}_samples")
            frac = g[f"{source}_portrait_fraction"]
            if frac is None or frac < 1.0:
                rejection.append(f"{source}_not_fully_portrait")
            gap = g[f"{source}_max_gap_ms"]
            if gap is not None and gap > config["max_sensor_gap_ms"]:
                rejection.append(f"large_{source}_gap")
        g["motion_covered"] = not rejection
        g["rejection_reason"] = ";".join(rejection)


def file_manifest(files: dict[str, Path], config: dict):
    file_stamps = {}
    for key, path in sorted(files.items()):
        stat = path.stat()
        file_stamps[key] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    payload = json.dumps({"version": CACHE_VERSION, "files": file_stamps,
                          "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def available_files(folder: Path) -> dict[str, Path]:
    result = {}
    for item in folder.iterdir():
        if item.is_file() and (key := FILE_KEYS.get(item.name.lower())):
            if key in result and key == "gyroscope" and item.name.lower() == "gyroscrope.csv":
                continue
            result[key] = item
    return result


def count_derived(path: Path | None, key: str, chunk_size: int) -> int:
    if path is None:
        return 0
    # Count CSV data records, not recognized gestures (scroll has many rows per gesture).
    n = 0
    for frame in csv_chunks(path, key, ["Systime"], chunk_size):
        n += len(frame)
    return n


def audit_session(user_id: str, folder: Path, files: dict[str, Path], config: dict):
    messages = []
    row = {"user_id": user_id, "session_dir": str(folder), "session_id": folder.name,
           "task": "unknown", "condition": "unknown", "status": "excluded",
           "reason": "", "activity_rows": 0}
    if "activity" not in files:
        row["reason"] = "missing_Activity.csv"
        return row, [], messages
    activities, activity_stats = read_activities(files["activity"])
    row["activity_rows"] = int(len(activities))
    row.update(activity_stats)
    if activities.empty:
        row["reason"] = "empty_Activity.csv"
        return row, [], messages
    if "sessionnumber" in activities:
        nums = sorted(set(int(v) for v in as_number(activities["sessionnumber"]) if np.isfinite(v)))
        if len(nums) == 1:
            row["session_id"] = str(nums[0])
        elif nums:
            messages.append(f"{user_id}/{folder.name}: Activity.csv lists multiple Session_number={nums}; review folder layout")
            row["reason"] = "multiple_sessions_in_one_folder"
            return row, [], messages
    if "starttime" in activities and "endtime" in activities:
        starts = normalize_systime(as_number(activities["starttime"]))
        ends = normalize_systime(as_number(activities["endtime"]))
        if np.any(np.isfinite(starts)) and np.any(np.isfinite(ends)):
            row["activity_duration_seconds"] = round(
                float((np.nanmax(ends) - np.nanmin(starts)) / 1000), 2
            )
    if "taskid" not in activities:
        row["reason"] = "missing_TaskID"
        return row, [], messages
    task_ids = [int(v) for v in as_number(activities["taskid"]) if np.isfinite(v)]
    tasks = {task_for_id(n) for n in task_ids}
    conditions = {activity_condition(n) for n in task_ids}
    row["task"] = next(iter(tasks)) if len(tasks) == 1 else "mixed"
    row["condition"] = next(iter(conditions)) if len(conditions) == 1 else "mixed"
    row["task_ids"] = ";".join(map(str, sorted(set(task_ids))))
    if row["task"] == "map":
        row["reason"] = "map_out_of_scope"
        return row, [], messages
    if row["task"] not in ("writing", "reading") or row["condition"] == "mixed":
        row["reason"] = "invalid_or_mixed_TaskID"
        return row, [], messages
    missing = [k for k in ("accelerometer", "gyroscope") if k not in files]
    if "touchevent" not in files and "onefinger" not in files:
        missing.append("touch_and_onefinger")
    row["missing_files"] = ";".join(missing)
    if missing:
        row["reason"] = "missing_required_file"
        return row, [], messages

    acc = sensor_data(files["accelerometer"], "accelerometer", config["chunk_size"])
    gyro = sensor_data(files["gyroscope"], "gyroscope", config["chunk_size"])
    tc, raw_gestures = touch_stats_and_gestures(
        files.get("touchevent"), config["chunk_size"], config)
    oc, onefinger_taps = onefinger_stats_and_taps(
        files.get("onefinger"), config["chunk_size"], config)
    gestures, dedup_stats = merge_gestures(raw_gestures, onefinger_taps, config)
    double_pairs=mark_double_taps(gestures, config)
    key_stats=keypress_stats(files.get("keypress"), config["chunk_size"])
    row["accel_rows"] = acc["rows"]
    row["gyro_rows"] = gyro["rows"]
    row["touch_rows"] = tc["touch_rows"]
    row["onefinger_rows"] = oc["onefinger_rows"]
    row["combined_touch_rows"] = tc["touch_rows"]+oc["onefinger_rows"]
    row["touch_raw_gesture_candidates"] = len(raw_gestures)
    row["onefinger_complete_taps"] = len(onefinger_taps)
    row["onefinger_annotation_rows_2_3"] = oc["onefinger_annotation_rows_2_3"]
    row["onefinger_unclosed"] = oc["onefinger_unclosed"]
    row["onefinger_moving_gesture_not_counted"] = oc["onefinger_moving_gesture_not_counted"]
    row["onefinger_long_press_not_counted"] = oc["onefinger_long_press_not_counted"]
    row["onefinger_deduplicated_against_touch"] = dedup_stats["onefinger_deduplicated_against_touch"]
    row["onefinger_duplicate_intervals"] = dedup_stats["onefinger_duplicate_intervals"]
    row["onefinger_suppressed_overlapping_swipe"] = dedup_stats["onefinger_suppressed_overlapping_swipe"]
    row["double_tap_pairs_detected"] = double_pairs
    row.update(key_stats)
    for name, data in (("accel", acc), ("gyro", gyro)):
        known = data["portrait"] + data["landscape"]
        row[f"{name}_portrait_fraction"] = round(data["portrait"] / known, 6) if known else None
        row[f"{name}_landscape_fraction"] = round(data["landscape"] / known, 6) if known else None
        row[f"{name}_unknown_fraction"] = round(data["unknown"] / data["rows"], 6) if data["rows"] else None
        row[f"{name}_time_nonmonotonic"] = data["nonmonotonic"]
    # Include OneFinger events: otherwise typing session orientation is biased
    # towards the few gestures occurring outside the virtual keyboard.
    raw_known = tc["touch_portrait_rows"]+tc["touch_landscape_rows"]
    one_known = oc["onefinger_portrait_rows"]+oc["onefinger_landscape_rows"]
    combined_known=raw_known+one_known
    row["raw_touch_portrait_fraction"]=(round(tc["touch_portrait_rows"]/raw_known,6)
        if raw_known else None)
    row["onefinger_portrait_fraction"]=(round(oc["onefinger_portrait_rows"]/one_known,6)
        if one_known else None)
    row["touch_portrait_fraction"]=(round((tc["touch_portrait_rows"]+
        oc["onefinger_portrait_rows"])/combined_known,6) if combined_known else None)
    row["touch_unknown_fraction"]=(round((tc["touch_unknown_rows"]+
        oc["onefinger_unknown_rows"])/row["combined_touch_rows"],6)
        if row["combined_touch_rows"] else None)
    known_total = sum(x["portrait"] + x["landscape"] for x in (acc, gyro))
    portrait_total = acc["portrait"] + gyro["portrait"]
    row["motion_portrait_fraction"] = round(portrait_total / known_total, 6) if known_total else None
    row["motion_landscape_fraction"] = round(1 - portrait_total / known_total, 6) if known_total else None
    row["touch_gesture_candidates"] = int(len(gestures))
    row["gesture_multitouch"] = tc["gesture_multitouch"]
    row["gesture_unclosed"] = tc["gesture_unclosed"]
    row["touch_time_nonmonotonic"] = tc["touch_out_of_order"]

    for key in ("accel", "gyro"):
        if row[f"{key}_portrait_fraction"] is None:
            row["reason"] = f"{key}_orientation_unknown"
            break
        if row[f"{key}_portrait_fraction"] < config["min_portrait_fraction"]:
            row["reason"] = f"{key}_mostly_landscape"
            break
        if row[f"{key}_unknown_fraction"] > config["max_unknown_orientation_fraction"]:
            row["reason"] = f"{key}_too_many_unknown_orientations"
            break
    if not row["reason"]:
        if row["touch_portrait_fraction"] is None:
            row["reason"] = "touch_orientation_unknown"
        elif row["touch_portrait_fraction"] < config["min_portrait_fraction"]:
            row["reason"] = "touch_mostly_landscape"
        elif row["touch_unknown_fraction"] > config["max_unknown_orientation_fraction"]:
            row["reason"] = "touch_too_many_unknown_orientations"

    gesture_sensor_coverage(gestures, acc, gyro, config)
    row["annotated_double_tap_events"] = oc["onefinger_annotation_rows_2_3"]
    row["raw_taps"] = sum(g["gesture_type"] == "tap" for g in gestures)
    row["raw_swipes"] = sum(g["gesture_type"] == "swipe" for g in gestures)
    row["covered_taps"] = sum(g["gesture_type"] == "tap" and g["motion_covered"] for g in gestures)
    row["covered_swipes"] = sum(g["gesture_type"] == "swipe" and g["motion_covered"] for g in gestures)
    row["covered_total"] = row["covered_taps"] + row["covered_swipes"]
    row["coverage_rate"] = (round(row["covered_total"] / len(gestures), 6)
                            if gestures else 0.0)
    row["double_tap_rejected"] = sum(g["double_tap_annotated"] for g in gestures)
    row["gesture_nonportrait"] = sum(not g["touch_portrait"] for g in gestures)
    row["low_accel_gestures"] = sum(g["accelerometer_samples"] < config["min_sensor_samples"] for g in gestures)
    row["low_gyro_gestures"] = sum(g["gyroscope_samples"] < config["min_sensor_samples"] for g in gestures)
    row["strokeevent_rows"] = count_derived(files.get("stroke"), "stroke", config["chunk_size"])
    row["scrollevent_rows"] = count_derived(files.get("scroll"), "scroll", config["chunk_size"])

    if not row["reason"] and row["covered_total"] < config["min_covered_gestures"]:
        row["reason"] = "too_few_motion_covered_gestures"
    if not row["reason"] and row["task"] == "reading" and row["covered_swipes"] < config["min_reading_swipes"]:
        row["reason"] = "too_few_reading_swipes"
    if not row["reason"] and row["task"] == "writing" and row["covered_taps"] < config["min_writing_taps"]:
        row["reason"] = "too_few_writing_taps"
    if not row["reason"]:
        row["status"] = "eligible"
        row["reason"] = ""

    row["below_100_covered"] = row["covered_total"] < 100
    row["below_20_swipes"] = row["covered_swipes"] < 20
    row["below_20_taps"] = row["covered_taps"] < 20
    # Convert gesture entries to cache records, including rejected candidates for auditing.
    for g in gestures:
        g.update({"user_id": user_id, "session_id": row["session_id"],
                  "task": row["task"], "condition": row["condition"]})
    if row["status"] != "eligible" and row["reason"]:
        messages.append(f"{user_id}/{row['session_id']} [{row['task']}]: {row['reason']}; "
                        f"portrait={row.get('motion_portrait_fraction')}, covered={row['covered_total']}")
    return row, gestures, messages


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def csv_gzip(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_GESTURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    os.replace(tmp, path)


def analyze_user(job):
    user_id, folder_paths, out_folder, config = job
    cache = Path(out_folder) / "cache" / "users" / f"{user_id}.json"
    gesture_cache = Path(out_folder) / "cache" / "gestures" / f"{user_id}.csv.gz"
    folders = [Path(x) for x in folder_paths]
    sources = {}
    for i, folder in enumerate(folders):
        for key, path in available_files(folder).items():
            sources[f"{i}/{key}"] = path
    manifest = file_manifest(sources, config)
    if not config["no_cache"] and cache.exists() and gesture_cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("manifest") == manifest:
                return {"user_id": user_id, "cached": True, "data": data}
        except (OSError, json.JSONDecodeError):
            pass
    sessions = []
    all_gestures = []
    messages = []
    for folder in folders:
        try:
            row, gestures, warnings = audit_session(user_id, folder, available_files(folder), config)
            sessions.append(row)
            all_gestures.extend(gestures)
            messages.extend(warnings)
        except Exception as ex:
            sessions.append({"user_id": user_id, "session_id": folder.name,
                             "session_dir": str(folder), "task": "unknown", "status": "error",
                             "reason": f"{type(ex).__name__}: {ex}"})
            messages.append(f"{user_id}/{folder.name}: {traceback.format_exc(limit=4)}")
    counts = Counter(s["task"] for s in sessions if s["status"] == "eligible")
    data = {"user_id": user_id, "manifest": manifest,
            "sessions": sessions, "messages": messages,
            "eligible_reading": counts["reading"], "eligible_writing": counts["writing"]}
    csv_gzip(gesture_cache, all_gestures)
    atomic_json(cache, data)
    return {"user_id": user_id, "cached": False, "data": data}


def discover_users(root: Path) -> dict[str, list[str]]:
    users: dict[str, list[str]] = defaultdict(list)
    for folder, subdirs, names in os.walk(root):
        if not any(name.lower() in FILE_KEYS for name in names):
            continue
        p = Path(folder)
        rel_parts = p.relative_to(root).parts
        user_id = next((part for part in rel_parts if re.fullmatch(r"\d{6}", part)), None)
        if user_id is None:
            user_id = rel_parts[0] if len(rel_parts) >= 2 else "UNKNOWN"
        users[user_id].append(str(p))
    return {uid: sorted(set(dirs)) for uid, dirs in sorted(users.items())}


def write_csv(path: Path, records: list[dict]):
    keys = sorted(set().union(*(r.keys() for r in records))) if records else []
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def ids_by_condition(sessions: list[dict], task: str, min_sessions: int):
    good = [s for s in sessions if s["status"] == "eligible" and s["task"] == task]
    return {"count": len(good), "sitting": sum(s["condition"] == "sitting" for s in good),
            "walking": sum(s["condition"] == "walking" for s in good),
            "ready": len(good) >= min_sessions,
            "both_conditions_ready": (sum(s["condition"] == "sitting" for s in good) >= min_sessions
                                      and sum(s["condition"] == "walking" for s in good) >= min_sessions)}


def write_ids(path: Path, ids: list[str]):
    path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def summarize(all_sessions: list[dict], users: list[dict], out: Path, config: dict):
    write_csv(out / "sessions.csv", all_sessions)
    excluded = [r for r in all_sessions if r["status"] != "eligible"]
    write_csv(out / "excluded_sessions.csv", excluded)
    eligible = [r for r in all_sessions if r["status"] == "eligible"]
    write_csv(out / "eligible_sessions.csv", eligible)
    ready_rows = []
    groups = defaultdict(list)
    for row in all_sessions:
        groups[row["user_id"]].append(row)
    for uid in sorted(groups):
        r = {"user_id": uid}
        for task in ("reading", "writing"):
            stats = ids_by_condition(groups[uid], task, config["min_sessions_per_task"])
            r.update({f"{task}_{k}": v for k, v in stats.items()})
        ready_rows.append(r)
    write_csv(out / "users.csv", ready_rows)
    reading = [r["user_id"] for r in ready_rows if r["reading_count"] >= 1]
    writing = [r["user_id"] for r in ready_rows if r["writing_count"] >= 1]
    read_ready = [r["user_id"] for r in ready_rows if r["reading_ready"]]
    write_ready = [r["user_id"] for r in ready_rows if r["writing_ready"]]
    for fname, ids in {
        "users_good_reading.txt": reading,
        "users_good_writing.txt": writing,
        "users_good_both.txt": sorted(set(reading) & set(writing)),
        "users_ready_reading.txt": read_ready,
        "users_ready_writing.txt": write_ready,
        "users_ready_both.txt": sorted(set(read_ready) & set(write_ready)),
        "users_ready_reading_sitting_walking.txt": [r["user_id"] for r in ready_rows if r["reading_both_conditions_ready"]],
        "users_ready_writing_sitting_walking.txt": [r["user_id"] for r in ready_rows if r["writing_both_conditions_ready"]],
    }.items():
        write_ids(out / fname, ids)
    reason_counts = Counter(s["reason"] for s in excluded)
    status_counts = Counter(s["status"] for s in all_sessions)
    by_task = {task: {
        "sessions_total": sum(s.get("task") == task for s in all_sessions),
        "sessions_eligible": sum(s.get("task") == task and s.get("status") == "eligible" for s in all_sessions),
        "covered_gestures": sum(int(s.get("covered_total", 0)) for s in eligible if s.get("task") == task),
        "covered_taps": sum(int(s.get("covered_taps", 0)) for s in eligible if s.get("task") == task),
        "covered_swipes": sum(int(s.get("covered_swipes", 0)) for s in eligible if s.get("task") == task),
    } for task in ("reading", "writing")}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_version": CACHE_VERSION,
        "criteria": {k: v for k, v in config.items() if k not in ("no_cache",)},
        "users_total": len(groups), "sessions_total": len(all_sessions),
        "sessions_by_status": dict(status_counts),
        "excluded_reasons": dict(reason_counts),
        "tasks": by_task,
        "gesture_diagnostics": {
            "onefinger_complete_taps": sum(int(s.get("onefinger_complete_taps",0)) for s in all_sessions),
            "onefinger_annotation_rows_2_3": sum(int(s.get("onefinger_annotation_rows_2_3",0)) for s in all_sessions),
            "onefinger_deduplicated_against_touch": sum(int(s.get("onefinger_deduplicated_against_touch",0)) for s in all_sessions),
            "onefinger_suppressed_overlapping_swipe": sum(int(s.get("onefinger_suppressed_overlapping_swipe",0)) for s in all_sessions),
            "double_tap_pairs_detected": sum(int(s.get("double_tap_pairs_detected",0)) for s in all_sessions),
            "keypress_down_rows": sum(int(s.get("keypress_down_rows",0)) for s in all_sessions),
        },
        "users_good_reading": len(reading), "users_good_writing": len(writing),
        "users_good_both": len(set(reading) & set(writing)),
        "users_ready_reading": len(read_ready), "users_ready_writing": len(write_ready),
        "users_ready_both": len(set(read_ready) & set(write_ready)),
        "users_ready_reading_sitting_walking": sum(r["reading_both_conditions_ready"] for r in ready_rows),
        "users_ready_writing_sitting_walking": sum(r["writing_both_conditions_ready"] for r in ready_rows),
    }
    atomic_json(out / "summary.json", summary)
    (out / "HOW_TO_READ_RESULTS.md").write_text("""# HMOG session audit — output guide

- `summary.json`: total users/sessions, reasons for exclusions, task counts. Send this back to ChatGPT.
- `sessions.csv`: one row per session; `motion_landscape_fraction`, `accel_landscape_fraction`,
  and `gyro_landscape_fraction` are proportions of recognized orientation rows.
- `eligible_sessions.csv` / `excluded_sessions.csv`: filtered session-level tables.
- `users.csv`: counts of eligible reading/writing, sitting/walking sessions per user.
- `users_good_reading.txt` / `users_good_writing.txt`: at least **one** eligible session.
- `users_ready_reading.txt` / `users_ready_writing.txt`: at least **N** eligible sessions
  (N is `--min-sessions-per-task`, default 2). This is only a preliminary readiness flag.
- `users_ready_*_sitting_walking.txt`: at least N good sessions in each motion condition.
- `cache/users/<id>.json`: per-user summary and warnings; reused when input unchanged.
- `cache/gestures/<id>.csv.gz`: reconstruction of single-finger tap/swipe gesture intervals,
  counts of both motion sensors, gesture-level portrait checks, and rejection reasons.
- `analysis.log`: per-user multiprocessing progress, warnings, and final counts.

## Important analysis conventions

A session is rejected if any of touch, accelerometer, or gyroscope is less than the
configured portrait-majority threshold among rows with recognized orientation codes.
For individual gestures, all available touch/accelerometer/gyroscope orientation codes
must be portrait (`0`), so predominantly portrait sessions can still contain rejected
gestures. Codes `1` and `3` mean landscape; other codes are unknown.

Session sufficiency thresholds are **exploratory choices**, not thresholds taken from
HMOG's published classifier evaluation. They can be changed on the CLI. Reading sessions
must have enough motion-covered swipes and writing sessions enough motion-covered taps.

`TouchEvent.csv` supplies raw strokes/swipes and taps; `OneFingerTouchEvent.csv`
adds completed one-finger DOWN/UP taps (often keyboard taps missing from raw touch).
Matching intervals/positions are deduplicated, prioritizing OneFingerTap details.
`KeyPressEvent.csv` supplies diagnostic press counts only, because it lacks touch
coordinates and pressure. We do not invent touch gestures from key press timestamps.
A suspected double-tap pair is excluded only when two taps occur very close
in time AND position, and at least one has the explicit `Tap_type=3` mark.
This conservative heuristic avoids excluding numerous adjacent keyboard taps.
Annotation rows coded `2` or `3` alone are NOT treated as distinct double-taps.
If the source files are unusual, inspect `onefinger_complete_taps`,
`onefinger_annotation_rows_2_3`, `double_tap_pairs_detected`,
`keypress_down_rows`, and `covered_taps` in `sessions.csv`.
`touch_portrait_fraction` is based on combined raw-touch and OneFinger rows.
`Systime` is used for cross-source timing because the HMOG documentation describes it as
an absolute timestamp in each source. Very low overlap should trigger manual review of
timestamp conventions and raw files before dataset exclusion decisions are finalized.

The program does not train authenticators or perform final train/validation/test splitting.
Sessions with ≥2 suitable blocks are *candidates* for chronological splitting; study-specific
minimum enrollment and test sample sizes must be checked in the next stage.
""", encoding="utf-8")
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="Audit portrait and motion-covered gestures in HMOG sessions")
    p.add_argument("--input", type=Path, default=Path("input_data/hmog_unarchived"))
    p.add_argument("--output", type=Path, default=Path("output_data/hmog_analysis"))
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2)//2)))
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--min-portrait-fraction", type=float, default=0.5,
                   help="Session is portrait-majority at or above this fraction, separately for touch/accel/gyro")
    p.add_argument("--max-unknown-orientation-fraction", type=float, default=0.05)
    p.add_argument("--min-covered-gestures", type=int, default=40)
    p.add_argument("--min-sessions-per-task", type=int, default=2)
    p.add_argument("--min-reading-swipes", type=int, default=20)
    p.add_argument("--min-writing-taps", type=int, default=40)
    p.add_argument("--min-sensor-samples", type=int, default=3)
    p.add_argument("--max-sensor-gap-ms", type=float, default=120.0)
    p.add_argument("--swipe-distance-px", type=float, default=30.0)
    p.add_argument("--swipe-path-px", type=float, default=50.0)
    p.add_argument("--max-tap-duration-ms", type=float, default=500.0)
    p.add_argument("--max-gesture-duration-ms", type=float, default=12_000.0)
    p.add_argument("--double-tap-match-ms", type=float, default=300.0)
    p.add_argument("--double-tap-match-px", type=float, default=20.0)
    p.add_argument("--dedup-time-ms", type=float, default=90.0,
                   help="Max start/end timestamp difference for cross-file gesture deduplication")
    p.add_argument("--dedup-distance-px", type=float, default=100.0)
    p.add_argument("--no-cache", action="store_true")
    return p


def main():
    # Also usable on Python 3.10+, designed for Python 3.14.
    args = build_parser().parse_args()
    root = args.input.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hmog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(out / "analysis.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt); logger.addHandler(h)
    if not root.is_dir():
        logger.error("Dataset folder does not exist: %s", root)
        return 2
    config = {
        "chunk_size": args.chunk_size,
        "min_portrait_fraction": args.min_portrait_fraction,
        "max_unknown_orientation_fraction": args.max_unknown_orientation_fraction,
        "min_covered_gestures": args.min_covered_gestures,
        "min_sessions_per_task": args.min_sessions_per_task,
        "min_reading_swipes": args.min_reading_swipes,
        "min_writing_taps": args.min_writing_taps,
        "min_sensor_samples": args.min_sensor_samples,
        "max_sensor_gap_ms": args.max_sensor_gap_ms,
        "swipe_distance_px": args.swipe_distance_px,
        "swipe_path_px": args.swipe_path_px,
        "max_tap_duration_ms": args.max_tap_duration_ms,
        "max_gesture_duration_ms": args.max_gesture_duration_ms,
        "double_tap_match_ms": args.double_tap_match_ms,
        "double_tap_match_px": args.double_tap_match_px,
        "dedup_time_ms": args.dedup_time_ms,
        "dedup_distance_px": args.dedup_distance_px,
        "no_cache": args.no_cache,
    }
    users = discover_users(root)
    if not users:
        logger.error("No HMOG CSV directories were found under %s", root)
        return 2
    logger.info("Found %s users, %s session directories; workers=%d", len(users),
                sum(map(len, users.values())), args.workers)
    logger.info("Dataset: %s", root)
    logger.info("Output: %s", out)
    logger.info("Portrait threshold: %.1f%% per source; minimum covered gestures/session: %d",
                100 * args.min_portrait_fraction, args.min_covered_gestures)
    logger.info("Gesture reconstruction: TouchEvent swipes + OneFingerTouchEvent taps; "
                "keypress diagnostics; cache_version=%s", CACHE_VERSION)
    jobs = [(uid, folders, str(out), config) for uid, folders in users.items()]
    results, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_user, job): job[0] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                response = future.result()
                data = response["data"]
                results.append(data)
                sessions = data["sessions"]
                eligible = [r for r in sessions if r.get("status") == "eligible"]
                logger.info("[%3d/%3d] user=%s %s eligible=%d/%d (R=%d W=%d) | "
                            "onefinger_taps=%d, covered_taps=%d, keypress_down=%d",
                            done, len(futures), uid, "CACHE" if response["cached"] else "DONE ",
                            len(eligible), len(sessions), data["eligible_reading"], data["eligible_writing"],
                            sum(int(s.get("onefinger_complete_taps",0)) for s in sessions),
                            sum(int(s.get("covered_taps",0)) for s in sessions),
                            sum(int(s.get("keypress_down_rows",0)) for s in sessions))
                for line in data["messages"][:5]:
                    logger.warning("%s", line)
                if len(data["messages"]) > 5:
                    logger.warning("user=%s: %d more messages saved in user JSON cache", uid, len(data["messages"])-5)
            except Exception:
                failed.append(uid)
                logger.exception("[%3d/%3d] FAILED user=%s", done, len(futures), uid)
    if not results:
        logger.error("No successful user analyses")
        return 1
    sessions = [s for item in results for s in item["sessions"]]
    sessions.sort(key=lambda s: (s["user_id"], int(s["session_id"]) if s["session_id"].isdigit() else 9999))
    summary = summarize(sessions, results, out, config)
    summary["failed_user_ids"] = failed
    atomic_json(out / "summary.json", summary)
    logger.info("DONE. %d users; %d sessions; eligible R=%d W=%d; failed users=%d",
                summary["users_total"], summary["sessions_total"],
                summary["tasks"]["reading"]["sessions_eligible"],
                summary["tasks"]["writing"]["sessions_eligible"], len(failed))
    report = {
        "users_total": summary["users_total"],
        "users_good_reading": summary["users_good_reading"],
        "users_good_writing": summary["users_good_writing"],
        "users_ready_reading": summary["users_ready_reading"],
        "users_ready_writing": summary["users_ready_writing"],
        "users_ready_both": summary["users_ready_both"],
        "tasks": summary["tasks"],
        "gesture_diagnostics": summary["gesture_diagnostics"],
        "excluded_reasons": summary["excluded_reasons"],
        "failed_users": failed,
    }
    print("\n===== COPY THIS BLOCK BACK TO CHAT =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("===== END BLOCK =====\n")
    logger.info("Send summary.json, users.csv and a sample of excluded_sessions.csv for next methodology decisions.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
