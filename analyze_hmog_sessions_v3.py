#!/usr/bin/env python3
"""HMOG v3: auditable Android-gesture-aware session quality analysis.

Python 3.14 (also compatible with 3.11+), numpy, pandas.

Run from the repository root:
    py -3.14 analyze_hmog_sessions_v3.py --workers 6

Input:  input_data/hmog_unarchived
Output: output_data/hmog_analysis_v3

IMPORTANT: This is an AUDIT, not a claim of bit-for-bit reproduction of the
HMOG collection application's GestureDetector. It reconciles raw contacts
with the detector's serialized callback records, keeps unknown matches visible,
and reports how gesture endpoints were determined. The raw dataset itself is
not available to this script's author: validate diagnostics on your real data.

Android mapping (HMOG PDF pages 4-10):
    TouchEvent: MotionEvent actions; OneFingerTouchEvent: tap callbacks;
    ScrollEvent: repeated GestureDetector.onScroll();
    StrokeEvent: GestureDetector.onFling();
    Accelerometer/Gyroscope: motion sensor samples.

No gestures from map-navigation sessions are accepted. Single-finger taps and
swipes are allowed. Pinch/multitouch, long press and positively matched double
taps are excluded from the PRIMARY eligible inventory. Uncertain events remain
in the cached gesture catalog for manual review.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CACHE_VERSION = "hmog-audit-v3.0-android-callbacks-20261107"
SCHEMAS = {
    "activity": ["ID", "SubjectID", "Session_number", "Start_time", "End_time", "Relative_Start_time", "Relative_End_time", "Gesture_scenario", "TaskID", "ContentID"],
    "accelerometer": ["Systime", "EventTime", "ActivityID", "X", "Y", "Z", "Phone_orientation"],
    "gyroscope": ["Systime", "EventTime", "ActivityID", "X", "Y", "Z", "Phone_orientation"],
    "touchevent": ["Systime", "EventTime", "ActivityID", "Pointer_count", "PointerID", "ActionID", "X", "Y", "Pressure", "Contact_size", "Phone_orientation"],
    "onefinger": ["Systime", "PressTime", "ActivityID", "TapID", "Tap_type", "Action_type", "X", "Y", "Pressure", "Contact_size", "Phone_orientation"],
    "keypress": ["Systime", "PressTime", "PressType", "ActivityID", "KeyID", "Phone_orientation"],
    "pinch": ["Systime", "PressTime", "ActivityID", "EventType", "PinchID", "Time_delta", "Focus_X", "Focus_Y", "Span", "Span_X", "Span_Y", "ScaleFactor", "Phone_orientation"],
    "scroll": ["Systime", "BeginTime", "CurrentTime", "ActivityID", "ScrollID", "Start_action_type", "Start_X", "Start_Y", "Start_pressure", "Start_size", "Current_action_type", "Current_X", "Current_Y", "Current_pressure", "Current_size", "Distance_X", "Distance_Y", "Phone_orientation"],
    "stroke": ["Systime", "Begin_time", "End_time", "ActivityID", "Start_action_type", "Start_X", "Start_Y", "Start_pressure", "Start_size", "End_action_type", "End_X", "End_Y", "End_pressure", "End_size", "Speed_X", "Speed_Y", "Phone_orientation"],
}
FILE_KEYS = {
    "activity.csv": "activity", "accelerometer.csv": "accelerometer",
    "gyroscope.csv": "gyroscope", "gyroscrope.csv": "gyroscope",
    "touchevent.csv": "touchevent", "onefingertouchevent.csv": "onefinger",
    "keypressevent.csv": "keypress", "pinchevent.csv": "pinch", "scrollevent.csv": "scroll",
    "strokeevent.csv": "stroke",
}
READING = {1, 2, 7, 8, 13, 14, 19, 20}
WRITING = {3, 4, 9, 10, 15, 16, 21, 22}
MAP = {5, 6, 11, 12, 17, 18, 23, 24}

GESTURE_FIELDS = [
    "user_id", "session_id", "task", "condition", "activity_id", "gesture_id",
    "gesture_type", "start_ms", "end_ms", "duration_ms", "start_x", "start_y",
    "end_x", "end_y", "path_px", "movement_px", "touch_points", "portrait_only",
    "source", "recognition", "endpoint_quality", "time_basis", "time_warning",
    "matched_touch", "matched_onefinger", "matched_scroll", "matched_stroke",
    "scroll_callback_rows", "fling", "onefinger_taptypes", "onefinger_tapid",
    "double_annotation", "confirmed_single", "double_tap_pair", "double_tap_suspect",
    "multitouch", "pinch_overlap", "accelerometer_samples", "gyroscope_samples",
    "accelerometer_max_gap_ms", "gyroscope_max_gap_ms",
    "accelerometer_portrait_fraction", "gyroscope_portrait_fraction",
    "motion_covered", "rejection_reason",
]


def canonical(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower().replace("\ufeff", ""))


def num(value: Any, default=float("nan")) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def text_id(value: Any) -> str:
    s = str(value)
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def number_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def norm_absolute(t: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(t, dtype=np.float64)
    good = arr[np.isfinite(arr)]
    if not good.size:
        return arr
    m = float(np.median(abs(good)))
    if m >= 1e17:
        return arr / 1e6
    if m >= 1e14:
        return arr / 1e3
    if 1e9 < m < 1e11:
        return arr * 1000
    return arr


def file_info(path: Path, schema: list[str]) -> tuple[str, bool]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.readline()
    if not sample.strip():
        return ",", False
    delim = max((",", "\t", ";"), key=sample.count)
    first = [canonical(x.strip(' "')) for x in sample.strip().split(delim)]
    expected = {canonical(x) for x in schema}
    header = len(set(first) & expected) >= 2 or bool(first and first[0] == canonical(schema[0]))
    return delim, header


def csv_chunks(path: Path | None, key: str, use: list[str] | None = None,
               chunk_size: int = 200_000):
    if path is None:
        return
    schema = SCHEMAS[key]
    delimiter, has_header = file_info(path, schema)
    if has_header:
        wanted = {canonical(x) for x in (use or schema)}
        opt = {"header": 0, "usecols": lambda name: canonical(name) in wanted}
    else:
        opt = {"header": None, "names": schema}
        if use:
            opt["usecols"] = use
    for frame in pd.read_csv(path, sep=delimiter, chunksize=chunk_size,
                             on_bad_lines="skip", encoding="utf-8-sig",
                             low_memory=False, **opt):
        frame.columns = [canonical(c) for c in frame.columns]
        if len(frame):
            yield frame


def load_table(path: Path | None, key: str, columns: list[str], chunk_size: int) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    pieces = list(csv_chunks(path, key, columns, chunk_size))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def col_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name in frame:
        return number_array(frame[name])
    return np.full(len(frame), np.nan)


def col_strings(frame: pd.DataFrame, name: str, default: str = "") -> np.ndarray:
    if name in frame:
        return frame[name].fillna(default).map(text_id).to_numpy()
    return np.repeat(default, len(frame))


def activity_type(task_id: int) -> str:
    return "reading" if task_id in READING else "writing" if task_id in WRITING else "map" if task_id in MAP else "unknown"


def condition(task_id: int) -> str:
    return "sitting" if task_id % 2 else "walking"


def available_files(folder: Path) -> dict[str, Path]:
    found = {}
    for p in folder.iterdir():
        if p.is_file() and p.name.lower() in FILE_KEYS:
            key = FILE_KEYS[p.name.lower()]
            if key not in found or p.name.lower() == "gyroscope.csv":
                found[key] = p
    return found


def discover_users(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for p, _, names in os.walk(root):
        if not any(n.lower() in FILE_KEYS for n in names):
            continue
        folder = Path(p)
        rel = folder.relative_to(root).parts
        uid = next((part for part in rel if re.fullmatch(r"\d{6}", part)), None)
        uid = uid or (rel[0] if len(rel) > 1 else "UNKNOWN")
        result[uid].append(str(folder))
    return {k: sorted(set(v)) for k, v in sorted(result.items())}


def sensor_data(path: Path, key: str, chunk_size: int) -> dict:
    time_parts, orientation_parts = [], []
    for frame in csv_chunks(path, key, ["Systime", "Phone_orientation"], chunk_size):
        if not {"systime", "phoneorientation"}.issubset(frame.columns):
            raise ValueError(f"Missing sensor timestamps/orientation in {path}")
        t = norm_absolute(number_array(frame["systime"]))
        o = number_array(frame["phoneorientation"])
        valid = np.isfinite(t)
        time_parts.append(t[valid])
        orientation_parts.append(o[valid])
    times = np.concatenate(time_parts) if time_parts else np.empty(0)
    orientations = np.concatenate(orientation_parts) if orientation_parts else np.empty(0)
    if len(times):
        order = np.argsort(times, kind="stable")
        times, orientations = times[order], orientations[order]
    rows = len(times)
    n_portrait = int(np.count_nonzero(orientations == 0))
    n_landscape = int(np.count_nonzero(np.isin(orientations, [1, 3])))
    n_unknown = rows - n_portrait - n_landscape
    return {"times": times, "orientations": orientations,
            "rows": rows, "portrait": n_portrait, "landscape": n_landscape,
            "unknown": n_unknown}


def calibration(frame: pd.DataFrame, relative: str, reference: dict | None = None) -> dict:
    """Determine within-stream relative-time scale and absolute-time offset.

    Android MotionEvent eventTime and sensor timestamps can be on different
    boot clocks. NEVER directly subtract their relative values. Only map a
    relative touch/gesture timestamp onto epoch ms via Systime pairs, and
    independently audit mapping residuals. Source time origins are not assumed.
    """
    if frame.empty or relative not in frame or "systime" not in frame:
        return {"valid": False, "scale": 1.0, "offset": 0.0,
                "spread_ms": None, "basis": "no_relative_clock", "reference_used": False}
    abs_t = norm_absolute(col_array(frame, "systime"))
    rel_t = col_array(frame, relative)
    mask = np.isfinite(abs_t) & np.isfinite(rel_t)
    abs_t, rel_t = abs_t[mask], rel_t[mask]
    if len(abs_t) < 4:
        # A single onFling callback can still use the RAW MotionEvent time
        # calibration: test all plausible units against its epoch clock.
        if reference and reference.get("valid") and len(abs_t):
            fits = [(float(np.median(abs(abs_t - rel_t*scale - reference["offset"]))), scale)
                    for scale in (1e-6, 1e-3, 1.0, 1e3)]
            error, chosen = min(fits)
            if error <= 1000:
                return {"valid": True, "scale": chosen,
                        "offset": float(reference["offset"]),
                        "spread_ms": round(error, 2), "basis": "raw_clock_reference",
                        "reference_used": True,
                        "delta_from_raw_ms": 0.0}
        return {"valid": False, "scale": 1.0, "offset": 0.0,
                "spread_ms": None, "basis": "too_few_time_pairs", "reference_used": False}
    if len(abs_t) > 3000:
        ix = np.linspace(0, len(abs_t)-1, 3000, dtype=int)
        abs_t, rel_t = abs_t[ix], rel_t[ix]
    span = float(np.percentile(rel_t, 95) - np.percentile(rel_t, 5))
    def calc(scale: float):
        offset = abs_t - rel_t * scale
        return float(np.percentile(offset, 85) - np.percentile(offset, 15)), float(np.median(offset)), float(np.percentile(offset, 10))
    candidates = [(calc(s)[0], s) for s in (1e-6, 1e-3, 1.0, 1e3)]
    candidates.sort()
    best_spread, best_scale = candidates[0]
    # Wide boot times and near-tied candidate fits can indicate callback delays
    # rather than a meaningful relative-time scale.
    reliable = span * best_scale >= 3000 and best_spread < 2000
    _, median_offset, low_offset = calc(best_scale)
    chosen_offset = low_offset if relative == "presstime" else median_offset
    ref_used = False
    if reference and reference.get("valid") and abs(reference["scale"]-best_scale) / best_scale < 1e-8:
        # The raw MotionEvent timeline is preferable for callback-derived press times,
        # but only if independently estimated source offsets are close.
        delta = abs(reference["offset"] - chosen_offset)
        if delta <= 1000:
            chosen_offset = float(reference["offset"])
            ref_used = True
        elif delta > 30_000:
            reliable = False
    return {"valid": bool(reliable), "scale": float(best_scale),
            "offset": float(chosen_offset), "spread_ms": round(best_spread, 2),
            "basis": "relative_anchored" if reliable else "relative_clock_uncertain",
            "reference_used": ref_used, "delta_from_raw_ms": (round(float(abs(reference["offset"] - chosen_offset)), 2)
                                                      if reference and reference.get("valid") else None)}


def absolute_event_time(frame: pd.DataFrame, relative: str, cal: dict) -> np.ndarray:
    sys_t = norm_absolute(col_array(frame, "systime"))
    rel_t = col_array(frame, relative)
    if cal.get("valid"):
        t = rel_t * cal["scale"] + cal["offset"]
        t[~np.isfinite(t)] = sys_t[~np.isfinite(t)]
        return t
    return sys_t


def source_orientation(counts: Counter, values: np.ndarray, prefix: str):
    counts[f"{prefix}_rows"] += len(values)
    counts[f"{prefix}_portrait_rows"] += int(np.count_nonzero(values == 0))
    counts[f"{prefix}_landscape_rows"] += int(np.count_nonzero(np.isin(values, [1, 3])))
    counts[f"{prefix}_unknown_rows"] += int(np.count_nonzero(~np.isin(values, [0, 1, 3])))


def physical_raw(frame: pd.DataFrame, cal: dict, config: dict) -> tuple[list[dict], Counter]:
    """Reconstruct physical DOWN -> UP touch intervals; action 5/6 are multitouch."""
    counts = Counter()
    if frame.empty:
        return [], counts
    t = absolute_event_time(frame, "eventtime", cal)
    a = col_strings(frame, "activityid")
    pc, pid, action = (col_array(frame, c) for c in ("pointercount", "pointerid", "actionid"))
    xs, ys, ori = (col_array(frame, c) for c in ("x", "y", "phoneorientation"))
    source_orientation(counts, ori, "touch")
    # Use event timestamps instead of callback record order for reconstruction.
    order = np.argsort(t, kind="stable")
    result, active = [], None
    def finish(st: dict, j: int):
        end = float(t[j]); delta = end - st["start_ms"]
        if not np.isfinite(delta) or delta < 0 or delta > config["max_gesture_duration_ms"]:
            counts["raw_bad_duration"] += 1
            return
        displacement = float(np.hypot(xs[j] - st["start_x"], ys[j] - st["start_y"]))
        path = st["path_px"] + float(np.hypot(xs[j] - st["last_x"], ys[j] - st["last_y"]))
        if displacement >= config["swipe_distance_px"] or path >= config["swipe_path_px"]:
            gesture_type = "swipe"
        else:
            gesture_type = "tap" if delta <= config["max_tap_duration_ms"] else "other"
        result.append({"activity_id": st["activity_id"], "gesture_type": gesture_type,
                       "start_ms": st["start_ms"], "end_ms": end,
                       "start_x": st["start_x"], "start_y": st["start_y"],
                       "end_x": float(xs[j]), "end_y": float(ys[j]),
                       "path_px": path, "movement_px": displacement,
                       "touch_points": st["touch_points"] + 1,
                       "orientation_codes": sorted(st["orient"] | ({int(ori[j])} if np.isfinite(ori[j]) else set())),
                       "multitouch": bool(st["multi"] or action[j] == 6 or pc[j] > 1 or pid[j] > 0),
                       "recognition": "raw_heuristic", "source": "touchevent",
                       "endpoint_quality": "physical_up", "time_basis": cal["basis"],
                       "time_warning": not cal["valid"], "matched_touch": True,
                       "matched_onefinger": False, "matched_scroll": False,
                       "matched_stroke": False, "scroll_callback_rows": 0,
                       "fling": False, "onefinger_taptypes": "",
                       "onefinger_tapid": "", "confirmed_single": False,
                       "double_annotation": False})
    for j in order:
        if not np.isfinite(t[j]) or not (np.isfinite(xs[j]) and np.isfinite(ys[j]) and np.isfinite(action[j])):
            counts["touch_invalid_rows"] += 1
            continue
        code = int(action[j]); acid = str(a[j])
        orient = int(ori[j]) if np.isfinite(ori[j]) else None
        is_multi = code in (5, 6) or pc[j] != 1 or pid[j] != 0
        if code in (0, 5):
            if active is not None:
                counts["touch_incomplete_previous"] += 1
            active = {"start_ms": float(t[j]), "start_x": float(xs[j]), "start_y": float(ys[j]),
                      "last_x": float(xs[j]), "last_y": float(ys[j]), "path_px": 0.0,
                      "touch_points": 1, "activity_id": acid,
                      "orient": {orient} if orient is not None else set(), "multi": is_multi}
        elif code in (1, 2, 6) and active:
            if acid != active["activity_id"]:
                counts["touch_activity_cross"] += 1
                active = None
                continue
            if orient is not None:
                active["orient"].add(orient)
            active["multi"] |= is_multi
            if code == 2:
                active["path_px"] += float(np.hypot(xs[j] - active["last_x"], ys[j] - active["last_y"]))
                active["last_x"], active["last_y"] = float(xs[j]), float(ys[j])
                active["touch_points"] += 1
            else:
                finish(active, j)
                active = None
        else:
            counts["touch_orphan_or_other_actions"] += 1
    if active is not None:
        counts["touch_unclosed"] += 1
    return result, counts


def onefinger_contacts(frame: pd.DataFrame, cal: dict, config: dict) -> tuple[list[dict], Counter]:
    """Recover unique physical touches from callback event times, not callback order.

    Android may emit onSingleTapUp(UP) before onSingleTapConfirmed(DOWN) is
    logged. We sort by PressTime (the event's own timestamp), then deduplicate
    repeated callback MotionEvent objects. TapID=-1/-2 identifies a UI area,
    not a globally unique physical-contact identifier.
    """
    counts = Counter()
    if frame.empty:
        return [], counts
    t = absolute_event_time(frame, "presstime", cal)
    a = col_strings(frame, "activityid")
    tid = col_strings(frame, "tapid")
    ty, action = (col_array(frame, c) for c in ("taptype", "actiontype"))
    xs, ys, ori = (col_array(frame, c) for c in ("x", "y", "phoneorientation"))
    source_orientation(counts, ori, "onefinger")
    counts["onefinger_annotation_rows_2_3"] = int(np.count_nonzero(np.isin(ty, [2, 3])))
    # A callback can repeat a previous MotionEvent. Collapse by event time,
    # activity, action and position (do NOT use callback Systime).
    order = np.argsort(t, kind="stable")
    records: list[dict] = []
    for j in order:
        if not all(map(np.isfinite, (t[j], action[j], xs[j], ys[j]))):
            counts["onefinger_invalid_rows"] += 1
            continue
        curr = {"t": float(t[j]), "act": int(action[j]), "x": float(xs[j]), "y": float(ys[j]),
                "ori": int(ori[j]) if np.isfinite(ori[j]) else -1,
                "acid": str(a[j]), "tid": str(tid[j]), "taptype": int(ty[j]) if np.isfinite(ty[j]) else -1}
        if records:
            prev = records[-1]
            if (curr["acid"] == prev["acid"] and curr["act"] == prev["act"] and
                abs(curr["t"]-prev["t"]) <= config["callback_duplicate_time_ms"] and
                np.hypot(curr["x"]-prev["x"], curr["y"]-prev["y"]) <= config["callback_duplicate_position_px"]):
                prev.setdefault("taptypes", set()).add(curr["taptype"])
                prev.setdefault("orientation_codes", set()).add(curr["ori"])
                counts["onefinger_duplicate_callback_events"] += 1
                continue
        curr["taptypes"] = {curr["taptype"]}
        curr["orientation_codes"] = {curr["ori"]}
        records.append(curr)
    result, active = [], None
    for record in records:
        code = record["act"]
        if code == 0:
            if active is not None:
                counts["onefinger_unclosed"] += 1
            active = {"start_ms": record["t"], "start_x": record["x"], "start_y": record["y"],
                      "last_x": record["x"], "last_y": record["y"], "path_px": 0.0,
                      "activity_id": record["acid"], "tapid": record["tid"],
                      "touch_points": 1, "orient": set(record["orientation_codes"]),
                      "taptypes": set(record["taptypes"])}
        elif code in (1, 2):
            if active is None or record["acid"] != active["activity_id"]:
                counts["onefinger_orphan_callback_actions"] += 1
                continue
            active["orient"].update(record["orientation_codes"])
            active["taptypes"].update(record["taptypes"])
            if code == 2:
                active["path_px"] += float(np.hypot(record["x"]-active["last_x"], record["y"]-active["last_y"]))
                active["last_x"], active["last_y"] = record["x"], record["y"]
                active["touch_points"] += 1
                continue
            dt = record["t"]-active["start_ms"]
            if not (0 <= dt <= config["max_gesture_duration_ms"]):
                counts["onefinger_invalid_duration"] += 1
                active = None
                continue
            movement = float(np.hypot(record["x"]-active["start_x"], record["y"]-active["start_y"]))
            path = active["path_px"] + float(np.hypot(record["x"]-active["last_x"], record["y"]-active["last_y"]))
            taptypes = sorted(active["taptypes"])
            gesture_type = ("tap" if dt <= config["max_tap_duration_ms"] and
                            movement < config["swipe_distance_px"] and path < config["swipe_path_px"]
                            else "other")
            result.append({"activity_id": active["activity_id"], "gesture_type": gesture_type,
                           "start_ms": active["start_ms"], "end_ms": record["t"],
                           "start_x": active["start_x"], "start_y": active["start_y"],
                           "end_x": record["x"], "end_y": record["y"],
                           "path_px": path, "movement_px": movement,
                           "touch_points": active["touch_points"]+1,
                           "orientation_codes": sorted(active["orient"]),
                           "multitouch": False, "source": "onefinger",
                           "recognition": "android_tap_callback" if (0 in taptypes or 1 in taptypes) else "onefinger_physical_pair",
                           "endpoint_quality": "physical_up", "time_basis": cal["basis"],
                           "time_warning": not cal["valid"], "matched_touch": False,
                           "matched_onefinger": True, "matched_scroll": False, "matched_stroke": False,
                           "scroll_callback_rows": 0, "fling": False,
                           "onefinger_taptypes": ";".join(map(str,taptypes)),
                           "onefinger_tapid": active["tapid"],
                           "confirmed_single": 1 in taptypes,
                           "double_annotation": 2 in taptypes or 3 in taptypes})
            counts["onefinger_complete_contacts"] += 1
            active = None
        else:
            counts["onefinger_other_action"] += 1
    if active is not None:
        counts["onefinger_unclosed"] += 1
    return result, counts


def scroll_groups(frame: pd.DataFrame, cal: dict, config: dict) -> tuple[list[dict], Counter]:
    counts = Counter()
    if frame.empty:
        return [], counts
    if not cal["valid"]:
        counts["scroll_unreliable_timebase"] = len(frame)
        return [], counts
    start = absolute_event_time(frame, "begintime", cal)
    current = absolute_event_time(frame, "currenttime", cal)
    ac = col_strings(frame, "activityid")
    sx, sy, ex, ey, ori = (col_array(frame, c) for c in
                          ("startx", "starty", "currentx", "currenty", "phoneorientation"))
    ids = col_strings(frame, "scrollid")
    source_orientation(counts, ori, "scroll")
    # Same Android DOWN and ActivityID = same continuous contact; ScrollID may
    # be a callback ID rather than a contact ID, so do not key solely on it.
    groups: dict[tuple, dict] = {}
    for j in range(len(frame)):
        if not all(np.isfinite(v) for v in (start[j], current[j], sx[j], sy[j], ex[j], ey[j])):
            counts["scroll_invalid_rows"] += 1
            continue
        key = (str(ac[j]), round(float(start[j]), 1), round(float(sx[j]), 1), round(float(sy[j]), 1))
        if key not in groups:
            groups[key] = {"activity_id": str(ac[j]), "gesture_type": "swipe",
                           "start_ms": float(start[j]), "end_ms": float(current[j]),
                           "start_x": float(sx[j]), "start_y": float(sy[j]),
                           "end_x": float(ex[j]), "end_y": float(ey[j]),
                           "path_px": 0.0, "movement_px": 0.0,
                           "touch_points": 1, "orientation_codes": set(),
                           "multitouch": False, "source": "scroll",
                           "recognition": "android_onScroll", "endpoint_quality": "last_move_only",
                           "time_basis": cal["basis"], "time_warning": False,
                           "matched_touch": False, "matched_onefinger": False,
                           "matched_scroll": True, "matched_stroke": False,
                           "scroll_callback_rows": 0, "fling": False,
                           "onefinger_taptypes": "", "onefinger_tapid": "",
                           "confirmed_single": False, "double_annotation": False,
                           "scroll_ids": set(), "last_move": None}
        g = groups[key]
        g["orientation_codes"].add(int(ori[j]) if np.isfinite(ori[j]) else -1)
        g["scroll_ids"].add(str(ids[j]))
        g["scroll_callback_rows"] += 1
        if current[j] >= g["end_ms"]:
            g["end_ms"] = float(current[j])
            g["end_x"], g["end_y"] = float(ex[j]), float(ey[j])
        g["touch_points"] += 1
    result = []
    for g in groups.values():
        g["movement_px"] = float(np.hypot(g["end_x"]-g["start_x"], g["end_y"]-g["start_y"]))
        g["path_px"] = g["movement_px"]  # Approximation only; later raw contact overrides.
        g["orientation_codes"] = sorted(g["orientation_codes"])
        del g["scroll_ids"], g["last_move"]
        if g["end_ms"] < g["start_ms"] or g["end_ms"]-g["start_ms"] > config["max_gesture_duration_ms"]:
            counts["scroll_bad_duration"] += 1
            continue
        result.append(g)
    counts["scroll_unique_contacts"] = len(result)
    return result, counts


def stroke_gestures(frame: pd.DataFrame, cal: dict, config: dict) -> tuple[list[dict], Counter]:
    counts = Counter()
    if frame.empty:
        return [], counts
    if not cal["valid"]:
        counts["stroke_unreliable_timebase"] = len(frame)
        return [], counts
    # Begin_time and End_time live on the same relative MotionEvent clock.
    rel_begin = col_array(frame, "begintime")
    rel_end = col_array(frame, "endtime")
    start = rel_begin * cal["scale"] + cal["offset"]
    end = rel_end * cal["scale"] + cal["offset"]
    ac = col_strings(frame, "activityid")
    sx, sy, ex, ey, ori = (col_array(frame, c) for c in
                          ("startx", "starty", "endx", "endy", "phoneorientation"))
    source_orientation(counts, ori, "stroke")
    result = []
    for j in range(len(frame)):
        if not all(np.isfinite(v) for v in (start[j], end[j], sx[j], sy[j], ex[j], ey[j])):
            counts["stroke_invalid_rows"] += 1
            continue
        if not 0 <= end[j]-start[j] <= config["max_gesture_duration_ms"]:
            counts["stroke_bad_duration"] += 1
            continue
        movement = float(np.hypot(ex[j]-sx[j], ey[j]-sy[j]))
        result.append({"activity_id": str(ac[j]), "gesture_type": "swipe",
                       "start_ms": float(start[j]), "end_ms": float(end[j]),
                       "start_x": float(sx[j]), "start_y": float(sy[j]),
                       "end_x": float(ex[j]), "end_y": float(ey[j]),
                       "path_px": movement, "movement_px": movement,
                       "touch_points": 2, "orientation_codes": [int(ori[j]) if np.isfinite(ori[j]) else -1],
                       "multitouch": False, "source": "stroke",
                       "recognition": "android_onFling", "endpoint_quality": "android_up",
                       "time_basis": cal["basis"], "time_warning": False,
                       "matched_touch": False, "matched_onefinger": False,
                       "matched_scroll": False, "matched_stroke": True,
                       "scroll_callback_rows": 0, "fling": True,
                       "onefinger_taptypes": "", "onefinger_tapid": "",
                       "confirmed_single": False, "double_annotation": False})
    counts["stroke_fling_records"] = len(result)
    return result, counts


def _by_activity_start(items: list[dict]) -> dict[str, tuple[list[float], list[dict]]]:
    groups = defaultdict(list)
    for g in items:
        groups[g["activity_id"]].append(g)
    result = {}
    for acid, seq in groups.items():
        seq.sort(key=lambda g: g["start_ms"])
        result[acid] = ([g["start_ms"] for g in seq], seq)
    return result


def match_gesture(target: dict, indexed: dict, config: dict, used: set[int] | None = None,
                  check_end: bool = False) -> tuple[dict | None, bool]:
    """Nearest contact match by timestamp AND location AND activity.

    Ambiguous matches are reported instead of arbitrarily merging gestures.
    """
    from bisect import bisect_left, bisect_right
    idx = indexed.get(target["activity_id"])
    if not idx:
        return None, False
    starts, gestures = idx
    tolerance = config["match_start_ms"]
    lo = bisect_left(starts, target["start_ms"] - tolerance)
    hi = bisect_right(starts, target["start_ms"] + tolerance)
    ranked = []
    for g in gestures[lo:hi]:
        if used is not None and id(g) in used:
            continue
        xy = float(np.hypot(g["start_x"] - target["start_x"],
                            g["start_y"] - target["start_y"]))
        if xy > config["match_position_px"]:
            continue
        dt = abs(g["start_ms"] - target["start_ms"])
        de = abs(g["end_ms"] - target["end_ms"])
        if check_end and de > config["match_end_ms"]:
            continue
        score = dt / max(tolerance, 1) + xy / max(config["match_position_px"], 1)
        if check_end:
            score += de / max(config["match_end_ms"], 1)
        ranked.append((score, g))
    if not ranked:
        return None, False
    ranked.sort(key=lambda v: v[0])
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.20:
        return None, True
    return ranked[0][1], False


def reconcile(raw: list[dict], onefinger: list[dict], scroll: list[dict], stroke: list[dict],
              config: dict) -> tuple[list[dict], Counter]:
    """Build 1 physical contact => 1 logical gesture, merging Android callbacks.

    1) Group repeated onScroll callback rows into a single scroll candidate;
    2) Attach onFling to that scroll, not another gesture;
    3) Attach a raw MotionEvent DOWN/UP (true end time) where available;
    4) Merge onefinger tap callbacks with raw taps and suppress tap-on-swipe;
    5) Keep independently observed contacts if their match is trustworthy.
    """
    counts = Counter()
    result = [dict(g) for g in scroll]
    used_strokes: set[int] = set()
    stroke_idx = _by_activity_start(stroke)
    for g in result:
        matched, ambiguous = match_gesture(g, stroke_idx, config, used_strokes)
        if ambiguous:
            counts["ambiguous_scroll_stroke"] += 1
            g["time_warning"] = True
        if matched is not None:
            used_strokes.add(id(matched))
            g["source"] = "scroll+stroke"
            g["matched_stroke"] = True
            g["fling"] = True
            g["endpoint_quality"] = "android_up"
            g["end_ms"], g["end_x"], g["end_y"] = (matched["end_ms"], matched["end_x"], matched["end_y"])
            g["orientation_codes"] = sorted(set(g["orientation_codes"]) | set(matched["orientation_codes"]))
            counts["scroll_stroke_same_gesture"] += 1
    for g in stroke:
        if id(g) not in used_strokes:
            # Avoid duplicate strokes from multiple callback rows at same endpoints.
            match, _ = match_gesture(g, _by_activity_start(result), config, check_end=True)
            if match is not None:
                counts["stroke_duplicate_records"] += 1
            else:
                result.append(dict(g))
                counts["stroke_without_scroll"] += 1

    raw_index = _by_activity_start(raw)
    used_raw: set[int] = set()
    for g in result:
        matched, ambiguous = match_gesture(g, raw_index, config, used_raw)
        if ambiguous:
            counts["ambiguous_derived_raw"] += 1
            g["time_warning"] = True
        if matched is None:
            continue
        used_raw.add(id(matched))
        g["matched_touch"] = True
        g["time_warning"] |= matched["time_warning"]
        g["source"] += "+touch"
        g["orientation_codes"] = sorted(set(g["orientation_codes"]) | set(matched["orientation_codes"]))
        g["multitouch"] |= matched["multitouch"]
        g["touch_points"] = matched["touch_points"]
        g["path_px"] = matched["path_px"]
        g["movement_px"] = matched["movement_px"]
        g["start_ms"], g["start_x"], g["start_y"] = (matched["start_ms"], matched["start_x"], matched["start_y"])
        if (g["endpoint_quality"] == "last_move_only" or
            abs(g["end_ms"] - matched["end_ms"]) <= config["match_end_ms"]):
            g["end_ms"], g["end_x"], g["end_y"] = (matched["end_ms"], matched["end_x"], matched["end_y"])
            g["endpoint_quality"] = "physical_up"
        else:
            g["time_warning"] = True
            counts["raw_vs_fling_end_disagreement"] += 1
        counts["derived_swipe_matched_raw"] += 1

    # Keep unclaimed raw swipes/taps; raw-only swipes are clearly labeled as
    # heuristic, not falsely attributed to GestureDetector.onScroll().
    for g in raw:
        if id(g) not in used_raw:
            result.append(dict(g))
            counts[f"raw_initially_unmatched_{g['gesture_type']}"] += 1

    one_idx = _by_activity_start(onefinger)
    used_one: set[int] = set()
    result_index = _by_activity_start(result)
    for g in onefinger:
        matched, ambiguous = match_gesture(g, result_index, config, check_end=False)
        if ambiguous:
            counts["ambiguous_onefinger_match"] += 1
            g = dict(g)
            g["time_warning"] = True
            result.append(g)
            continue
        if matched is None:
            result.append(dict(g))
            counts["onefinger_without_raw"] += 1
            continue
        used_one.add(id(g))
        if matched["gesture_type"] == "swipe":
            counts["onefinger_callback_belongs_to_swipe"] += 1
            matched["matched_onefinger"] = True
            matched["orientation_codes"] = sorted(set(matched["orientation_codes"]) | set(g["orientation_codes"]))
            continue
        if matched["gesture_type"] == "other" and g["gesture_type"] == "tap":
            # Raw event lasted too long to be a tap; do not overwrite it based
            # solely on a callback. Keep evidence for audit.
            counts["onefinger_raw_type_conflict"] += 1
            matched["time_warning"] = True
            continue
        matched["matched_onefinger"] = True
        matched["source"] += "+onefinger"
        matched["orientation_codes"] = sorted(set(matched["orientation_codes"]) | set(g["orientation_codes"]))
        matched["double_annotation"] |= g["double_annotation"]
        matched["confirmed_single"] |= g["confirmed_single"]
        matched["onefinger_taptypes"] = g["onefinger_taptypes"]
        matched["onefinger_tapid"] = g["onefinger_tapid"]
        matched["recognition"] = "android_tap_callback"
        counts["onefinger_matched_raw_tap"] += 1

    result.sort(key=lambda g: (g["start_ms"], g["end_ms"], g["activity_id"]))
    # Exact duplicate physical taps can appear after callback dedup; collapse
    # even if the last two touch sources differ. Never merge merely because
    # two taps occur within 300 ms on the same keyboard key.
    collapsed = []
    for g in result:
        prev = collapsed[-1] if collapsed else None
        if (prev and prev["gesture_type"] == g["gesture_type"] and
            prev["activity_id"] == g["activity_id"] and
            abs(prev["start_ms"]-g["start_ms"]) <= config["callback_duplicate_time_ms"] and
            abs(prev["end_ms"]-g["end_ms"]) <= config["callback_duplicate_time_ms"] and
            np.hypot(prev["start_x"]-g["start_x"], prev["start_y"]-g["start_y"]) <=
                 config["callback_duplicate_position_px"]):
            # Prefer full DOWN/UP over callback-only and retain source evidence.
            if prev["endpoint_quality"] != "physical_up" and g["endpoint_quality"] == "physical_up":
                collapsed[-1] = g
                prev, g = g, prev
            prev["double_annotation"] |= g["double_annotation"]
            prev["confirmed_single"] |= g["confirmed_single"]
            prev["matched_onefinger"] |= g["matched_onefinger"]
            prev["matched_touch"] |= g["matched_touch"]
            prev["matched_scroll"] |= g["matched_scroll"]
            prev["matched_stroke"] |= g["matched_stroke"]
            prev["fling"] |= g["fling"]
            counts["collapsed_exact_contact_duplicates"] += 1
            continue
        collapsed.append(g)
    counts["gestures_after_reconciliation"] = len(collapsed)
    return collapsed, counts


def annotate_doubles(gestures: list[dict], config: dict) -> Counter:
    """Exclude only double-tap pairs with coherent callback evidence.

    Fast consecutive keyboard presses are NOT automatically double taps.
    Tap_type=2/3 alone does not imply a physical double gesture; Android
    callbacks can describe first/second actions or repeat the same event.
    """
    counts = Counter()
    previous: dict[str, dict] = {}
    for g in gestures:
        g["double_tap_pair"] = False
        g["double_tap_suspect"] = bool(g["double_annotation"] and not g["confirmed_single"])
        if g["gesture_type"] != "tap":
            continue
        p = previous.get(g["activity_id"])
        if p and not (p["confirmed_single"] or g["confirmed_single"]):
            dt = g["start_ms"] - p["end_ms"]
            distance = float(np.hypot(g["start_x"]-p["start_x"], g["start_y"]-p["start_y"]))
            if (0 <= dt <= config["double_tap_match_ms"] and
                distance <= config["double_tap_match_px"] and
                p["double_annotation"] and g["double_annotation"]):
                p["double_tap_pair"] = True
                g["double_tap_pair"] = True
                counts["double_tap_pairs_detected"] += 1
        previous[g["activity_id"]] = g
    counts["double_annotation_ambiguous_contacts"] = sum(
        g["double_tap_suspect"] and not g["double_tap_pair"] for g in gestures)
    return counts


def pinch_intervals(frame: pd.DataFrame, cal: dict) -> list[tuple[str, float, float]]:
    """Extract ScaleGestureDetector-style pinch bounds when time is reliable."""
    if frame.empty or not cal.get("valid"):
        return []
    times = absolute_event_time(frame, "presstime", cal)
    acid = col_strings(frame, "activityid")
    pinch_id = col_strings(frame, "pinchid")
    groups = defaultdict(list)
    for j, t in enumerate(times):
        if np.isfinite(t):
            groups[(acid[j], pinch_id[j])].append(float(t))
    return [(act, min(values), max(values)) for (act, _), values in groups.items()
            if len(values) >= 2 and max(values) >= min(values)]


def mark_pinch_overlaps(gestures: list[dict], pinches: list[tuple[str, float, float]]) -> int:
    by_activity = defaultdict(list)
    for act, start, end in pinches:
        by_activity[act].append((start, end))
    overlaps = 0
    for g in gestures:
        g["pinch_overlap"] = any(max(g["start_ms"], st) <= min(g["end_ms"], en)
                                 for st, en in by_activity.get(g["activity_id"], []))
        if g["pinch_overlap"]:
            overlaps += 1
            g["multitouch"] = True
    return overlaps


def coverage(gestures: list[dict], sensors: dict, config: dict):
    if not gestures:
        return
    st = np.array([g["start_ms"] for g in gestures], dtype=float)
    en = np.array([g["end_ms"] for g in gestures], dtype=float)
    for name in ("accelerometer", "gyroscope"):
        d = sensors[name]
        ts, orientation = d["times"], d["orientations"]
        lo = np.searchsorted(ts, st, side="left")
        hi = np.searchsorted(ts, en, side="right")
        for i, g in enumerate(gestures):
            t = ts[lo[i]:hi[i]]
            o = orientation[lo[i]:hi[i]]
            known = np.isin(o, [0, 1, 3])
            g[f"{name}_samples"] = int(len(t))
            g[f"{name}_max_gap_ms"] = float(np.diff(t).max()) if len(t) >= 2 else None
            g[f"{name}_portrait_fraction"] = float(np.mean(o[known] == 0)) if np.any(known) else None
    for g in gestures:
        reasons = []
        ori = set(g["orientation_codes"])
        g["portrait_only"] = bool(ori) and ori == {0}
        if g["gesture_type"] not in ("tap", "swipe"):
            reasons.append("unsupported_or_long_press")
        if g["multitouch"]:
            reasons.append("multitouch")
        if not g["portrait_only"]:
            reasons.append("touch_not_fully_portrait")
        if g["double_tap_pair"]:
            reasons.append("confirmed_double_tap_pair")
        if g["time_warning"]:
            reasons.append("untrusted_cross_file_time")
        if g["endpoint_quality"] == "last_move_only" and not config["allow_open_scroll"]:
            reasons.append("scroll_missing_up_timestamp")
        duration = g["end_ms"]-g["start_ms"]
        g["duration_ms"] = round(duration, 3)
        if duration < 0 or duration > config["max_gesture_duration_ms"]:
            reasons.append("invalid_duration")
        for name in ("accelerometer", "gyroscope"):
            if g[f"{name}_samples"] < config["min_sensor_samples"]:
                reasons.append(f"too_few_{name}_samples")
            if g[f"{name}_portrait_fraction"] != 1.0:
                reasons.append(f"{name}_not_fully_portrait")
            gap = g[f"{name}_max_gap_ms"]
            if gap is not None and gap > config["max_sensor_gap_ms"]:
                reasons.append(f"{name}_large_sample_gap")
        g["rejection_reason"] = ";".join(reasons)
        g["motion_covered"] = not reasons


def keypress_counts(frame: pd.DataFrame) -> Counter:
    counts = Counter()
    if frame.empty:
        return counts
    counts["keypress_rows"] = len(frame)
    pt = col_array(frame, "presstype")
    counts["keypress_down_rows"] = int(np.count_nonzero(pt == 0))
    counts["keypress_up_rows"] = int(np.count_nonzero(pt == 1))
    return counts


def orient_summary(count: Counter, prefix: str) -> dict[str, float | None]:
    pr, ls, un = (count[f"{prefix}_{x}_rows"] for x in ("portrait", "landscape", "unknown"))
    return {
        f"{prefix}_portrait_fraction": round(pr / (pr+ls), 6) if pr+ls else None,
        f"{prefix}_landscape_fraction": round(ls / (pr+ls), 6) if pr+ls else None,
        f"{prefix}_unknown_fraction": round(un / (pr+ls+un), 6) if pr+ls+un else None,
    }


def audit_session(uid: str, folder: Path, files: dict[str, Path], config: dict):
    row = {"user_id": uid, "session_dir": str(folder), "session_id": folder.name,
           "task": "unknown", "condition": "unknown", "status": "excluded",
           "reason": "", "activity_rows": 0}
    warnings = []
    calibrations = []
    if "activity" not in files:
        row["reason"] = "missing_Activity.csv"
        return row, [], calibrations, warnings
    activities = load_table(files["activity"], "activity", SCHEMAS["activity"], 10000)
    if activities.empty or "taskid" not in activities:
        row["reason"] = "empty_or_invalid_Activity.csv"
        return row, [], calibrations, warnings
    row["activity_rows"] = len(activities)
    if "sessionnumber" in activities:
        session_nums = {int(v) for v in col_array(activities, "sessionnumber") if np.isfinite(v)}
        if len(session_nums) == 1:
            row["session_id"] = str(next(iter(session_nums)))
        elif session_nums:
            row["reason"] = "multiple_sessions_in_one_folder"
            return row, [], calibrations, warnings
    task_ids = {int(v) for v in col_array(activities, "taskid") if np.isfinite(v)}
    tasks = {activity_type(x) for x in task_ids}
    conds = {condition(x) for x in task_ids}
    row["task_ids"] = ";".join(map(str, sorted(task_ids)))
    row["task"] = next(iter(tasks)) if len(tasks) == 1 else "mixed"
    row["condition"] = next(iter(conds)) if len(conds) == 1 else "mixed"
    if row["task"] == "map":
        row["reason"] = "map_out_of_scope"
        return row, [], calibrations, warnings
    if row["task"] not in ("reading", "writing") or row["condition"] == "mixed":
        row["reason"] = "unknown_or_mixed_task"
        return row, [], calibrations, warnings
    missing = [x for x in ("accelerometer", "gyroscope") if x not in files]
    if all(x not in files for x in ("touchevent", "onefinger", "scroll", "stroke")):
        missing.append("no_touch_or_gesture_file")
    if missing:
        row["reason"] = "missing_required_files:" + ",".join(missing)
        return row, [], calibrations, warnings

    sensors = {k: sensor_data(files[k], k, config["chunk_size"])
               for k in ("accelerometer", "gyroscope")}
    row["accel_rows"] = sensors["accelerometer"]["rows"]
    row["gyro_rows"] = sensors["gyroscope"]["rows"]
    for short, key in (("accel", "accelerometer"), ("gyro", "gyroscope")):
        sensor = sensors[key]
        known = sensor["portrait"] + sensor["landscape"]
        row[f"{short}_portrait_fraction"] = round(sensor["portrait"] / known, 6) if known else None
        row[f"{short}_landscape_fraction"] = round(sensor["landscape"] / known, 6) if known else None
        row[f"{short}_unknown_fraction"] = round(sensor["unknown"] / sensor["rows"], 6) if sensor["rows"] else None

    columns = {"touchevent": SCHEMAS["touchevent"], "onefinger": SCHEMAS["onefinger"],
               "scroll": SCHEMAS["scroll"], "stroke": SCHEMAS["stroke"],
               "keypress": SCHEMAS["keypress"], "pinch": SCHEMAS["pinch"]}
    tables = {k: load_table(files.get(k), k, cols, config["chunk_size"])
              for k, cols in columns.items()}
    raw_cal = calibration(tables["touchevent"], "eventtime")
    one_cal = calibration(tables["onefinger"], "presstime", raw_cal)
    scroll_cal = calibration(tables["scroll"], "currenttime", raw_cal)
    stroke_cal = calibration(tables["stroke"], "endtime", raw_cal)
    pinch_cal = calibration(tables["pinch"], "presstime", raw_cal)
    for source, c in (("touch", raw_cal), ("onefinger", one_cal),
                      ("scroll", scroll_cal), ("stroke", stroke_cal), ("pinch", pinch_cal)):
        row[f"{source}_time_basis"] = c["basis"]
        row[f"{source}_time_scale"] = c["scale"]
        row[f"{source}_time_spread_ms"] = c["spread_ms"]
        row[f"{source}_time_delta_from_touch_ms"] = c.get("delta_from_raw_ms")
        calibrations.append({"user_id": uid, "session_id": row["session_id"], "task": row["task"],
                             "source": source, **c})
        if not c["valid"] and not tables[({"touch":"touchevent"}.get(source,source))].empty:
            warnings.append(f"{uid}/{row['session_id']}: {source} relative clock uncertain (spread={c['spread_ms']}); check time_calibration.csv")

    raw, raw_counts = physical_raw(tables["touchevent"], raw_cal, config)
    onefinger, one_counts = onefinger_contacts(tables["onefinger"], one_cal, config)
    scroll, scroll_counts = scroll_groups(tables["scroll"], scroll_cal, config)
    stroke, stroke_counts = stroke_gestures(tables["stroke"], stroke_cal, config)
    gestures, recon_counts = reconcile(raw, onefinger, scroll, stroke, config)
    double_counts = annotate_doubles(gestures, config)
    pinch_bounds = pinch_intervals(tables["pinch"], pinch_cal)
    pinch_overlaps = mark_pinch_overlaps(gestures, pinch_bounds)
    coverage(gestures, sensors, config)
    key_counts = keypress_counts(tables["keypress"])

    row["touch_rows"] = len(tables["touchevent"])
    row["onefinger_rows"] = len(tables["onefinger"])
    row["scrollevent_rows"] = len(tables["scroll"])
    row["strokeevent_rows"] = len(tables["stroke"])
    row["pinchevent_rows"] = len(tables["pinch"])
    row["pinch_overlap_contacts"] = pinch_overlaps
    row["touch_raw_contact_candidates"] = len(raw)
    row["onefinger_complete_contacts"] = len(onefinger)
    row["scroll_unique_contacts"] = len(scroll)
    row["stroke_fling_records"] = len(stroke)
    row.update({k: int(v) for d in (raw_counts, one_counts, scroll_counts,
                                       stroke_counts, recon_counts, double_counts, key_counts)
                for k, v in d.items()})
    row.update(orient_summary(raw_counts, "touch"))
    row.update(orient_summary(one_counts, "onefinger"))
    row.update(orient_summary(scroll_counts, "scroll"))
    row.update(orient_summary(stroke_counts, "stroke"))
    # Orientation for primary touch data is based on DISTINCT reconstructed
    # gestures, not repeated GestureDetector callbacks (which would overweight
    # scrolls and delayed confirmations). Empty => fallback to raw rows.
    candidate_orient = [g for g in gestures if g["gesture_type"] in ("tap", "swipe")]
    known_g = [g for g in candidate_orient if set(g["orientation_codes"]) <= {0, 1, 3}]
    row["touch_portrait_fraction"] = (round(sum(set(g["orientation_codes"]) == {0} for g in known_g) / len(known_g), 6)
                                       if known_g else (row.get("touch_portrait_fraction") or
                                                      row.get("onefinger_portrait_fraction")))
    row["touch_unknown_fraction"] = (round(sum(not set(g["orientation_codes"]) <= {0, 1, 3} for g in candidate_orient)/len(candidate_orient), 6)
                                      if candidate_orient else None)
    row["touch_gesture_candidates"] = len(candidate_orient)
    row["raw_taps"] = sum(g["gesture_type"] == "tap" for g in gestures)
    row["raw_swipes"] = sum(g["gesture_type"] == "swipe" for g in gestures)
    row["covered_taps"] = sum(g["gesture_type"] == "tap" and g["motion_covered"] for g in gestures)
    row["covered_swipes"] = sum(g["gesture_type"] == "swipe" and g["motion_covered"] for g in gestures)
    row["covered_total"] = row["covered_taps"] + row["covered_swipes"]
    row["coverage_rate"] = round(row["covered_total"] / len(candidate_orient), 6) if candidate_orient else 0.0
    row["covered_raw_heuristic"] = sum(g["motion_covered"] and g["recognition"] == "raw_heuristic" for g in gestures)
    row["covered_android_recognized"] = sum(g["motion_covered"] and g["recognition"] != "raw_heuristic" for g in gestures)
    row["uncertain_time_gestures"] = sum(g["time_warning"] for g in gestures)
    row["incomplete_scroll_endpoints"] = sum(g["endpoint_quality"] == "last_move_only" for g in gestures)
    row["double_tap_pair_contacts"] = sum(g["double_tap_pair"] for g in gestures)
    row["gesture_rejection_categories"] = ";".join(f"{k}:{v}" for k,v in
        sorted(Counter(reason for g in gestures for reason in g["rejection_reason"].split(";") if reason).items()))

    for short in ("accel", "gyro"):
        val = row.get(f"{short}_portrait_fraction")
        unknown = row.get(f"{short}_unknown_fraction")
        if val is None:
            row["reason"] = f"{short}_orientation_unknown"
            break
        if val < config["min_portrait_fraction"]:
            row["reason"] = f"{short}_mostly_landscape"
            break
        if unknown is not None and unknown > config["max_unknown_orientation_fraction"]:
            row["reason"] = f"{short}_orientation_unknown_excess"
            break
    if not row["reason"]:
        tpr = row.get("touch_portrait_fraction")
        tun = row.get("touch_unknown_fraction")
        if tpr is None:
            row["reason"] = "touch_orientation_unknown"
        elif tpr < config["min_portrait_fraction"]:
            row["reason"] = "touch_mostly_landscape"
        elif tun is not None and tun > config["max_unknown_orientation_fraction"]:
            row["reason"] = "touch_orientation_unknown_excess"
    if not row["reason"] and row["covered_total"] < config["min_covered_gestures"]:
        row["reason"] = "too_few_motion_covered_gestures"
    if not row["reason"] and row["task"] == "reading" and row["covered_swipes"] < config["min_reading_swipes"]:
        row["reason"] = "too_few_reading_swipes"
    if not row["reason"] and row["task"] == "writing" and row["covered_taps"] < config["min_writing_taps"]:
        row["reason"] = "too_few_writing_taps"
    if not row["reason"]:
        row["status"] = "eligible"
    for i, g in enumerate(gestures):
        g["gesture_id"] = f"{uid}:{row['session_id']}:{i:06d}"
        g.update({"user_id": uid, "session_id": row["session_id"],
                  "task": row["task"], "condition": row["condition"]})
        g.pop("orientation_codes", None)
    if row["reason"]:
        warnings.append(f"{uid}/{row['session_id']} ({row['task']}): {row['reason']}; covered={row['covered_total']}; "
                        f"raw={len(raw)} onefinger={len(onefinger)} scroll={len(scroll)} stroke={len(stroke)}")
    return row, gestures, calibrations, warnings


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def atomic_json(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                              default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gzip_csv(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GESTURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def digest_manifest(files: dict[str, Path], config: dict) -> str:
    stamps = []
    for name, path in sorted(files.items()):
        st = path.stat()
        stamps.append((name, str(path.resolve()), st.st_size, st.st_mtime_ns))
    s = json.dumps({"version": CACHE_VERSION, "config": config, "files": stamps},
                   ensure_ascii=False, sort_keys=True, default=json_default)
    return hashlib.sha256(s.encode()).hexdigest()


def audit_user(job: tuple):
    uid, paths, out_str, cfg = job
    out = Path(out_str)
    session_dirs = [Path(x) for x in paths]
    manifests = {}
    for i, folder in enumerate(session_dirs):
        for key, p in available_files(folder).items():
            manifests[f"{i}/{key}"] = p
    manifest = digest_manifest(manifests, cfg)
    cache = out / "cache" / "v3" / "users" / f"{uid}.json"
    cache_gestures = out / "cache" / "v3" / "gestures" / f"{uid}.csv.gz"
    if not cfg["no_cache"] and cache.is_file() and cache_gestures.is_file():
        try:
            item = json.loads(cache.read_text(encoding="utf-8"))
            if item.get("manifest") == manifest:
                return {"cached": True, "data": item}
        except (OSError, json.JSONDecodeError):
            pass
    sessions, gest, cal, messages = [], [], [], []
    for folder in session_dirs:
        try:
            s, g, time_rows, w = audit_session(uid, folder, available_files(folder), cfg)
            sessions.append(s)
            gest.extend(g)
            cal.extend(time_rows)
            messages.extend(w)
        except Exception as ex:
            sessions.append({"user_id": uid, "session_id": folder.name,
                             "session_dir": str(folder), "task": "unknown", "status": "error",
                             "reason": f"{type(ex).__name__}: {ex}"})
            messages.append(f"{uid}/{folder.name}: {traceback.format_exc(limit=5)}")
    cnt = Counter(s["task"] for s in sessions if s["status"] == "eligible")
    data = {"user_id": uid, "manifest": manifest, "sessions": sessions,
            "calibrations": cal, "messages": messages,
            "eligible_reading": cnt["reading"], "eligible_writing": cnt["writing"]}
    gzip_csv(cache_gestures, gest)
    atomic_json(cache, data)
    return {"cached": False, "data": data}


def ids_by_task(sessions: list[dict], task: str, threshold: int) -> dict:
    matched = [s for s in sessions if s.get("task") == task and s.get("status") == "eligible"]
    sit = sum(s.get("condition") == "sitting" for s in matched)
    walk = sum(s.get("condition") == "walking" for s in matched)
    return {"count": len(matched), "sitting": sit, "walking": walk,
            "ready": len(matched) >= threshold,
            "both_conditions_ready": sit >= threshold and walk >= threshold,
            "at_least_four": len(matched) >= 4,
            "at_least_six": len(matched) >= 6}


def write_ids(path: Path, ids: list[str]):
    path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def summary_report(sessions: list[dict], results: list[dict], out: Path,
                   cfg: dict, failed: list[str]) -> dict:
    sessions.sort(key=lambda s: (str(s.get("user_id", "")), int(s["session_id"]) if
                   str(s.get("session_id", "")).isdigit() else 9999))
    write_csv(out / "sessions.csv", sessions)
    eligible = [s for s in sessions if s.get("status") == "eligible"]
    write_csv(out / "eligible_sessions.csv", eligible)
    write_csv(out / "excluded_sessions.csv", [s for s in sessions if s.get("status") != "eligible"])
    calibrations = [c for r in results for c in r.get("calibrations", [])]
    write_csv(out / "time_calibration.csv", calibrations)
    groups = defaultdict(list)
    for s in sessions:
        groups[s["user_id"]].append(s)
    users = []
    for uid, items in sorted(groups.items()):
        r = {"user_id": uid}
        for task in ("reading", "writing"):
            stat = ids_by_task(items, task, cfg["min_sessions_per_task"])
            r.update({f"{task}_{k}": v for k, v in stat.items()})
        users.append(r)
    write_csv(out / "users.csv", users)
    lists = {
        "users_good_reading.txt": [r["user_id"] for r in users if r["reading_count"] >= 1],
        "users_good_writing.txt": [r["user_id"] for r in users if r["writing_count"] >= 1],
        "users_ready_reading.txt": [r["user_id"] for r in users if r["reading_ready"]],
        "users_ready_writing.txt": [r["user_id"] for r in users if r["writing_ready"]],
        "users_ready_both.txt": [r["user_id"] for r in users if r["writing_ready"] and r["reading_ready"]],
        "users_ready_reading_sitting_walking.txt": [r["user_id"] for r in users if r["reading_both_conditions_ready"]],
        "users_ready_writing_sitting_walking.txt": [r["user_id"] for r in users if r["writing_both_conditions_ready"]],
    }
    for name, ids in lists.items():
        write_ids(out / name, ids)
    tasks = {}
    for task in ("reading", "writing"):
        included = [s for s in eligible if s.get("task") == task]
        tasks[task] = {
            "sessions_total": sum(s.get("task") == task for s in sessions),
            "sessions_eligible": len(included),
            "covered_gestures": sum(int(s.get("covered_total", 0)) for s in included),
            "covered_taps": sum(int(s.get("covered_taps", 0)) for s in included),
            "covered_swipes": sum(int(s.get("covered_swipes", 0)) for s in included),
        }
    keys = ("onefinger_annotation_rows_2_3", "onefinger_complete_contacts",
            "onefinger_duplicate_callback_events", "onefinger_matched_raw_tap",
            "onefinger_without_raw", "onefinger_callback_belongs_to_swipe",
            "scroll_unique_contacts", "stroke_fling_records", "scroll_stroke_same_gesture",
            "derived_swipe_matched_raw", "raw_initially_unmatched_swipe", "raw_initially_unmatched_tap",
            "collapsed_exact_contact_duplicates", "ambiguous_derived_raw",
            "ambiguous_onefinger_match", "double_tap_pairs_detected",
            "double_annotation_ambiguous_contacts", "keypress_rows", "keypress_down_rows",
            "covered_raw_heuristic", "covered_android_recognized", "uncertain_time_gestures",
            "incomplete_scroll_endpoints", "pinch_overlap_contacts")
    diagnosis = {key: sum(int(s.get(key, 0) or 0) for s in sessions) for key in keys}
    reasons = Counter(s.get("reason", "") for s in sessions if s.get("status") != "eligible")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "cache_version": CACHE_VERSION,
        "criteria": {k: v for k, v in cfg.items() if k != "no_cache"},
        "users_total": len(groups), "sessions_total": len(sessions),
        "sessions_by_status": dict(Counter(s.get("status", "unknown") for s in sessions)),
        "excluded_reasons": dict(reasons), "tasks": tasks,
        "users_good_reading": len(lists["users_good_reading.txt"]),
        "users_good_writing": len(lists["users_good_writing.txt"]),
        "users_ready_reading": len(lists["users_ready_reading.txt"]),
        "users_ready_writing": len(lists["users_ready_writing.txt"]),
        "users_ready_both": len(lists["users_ready_both.txt"]),
        "gesture_diagnostics": diagnosis,
        "time_calibration_diagnostics": {
            "sources_unreliable": dict(Counter(c["source"] for c in calibrations if not c.get("valid") and c.get("basis") != "no_relative_clock")),
            "total_calibration_rows": len(calibrations)},
        "failed_user_ids": failed,
    }
    atomic_json(out / "summary.json", result)
    guidance = """# HMOG Android gesture audit — how to interpret v3

This is an exploratory quality audit, not a published HMOG replication.

- `sessions.csv`: one row/session with all quality decisions and reasons.
- `users.csv`: number of eligible reading/writing sessions per user, separately.
- `summary.json`: totals and diagnostic counters. Send to ChatGPT with `sessions.csv`.
- `time_calibration.csv`: each stream's relative-to-absolute timestamp fit;
  `valid=false` indicates cross-source alignment could not be established.
- `cache/v3/gestures/*.csv.gz`: full inventory of physical contacts and
  reconstructed Android-recognized gestures, including rejected candidates.
- `cache/v3/users/*.json`: per-user cached summaries. Matching input size/mtime,
  analysis parameters and CACHE_VERSION ensures automatic cache invalidation.
- The v2 cache is NOT reused. No manual cache deletion is needed.

Semantics:
- A single physical swipe may generate repeated `onScroll` callback rows and
  one `onFling`/`StrokeEvent` record. They are reconciled, not counted twice.
- `onSingleTapConfirmed` can arrive later than its original DOWN event. For
  OneFinger, `PressTime` is calibrated to the original event timeline, when
  the calibration is reliable. `Systime` is a fallback only and marked uncertain.
- `onefinger_annotation_rows_2_3` is a number of callback ROWS, not a number
  of double taps. Confirmed double tap pairs require coherent timing, positions,
  and annotations for both contacts. Other annotated pairs remain flagged.
- For gesture-level sensor coverage, BOTH accelerometer and gyroscope must
  have samples inside [start_ms, end_ms]. No inter-gesture motion is used.
- PinchEvent intervals additionally reject overlapping contacts as multitouch.
- A scroll without a physical UP or onFling end is `last_move_only` and excluded
  by default because its physical gesture end cannot be proven. Use
  `--allow-open-scroll` only for a declared sensitivity analysis.
- `raw_heuristic` denotes a swipe detected from a DOWN/MOVE/UP trace without
  positive evidence from Android's `onScroll`/`onFling` callbacks. Compare
  its accuracy separately from `android_onScroll`/`android_onFling` gestures.
- Orientation majority: session-level portrait (0) fraction must be at least
  0.50 separately for accelerometer, gyroscope and unique contact orientations;
  gesture-level samples and touch records must ALL be portrait (0).
- A user with >=2 eligible sessions is a *candidate* for chronological splitting;
  actual enrollment, validation and test independence must be designed later.

Caveats and audit checkpoints:
- Android `MotionEvent.getEventTime()` and `SensorEvent.timestamp` need NOT use
  the same boot clock. This script aligns gesture relative times to their
  recorded `Systime` pairs, then uses sensor `Systime`. The fit is empirical.
- HMOG collection application source code is not available here. Callback
  interpretations are well-supported by Android API field semantics, but exact
  application behavior needs inspection of representative original CSV rows.
- Row counts and session eligibility may change substantially from v2.
- `OneFingerTouchEvent.TapID` values -1/-2 are UI region codes, not unique IDs.
- Check sessions with `uncertain_time_gestures>0`, large mismatch counts, or
  unexpected missing KeyPressEvent rows before finalizing doctoral experiments.

Recommended next step: send `summary.json`, `sessions.csv` and
`time_calibration.csv`, plus *anonymized 15-30 original rows* from each gesture
file for 1 reading session and 1 writing session if timeline flags are widespread.
"""
    (out / "HOW_TO_READ_RESULTS.md").write_text(guidance, encoding="utf-8")
    return result


def parser():
    p = argparse.ArgumentParser(description="HMOG v3 Android-GestureDetector-aware session audit")
    p.add_argument("--input", type=Path, default=Path("input_data/hmog_unarchived"))
    p.add_argument("--output", type=Path, default=Path("output_data/hmog_analysis_v3"))
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2)//2)))
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--min-portrait-fraction", type=float, default=0.5)
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
    p.add_argument("--match-start-ms", type=float, default=125.0)
    p.add_argument("--match-end-ms", type=float, default=180.0)
    p.add_argument("--match-position-px", type=float, default=65.0)
    p.add_argument("--callback-duplicate-time-ms", type=float, default=2.0)
    p.add_argument("--callback-duplicate-position-px", type=float, default=6.0)
    p.add_argument("--allow-open-scroll", action="store_true",
                   help="Include Android onScroll traces with only the last MOVE as endpoint (sensitivity analysis only)")
    p.add_argument("--no-cache", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hmog_v3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    for handler in (logging.FileHandler(out / "analysis.log", mode="w", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if not root.is_dir():
        logger.error("Input directory missing: %s", root)
        return 2
    config = {k: v for k, v in vars(args).items() if k not in {"input", "output", "workers"}}
    if args.workers < 1 or args.min_sensor_samples < 1 or not (0 <= args.min_portrait_fraction <= 1):
        logger.error("Invalid worker count or filtering parameters")
        return 2
    users = discover_users(root)
    if not users:
        logger.error("No HMOG session directories found under %s", root)
        return 2
    logger.info("HMOG v3; input=%s; output=%s; users=%d; sessions=%d; workers=%d",
                root, out, len(users), sum(map(len, users.values())), args.workers)
    logger.info("Android gesture audit: onScroll grouped, onFling merged, OneFinger callback event-time sorted")
    logger.info("Main criteria: portrait>=%.0f%%; covered>=%d; reading swipes>=%d; writing taps>=%d",
                100*args.min_portrait_fraction, args.min_covered_gestures,
                args.min_reading_swipes, args.min_writing_taps)
    jobs = [(uid, folders, str(out), config) for uid, folders in users.items()]
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_user, j): j[0] for j in jobs}
        for step, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                output = future.result()
                data = output["data"]
                results.append(data)
                ss = data["sessions"]
                logger.info("[%3d/%3d] user=%s %s eligible=%d/%d R=%d W=%d; taps=%d swipes=%d; time_warnings=%d",
                            step, len(futures), uid, "CACHE" if output["cached"] else "DONE ",
                            sum(s.get("status") == "eligible" for s in ss), len(ss),
                            data["eligible_reading"], data["eligible_writing"],
                            sum(int(s.get("covered_taps", 0)) for s in ss),
                            sum(int(s.get("covered_swipes", 0)) for s in ss),
                            sum(int(s.get("uncertain_time_gestures", 0)) for s in ss))
                for warning in data["messages"][:3]:
                    logger.warning("%s", warning)
                if len(data["messages"]) > 3:
                    logger.warning("user=%s: %d more warning(s), see cache/users JSON", uid, len(data["messages"])-3)
            except Exception:
                logger.exception("[%3d/%3d] FAILED user=%s", step, len(futures), uid)
                failures.append(uid)
    if not results:
        return 1
    sessions = [s for r in results for s in r["sessions"]]
    report = summary_report(sessions, results, out, config, failures)
    compact = {k: report[k] for k in ("users_total", "users_good_reading", "users_good_writing",
                                     "users_ready_reading", "users_ready_writing", "users_ready_both",
                                     "tasks", "gesture_diagnostics", "time_calibration_diagnostics",
                                     "excluded_reasons", "failed_user_ids")}
    logger.info("FINISHED: eligible reading=%d/%d, writing=%d/%d; failed_users=%d",
                report["tasks"]["reading"]["sessions_eligible"],
                report["tasks"]["reading"]["sessions_total"],
                report["tasks"]["writing"]["sessions_eligible"],
                report["tasks"]["writing"]["sessions_total"], len(failures))
    print("\n===== COPY THIS BLOCK BACK TO CHAT =====")
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=json_default))
    print("===== END BLOCK =====\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
