#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit BB-MAS HandPhone CSV files, Android touch contacts and motion coverage.

Python 3.14; dependencies: numpy, pandas. Run from repository root:
    python analyze_bbmas_handphone.py --workers 6

The file format follows SU-AIS BB-MAS README (raw data section).
IMPORTANT: No ground-truth Android GestureDetector callbacks are provided by BB-MAS;
          tap/swipe/possible-double-tap labels below are transparent heuristics.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "bbmas-handphone-audit-v1.0.0-20261107"
KINDS = ("Accelerometer", "Gyroscope", "TouchEvent", "Keyboard", "Checkpoints")
FILENAME_RE = re.compile(
    r"^(?P<uid>\d+)_HandPhone_(?P<kind>Accelerometer|Gyroscope|TouchEvent|Keyboard|Checkpoints)"
    r"(?:_\((?P<model>[^)]+)\))?\.csv$", re.I,
)
FREQUENCIES = ("Accelerometer", "Gyroscope")
TIMESTAMP_NAT = np.iinfo(np.int64).min
MSEC = 1_000_000


def log_setup(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, encoding="utf-8", mode="w"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(val, decimals: int = 4):
    if val is None:
        return None
    try:
        x = float(val)
        return round(x, decimals) if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def col_lookup(columns) -> dict[str, str]:
    return {norm_col(c): c for c in columns}


def csv_table(path: Path, required: set[str], optional: set[str] = frozenset(), chunksize: int = 200_000):
    """Read only needed fields and normalize column names. Caller receives dataframe and count metadata."""
    frame_chunks = []
    invalid_rows = 0
    raw_rows = 0
    try:
        head = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Cannot read header of {path.name}: {exc}") from exc
    lookup = col_lookup(head.columns)
    if not required.issubset(lookup):
        raise ValueError(f"{path.name}: missing required columns {sorted(required - set(lookup))}; "
                         f"actual headers = {list(head.columns)}")
    needed = required | optional
    available = {n: lookup[n] for n in needed if n in lookup}
    for chunk in pd.read_csv(path, usecols=list(available.values()), chunksize=chunksize,
                             encoding="utf-8-sig", low_memory=False, on_bad_lines="warn"):
        raw_rows += len(chunk)
        chunk = chunk.rename(columns={c: n for n, c in available.items()})
        frame_chunks.append(chunk)
    columns = [n for n in needed if n in available]
    if not frame_chunks:
        return pd.DataFrame(columns=columns), {"raw_rows": 0, "invalid_rows": 0, "columns": sorted(lookup)}
    df = pd.concat(frame_chunks, ignore_index=True)
    return df, {"raw_rows": raw_rows, "invalid_rows": invalid_rows, "columns": sorted(lookup)}


def time_ns(series: pd.Series) -> np.ndarray:
    """Datetime strings -> absolute nanoseconds. UTC is a parsing convention, not an inferred source timezone."""
    s = series.astype("string").str.strip()
    try:
        parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    except TypeError:  # pandas older than 2.0
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
    out = parsed.astype("int64").to_numpy(dtype=np.int64, copy=True)
    return out


def numeric_col(df: pd.DataFrame, field: str, default=float("nan")) -> np.ndarray:
    if field not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=np.float64)


def iso_ns(n: int) -> str:
    if n == TIMESTAMP_NAT:
        return ""
    return pd.Timestamp(int(n), tz="UTC").isoformat()


def metadata_for(path: Path, root: Path, kind: str, model: str | None) -> dict:
    st = path.stat()
    return {"kind": kind, "model": model or "UNKNOWN", "path": str(path.relative_to(root)),
            "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def scan(root: Path) -> tuple[dict[str, dict[str, list]], list[dict]]:
    """Enumerate numbered user directories AND HandPhone files. Never silently mix suffix models."""
    users: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    unexpected = []
    for base, dirs, files in os.walk(root):
        basep = Path(base)
        if basep.name.isdigit() and basep != root:
            users[str(int(basep.name))]
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            match = FILENAME_RE.fullmatch(fname)
            if match:
                uid = str(int(match.group("uid")))
                kind = next(k for k in KINDS if k.lower() == match.group("kind").lower())
                model = match.group("model")
                users[uid][kind].append({"path": str(basep / fname), "model": model or "UNKNOWN"})
                parent_num = next((p.name for p in [basep, *basep.parents] if p != root and p.name.isdigit()), None)
                if parent_num and str(int(parent_num)) != uid:
                    unexpected.append({"path": str((basep / fname).relative_to(root)), "issue": "user_id_directory_mismatch", "folder_user": parent_num, "file_user": uid})
            elif "handphone" in fname.lower():
                unexpected.append({"path": str((basep / fname).relative_to(root)), "issue": "unrecognized_handphone_filename"})
    return dict(users), unexpected


def signature(uid: str, entries: dict[str, list], root: Path, cfg: dict) -> str:
    files = []
    for kind in KINDS:
        for desc in entries.get(kind, []):
            p = Path(desc["path"])
            files.append(metadata_for(p, root, kind, desc["model"]))
    packed = json.dumps({"version": VERSION, "uid": uid, "files": sorted(files, key=lambda x: x["path"]),
                         "config": cfg}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def sorted_times(df: pd.DataFrame, time_field: str) -> tuple[np.ndarray, dict]:
    raw_t = time_ns(df[time_field])
    valid = raw_t != TIMESTAMP_NAT
    out = np.sort(raw_t[valid])
    diffs = np.diff(out) / MSEC
    stats = {
        "rows": len(df), "valid_timestamps": int(valid.sum()), "invalid_timestamps": int((~valid).sum()),
        "first_utc": iso_ns(int(out[0])) if len(out) else "", "last_utc": iso_ns(int(out[-1])) if len(out) else "",
        "median_gap_ms": safe_float(np.median(diffs)) if len(diffs) else None,
        "p95_gap_ms": safe_float(np.percentile(diffs, 95)) if len(diffs) else None,
        "max_gap_ms": safe_float(np.max(diffs)) if len(diffs) else None,
        "same_timestamp_neighbor_pairs": int(np.sum(diffs == 0)),
        "negative_order_jumps_original": int(np.sum(np.diff(raw_t[valid]) < 0)) if valid.sum() > 1 else 0,
        "gaps_over_120ms": int((diffs > 120).sum()),
        "gaps_over_1000ms": int((diffs > 1000).sum()),
    }
    return out, stats


def near_sensor(times: np.ndarray, queries: np.ndarray, tol_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """For each event, nearest sensor sample <= tolerance. Return boolean and delta in ms."""
    answer = np.zeros(len(queries), dtype=bool)
    delays = np.full(len(queries), np.inf)
    if not len(times) or not len(queries):
        return answer, delays
    valid = queries != TIMESTAMP_NAT
    q = queries[valid]
    pos = np.searchsorted(times, q)
    left = np.clip(pos - 1, 0, len(times) - 1)
    right = np.clip(pos, 0, len(times) - 1)
    d = np.minimum(np.abs(q - times[left]), np.abs(q - times[right])) / MSEC
    delays[valid] = d
    answer[valid] = d <= tol_ms
    return answer, delays


def coverage(times: np.ndarray, start: int, end: int, cfg: dict) -> dict:
    """Coverage strictly using samples IN the gesture interval (not outside feature window)."""
    if not len(times) or start == TIMESTAMP_NAT or end == TIMESTAMP_NAT or end < start:
        return {"samples": 0, "first_delay_ms": None, "last_delay_ms": None,
                "max_inside_gap_ms": None, "interval_covered": False, "any_inside": False}
    left, right = int(np.searchsorted(times, start, side="left")), int(np.searchsorted(times, end, side="right"))
    n = right - left
    if not n:
        return {"samples": 0, "first_delay_ms": None, "last_delay_ms": None,
                "max_inside_gap_ms": None, "interval_covered": False, "any_inside": False}
    chunk = times[left:right]
    first_delay = float((chunk[0] - start) / MSEC)
    last_delay = float((end - chunk[-1]) / MSEC)
    max_gap = float(np.max(np.diff(chunk)) / MSEC) if n > 1 else None
    ok = (n >= cfg["min_sensor_samples"]
          and first_delay <= cfg["max_edge_gap_ms"]
          and last_delay <= cfg["max_edge_gap_ms"]
          and (max_gap is None or max_gap <= cfg["max_sensor_gap_ms"]))
    return {"samples": n, "first_delay_ms": safe_float(first_delay), "last_delay_ms": safe_float(last_delay),
            "max_inside_gap_ms": safe_float(max_gap), "interval_covered": bool(ok), "any_inside": True}


def pairs_inside(accel: np.ndarray, gyro: np.ndarray, start: int, end: int, pair_tolerance_ms: float) -> int:
    """Count accel samples with temporally nearby gyro reading inside same window."""
    a = accel[np.searchsorted(accel, start):np.searchsorted(accel, end, side="right")]
    g = gyro[np.searchsorted(gyro, start):np.searchsorted(gyro, end, side="right")]
    if not len(a) or not len(g):
        return 0
    matches, _ = near_sensor(g, a, pair_tolerance_ms)
    return int(matches.sum())


def construct_touch(df: pd.DataFrame, cfg: dict):
    """One physical contact per DOWN->UP, not one event row. Multi-pointer overlaps flagged."""
    t = time_ns(df["time"])
    action = numeric_col(df, "actiontype")
    pointer = numeric_col(df, "pointerid", 0)
    xs = numeric_col(df, "xvalue")
    ys = numeric_col(df, "yvalue")
    valid = (t != TIMESTAMP_NAT) & np.isin(action, [0, 1, 2]) & np.isfinite(pointer)
    raw_meta = {"rows": len(df), "valid_action_time_rows": int(valid.sum()),
                "unrecognized_action_rows": int((~np.isin(action, [0, 1, 2])).sum()),
                "invalid_timestamp_rows": int((t == TIMESTAMP_NAT).sum()),
                "down_rows": int((action == 0).sum()), "move_rows": int((action == 2).sum()),
                "up_rows": int((action == 1).sum()), "nonzero_pointer_rows": int((pointer > 0).sum()),
                "reversed_timestamp_pairs": int(np.sum(np.diff(t[t != TIMESTAMP_NAT]) < 0))}
    idx = np.flatnonzero(valid)
    idx = idx[np.lexsort((idx, t[idx]))]  # chronological; preserve original position on ties
    opened = {}
    finished = []
    anomalies = Counter()

    def end_contact(c, endt, status):
        c["end_ns"] = int(endt)
        c["status"] = status
        elapsed = (endt - c["start_ns"]) / MSEC
        c["duration_ms"] = safe_float(elapsed)
        c["net_distance_px"] = safe_float(math.hypot(c["last_x"] - c["start_x"], c["last_y"] - c["start_y"]))
        c["path_length_px"] = safe_float(c["path_length_px"])
        if status != "complete" or elapsed < 0 or elapsed > cfg["max_gesture_duration_ms"]:
            c["gesture_type"] = "unusable"
        elif ((c["net_distance_px"] is not None and c["net_distance_px"] >= cfg["swipe_distance_px"])
              or (c["path_length_px"] is not None and c["path_length_px"] >= cfg["swipe_path_px"])):
            c["gesture_type"] = "swipe"
        elif elapsed <= cfg["max_tap_duration_ms"]:
            c["gesture_type"] = "tap"
        else:
            c["gesture_type"] = "stationary_long_contact"
        c["multitouch"] = False
        finished.append(c)

    for ix in idx:
        action_i = int(action[ix])
        p = int(pointer[ix])
        ti = int(t[ix])
        xi = float(xs[ix])
        yi = float(ys[ix])
        if action_i == 0:
            if p in opened:
                anomalies["down_before_previous_up"] += 1
                end_contact(opened.pop(p), ti, "missing_up_before_next_down")
            opened[p] = {"start_ns": ti, "start_x": xi, "start_y": yi, "last_x": xi, "last_y": yi,
                         "path_length_px": 0.0, "pointer_id": p, "event_rows": 1, "move_rows": 0}
        elif action_i in (1, 2):
            if p not in opened:
                anomalies["orphan_up" if action_i == 1 else "orphan_move"] += 1
                continue
            c = opened[p]
            if math.isfinite(xi) and math.isfinite(yi) and math.isfinite(c["last_x"]) and math.isfinite(c["last_y"]):
                c["path_length_px"] += math.hypot(xi - c["last_x"], yi - c["last_y"])
            c["last_x"], c["last_y"] = xi, yi
            c["event_rows"] += 1
            if action_i == 2:
                c["move_rows"] += 1
            else:
                end_contact(opened.pop(p), ti, "complete")
    for p, c in opened.items():
        anomalies["missing_up_at_file_end"] += 1
        end_contact(c, c["start_ns"], "missing_up_at_file_end")

    # Overlap = multi-touch; using time intervals across DIFFERENT pointer IDs.
    by_start = sorted(range(len(finished)), key=lambda i: finished[i]["start_ns"])
    active = []
    for ix in by_start:
        cur = finished[ix]
        active = [j for j in active if finished[j]["end_ns"] > cur["start_ns"]]
        for j in active:
            if finished[j]["pointer_id"] != cur["pointer_id"]:
                finished[j]["multitouch"] = True
                cur["multitouch"] = True
        active.append(ix)
    for i, c in enumerate(finished):
        c["contact_id"] = i + 1
        c["possible_double_tap"] = False
        c["matched_keypress"] = False
        c["matched_keypress_id"] = ""
    valid_t = t[t != TIMESTAMP_NAT]
    raw_meta["first_timestamp_utc"] = iso_ns(int(valid_t.min())) if len(valid_t) else ""
    raw_meta["last_timestamp_utc"] = iso_ns(int(valid_t.max())) if len(valid_t) else ""
    return finished, raw_meta, dict(anomalies), t


def construct_keys(df: pd.DataFrame, cfg: dict):
    t = time_ns(df["time"])
    dirs = numeric_col(df, "direction")
    keynames = df["key"].fillna("").astype(str).to_numpy()
    valid = (t != TIMESTAMP_NAT) & np.isin(dirs, [0, 1])
    idx = np.flatnonzero(valid)
    idx = idx[np.lexsort((idx, t[idx]))]
    stacks = defaultdict(list)
    finished = []
    anomalies = Counter()
    for ix in idx:
        direction = int(dirs[ix])
        key = str(keynames[ix])
        if direction == 0:
            stacks[key].append(int(t[ix]))
        else:
            if stacks[key]:
                begin = stacks[key].pop()
                end = int(t[ix])
                finished.append({"keypress_id": len(finished) + 1, "key": key,
                                 "start_ns": begin, "end_ns": end,
                                 "duration_ms": safe_float((end - begin) / MSEC),
                                 "matched_contact_id": "", "matched_touch_type": "",
                                 "touch_start_delta_ms": None, "touch_end_delta_ms": None})
            else:
                anomalies["release_without_press"] += 1
    anomalies["press_without_release"] = sum(len(s) for s in stacks.values())
    finished.sort(key=lambda x: x["start_ns"])
    for i, x in enumerate(finished):
        x["keypress_id"] = i + 1
    meta = {"rows": len(df), "valid_event_rows": int(valid.sum()),
            "press_rows": int((dirs == 0).sum()), "release_rows": int((dirs == 1).sum()),
            "unknown_direction_rows": int((~np.isin(dirs, [0, 1])).sum()),
            "invalid_timestamp_rows": int((t == TIMESTAMP_NAT).sum()),
            "unique_key_names": len(set(keynames)),
            "reversed_timestamp_pairs": int(np.sum(np.diff(t[t != TIMESTAMP_NAT]) < 0))}
    valid_t = t[t != TIMESTAMP_NAT]
    meta["first_timestamp_utc"] = iso_ns(int(valid_t.min())) if len(valid_t) else ""
    meta["last_timestamp_utc"] = iso_ns(int(valid_t.max())) if len(valid_t) else ""
    return finished, meta, dict(anomalies), t, dirs


def link_keystrokes(contacts: list[dict], keys: list[dict], tol_ms: float) -> dict:
    """Timestamp-only, one-to-one linkage from keydown to complete physical touch contact.

    A raw keyboard contact can be classified as a swipe by movement thresholds,
    so do not restrict this matching step to taps. Report touch type separately.
    """
    valid_ids = [i for i, c in enumerate(contacts)
                 if c["status"] == "complete" and not c["multitouch"]]
    valid_ids.sort(key=lambda i: contacts[i]["start_ns"])
    starts = np.array([contacts[i]["start_ns"] for i in valid_ids], dtype=np.int64)
    used = set()
    matched = 0
    types = Counter()
    matching_deltas = []
    for k in keys:
        if not len(starts):
            break
        p = int(np.searchsorted(starts, k["start_ns"]))
        cand = list(range(max(0, p - 6), min(len(starts), p + 7)))
        cand = [j for j in cand if j not in used and abs(starts[j] - k["start_ns"]) <= tol_ms * MSEC]
        if not cand:
            continue
        ci = min(cand, key=lambda j: (abs(starts[j] - k["start_ns"]),
                                     0 if contacts[valid_ids[j]]["gesture_type"] == "tap" else 1))
        used.add(ci)
        contact = contacts[valid_ids[ci]]
        delta = (contact["start_ns"] - k["start_ns"]) / MSEC
        contact["matched_keypress"] = True
        contact["matched_keypress_id"] = k["keypress_id"]
        k["matched_contact_id"] = contact["contact_id"]
        k["matched_touch_type"] = contact["gesture_type"]
        k["touch_start_delta_ms"] = safe_float(delta)
        k["touch_end_delta_ms"] = safe_float((contact["end_ns"] - k["end_ns"]) / MSEC)
        types[contact["gesture_type"]] += 1
        matched += 1
        matching_deltas.append(abs(delta))
    return {"matched_key_intervals": matched,
            "matched_touch_taps": types["tap"],
            "matched_touch_swipes": types["swipe"],
            "matched_touch_other": matched - types["tap"] - types["swipe"],
            "matched_fraction_of_completed_keys": safe_float(matched / len(keys)) if keys else None,
            "median_absolute_key_to_touch_start_ms": safe_float(np.median(matching_deltas)) if matching_deltas else None,
            "unmatched_key_intervals": len(keys) - matched}


def flag_double_candidates(contacts: list[dict], cfg: dict) -> int:
    """Conservative DOUBLE candidate, NOT a claim of an Android-recognized double tap.

    Two nearby rapid taps without individual keyboard matches. We exclude candidate
    pairs by default but store flags and counts so this subjective assumption is auditable.
    """
    taps = [i for i, c in enumerate(contacts) if c["gesture_type"] == "tap" and c["status"] == "complete"
            and not c["multitouch"] and not c["matched_keypress"]]
    taps.sort(key=lambda i: contacts[i]["start_ns"])
    used = set()
    pairs = 0
    for a, b in zip(taps, taps[1:]):
        if a in used or b in used:
            continue
        left, right = contacts[a], contacts[b]
        delay = (right["start_ns"] - left["end_ns"]) / MSEC
        dx = left["start_x"] - right["start_x"]
        dy = left["start_y"] - right["start_y"]
        if (0 <= delay <= cfg["double_tap_gap_ms"] and math.isfinite(dx) and math.isfinite(dy)
                and math.hypot(dx, dy) <= cfg["double_tap_distance_px"]):
            left["possible_double_tap"] = True
            right["possible_double_tap"] = True
            pairs += 1
            used.update((a, b))
    return pairs


def event_coverage(t: np.ndarray, acc: np.ndarray, gyro: np.ndarray, cfg: dict):
    a, ad = near_sensor(acc, t, cfg["event_tolerance_ms"])
    g, gd = near_sensor(gyro, t, cfg["event_tolerance_ms"])
    valid = t != TIMESTAMP_NAT
    both = valid & a & g
    return {"valid": int(valid.sum()), "near_accel": int((valid & a).sum()),
            "near_gyro": int((valid & g).sum()), "near_both": int(both.sum()),
            "near_both_pct": safe_float(100 * both.sum() / valid.sum()) if valid.sum() else None,
            "median_nearest_accel_ms": safe_float(np.median(ad[valid & np.isfinite(ad)])) if np.any(valid & np.isfinite(ad)) else None,
            "median_nearest_gyro_ms": safe_float(np.median(gd[valid & np.isfinite(gd)])) if np.any(valid & np.isfinite(gd)) else None}


def augment_interval(c: dict, acc: np.ndarray, gyro: np.ndarray, cfg: dict) -> None:
    a = coverage(acc, c["start_ns"], c["end_ns"], cfg)
    g = coverage(gyro, c["start_ns"], c["end_ns"], cfg)
    pairs = pairs_inside(acc, gyro, c["start_ns"], c["end_ns"], cfg["pair_tolerance_ms"])
    c.update({
        "accel_samples": a["samples"], "gyro_samples": g["samples"],
        "accel_first_delay_ms": a["first_delay_ms"], "gyro_first_delay_ms": g["first_delay_ms"],
        "accel_last_delay_ms": a["last_delay_ms"], "gyro_last_delay_ms": g["last_delay_ms"],
        "accel_max_inside_gap_ms": a["max_inside_gap_ms"], "gyro_max_inside_gap_ms": g["max_inside_gap_ms"],
        "accel_any_inside": a["any_inside"], "gyro_any_inside": g["any_inside"],
        "accel_interval_covered": a["interval_covered"], "gyro_interval_covered": g["interval_covered"],
        "paired_accel_gyro_samples": pairs, "both_interval_covered": a["interval_covered"] and g["interval_covered"],
        "both_strict_covered": a["interval_covered"] and g["interval_covered"]
                               and pairs >= cfg["min_sensor_pairs"],
    })


def norm_model(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def quick_file_report(uid: str, kind: str, descs: list, root: Path) -> dict:
    return {"user_id": uid, "kind": kind, "count": len(descs),
            "present": bool(descs), "models": ";".join(sorted({d["model"] for d in descs})),
            "paths": ";".join(str(Path(d["path"]).relative_to(root)) for d in descs),
            "total_bytes": sum(Path(d["path"]).stat().st_size for d in descs)}


def checkpoint_report(df: pd.DataFrame, acc: np.ndarray, gyro: np.ndarray, t_touch: np.ndarray, t_key: np.ndarray, cfg: dict):
    t = time_ns(df["time"])
    event = df["eventtype"].astype(str).str.strip().to_numpy()
    valid = t != TIMESTAMP_NAT
    first = int(t[valid].min()) if np.any(valid) else None
    stats = {"rows": len(df), "valid_rows": int(valid.sum()), "first_checkpoint_utc": iso_ns(first) if first else "",
             "checkpoint_types": dict(Counter(event))}
    if first is not None:
        for label, arr in (("accel", acc), ("gyro", gyro), ("touch", t_touch), ("keyboard", t_key)):
            valid_arr = arr[arr != TIMESTAMP_NAT]
            stats[f"{label}_rows_pre_first_checkpoint"] = int(np.sum(valid_arr < first))
            stats[f"{label}_rows_post_first_checkpoint"] = int(np.sum(valid_arr >= first))
        # This diagnostic does not assume Checkpoints identify the exact start of task h.
    return stats


def summarize_count(rows, feature, condition):
    return sum(1 for r in rows if r.get(feature) == condition)


def process_user(uid: str, entries: dict, rootstr: str, outstr: str, cfg: dict, no_cache: bool):
    root, output = Path(rootstr), Path(outstr)
    cache_dir = output / "cache" / "users"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / f"{uid}"
    jsfile = base.with_suffix(".json")
    gestures_path = cache_dir / f"{uid}_gestures.csv.gz"
    keys_path = cache_dir / f"{uid}_keys.csv.gz"
    sig = signature(uid, entries, root, cfg)
    if not no_cache and jsfile.exists() and gestures_path.exists() and keys_path.exists():
        try:
            data = json.loads(jsfile.read_text(encoding="utf-8"))
            if data.get("signature") == sig and data.get("version") == VERSION:
                data["cache_hit"] = True
                return data
        except (ValueError, OSError):
            pass

    started = time.perf_counter()
    result = {"version": VERSION, "user_id": uid, "signature": sig, "cache_hit": False,
              "files": [quick_file_report(uid, k, entries.get(k, []), root) for k in KINDS],
              "errors": [], "warnings": [], "sensor_stats": {}, "events": {},
              "models": sorted({d["model"] for k in KINDS for d in entries.get(k, [])}),
              "file_models_consistent": False, "processed": False}
    missing = [k for k in KINDS if len(entries.get(k, [])) == 0]
    dupes = [k for k in KINDS if len(entries.get(k, [])) > 1]
    result["missing_kinds"] = missing
    result["duplicate_kinds"] = dupes
    norm_models = {norm_model(d["model"]) for k in KINDS for d in entries.get(k, []) if d["model"] != "UNKNOWN"}
    has_model_files = any(entries.get(k, []) for k in KINDS)
    has_unknown_model = any(d["model"] == "UNKNOWN" for k in KINDS for d in entries.get(k, []))
    result["model_suffix_mismatch"] = len(norm_models) > 1
    result["unknown_model_suffix"] = bool(has_unknown_model)
    result["file_models_consistent"] = bool(has_model_files and len(norm_models) == 1
                                            and not dupes and not has_unknown_model)
    if has_unknown_model:
        result["warnings"].append("At least one HandPhone filename lacks a model suffix; do not assume a model")
    if dupes:
        result["errors"].append(f"Duplicate HandPhone files for: {', '.join(dupes)}. Refuse mixing recordings.")
    if len(norm_models) > 1:
        result["errors"].append(f"Different model suffixes across HandPhone files: {result['models']}")
    if missing:
        result["warnings"].append(f"Missing HandPhone files: {', '.join(missing)}")
    valid_for_matching = not dupes and len(norm_models) == 1 and not has_unknown_model

    tables = {}
    stats_for_files = {}
    for kind in KINDS:
        if len(entries.get(kind, [])) != 1:
            continue
        path = Path(entries[kind][0]["path"])
        required = {
            "Accelerometer": {"time", "xvalue", "yvalue", "zvalue"},
            "Gyroscope": {"time", "xvalue", "yvalue", "zvalue"},
            "TouchEvent": {"time", "xvalue", "yvalue", "actiontype", "pointerid"},
            "Keyboard": {"time", "key", "direction"},
            "Checkpoints": {"time", "eventtype"},
        }[kind]
        optional = {
            "TouchEvent": {"pressure", "touchmajor", "touchminor", "fingerorientation", "eid"},
            "Keyboard": {"eid"}, "Accelerometer": {"eid"}, "Gyroscope": {"eid"},
            "Checkpoints": {"eid"},
        }.get(kind, set())
        try:
            df, diag = csv_table(path, required, optional, cfg["chunk_size"])
            tables[kind] = df
            stats_for_files[kind] = diag
        except Exception as exc:
            result["errors"].append(f"{kind}: {exc}")
    result["file_parse"] = stats_for_files

    acc = np.array([], dtype=np.int64)
    gyro = np.array([], dtype=np.int64)
    for kind in FREQUENCIES:
        if kind in tables:
            raw = tables[kind]
            acc_or_gyro, diag = sorted_times(raw, "time")
            if kind == "Accelerometer":
                acc = acc_or_gyro
            else:
                gyro = acc_or_gyro
            # Sensor units are defined in the supplied README. The audit does not invent units.
            result["sensor_stats"][kind.lower()] = diag
            for axis in ("xvalue", "yvalue", "zvalue"):
                vals = numeric_col(raw, axis)
                diag[f"{axis}_nonfinite_rows"] = int((~np.isfinite(vals)).sum())
            del tables[kind]

    contacts, keys = [], []
    t_touch, t_key, dirs = np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([])
    raw_touch_down_times = np.array([], dtype=np.int64)
    raw_touch_up_times = np.array([], dtype=np.int64)
    if "TouchEvent" in tables:
        try:
            raw_actions = numeric_col(tables["TouchEvent"], "actiontype")
            touch_raw_times = time_ns(tables["TouchEvent"]["time"])
            raw_touch_down_times = np.sort(touch_raw_times[raw_actions == 0])
            raw_touch_up_times = np.sort(touch_raw_times[raw_actions == 1])
            raw_touch_down_times = raw_touch_down_times[raw_touch_down_times != TIMESTAMP_NAT]
            raw_touch_up_times = raw_touch_up_times[raw_touch_up_times != TIMESTAMP_NAT]
            contacts, raw_meta, contact_errors, t_touch = construct_touch(tables["TouchEvent"], cfg)
            result["events"]["touch"] = {**raw_meta, **contact_errors}
            result["events"]["touch_row_coverage"] = event_coverage(t_touch, acc, gyro, cfg)
            del tables["TouchEvent"]
        except Exception as exc:
            result["errors"].append(f"Touch event reconstruction: {exc}")
    if "Keyboard" in tables:
        try:
            keys, raw_meta, key_errors, t_key, dirs = construct_keys(tables["Keyboard"], cfg)
            result["events"]["keyboard"] = {**raw_meta, **key_errors}
            result["events"]["keypress_row_coverage"] = event_coverage(t_key, acc, gyro, cfg)
            for name, digit in (("keypress_down_point_coverage", 0), ("keypress_up_point_coverage", 1)):
                result["events"][name] = event_coverage(t_key[dirs == digit], acc, gyro, cfg)
            # Raw key-event vs raw touch-event cross-check, independent of whether
            # a full touch gesture could be reconstructed from DOWN/MOVE/UP.
            for name, digit, touch_times in (
                ("keydown_raw_touchdown_proximity", 0, raw_touch_down_times),
                ("keyup_raw_touchup_proximity", 1, raw_touch_up_times),
            ):
                nearby, delays = near_sensor(touch_times, t_key[dirs == digit], cfg["key_touch_tolerance_ms"])
                result["events"][name] = {
                    "key_rows": int((dirs == digit).sum()),
                    "within_tolerance": int(nearby.sum()),
                    "within_tolerance_pct": safe_float(100 * nearby.mean()) if len(nearby) else None,
                    "median_delta_ms": safe_float(np.median(delays[np.isfinite(delays)]))
                    if np.isfinite(delays).any() else None,
                }
            del tables["Keyboard"]
        except Exception as exc:
            result["errors"].append(f"Keyboard event reconstruction: {exc}")
    if "Checkpoints" in tables:
        try:
            result["events"]["checkpoints"] = checkpoint_report(tables["Checkpoints"], acc, gyro, t_touch, t_key, cfg)
            del tables["Checkpoints"]
        except Exception as exc:
            result["errors"].append(f"Checkpoints analysis: {exc}")

    if valid_for_matching:
        result["events"]["keyboard_touch_linkage"] = link_keystrokes(contacts, keys, cfg["key_touch_tolerance_ms"])
    else:
        result["warnings"].append("Skipped key-to-touch linkage due to inconsistent model suffixes or duplicates")
    result["events"]["possible_double_tap_pairs"] = flag_double_candidates(contacts, cfg)

    for contact in contacts:
        augment_interval(contact, acc, gyro, cfg)
        is_supported = contact["status"] == "complete" and contact["gesture_type"] in ("tap", "swipe")
        if contact["multitouch"]:
            is_supported = False
        if contact["possible_double_tap"] and cfg["double_tap_policy"] == "exclude":
            is_supported = False
        if not valid_for_matching:
            is_supported = False
        contact["structurally_eligible"] = bool(is_supported)
        contact["motion_eligible"] = bool(is_supported and contact["both_strict_covered"])
        if not is_supported:
            if not valid_for_matching:
                reason = "inconsistent_device_models"
            elif contact["multitouch"]:
                reason = "multitouch"
            elif contact["possible_double_tap"] and cfg["double_tap_policy"] == "exclude":
                reason = "possible_double_tap"
            else:
                reason = "unusable_or_unsupported_gesture"
        elif not contact["both_strict_covered"]:
            reason = "insufficient_motion_coverage"
        else:
            reason = "eligible"
        contact["eligibility_reason"] = reason
        contact["user_id"] = uid
        contact["start_utc"] = iso_ns(contact["start_ns"])
        contact["end_utc"] = iso_ns(contact["end_ns"])
        contact["start_ms_unix"] = int(contact["start_ns"] // MSEC)
        contact["end_ms_unix"] = int(contact["end_ns"] // MSEC)
        del contact["start_ns"], contact["end_ns"]

    for k in keys:
        augment_interval(k, acc, gyro, cfg)
        k["user_id"] = uid
        k["start_utc"] = iso_ns(k["start_ns"])
        k["end_utc"] = iso_ns(k["end_ns"])
        k["start_ms_unix"] = int(k["start_ns"] // MSEC)
        k["end_ms_unix"] = int(k["end_ns"] // MSEC)
        del k["start_ns"], k["end_ns"]

    def count_contacts(condition):
        return sum(1 for c in contacts if condition(c))

    def count_keys(condition):
        return sum(1 for k in keys if condition(k))

    summary = {
        "contact_count": len(contacts),
        "complete_contacts": count_contacts(lambda c: c["status"] == "complete"),
        "touch_taps": count_contacts(lambda c: c["gesture_type"] == "tap"),
        "touch_swipes": count_contacts(lambda c: c["gesture_type"] == "swipe"),
        "touch_multitouch_contacts": count_contacts(lambda c: c["multitouch"]),
        "touch_long_stationary_contacts": count_contacts(lambda c: c["gesture_type"] == "stationary_long_contact"),
        "touch_incomplete_contacts": count_contacts(lambda c: c["status"] != "complete"),
        "touch_possible_double_tap_contacts": count_contacts(lambda c: c["possible_double_tap"]),
        "touch_structurally_eligible": count_contacts(lambda c: c["structurally_eligible"]),
        "touch_any_accel_and_gyro": count_contacts(lambda c: c["accel_any_inside"] and c["gyro_any_inside"]),
        "touch_both_interval_covered": count_contacts(lambda c: c["both_interval_covered"]),
        "touch_strict_motion_covered": count_contacts(lambda c: c["both_strict_covered"]),
        "touch_eligible_motion_covered": count_contacts(lambda c: c["motion_eligible"]),
        "eligible_taps": count_contacts(lambda c: c["motion_eligible"] and c["gesture_type"] == "tap"),
        "eligible_swipes": count_contacts(lambda c: c["motion_eligible"] and c["gesture_type"] == "swipe"),
        "key_intervals_paired": len(keys),
        "key_intervals_both_any": count_keys(lambda k: k["accel_any_inside"] and k["gyro_any_inside"]),
        "key_intervals_both_covered": count_keys(lambda k: k["both_interval_covered"]),
        "key_intervals_both_strict": count_keys(lambda k: k["both_strict_covered"]),
        "key_intervals_matched_touch": count_keys(lambda k: bool(k["matched_contact_id"])),
        "key_intervals_matched_touch_and_strict_covered": count_keys(
            lambda k: bool(k["matched_contact_id"]) and k["both_strict_covered"]),
        "contact_exclusion_reasons": dict(Counter(c["eligibility_reason"] for c in contacts)),
        "no_keys_or_touches_overlap_sensor_time": bool((len(keys) or len(contacts)) and not any(c["both_interval_covered"] for c in contacts) and not any(k["both_interval_covered"] for k in keys)),
    }
    result["counts"] = summary
    result["processed"] = bool("TouchEvent" in stats_for_files and "Keyboard" in stats_for_files and len(acc) and len(gyro)
                               and valid_for_matching)
    result["runtime_seconds"] = safe_float(time.perf_counter() - started)

    # Cache reusable physical gestures and key intervals (not source sensor XYZ values).
    cdf = pd.DataFrame(contacts)
    kdf = pd.DataFrame(keys)
    if cdf.empty:
        cdf = pd.DataFrame(columns=["user_id", "contact_id", "gesture_type", "start_utc", "end_utc",
                                    "motion_eligible", "eligibility_reason"])
    if kdf.empty:
        kdf = pd.DataFrame(columns=["user_id", "keypress_id", "start_utc", "end_utc",
                                    "both_strict_covered", "matched_contact_id"])
    # Atomic-ish: unique file per worker, temp files then rename.
    for df, destination in ((cdf, gestures_path), (kdf, keys_path)):
        temp = destination.with_name(destination.name + ".tmp")
        df.to_csv(temp, index=False, compression="gzip")
        temp.replace(destination)
    temp_json = jsfile.with_suffix(".json.tmp")
    temp_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp_json.replace(jsfile)
    return result


def write_table(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def combine_cached_csv(cached_paths: list[Path], output: Path):
    """Merge cached .csv.gz with union schema, streaming rows, including empty users."""
    import gzip
    union: list[str] = []
    seen: set[str] = set()
    for source in cached_paths:
        if not source.exists():
            continue
        with gzip.open(source, "rt", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader, [])
            for field in columns:
                if field not in seen:
                    union.append(field)
                    seen.add(field)
    with gzip.open(output, "wt", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=union, extrasaction="ignore")
        writer.writeheader()
        for source in cached_paths:
            if not source.exists():
                continue
            with gzip.open(source, "rt", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow(row)
    return output


def main():
    parser = argparse.ArgumentParser(description="BB-MAS HandPhone audit: file integrity, models, touch gestures, keyboard & motion coverage.")
    parser.add_argument("--input", type=Path, default=Path("input_data/BB-MAS_Dataset/BB-MAS_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("output_data/bbmas_handphone_analysis"))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--no-cache", action="store_true", help="Reanalyze all users, update cache")
    parser.add_argument("--clear-cache", action="store_true", help="Remove ONLY this audit's cached per-user files before running")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--event-tolerance-ms", type=float, default=30,
                        help="Maximum nearest accel/gyro distance to one touch or key event")
    parser.add_argument("--min-sensor-samples", type=int, default=2,
                        help="Minimum readings from EACH motion sensor INSIDE gesture/key interval")
    parser.add_argument("--max-edge-gap-ms", type=float, default=40,
                        help="Maximum unsampled start/end tail inside gesture interval")
    parser.add_argument("--max-sensor-gap-ms", type=float, default=120,
                        help="Maximum internal gap between readings in gesture interval")
    parser.add_argument("--pair-tolerance-ms", type=float, default=25,
                        help="Accelerometer-gyro temporal coincidence tolerance")
    parser.add_argument("--min-sensor-pairs", type=int, default=1)
    parser.add_argument("--swipe-distance-px", type=float, default=30)
    parser.add_argument("--swipe-path-px", type=float, default=50)
    parser.add_argument("--max-tap-duration-ms", type=float, default=500)
    parser.add_argument("--max-gesture-duration-ms", type=float, default=12_000)
    parser.add_argument("--key-touch-tolerance-ms", type=float, default=80)
    parser.add_argument("--double-tap-gap-ms", type=float, default=300)
    parser.add_argument("--double-tap-distance-px", type=float, default=20)
    parser.add_argument("--double-tap-policy", choices=["exclude", "report"], default="exclude",
                        help="Exclude ambiguous probable double taps, or only report them")
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)
    log_setup(output / "analysis.log", args.verbose)
    config = {k.replace("-", "_"): v for k, v in vars(args).items()
              if k not in ("input", "output", "workers", "no_cache", "clear_cache", "verbose")}
    if args.clear_cache:
        shutil.rmtree(output / "cache", ignore_errors=True)
        logging.warning("Cache cleared: %s", output / "cache")
    logging.info("BB-MAS HandPhone audit %s", VERSION)
    logging.info("Input: %s | Output: %s | Workers: %s", root, output, args.workers)
    logging.info("Interpretation: touchscreen 0=DOWN 1=UP 2=MOVE; gesture classes are inferred, not Android API annotations")
    logging.info("Coverage: nearby event samples (±%.1f ms); interval strict min %d samples/sensor, edge ≤%.1f ms, max gap ≤%.1f ms, pair within %.1f ms",
                 args.event_tolerance_ms, args.min_sensor_samples, args.max_edge_gap_ms,
                 args.max_sensor_gap_ms, args.pair_tolerance_ms)
    all_users, unknown = scan(root)
    logging.info("Found %d subject IDs and %d unrecognized/misplaced HandPhone filenames", len(all_users), len(unknown))
    write_table(output / "unexpected_filenames.csv", unknown, ["path", "issue", "folder_user", "file_user"])
    if not all_users:
        logging.error("No HandPhone files or numbered participant folders under %s", root)
        return 2
    user_ids = sorted(all_users, key=lambda x: int(x))
    results = []
    failures = []
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut_to_id = {pool.submit(process_user, uid, all_users[uid], str(root), str(output), config,
                                 args.no_cache): uid for uid in user_ids}
        for idx, future in enumerate(concurrent.futures.as_completed(fut_to_id), 1):
            uid = fut_to_id[future]
            try:
                result = future.result()
                results.append(result)
                s = result.get("counts", {})
                logging.info("[%d/%d] ID %-5s %s model=%s taps=%s swipes=%s eligible=%s keyPairs=%s keyTouch=%s (%.1fs)%s",
                             idx, len(user_ids), uid, "OK" if result.get("processed") else "CHECK",
                             ";".join(result.get("models", [])), s.get("touch_taps", "?"),
                             s.get("touch_swipes", "?"), s.get("touch_eligible_motion_covered", "?"),
                             s.get("key_intervals_paired", "?"), s.get("key_intervals_matched_touch", "?"),
                             result.get("runtime_seconds", 0) or 0, " [cache]" if result.get("cache_hit") else "")
                for err in result.get("errors", []):
                    logging.error("User %s: %s", uid, err)
                if result.get("missing_kinds"):
                    logging.warning("User %s missing: %s", uid, result["missing_kinds"])
            except Exception as exc:
                failures.append(uid)
                logging.error("User %s failed: %s\n%s", uid, exc, traceback.format_exc())
    results.sort(key=lambda r: int(r["user_id"]))
    file_rows = [f for r in results for f in r["files"]]
    write_table(output / "handphone_files.csv", file_rows)

    user_rows = []
    for r in results:
        flat = {"user_id": r["user_id"], "models": ";".join(r.get("models", [])),
                "file_models_consistent": r.get("file_models_consistent"),
                "model_suffix_mismatch": r.get("model_suffix_mismatch"),
                "unknown_model_suffix": r.get("unknown_model_suffix"),
                "missing_kinds": ";".join(r.get("missing_kinds", [])),
                "duplicate_kinds": ";".join(r.get("duplicate_kinds", [])),
                "processed": r.get("processed"), "cache_hit": r.get("cache_hit"),
                "errors": "; ".join(r.get("errors", [])), "warnings": "; ".join(r.get("warnings", [])),
                **r.get("counts", {})}
        for sect in ("touch_row_coverage", "keypress_row_coverage", "keypress_down_point_coverage", "keypress_up_point_coverage",
                     "keydown_raw_touchdown_proximity", "keyup_raw_touchup_proximity"):
            for k, v in r.get("events", {}).get(sect, {}).items():
                flat[f"{sect}_{k}"] = v
        for sect in ("touch", "keyboard", "keyboard_touch_linkage"):
            for k, v in r.get("events", {}).get(sect, {}).items():
                flat[f"{sect}_{k}"] = v
        for kind in ("accelerometer", "gyroscope"):
            for k, v in r.get("sensor_stats", {}).get(kind, {}).items():
                flat[f"{kind}_{k}"] = v
        user_rows.append(flat)
    write_table(output / "users.csv", user_rows)

    model_users = defaultdict(set)
    for r in results:
        for mod in r.get("models", []):
            model_users[mod].add(r["user_id"])
    model_rows = [{"model_suffix": model, "users": len(ids), "user_ids": ";".join(sorted(ids, key=int))}
                  for model, ids in sorted(model_users.items())]
    write_table(output / "model_inventory.csv", model_rows)
    stats_rows = []
    for r in results:
        stats = r.get("events", {}).get("checkpoints", {})
        stats_rows.append({"user_id": r["user_id"], "checkpoint_rows": stats.get("rows"),
                           "first_checkpoint_utc": stats.get("first_checkpoint_utc"),
                           "checkpoint_types": stats.get("checkpoint_types"),
                           **{k: v for k, v in stats.items() if "_rows_pre_" in k or "_rows_post_" in k}})
    write_table(output / "checkpoints_timing.csv", stats_rows)

    cache_users = output / "cache" / "users"
    # Per-user compressed CSV are always present after a successful worker.
    try:
        combine_cached_csv([cache_users / f"{r['user_id']}_gestures.csv.gz" for r in results],
                           output / "touch_contacts.csv.gz")
        combine_cached_csv([cache_users / f"{r['user_id']}_keys.csv.gz" for r in results],
                           output / "keystroke_intervals.csv.gz")
    except Exception as exc:
        logging.error("Cannot combine cached event catalogues: %s", exc)
        failures.append("catalogue_merge")

    counters = Counter()
    for r in results:
        for k, v in r.get("counts", {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                counters[k] += v
    summary = {
        "version": VERSION, "created_utc": now_utc(), "input": str(root), "output": str(output),
        "parameters": config, "participant_count": len(all_users),
        "processed_users": sum(bool(r.get("processed")) for r in results),
        "users_missing_handphone_files": sum(bool(r.get("missing_kinds")) for r in results),
        "users_inconsistent_model_suffix": sum(bool(r.get("model_suffix_mismatch")) for r in results),
        "users_unknown_model_suffix": sum(bool(r.get("unknown_model_suffix")) for r in results),
        "users_with_errors": sum(bool(r.get("errors")) for r in results),
        "model_inventory": {k: sorted(v, key=int) for k, v in model_users.items()},
        "missing_by_kind": {k: sum(k in r.get("missing_kinds", []) for r in results) for k in KINDS},
        "unrecognized_handphone_filenames": len(unknown),
        "totals": dict(counters), "failed_users": failures,
        "elapsed_seconds": safe_float(time.perf_counter() - started),
        "interpretation_warnings": [
            "TouchEvent rows are raw Android-style motion events, not gestures; tap/swipe labels are heuristics.",
            "Keyboard keydown/up matching to touchscreen is timestamp-based and does not prove exact semantic correspondence.",
            "Point coverage and complete interval coverage are separate; sensor coincidence uses a time tolerance.",
            "A same-model suffix does NOT prove that the exact same physical handset was used by all participants.",
            "Checkpoints describe walking segments; first checkpoint is not a confirmed typing-session boundary.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# BB-MAS HandPhone — результати аудиту

Generated: {summary['created_utc']}  
Version: `{VERSION}`

## Files

- `summary.json`: суми, моделі, проблеми.
- `users.csv`: детальна статистика по кожному учаснику.
- `handphone_files.csv`: наявність усіх п'яти HandPhone файлів по учасниках.
- `model_inventory.csv`: усі суфікси моделей у фактичних іменах файлів.
- `touch_contacts.csv.gz`: один реконструйований фізичний контакт на рядок, з позначенням tap/swipe, мультидотику, можливої double-tap пари й часових показників сенсорного покриття.
- `keystroke_intervals.csv.gz`: реконструйовані інтервали keydown→keyup, зіставлені контакти та сенсорне покриття.
- `checkpoints_timing.csv`: події контрольних точок і розподіл записів до/після першого checkpoint (не є гарантованим поділом на задачі).
- `unexpected_filenames.csv`: підозрілі HandPhone назви/місця файлів.
- `cache/users/`: окремий кеш по кожному учаснику, автоматично інвалідується за версією коду, параметрами та розміром/mtime вхідних CSV.
- `analysis.log`: прогрес і попередження.

## Definitions

* **touch row nearby coverage**: for the exact raw event timestamp each sensor independently has at least one reading within ±`{config['event_tolerance_ms']}` ms; this does **not** demonstrate full gesture coverage.
* **interval coverage**: at least `{config['min_sensor_samples']}` readings per sensor **inside** the start/end interval, each endpoint gap at most `{config['max_edge_gap_ms']}` ms and internal gap at most `{config['max_sensor_gap_ms']}` ms.
* **both_strict_covered**: interval coverage for both sensors and at least `{config['min_sensor_pairs']}` accelerometer readings paired with gyroscope within `{config['pair_tolerance_ms']}` ms inside the same interval.
* **matched_keypress**: a timestamp-based *one-to-one* match between keydown and tap DOWN within ±`{config['key_touch_tolerance_ms']}` ms. Not necessarily proof that a raw tap is the actual keyboard key event.
* **double-tap**: inferred ambiguous near-repeat between **non-keyboard-matched** taps (gap ≤{config['double_tap_gap_ms']}ms, start-point distance ≤{config['double_tap_distance_px']}px), `{config['double_tap_policy']}` policy. No Android GestureDetector annotations are available.
* **eligible gesture**: complete single-finger tap/swipe, excluding uncertain double-tap candidates according to policy, with simultaneous motion coverage and consistent device model suffixes.

Sensor samples outside touch gesture boundaries are NEVER used to satisfy interval coverage.

## Share back with assistant

Send `summary.json`, `users.csv`, `model_inventory.csv`, and (if troubleshooting timestamps) a few rows of `touch_contacts.csv.gz` and `keystroke_intervals.csv.gz`.

## Caveats

The README's reported ≈9,415 phone keystrokes/user and ≈8,890 phone swipe data points/user are dataset-level descriptive statistics, NOT guaranteed equal to our reconstructed `keydown→keyup` pairs or physical contacts. Sensor data may include activities other than typing; overall file-overlap therefore differs from touch interval coverage.
"""
    (output / "HOW_TO_READ_RESULTS.md").write_text(readme, encoding="utf-8")
    logging.info("Summary: users=%d processed=%d inconsistent_model=%d missing_files=%d",
                 summary["participant_count"], summary["processed_users"], summary["users_inconsistent_model_suffix"],
                 summary["users_missing_handphone_files"])
    logging.info("Totals: physical contacts=%s, taps=%s, swipes=%s, eligible=%s, paired key intervals=%s, key-touch matches=%s",
                 counters["contact_count"], counters["touch_taps"], counters["touch_swipes"],
                 counters["touch_eligible_motion_covered"], counters["key_intervals_paired"],
                 counters["key_intervals_matched_touch"])
    logging.info("Wrote %s", output)
    small = {"participants": summary["participant_count"], "processed_users": summary["processed_users"],
             "model_inventory": {k: len(v) for k, v in summary["model_inventory"].items()},
             "missing_by_kind": summary["missing_by_kind"],
             "touch_contacts": counters["contact_count"], "tap_contacts": counters["touch_taps"],
             "swipe_contacts": counters["touch_swipes"], "usable_multimodal_gestures": counters["touch_eligible_motion_covered"],
             "paired_keyboard_intervals": counters["key_intervals_paired"],
             "keyboard_to_touch_matches": counters["key_intervals_matched_touch"],
             "keypress_interval_both_covered": counters["key_intervals_both_covered"],
             "failures": summary["failed_users"]}
    block = json.dumps(small, ensure_ascii=False, indent=2)
    print("\n===== COPY THIS BLOCK BACK TO CHAT =====\n" + block + "\n===== END BLOCK =====", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
