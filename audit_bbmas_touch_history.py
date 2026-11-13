#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BB-MAS HandPhone TouchEvent structure / Android MotionEvent batching audit.

Python 3.14; standard library only (no pandas/numpy needed).

From repository root:
    py -3.14 audit_bbmas_touch_history.py --workers 6

This is a diagnostic audit, NOT reconstruction of missing exact timestamps.
The possible Android historical MotionEvent samples hypothesis is tested, NOT assumed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

VERSION = 'bbmas-touch-history-audit-v1.1.0'
DEFAULT_INPUT = Path('input_data/BB-MAS_Dataset/BB-MAS_Dataset')
DEFAULT_OUTPUT = Path('output_data/bbmas_touch_history_audit')
FILE_RE = re.compile(r'^(?P<uid>\d+)_HandPhone_TouchEvent_\((?P<model>[^)]+)\)\.csv$', re.I)
MILLISECONDS_PER_DAY = 86_400_000
EPOCH = datetime(1970, 1, 1)
TIME_REGEX = re.compile(r'^(\d{4}-\d{1,2}-\d{1,2})[ T](\d{1,2}:\d{2}:\d{2})[.:](\d{1,6})$')
REPORT_FIELDS = [
    'user_id', 'model', 'filename', 'rows', 'down', 'up', 'move', 'other_actions',
    'missing_both', 'missing_action_only', 'missing_time_only', 'present_both',
    'time_parse_failures', 'down_missing_time', 'up_missing_time', 'move_missing_time',
    'missing_xy', 'missing_pointer', 'eid_nonmonotonic', 'orphan_move_or_blank',
    'orphan_up', 'unclosed_contacts', 'completed_contacts', 'reliable_bounds_contacts',
    'unreliable_bounds_contacts', 'contacts_with_missing_interior', 'contacts_with_missing_endpoint_time', 'missing_both_in_contact',
    'missing_both_inside_reliable_contacts',
    'missing_time_in_contact', 'missing_both_before_move',
    'missing_both_before_timestamped_move', 'missing_both_before_move_inside_reliable_contacts',
    'missing_both_before_up',
    'candidate_history_batches_before_move', 'candidate_history_batches_timestamped_move',
    'bracketed_missing_inside_contact', 'bracketed_missing_both_inside_contact',
    'timestamp_order_reversals', 'long_or_negative_contact_durations',
    'contact_steps_total', 'contact_steps_below_jump_threshold',
    'median_contact_duration_ms', 'p95_contact_duration_ms',
    'median_coordinate_step_px', 'p95_coordinate_step_px',
    'median_history_batch_size', 'p95_history_batch_size',
]
CONTACT_FIELDS = [
    'user_id', 'model', 'contact_no', 'pointer_id', 'down_row', 'up_row',
    'start_time', 'end_time', 'duration_ms', 'bounds_valid', 'rows',
    'missing_both', 'missing_time', 'move_rows', 'timestamped_move_rows',
    'path_px', 'displacement_px', 'max_coordinate_step_px',
    'time_reversals',
]


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode='w', encoding='utf-8')],
        force=True,
    )


def norm_header(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def int_or_none(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in ('', 'nan', 'null', 'none'):
        return None
    try:
        n = float(value)
        return int(n) if math.isfinite(n) and n.is_integer() else None
    except (ValueError, TypeError):
        return None


def float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in ('', 'nan', 'null', 'none'):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def parse_time_ms(value: Any) -> int | None:
    if value is None:
        return None
    raw = str(value).strip().strip('"\'')
    if not raw or raw.lower() in ('nan', 'null', 'none'):
        return None
    match = TIME_REGEX.fullmatch(raw)
    if match:
        d, hms, fraction = match.groups()
        raw = f'{d} {hms}.{fraction}'
    try:
        dt = datetime.strptime(raw, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            dt = datetime.strptime(raw, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    delta = dt - EPOCH
    return delta.days * MILLISECONDS_PER_DAY + delta.seconds * 1000 + delta.microseconds // 1000


def percentile(values: list[float | int], frac: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * frac
    a = int(rank)
    b = min(a + 1, len(ordered) - 1)
    return round(float(ordered[a] + (ordered[b] - ordered[a]) * (rank - a)), 3)


def iso_time(time_ms: int | None) -> str:
    if time_ms is None:
        return ''
    # Store original time fields in samples; this is display only.
    return datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


@dataclass(slots=True)
class Row:
    i: int
    eid: int | None
    x: float | None
    y: float | None
    p: int | None
    a: int | None
    t: int | None
    raw_a: str
    raw_t: str
    raw_line: str
    cid: int = -1

    @property
    def missing_both(self) -> bool:
        return self.a is None and self.t is None


def read_touch_csv(path: Path) -> tuple[list[Row], dict[str, Any]]:
    # DictReader preserves embedded commas in correctly quoted CSV cells.
    # Physical-line snippets are preserved only when each CSV record occupies one line.
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        raw_lines = f.readlines()
    reader = csv.DictReader(raw_lines)
    headers = reader.fieldnames or []
    names = {norm_header(c): c for c in headers}
    need = {'eid', 'xvalue', 'yvalue', 'actiontype', 'time'}
    if not need.issubset(names):
        raise ValueError(f'{path.name}: missing columns {sorted(need - names.keys())}; headers={headers}')
    rows = []
    malformed = 0
    bad_times = 0
    row_to_line_map_exact = True
    prev_line = reader.line_num
    for i, rec in enumerate(reader):
        physical_line_end = reader.line_num
        if physical_line_end != prev_line + 1:
            row_to_line_map_exact = False
        prev_line = physical_line_end
        if None in rec:
            malformed += 1
        a_raw = str(rec.get(names['actiontype']) or '').strip()
        t_raw = str(rec.get(names['time']) or '').strip()
        t = parse_time_ms(t_raw)
        if t_raw and t is None:
            bad_times += 1
        rows.append(Row(
            i=i, eid=int_or_none(rec.get(names['eid'])),
            x=float_or_none(rec.get(names['xvalue'])), y=float_or_none(rec.get(names['yvalue'])),
            p=int_or_none(rec.get(names.get('pointerid', ''), '')),
            a=int_or_none(a_raw), t=t, raw_a=a_raw, raw_t=t_raw,
            raw_line=raw_lines[physical_line_end - 1].rstrip('\r\n') if physical_line_end <= len(raw_lines) else '',
        ))
    return rows, {'headers': headers, 'malformed_records': malformed,
                  'time_parse_failures': bad_times, 'raw_line_mapping_1to1': row_to_line_map_exact}


def contact_analysis(rows: list[Row], uid: str, model: str, max_duration_ms: float,
                     jump_threshold_px: float) -> tuple[list[dict], dict, list[float]]:
    active: dict[int, tuple[int, list[int]]] = {}
    contacts: list[dict] = []
    stats: Counter = Counter()
    all_steps: list[float] = []
    duration_values: list[int] = []
    prev_t = None
    def finalize(indices: list[int], end_found: bool) -> None:
        cid = len(contacts)
        for j in indices:
            rows[j].cid = cid
        events = [rows[j] for j in indices]
        first = events[0]
        last = events[-1]
        bound = end_found and first.a == 0 and last.a == 1 and first.t is not None and last.t is not None
        duration = (last.t - first.t) if bound else None
        bounds_valid = (bound and first.p is not None and last.p is not None
                        and first.p == last.p and 0 <= duration <= max_duration_ms)
        if not bounds_valid and end_found:
            stats['unreliable_bounds_contacts'] += 1
        if bounds_valid:
            stats['reliable_bounds_contacts'] += 1
            duration_values.append(duration)
        if bound and not (0 <= duration <= max_duration_ms):
            stats['long_or_negative_contact_durations'] += 1
        if end_found:
            stats['completed_contacts'] += 1
        else:
            stats['unclosed_contacts'] += 1
        coords = [(r.x, r.y) for r in events if r.x is not None and r.y is not None]
        steps = [math.hypot(x1-x0, y1-y0) for (x0,y0),(x1,y1) in zip(coords, coords[1:])]
        all_steps.extend(steps)
        stats['contact_steps_total'] += len(steps)
        stats['contact_steps_below_jump_threshold'] += sum(d <= jump_threshold_px for d in steps)
        path = sum(steps)
        displacement = math.hypot(coords[-1][0]-coords[0][0], coords[-1][1]-coords[0][1]) if len(coords)>1 else 0.0
        times = [r.t for r in events if r.t is not None]
        reversals = sum(b<a for a,b in zip(times, times[1:]))
        stats['timestamp_order_reversals'] += reversals
        missing_both = sum(r.missing_both for r in events)
        missing_time = sum(r.t is None for r in events)
        stats['missing_both_in_contact'] += missing_both
        stats['missing_time_in_contact'] += missing_time
        if any(r.t is None for r in events[1:-1]):
            stats['contacts_with_missing_interior'] += 1
        if first.t is None or (end_found and last.t is None):
            stats['contacts_with_missing_endpoint_time'] += 1
        if bounds_valid:
            stats['missing_both_inside_reliable_contacts'] += missing_both
        contacts.append({
            'user_id': uid, 'model': model, 'contact_no': cid,
            'pointer_id': first.p if first.p is not None else 0,
            'down_row': first.i, 'up_row': last.i if end_found else '',
            'start_time': first.raw_t, 'end_time': last.raw_t if end_found else '',
            'duration_ms': duration if bound else '', 'bounds_valid': int(bounds_valid),
            'rows': len(events), 'missing_both': missing_both, 'missing_time': missing_time,
            'move_rows': sum(r.a == 2 for r in events),
            'timestamped_move_rows': sum(r.a == 2 and r.t is not None for r in events),
            'path_px': round(path, 3), 'displacement_px': round(displacement, 3),
            'max_coordinate_step_px': round(max(steps), 3) if steps else '',
            'time_reversals': reversals,
        })

    for row in rows:
        pointer = row.p if row.p is not None else 0
        if row.a == 0:
            if pointer in active:
                finalize(active.pop(pointer)[1], False)
            active[pointer] = (row.i, [row.i])
        elif row.a == 1:
            if pointer in active:
                active[pointer][1].append(row.i)
                finalize(active.pop(pointer)[1], True)
            else:
                stats['orphan_up'] += 1
        elif row.a == 2 or row.a is None:
            if pointer in active:
                active[pointer][1].append(row.i)
            else:
                stats['orphan_move_or_blank'] += 1
        elif row.a is not None and pointer in active:
            # Preserve a classified but nonstandard event inside current contact.
            active[pointer][1].append(row.i)
    for start, indices in active.values():
        finalize(indices, False)
    stats['median_contact_duration_ms'] = percentile(duration_values, 0.5)
    stats['p95_contact_duration_ms'] = percentile(duration_values, 0.95)
    stats['median_coordinate_step_px'] = percentile(all_steps, 0.5)
    stats['p95_coordinate_step_px'] = percentile(all_steps, 0.95)
    return contacts, dict(stats), all_steps


def patterns_and_examples(rows: list[Row], uid: str, model: str, *, max_bracket_gap_ms: int,
                          samples_per_user: int, reliable_contact_ids: set[int]) -> tuple[dict, list[dict], list[int]]:
    counts: Counter = Counter()
    sample_buckets: dict[str, list[int]] = defaultdict(list)
    all_history_batch_sizes: list[int] = []
    # Map valid-time anchors within each reconstructed contact.
    groups: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        if row.cid != -1:
            groups[row.cid].append(row.i)
    for indices in groups.values():
        last_known = None
        next_known_by_idx: dict[int, int] = {}
        next_known = None
        for j in reversed(indices):
            if rows[j].t is not None:
                next_known = j
            else:
                if next_known is not None:
                    next_known_by_idx[j] = next_known
        for j in indices:
            rr = rows[j]
            if rr.t is None and last_known is not None and j in next_known_by_idx:
                nxt = rows[next_known_by_idx[j]]
                prev = rows[last_known]
                if prev.t is not None and nxt.t is not None and 0 <= nxt.t - prev.t <= max_bracket_gap_ms:
                    counts['bracketed_missing_inside_contact'] += 1
                    if rr.missing_both:
                        counts['bracketed_missing_both_inside_contact'] += 1
                        if len(sample_buckets['bracketed_missing_both']) < samples_per_user:
                            sample_buckets['bracketed_missing_both'].append(j)
            if rr.t is not None:
                last_known = j

    for i, row in enumerate(rows):
        if row.missing_both and row.cid >= 0:
            # Nearest subsequent row with an explicit action (no assertion about its time).
            j = i + 1
            while j < len(rows) and rows[j].cid == row.cid and rows[j].a is None:
                j += 1
            if j < len(rows) and rows[j].cid == row.cid and rows[j].a == 2:
                counts['missing_both_before_move'] += 1
                if row.cid in reliable_contact_ids:
                    counts['missing_both_before_move_inside_reliable_contacts'] += 1
                if rows[j].t is not None:
                    counts['missing_both_before_timestamped_move'] += 1
                    if len(sample_buckets['missing_before_timestamped_move']) < samples_per_user:
                        sample_buckets['missing_before_timestamped_move'].append(i)
                elif len(sample_buckets['missing_before_move_without_time']) < samples_per_user:
                    sample_buckets['missing_before_move_without_time'].append(i)
            elif j < len(rows) and rows[j].cid == row.cid and rows[j].a == 1:
                counts['missing_both_before_up'] += 1
        if row.a is not None and row.t is None and len(sample_buckets['action_without_time']) < samples_per_user:
            sample_buckets['action_without_time'].append(i)
        if row.a in (0, 1) and row.t is None and len(sample_buckets['endpoint_without_time']) < samples_per_user:
            sample_buckets['endpoint_without_time'].append(i)
        if row.missing_both and row.cid == -1 and len(sample_buckets['orphan_missing_both']) < samples_per_user:
            sample_buckets['orphan_missing_both'].append(i)

    # Consecutive missing-both rows immediately followed by MOVE with same contact.
    idx = 0
    while idx < len(rows):
        if not rows[idx].missing_both or rows[idx].cid == -1:
            idx += 1
            continue
        stop = idx
        cid = rows[idx].cid
        while stop < len(rows) and rows[stop].cid == cid and rows[stop].missing_both:
            stop += 1
        if stop < len(rows) and rows[stop].cid == cid and rows[stop].a == 2:
            counts['candidate_history_batches_before_move'] += 1
            all_history_batch_sizes.append(stop-idx)
            if rows[stop].t is not None:
                counts['candidate_history_batches_timestamped_move'] += 1
        idx = max(stop, idx+1)
    counts['median_history_batch_size'] = percentile(all_history_batch_sizes, 0.5)
    counts['p95_history_batch_size'] = percentile(all_history_batch_sizes, 0.95)

    examples: list[dict] = []
    for kind, indices in sample_buckets.items():
        for i in indices[:samples_per_user]:
            nearby = []
            for q in range(max(0, i-3), min(len(rows), i+5)):
                r = rows[q]
                nearby.append({'row_index': q, 'eid': r.eid, 'pointer_id': r.p,
                               'x': r.x, 'y': r.y, 'action': r.a, 'time': r.raw_t,
                               'contact_no': r.cid, 'csv_record': r.raw_line})
            examples.append({'user_id': uid, 'model': model, 'pattern': kind, 'center_row': i,
                             'window': nearby})
    return dict(counts), examples, all_history_batch_sizes


def analyze_one(path_str: str, output_str: str, cfg: dict) -> dict:
    p = Path(path_str)
    out = Path(output_str)
    m = FILE_RE.fullmatch(p.name)
    if not m:
        raise ValueError(f'Unrecognized HandPhone TouchEvent filename: {p.name}')
    uid = str(int(m.group('uid')))
    model = m.group('model')
    cache_dir = out / 'cache' / 'users'
    cache_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = out / 'cache' / 'contacts'
    contact_dir.mkdir(parents=True, exist_ok=True)
    st = p.stat()
    analytical_cfg = {k: cfg[k] for k in ('max_bracket_gap_ms', 'max_contact_ms',
                                          'jump_threshold_px', 'samples_per_user')}
    sign = hashlib.sha256(json.dumps({
        'version': VERSION, 'name': p.name, 'path': str(p.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
        'cfg': analytical_cfg,
    }, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / f'{uid}.json'
    contact_file = contact_dir / f'{uid}.csv.gz'
    if not cfg['no_cache'] and cache_file.exists() and contact_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            if cached.get('signature') == sign:
                cached['from_cache'] = True
                return cached
        except (ValueError, OSError):
            pass

    rows, file_info = read_touch_csv(p)
    counts: Counter = Counter()
    counts['rows'] = len(rows)
    counts['time_parse_failures'] = file_info['time_parse_failures']
    prev_eid = None
    for r in rows:
        if r.a == 0:
            counts['down'] += 1
        elif r.a == 1:
            counts['up'] += 1
        elif r.a == 2:
            counts['move'] += 1
        elif r.a is not None:
            counts['other_actions'] += 1
        if r.a is None and r.t is None:
            counts['missing_both'] += 1
        elif r.a is None:
            counts['missing_action_only'] += 1
        elif r.t is None:
            counts['missing_time_only'] += 1
        else:
            counts['present_both'] += 1
        if r.a == 0 and r.t is None:
            counts['down_missing_time'] += 1
        elif r.a == 1 and r.t is None:
            counts['up_missing_time'] += 1
        elif r.a == 2 and r.t is None:
            counts['move_missing_time'] += 1
        if r.x is None or r.y is None:
            counts['missing_xy'] += 1
        if r.p is None:
            counts['missing_pointer'] += 1
        if r.eid is not None:
            if prev_eid is not None and r.eid <= prev_eid:
                counts['eid_nonmonotonic'] += 1
            prev_eid = r.eid
    contacts, contact_stats, _ = contact_analysis(rows, uid, model, cfg['max_contact_ms'], cfg['jump_threshold_px'])
    counts.update(contact_stats)
    reliable_ids = {int(c['contact_no']) for c in contacts if c['bounds_valid']}
    pattern_stats, examples, _ = patterns_and_examples(rows, uid, model, max_bracket_gap_ms=cfg['max_bracket_gap_ms'], samples_per_user=cfg['samples_per_user'], reliable_contact_ids=reliable_ids)
    counts.update(pattern_stats)
    summary = {'user_id': uid, 'model': model, 'filename': p.name,
               'file_info': file_info, 'stats': dict(counts), 'examples': examples,
               'signature': sign, 'from_cache': False}

    # Write atomically to protect against a terminated run.
    temp_contacts = contact_file.with_name(contact_file.name + '.tmp')
    with gzip.open(temp_contacts, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CONTACT_FIELDS)
        writer.writeheader()
        writer.writerows(contacts)
    temp_contacts.replace(contact_file)
    temp_summary = cache_file.with_name(cache_file.name + '.tmp')
    temp_summary.write_text(json.dumps(summary, ensure_ascii=False), encoding='utf-8')
    temp_summary.replace(cache_file)
    return summary


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def add_counts(items: list[dict]) -> dict[str, int]:
    out: Counter = Counter()
    numeric_keys = [k for k in REPORT_FIELDS if k not in ('user_id', 'model', 'filename')
                    and not k.startswith('median_') and not k.startswith('p95_')]
    for item in items:
        stats = item['stats']
        for k in numeric_keys:
            if isinstance(stats.get(k), int):
                out[k] += stats[k]
    return dict(out)


def run(cfg: dict) -> dict:
    root = Path(cfg['input']).resolve()
    out = Path(cfg['output']).resolve()
    if not root.is_dir():
        raise SystemExit(f'Input directory not found: {root}')
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / 'analysis.log')
    files = sorted((p for p in root.rglob('*.csv') if FILE_RE.fullmatch(p.name)),
                   key=lambda p: int(FILE_RE.fullmatch(p.name).group('uid')))
    if cfg['users']:
        keep = set(cfg['users'])
        files = [p for p in files if str(int(FILE_RE.fullmatch(p.name).group('uid'))) in keep]
    if not files:
        raise SystemExit(f'No matching HandPhone TouchEvent files found in {root}')
    seen_uid: set[str] = set()
    for p in files:
        uid = str(int(FILE_RE.fullmatch(p.name).group('uid')))
        if uid in seen_uid:
            raise SystemExit(f'More than one HandPhone TouchEvent file for participant {uid}; inspect duplicates before analysis')
        seen_uid.add(uid)
    logging.info('BB-MAS TouchEvent Android batching hypothesis audit, %s', VERSION)
    logging.info('Input: %s | files: %d | workers: %d', root, len(files), cfg['workers'])
    logging.info('Output: %s', out)
    start = time.monotonic()
    results = []
    failures = []
    if cfg['workers'] == 1:
        for k, p in enumerate(files, 1):
            try:
                res = analyze_one(str(p), str(out), cfg)
                results.append(res)
                logging.info('[%d/%d] user %s / %s | %d rows | %d reliable contacts | cache=%s',
                             k, len(files), res['user_id'], res['model'], res['stats'].get('rows', 0),
                             res['stats'].get('reliable_bounds_contacts', 0), res['from_cache'])
            except Exception as exc:
                failures.append({'path': str(p), 'error': repr(exc)})
                logging.exception('Failed processing %s', p.name)
    else:
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            jobs = {pool.submit(analyze_one, str(p), str(out), cfg): p for p in files}
            for k, task in enumerate(as_completed(jobs), 1):
                p = jobs[task]
                try:
                    res = task.result()
                    results.append(res)
                    logging.info('[%d/%d] user %s / %s | %d rows | %d reliable contacts | cache=%s',
                                 k, len(files), res['user_id'], res['model'], res['stats'].get('rows', 0),
                                 res['stats'].get('reliable_bounds_contacts', 0), res['from_cache'])
                except Exception as exc:
                    failures.append({'path': str(p), 'error': repr(exc)})
                    logging.exception('Failed processing %s: %r', p.name, exc)
    results.sort(key=lambda d: int(d['user_id']))
    rows = []
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        row = {'user_id': r['user_id'], 'model': r['model'], 'filename': r['filename']}
        row.update(r['stats'])
        rows.append(row)
        by_model[r['model']].append(r)
    write_csv(out / 'users.csv', rows, REPORT_FIELDS)

    model_rows = []
    for model, group in sorted(by_model.items()):
        totals = add_counts(group)
        model_rows.append({'model': model, 'users': len(group), **totals,
                           'missing_both_pct': round(100*totals.get('missing_both',0)/max(1,totals.get('rows',0)),3),
                           'missing_both_in_contacts_pct': round(100*totals.get('missing_both_in_contact',0)/max(1,totals.get('missing_both',0)),3),
                           'missing_both_before_move_pct': round(100*totals.get('missing_both_before_move',0)/max(1,totals.get('missing_both',0)),3),
                           'reliable_bounds_pct': round(100*totals.get('reliable_bounds_contacts',0)/max(1,totals.get('completed_contacts',0)),3)})
    mod_fields = ['model','users',*[
        k for k in REPORT_FIELDS if k not in ('user_id','model','filename') and not k.startswith('median_') and not k.startswith('p95_')
    ],'missing_both_pct','missing_both_in_contacts_pct','missing_both_before_move_pct','reliable_bounds_pct']
    write_csv(out / 'models.csv', model_rows, mod_fields)

    examples = []
    for r in results:
        examples.extend(r['examples'])
    with (out / 'examples.jsonl').open('w', encoding='utf-8') as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    # Merge per-user contacts into an analysis-friendly archive.
    with gzip.open(out / 'contacts.csv.gz', 'wt', encoding='utf-8', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=CONTACT_FIELDS)
        writer.writeheader()
        for r in results:
            contact_path = out / 'cache' / 'contacts' / f"{r['user_id']}.csv.gz"
            with gzip.open(contact_path, 'rt', encoding='utf-8', newline='') as src:
                reader = csv.DictReader(src)
                writer.writerows(reader)
    totals = add_counts(results)
    def ratio(num: str, denom: str) -> float:
        return round(100 * totals.get(num, 0) / max(1, totals.get(denom, 0)), 3)
    summary = {
        'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(root), 'output': str(out), 'users': len(results),
        'model_users': {model: len(v) for model, v in by_model.items()},
        'parameters': {k:v for k,v in cfg.items() if k not in ('input','output','users')},
        'totals': totals,
        'diagnostic_ratios_pct': {
            'missing_both_of_all_rows': ratio('missing_both','rows'),
            'missing_both_inside_reconstructed_contacts': ratio('missing_both_in_contact','missing_both'),
            'missing_both_inside_contacts_with_valid_bounds': ratio('missing_both_inside_reliable_contacts','missing_both'),
            'missing_both_directly_before_move': ratio('missing_both_before_move','missing_both'),
            'missing_both_directly_before_timestamped_move': ratio('missing_both_before_timestamped_move','missing_both'),
            'missing_times_bracketed_by_close_known_anchors': ratio('bracketed_missing_inside_contact','missing_time_in_contact'),
            'contacts_with_valid_down_up_times': ratio('reliable_bounds_contacts','completed_contacts'),
            'contacts_with_some_missing_time': ratio('contacts_with_missing_interior','completed_contacts'),
            'coordinate_steps_below_threshold': ratio('contact_steps_below_jump_threshold','contact_steps_total'),
        },
        'failed_files': failures, 'examples_saved': len(examples),
        'elapsed_seconds': round(time.monotonic()-start, 2),
        'interpretation': [
            'Patterns consistent with MotionEvent history do not prove the original BB-MAS logger called getHistoricalX/getHistoricalEventTime.',
            'Android exposes getHistoricalEventTime for each historical sample: missing CSV time is a logging/export choice, not mandated by Android.',
            'The audit NEVER invents per-row timestamps. Temporal interpolation is NOT applied to gesture features.',
            'Timestamped DOWN/UP can delimit motion-sensor windows even if intermediate touch points have no time.',
            'Short tapping events from a custom keyboard may not appear in TouchEvent.csv.',
            'Unrecognized or unpaired events are not treated as verified gestures.',
        ],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    logging.info('=========================================================')
    logging.info('COPY THIS JSON SUMMARY BACK TO CHAT:')
    logging.info(json.dumps({'users': summary['users'], 'models': summary['model_users'],
                             'totals': {k: totals.get(k,0) for k in [
                                 'rows','missing_both','missing_time_only','missing_action_only','down','up','move',
                                 'missing_both_in_contact','missing_both_before_move',
                                 'missing_both_before_timestamped_move',
                                 'reliable_bounds_contacts','completed_contacts',
                                 'bracketed_missing_inside_contact',
                                 'candidate_history_batches_timestamped_move',
                                 'down_missing_time','up_missing_time']},
                             'diagnostic_ratios_pct': summary['diagnostic_ratios_pct'],
                             'failed_files': summary['failed_files']}, indent=2, ensure_ascii=False))
    logging.info('Saved: %s', out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=str(DEFAULT_INPUT), help='Unpacked BB-MAS root')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='Analysis output directory')
    parser.add_argument('--workers', type=int, default=max(1,min(6,os.cpu_count() or 1)), help='Number of CPU worker processes')
    parser.add_argument('--users', nargs='*', default=[], help='Optional user IDs, e.g. --users 1 9')
    parser.add_argument('--max-bracket-gap-ms', type=int, default=500, help='Maximum gap across an untimed row for plausible interpolation feasibility (NOT performed)')
    parser.add_argument('--max-contact-ms', type=int, default=12000, help='Maximum contact length for trustworthy DOWN-UP bounds')
    parser.add_argument('--jump-threshold-px', type=float, default=150.0, help='Coordinate step continuity diagnostic threshold')
    parser.add_argument('--samples-per-user', type=int, default=2, help='Examples per anomaly category for each user')
    parser.add_argument('--no-cache', action='store_true', help='Reanalyze all files ignoring cache')
    args = parser.parse_args()
    if args.workers < 1 or args.samples_per_user < 0:
        parser.error('--workers must be >=1 and --samples-per-user >=0')
    cfg = vars(args)
    run(cfg)


if __name__ == '__main__':
    main()
