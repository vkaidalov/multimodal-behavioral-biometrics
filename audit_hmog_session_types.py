#!/usr/bin/env python3
"""Audit HMOG session directory numbers, Activity.csv labels, and answer files.

No pandas, no sensor processing, no network. Python 3.10+.

Usage:
  python audit_hmog_session_types.py --input input_data/hmog_unarchived
  python audit_hmog_session_types.py --input input_data/hmog_unarchived \
       --previous-sessions output_data/hmog_raw_feta_audit/sessions.csv

CRITICAL: A folder's session number is NOT necessarily Activity.TaskID.
The dataset authors describe randomly assigning tasks when subjects log in.
The numeric directory-to-TaskID comparison is an *assumption audit*, not proof
that any dataset record is wrong.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = '1.0.0'
ACTIVITY_COLUMNS = (
    'ID', 'SubjectID', 'Session_number', 'Start_time', 'End_time',
    'Relative_Start_time', 'Relative_End_time', 'Gesture_scenario',
    'TaskID', 'ContentID',
)
ACTIVITY_HEADER = {re.sub(r'[^a-z0-9]', '', v.lower()): v for v in ACTIVITY_COLUMNS}
ACTIVITY_TYPES = {
    1: ('reading', 'sitting'), 2: ('reading', 'walking'),
    3: ('writing', 'sitting'), 4: ('writing', 'walking'),
    5: ('map', 'sitting'), 6: ('map', 'walking'),
}
TASKS = {task_id: ACTIVITY_TYPES[(task_id - 1) % 6 + 1] for task_id in range(1, 25)}
SESSION_RE = re.compile(r'^(?:(?P<uid>\d{6})[_-])?session[_-]?(?P<number>\d+)$', re.I)
ANSWER_RE = re.compile(r'^(reading|writing)[_\s-]*([123])[_\s-]*answer(?:\..+)?$', re.I)
EXPECTED_PER_TASK = 4


def normalize_header(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.strip().lower().lstrip('\ufeff'))


def as_int(value: object) -> int | None:
    try:
        x = str(value).strip()
        if x == '':
            return None
        if x.endswith('.0'):
            x = x[:-2]
        return int(x)
    except (ValueError, TypeError):
        return None


def stable_unique(values):
    return sorted(set(values), key=lambda s: (str(s)))


def fmt_set(values):
    return ';'.join(str(x) for x in sorted(set(values)))


def choose_delimiter(first_line):
    return max((',', ';', '\t'), key=first_line.count)


def parse_activity(path: Path):
    """Read positional or headered Activity.csv; flag suspicious schemas, never guess."""
    result = {
        'activity_file_exists': path.is_file(), 'activity_row_count': 0,
        'activity_invalid_rows': 0, 'activity_header_type': 'missing',
        'activity_task_ids': '', 'activity_category': 'unknown',
        'activity_posture': 'unknown', 'activity_gesture_scenarios': '',
        'activity_session_numbers': '', 'activity_subject_ids': '',
        'activity_content_ids': '', 'activity_task_category_consistent': '',
        'activity_task_posture_consistent': '',
        'activity_scenario_vs_taskid': 'not_checkable',
        'activity_start_time_min': '',
    }
    if not path.is_file():
        return result, ['activity_missing']
    issues = []
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        sample = f.readline()
        if not sample.strip():
            result['activity_header_type'] = 'empty'
            return result, ['activity_empty']
        delimiter = choose_delimiter(sample)
        f.seek(0)
        rows = csv.reader(f, delimiter=delimiter)
        first = next(rows, None)
        if first is None:
            result['activity_header_type'] = 'empty'
            return result, ['activity_empty']
        normalized = [normalize_header(c) for c in first]
        headered = 'taskid' in normalized and ('sessionnumber' in normalized or 'id' in normalized)
        if headered:
            result['activity_header_type'] = 'headered'
            indexes = {ACTIVITY_HEADER.get(key): i for i, key in enumerate(normalized) if key in ACTIVITY_HEADER}
            required = ['ID', 'SubjectID', 'Session_number', 'Gesture_scenario', 'TaskID']
            missing = [c for c in required if c not in indexes]
            if missing:
                issues.append('activity_header_missing_fields')
            parsed_rows = rows
        else:
            result['activity_header_type'] = 'positional'
            indexes = {name: idx for idx, name in enumerate(ACTIVITY_COLUMNS)}
            parsed_rows = iter([first, *rows])  # Activity.csv is small (about 3 rows per session)
        taskids = []
        scenarios = []
        sessionnums = []
        subjects = []
        contents = []
        starts = []
        types = []
        postures = []
        observed_disagreements = 0
        bad_task = 0
        for raw in parsed_rows:
            if not raw or not any(str(x).strip() for x in raw):
                continue
            result['activity_row_count'] += 1
            if len(raw) < max(indexes.values(), default=-1) + 1:
                result['activity_invalid_rows'] += 1
                continue
            def val(name):
                idx = indexes.get(name)
                return raw[idx].strip() if idx is not None and idx < len(raw) else ''
            task = as_int(val('TaskID'))
            scen = as_int(val('Gesture_scenario'))
            sess = as_int(val('Session_number'))
            subject = str(val('SubjectID')).strip()
            content = as_int(val('ContentID'))
            start = val('Start_time')
            if task is None or task not in TASKS:
                bad_task += 1
            else:
                taskids.append(task)
                cat, posture = TASKS[task]
                types.append(cat)
                postures.append(posture)
                if scen in (1, 2) and posture != ('sitting' if scen == 1 else 'walking'):
                    observed_disagreements += 1
            if scen is not None:
                scenarios.append(scen)
            if sess is not None:
                sessionnums.append(sess)
            if subject:
                subjects.append(subject)
            if content is not None:
                contents.append(content)
            if start:
                starts.append(start)
        result['activity_task_ids'] = fmt_set(taskids)
        result['activity_gesture_scenarios'] = fmt_set(scenarios)
        result['activity_session_numbers'] = fmt_set(sessionnums)
        result['activity_subject_ids'] = fmt_set(subjects)
        result['activity_content_ids'] = fmt_set(contents)
        result['activity_start_time_min'] = min(starts) if starts else ''
        result['activity_category'] = next(iter(set(types))) if len(set(types)) == 1 else ('mixed' if len(set(types)) > 1 else 'unknown')
        result['activity_posture'] = next(iter(set(postures))) if len(set(postures)) == 1 else ('mixed' if len(set(postures)) > 1 else 'unknown')
        result['activity_task_category_consistent'] = int(len(set(types)) <= 1) if types else ''
        result['activity_task_posture_consistent'] = int(len(set(postures)) <= 1) if postures else ''
        result['activity_scenario_vs_taskid'] = ('mismatch' if observed_disagreements else
            'match' if taskids and scenarios else 'not_checkable')
        if bad_task:
            issues.append('activity_invalid_task_ids')
        if result['activity_invalid_rows']:
            issues.append('activity_malformed_rows')
        if result['activity_row_count'] == 0:
            issues.append('activity_empty')
        if result['activity_category'] == 'mixed' or result['activity_posture'] == 'mixed':
            issues.append('activity_mixed_taskid_types')
        if observed_disagreements:
            issues.append('activity_gesture_scenario_conflicts_with_taskid')
        if any(s not in (1, 2) for s in scenarios):
            issues.append('activity_unknown_gesture_scenario')
        if any(c not in (1, 2, 3) for c in contents):
            issues.append('activity_unexpected_content_id')
        if len(set(sessionnums)) > 1:
            issues.append('activity_inconsistent_session_number')
        if len(set(subjects)) > 1:
            issues.append('activity_inconsistent_subject_id')
    return result, issues


def detect_answers(session: Path, recursive: bool):
    marker_map = {'reading': defaultdict(list), 'writing': defaultdict(list)}
    files = session.rglob('*') if recursive else session.iterdir()
    for entry in files:
        if not entry.is_file():
            continue
        match = ANSWER_RE.fullmatch(entry.name)
        if match:
            cat, idx = match.group(1).lower(), int(match.group(2))
            marker_map[cat][idx].append(str(entry.relative_to(session)))
    reading = sum(len(v) for v in marker_map['reading'].values())
    writing = sum(len(v) for v in marker_map['writing'].values())
    if reading and writing:
        evidence = 'mixed'
    elif reading:
        evidence = 'reading'
    elif writing:
        evidence = 'writing'
    else:
        evidence = 'map_inferred'  # Negative evidence, not proof of map session
    return {
        'answer_evidence': evidence,
        'answer_reading_files': reading, 'answer_writing_files': writing,
        'answer_reading_indices': fmt_set(marker_map['reading'].keys()),
        'answer_writing_indices': fmt_set(marker_map['writing'].keys()),
        'answer_reading_names': ';'.join(sorted(v for group in marker_map['reading'].values() for v in group)),
        'answer_writing_names': ';'.join(sorted(v for group in marker_map['writing'].values() for v in group)),
    }


def session_identity(path: Path, root: Path):
    m = SESSION_RE.fullmatch(path.name)
    if not m:
        return None
    uid = m.group('uid')
    if uid is None:
        # Match parent directory name only; don't use some arbitrary numeric ancestor.
        if re.fullmatch(r'\d{6}', path.parent.name):
            uid = path.parent.name
        else:
            return None
    return uid, int(m.group('number'))


def audit_session(path: Path, root: Path, recursive: bool):
    identity = session_identity(path, root)
    if not identity:
        return None
    user, number = identity
    answers = detect_answers(path, recursive)
    activity, issues = parse_activity(path / 'Activity.csv')
    expected_cat, expected_posture = TASKS.get(number, ('unknown', 'unknown'))
    result = {
        'user_id': user, 'session_number': number,
        'session_directory': path.name,
        'session_path': str(path.resolve()),
        'directory_number_as_task_category': expected_cat,
        'directory_number_as_task_posture': expected_posture,
        **answers, **activity,
    }
    actual_cat = activity['activity_category']
    actual_post = activity['activity_posture']
    markers = answers['answer_evidence']
    if markers == 'mixed':
        result['activity_vs_answers'] = 'ambiguous_both_answer_types'
        result['directory_number_vs_answers'] = 'ambiguous_both_answer_types'
        issues.append('both_reading_and_writing_answer_files')
    else:
        files_cat = 'map' if markers == 'map_inferred' else markers
        if actual_cat in ('reading', 'writing', 'map'):
            result['activity_vs_answers'] = ('match' if actual_cat == files_cat else 'mismatch')
            if result['activity_vs_answers'] == 'mismatch':
                issues.append('activity_taskid_conflicts_with_answer_files')
        else:
            result['activity_vs_answers'] = 'not_checkable'
        result['directory_number_vs_answers'] = ('match' if expected_cat == files_cat else 'mismatch')
        if result['directory_number_vs_answers'] == 'mismatch':
            issues.append('folder_number_conflicts_with_answer_files')
    result['directory_number_vs_activity_taskid'] = (
        'match' if (expected_cat, expected_posture) == (actual_cat, actual_post) else
        'not_checkable' if actual_cat not in ('reading', 'writing', 'map') or actual_post not in ('sitting', 'walking') else
        'mismatch'
    )
    if result['directory_number_vs_activity_taskid'] == 'mismatch':
        issues.append('folder_number_differs_from_taskid')
    # Verify physical folder identity against Activity.csv metadata, not TaskID.
    report_sessions = [as_int(v) for v in activity['activity_session_numbers'].split(';') if v]
    report_subjects = activity['activity_subject_ids'].split(';') if activity['activity_subject_ids'] else []
    result['folder_number_vs_activity_session_number'] = (
        'match' if report_sessions == [number] else 'not_checkable' if not report_sessions else 'mismatch'
    )
    result['folder_user_vs_activity_subject_id'] = (
        'match' if report_subjects == [user] else 'not_checkable' if not report_subjects else 'mismatch'
    )
    if result['folder_number_vs_activity_session_number'] == 'mismatch':
        issues.append('folder_number_conflicts_with_activity_session_number')
    if result['folder_user_vs_activity_subject_id'] == 'mismatch':
        issues.append('folder_user_conflicts_with_activity_subject_id')
    # Presence of 1/2/3 is a quality signal, not a categorical requirement.
    if markers in ('reading', 'writing'):
        count = len(answers[f'answer_{markers}_indices'].split(';')) if answers[f'answer_{markers}_indices'] else 0
        if count < 3:
            issues.append('partial_answer_file_set')
    if markers == 'map_inferred':
        issues.append('map_inference_is_based_on_absence_only')
    result['structural_category'] = markers
    result['structural_posture'] = 'unobservable'
    result['issue_codes'] = ';'.join(sorted(set(issues)))
    result['issue_count'] = len(set(issues))
    result['critical_issue_count'] = sum(1 for issue in set(issues) if issue in {
        'activity_taskid_conflicts_with_answer_files',
        'both_reading_and_writing_answer_files',
        'activity_mixed_taskid_types',
        'activity_gesture_scenario_conflicts_with_taskid',
        'folder_number_conflicts_with_activity_session_number',
        'folder_user_conflicts_with_activity_subject_id',
        'activity_invalid_task_ids',
    })
    return result


def discover_sessions(root: Path):
    # User folders are exactly six digits; session folders are 123456_session_1.
    # Also support user/Session_1. Restrict walk to known user directories.
    users = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'\d{6}', p.name)]
    for user_dir in sorted(users, key=lambda p:p.name):
        for candidate in sorted(user_dir.iterdir(), key=lambda p:p.name):
            if candidate.is_dir() and session_identity(candidate,root):
                yield candidate


def write_tsv(path: Path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, delimiter='\t', fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_previous(path: Path):
    old = {}
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = str(row.get('user_id','')).strip()
            idx = as_int(row.get('session_number',''))
            if uid and idx is not None:
                old[(uid,idx)] = row
    return old


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent HMOG session type and answer-file audit (stdlib only).')
    parser.add_argument('--input', type=Path, default=Path('input_data/hmog_unarchived'))
    parser.add_argument('--output', type=Path, default=Path('output_data/hmog_session_type_audit'))
    parser.add_argument('--recursive-markers', action='store_true',
                        help='Also look for answer files in subdirectories of each session.')
    parser.add_argument('--previous-sessions', type=Path,
                        help='Optional sessions.csv from HMOG Raw/FETA v1.2, for classification comparison.')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f'Input directory does not exist: {args.input}')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in discover_sessions(args.input):
        row = audit_session(path, args.input, args.recursive_markers)
        if row:
            rows.append(row)
    rows.sort(key=lambda r:(r['user_id'],r['session_number']))
    if not rows:
        parser.error('No session directories found: expected USER_ID/USER_ID_session_N (or USER_ID/session_N).')
    output_cols = list(rows[0].keys())
    write_tsv(args.output/'all_sessions.tsv', rows, output_cols)
    mismatches = [r for r in rows if r['activity_vs_answers'] == 'mismatch' or
                  r['activity_vs_answers'] == 'ambiguous_both_answer_types']
    write_tsv(args.output/'activity_vs_answer_mismatches.tsv', mismatches, output_cols)
    folder_mismatch = [r for r in rows if r['directory_number_vs_answers'] == 'mismatch' or
                       r['directory_number_vs_activity_taskid'] == 'mismatch']
    write_tsv(args.output/'folder_number_mismatches.tsv', folder_mismatch, output_cols)
    serious = [r for r in rows if r['critical_issue_count'] > 0]
    write_tsv(args.output/'other_metadata_problems.tsv', serious, output_cols)

    # Per-user task availability and internal metadata consistency.
    by_user = defaultdict(list)
    for row in rows:
        by_user[row['user_id']].append(row)
    user_rows = []
    for uid, group in sorted(by_user.items()):
        cats = Counter(r['activity_category'] for r in group)
        evid = Counter(r['answer_evidence'] for r in group)
        tp = Counter((r['activity_category'],r['activity_posture']) for r in group)
        task_ids = Counter(int(t) for r in group for t in r['activity_task_ids'].split(';') if t)
        missing_taskids = [i for i in range(1,25) if i not in task_ids]
        duplicates = [i for i,n in task_ids.items() if n>1]
        user_rows.append({
            'user_id':uid, 'session_count':len(group),
            'reading_files_sessions': evid['reading'], 'writing_files_sessions':evid['writing'],
            'map_inferred_sessions':evid['map_inferred'], 'mixed_answer_sessions':evid['mixed'],
            'activity_reading_sessions':cats['reading'],
            'activity_writing_sessions':cats['writing'],
            'activity_map_sessions':cats['map'],
            'reading_sitting_taskid_sessions':tp[('reading','sitting')],
            'reading_walking_taskid_sessions':tp[('reading','walking')],
            'writing_sitting_taskid_sessions':tp[('writing','sitting')],
            'writing_walking_taskid_sessions':tp[('writing','walking')],
            'map_sitting_taskid_sessions':tp[('map','sitting')],
            'map_walking_taskid_sessions':tp[('map','walking')],
            'activity_vs_answers_mismatches':sum(r['activity_vs_answers']=='mismatch' for r in group),
            'folder_number_vs_activity_taskid_mismatches':sum(r['directory_number_vs_activity_taskid']=='mismatch' for r in group),
            'critical_issue_sessions':sum(r['critical_issue_count']>0 for r in group),
            'missing_task_ids':fmt_set(missing_taskids),
            'repeated_task_ids_across_directories':fmt_set(duplicates),
            'missing_session_folder_numbers':fmt_set([i for i in range(1,25) if i not in {r['session_number'] for r in group}]),
        })
    write_tsv(args.output/'per_user.tsv',user_rows,list(user_rows[0]))

    counts = Counter(r['answer_evidence'] for r in rows)
    av = Counter(r['activity_vs_answers'] for r in rows)
    dv = Counter(r['directory_number_vs_answers'] for r in rows)
    df = Counter(r['directory_number_vs_activity_taskid'] for r in rows)
    fc = Counter(r['folder_number_vs_activity_session_number'] for r in rows)
    task_counts = Counter((r['activity_category'],r['activity_posture']) for r in rows)
    summary = {
        'version':VERSION, 'generated_utc':datetime.now(timezone.utc).isoformat(),
        'input':str(args.input.resolve()),
        'users':len(by_user), 'session_directories':len(rows),
        'answer_file_evidence':dict(counts),
        'activity_task_id_vs_answer_evidence':dict(av),
        'folder_number_interpreted_as_task_id_vs_answer_evidence':dict(dv),
        'folder_number_interpreted_as_task_id_vs_actual_activity_task_id':dict(df),
        'folder_number_vs_activity_session_number':dict(fc),
        'actual_activity_task_counts':{f'{a}_{b}':n for (a,b),n in sorted(task_counts.items())},
        'sessions_with_critical_metadata_issues':len(serious),
        'activity_vs_answer_mismatches':len(mismatches),
        'folder_number_assumption_mismatches':len(folder_mismatch),
        'sessions_missing_activity_csv':sum(not r['activity_file_exists'] for r in rows),
        'sessions_without_any_answer_markers':counts['map_inferred'],
        'users_with_any_activity_answer_mismatch':sum(any(r['activity_vs_answers']=='mismatch' for r in g) for g in by_user.values()),
        'important_caveats':[
            'Session_number denotes a session number; TaskID denotes task type. Folder-number mismatch is not necessarily a data error.',
            'The HMOG poster describes randomized task assignment on user login.',
            'Writing/Reading answer files identify task CATEGORY, not sitting/walking.',
            'Absence of answer files is only a heuristic for map navigation; missing data is possible.',
            'Gesture_scenario in Activity.csv (1 sitting / 2 walking) provides non-sensor corroboration for TaskID posture.',
            'No motion signals or touch files are read.',
        ],
    }
    if args.previous_sessions:
        if not args.previous_sessions.is_file():
            parser.error(f'--previous-sessions not found: {args.previous_sessions}')
        old = load_previous(args.previous_sessions)
        comp = []
        for r in rows:
            prior = old.get((r['user_id'],r['session_number']))
            if prior is None:
                continue
            old_type = prior.get('task','').strip().lower()
            expected_type = f"{r['activity_category']}_{r['activity_posture']}"
            match = old_type == expected_type
            comp.append({
                'user_id':r['user_id'], 'session_number':r['session_number'],
                'previous_analyzer_task':old_type, 'activity_csv_task':expected_type,
                'file_evidence':r['answer_evidence'],
                'old_vs_activity': 'match' if match else 'mismatch',
                'activity_vs_answers':r['activity_vs_answers'],
            })
        write_tsv(args.output/'previous_analyzer_comparison.tsv', comp,
                  ['user_id','session_number','previous_analyzer_task','activity_csv_task',
                   'file_evidence','old_vs_activity','activity_vs_answers'])
        summary['previous_analyzer_comparison'] = {
            'sessions_joined':len(comp),
            'classification_disagreements':sum(r['old_vs_activity']=='mismatch' for r in comp),
        }
    with (args.output/'summary.json').open('w',encoding='utf-8') as fh:
        json.dump(summary,fh,ensure_ascii=False,indent=2)
    print(f"Users: {summary['users']}, session directories: {summary['session_directories']}")
    print(f'Answer evidence: {dict(counts)}')
    print(f'Activity.TaskID vs answer files: {dict(av)}')
    print(f'Folder number vs TaskID (ASSUMPTION ONLY): {dict(df)}')
    print(f'Sessions with substantive Activity.csv/answer conflicts: {len(mismatches)}')
    print(f'Sessions with other critical metadata issues: {len(serious)}')
    print(f'Reports written to: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
