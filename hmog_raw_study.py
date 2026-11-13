#!/usr/bin/env python3
"""Explore the raw HMOG dataset with pandas.

The script is intentionally split into reusable functions so it can be imported
from a notebook as well as executed from the command line.

Typical command-line use
------------------------
python hmog_raw_study.py \
    --data-root /path/to/hmog_dataset \
    --output-dir ./hmog_raw_stats

Typical notebook use
--------------------
from pathlib import Path
from hmog_raw_study import (
    discover_sessions,
    load_session_data,
    build_dataset_statistics,
)

root = Path("/path/to/hmog_dataset")
sessions = discover_sessions(root)

# Full raw pandas DataFrames for one session:
raw = load_session_data(sessions.iloc[0].session_dir)
raw["accelerometer"].head()
raw["touch_event"].head()

# Dataset-level summary DataFrames:
stats = build_dataset_statistics(root)
stats["sessions"].head()
stats["activities"].head()
stats["writing"].head()

Notes
-----
* HMOG is large. `load_session_data()` loads one session at a time so you do
  not need to hold the whole dataset in RAM.
* `build_dataset_statistics()` also processes one session at a time and reads
  only the columns needed for counts wherever possible.
* The loader accepts CSV files with or without a header row.
* The data-description document spells the gyroscope filename
  "Gyroscrope.csv". This script accepts both that spelling and
  "Gyroscope.csv".
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Raw HMOG table schemas from the supplied data-description document.
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, list[str]] = {
    "activity": [
        "ID",
        "SubjectID",
        "Session_number",
        "Start_time",
        "End_time",
        "Relative_Start_time",
        "Relative_End_time",
        "Gesture_scenario",
        "TaskID",
        "ContentID",
    ],
    "accelerometer": [
        "Systime",
        "EventTime",
        "ActivityID",
        "X",
        "Y",
        "Z",
        "Phone_orientation",
    ],
    "gyroscope": [
        "Systime",
        "EventTime",
        "ActivityID",
        "X",
        "Y",
        "Z",
        "Phone_orientation",
    ],
    "magnetometer": [
        "Systime",
        "EventTime",
        "ActivityID",
        "X",
        "Y",
        "Z",
        "Phone_orientation",
    ],
    "touch_event": [
        "Systime",
        "EventTime",
        "ActivityID",
        "Pointer_count",
        "PointerID",
        "ActionID",
        "X",
        "Y",
        "Pressure",
        "Contact_size",
        "Phone_orientation",
    ],
    "key_press_event": [
        "Systime",
        "PressTime",
        "ActivityID",
        "PressType",
        "KeyID",
        "Phone_orientation",
    ],
    "one_finger_touch_event": [
        "Systime",
        "PressTime",
        "ActivityID",
        "TapID",
        "Tap_type",
        "Action_type",
        "X",
        "Y",
        "Pressure",
        "Contact_size",
        "Phone_orientation",
    ],
    "pinch_event": [
        "Systime",
        "PressTime",
        "ActivityID",
        "EventType",
        "PinchID",
        "Time_delta",
        "Focus_X",
        "Focus_Y",
        "Span",
        "Span_X",
        "Span_Y",
        "ScaleFactor",
        "Phone_orientation",
    ],
    "scroll_event": [
        "Systime",
        "BeginTime",
        "CurrentTime",
        "ActivityID",
        "ScrollID",
        "Start_action_type",
        "Start_X",
        "Start_Y",
        "Start_pressure",
        "Start_size",
        "Current_action_type",
        "Current_X",
        "Current_Y",
        "Current_pressure",
        "Current_size",
        "Distance_X",
        "Distance_Y",
        "Phone_orientation",
    ],
    "stroke_event": [
        "Systime",
        "Begin_time",
        "End_time",
        "ActivityID",
        "Start_action_type",
        "Start_X",
        "Start_Y",
        "Start_pressure",
        "Start_size",
        "End_action_type",
        "End_X",
        "End_Y",
        "End_pressure",
        "End_size",
        "Speed_X",
        "Speed_Y",
        "Phone_orientation",
    ],
}

TABLE_FILENAMES: dict[str, tuple[str, ...]] = {
    "activity": ("Activity.csv",),
    "accelerometer": ("Accelerometer.csv",),
    "gyroscope": ("Gyroscope.csv", "Gyroscrope.csv"),
    "magnetometer": ("Magnetometer.csv",),
    "touch_event": ("TouchEvent.csv",),
    "key_press_event": ("KeyPressEvent.csv",),
    "one_finger_touch_event": ("OneFingerTouchEvent.csv",),
    "pinch_event": ("PinchEvent.csv",),
    "scroll_event": ("ScrollEvent.csv",),
    "stroke_event": ("StrokeEvent.csv",),
}

# Minimal columns used by the dataset-wide statistics pass.  This makes the
# pass much lighter than loading X/Y/Z etc. for every session.
STAT_COLUMNS: dict[str, list[str]] = {
    "activity": TABLE_SCHEMAS["activity"],
    "accelerometer": ["ActivityID"],
    "gyroscope": ["ActivityID"],
    "magnetometer": ["ActivityID"],
    "touch_event": ["ActivityID"],
    "key_press_event": ["ActivityID", "PressType"],
    "one_finger_touch_event": ["ActivityID", "TapID"],
    "pinch_event": ["ActivityID", "PinchID"],
    "scroll_event": ["ActivityID", "ScrollID"],
    "stroke_event": ["ActivityID"],
}

# Event files are excluded from the small-file scan used to find writing
# answers.
RAW_EVENT_FILENAMES = {
    name.lower()
    for names in TABLE_FILENAMES.values()
    for name in names
}


# ---------------------------------------------------------------------------
# Generic CSV loading
# ---------------------------------------------------------------------------


def _norm_name(value: object) -> str:
    """Normalize a column/file label for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_first_csv_row(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        line = f.readline()
    if not line:
        return []
    try:
        return next(csv.reader([line]))
    except Exception:
        return [part.strip() for part in line.split(",")]


def _looks_like_header(first_row: Sequence[str], schema: Sequence[str]) -> bool:
    if not first_row:
        return False
    schema_norm = {_norm_name(c) for c in schema}
    row_norm = {_norm_name(c) for c in first_row}
    overlap = len(schema_norm & row_norm)
    return overlap >= 2 or _norm_name(first_row[0]) == _norm_name(schema[0])


def read_hmog_csv(
    path: str | Path,
    table: str,
    *,
    usecols: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read one HMOG CSV into a pandas DataFrame.

    This is the main reusable raw-data loader.  It handles both headered and
    headerless HMOG CSV files and returns the canonical column names used in
    TABLE_SCHEMAS.

    Parameters
    ----------
    path:
        Path to a raw HMOG CSV.
    table:
        Canonical table key, e.g. "accelerometer" or "touch_event".
    usecols:
        Optional canonical column names to load.  Useful for reducing memory.
    nrows:
        Optional pandas `nrows`, useful during interactive study.
    """
    path = Path(path)
    if table not in TABLE_SCHEMAS:
        raise KeyError(f"Unknown HMOG table {table!r}")
    schema = TABLE_SCHEMAS[table]

    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=list(usecols or schema))

    first = _read_first_csv_row(path)
    has_header = _looks_like_header(first, schema)
    wanted = None if usecols is None else list(usecols)
    wanted_norm = None if wanted is None else {_norm_name(c) for c in wanted}

    if has_header:
        # Filter at parse time so large sensor CSVs do not load unnecessary
        # columns during the statistics pass.
        raw = pd.read_csv(
            path,
            header=0,
            nrows=nrows,
            usecols=(
                None
                if wanted_norm is None
                else lambda c: _norm_name(c) in wanted_norm
            ),
            encoding="utf-8-sig",
            low_memory=True,
        )

        canonical_by_norm = {_norm_name(c): c for c in schema}
        raw = raw.rename(
            columns={
                c: canonical_by_norm.get(_norm_name(c), str(c).strip())
                for c in raw.columns
            }
        )
    else:
        n_fields = len(first)
        names = list(schema[:n_fields])
        if n_fields > len(schema):
            names.extend(
                f"_extra_{i}" for i in range(n_fields - len(schema))
            )

        selected = None
        if wanted is not None:
            selected = [c for c in wanted if c in names]

        raw = pd.read_csv(
            path,
            header=None,
            names=names,
            usecols=selected,
            nrows=nrows,
            encoding="utf-8-sig",
            low_memory=True,
        )

    # Preserve the requested canonical column order and add missing columns as
    # NA instead of failing on a partially populated file.
    if wanted is not None:
        for c in wanted:
            if c not in raw.columns:
                raw[c] = pd.NA
        raw = raw[wanted]

    return raw


def _session_file_map(session_dir: str | Path) -> dict[str, Path]:
    """Map lowercase filenames to paths for one session directory."""
    session_dir = Path(session_dir)
    return {
        p.name.lower(): p
        for p in session_dir.iterdir()
        if p.is_file()
    }


def find_table_file(session_dir: str | Path, table: str) -> Optional[Path]:
    file_map = _session_file_map(session_dir)
    for expected in TABLE_FILENAMES[table]:
        match = file_map.get(expected.lower())
        if match is not None:
            return match
    return None


def load_session_data(
    session_dir: str | Path,
    *,
    tables: Optional[Iterable[str]] = None,
    usecols: Optional[Mapping[str, Sequence[str]]] = None,
    nrows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Build pandas DataFrames for the raw CSV files in one HMOG session.

    Missing files are represented by empty DataFrames with the expected
    columns.  The returned dictionary is convenient for notebook work:

        raw = load_session_data(session_dir)
        raw["accelerometer"]
        raw["touch_event"]
    """
    session_dir = Path(session_dir)
    requested = list(tables or TABLE_SCHEMAS.keys())
    result: dict[str, pd.DataFrame] = {}

    for table in requested:
        path = find_table_file(session_dir, table)
        cols = None if usecols is None else usecols.get(table)
        if path is None:
            result[table] = pd.DataFrame(columns=list(cols or TABLE_SCHEMAS[table]))
        else:
            result[table] = read_hmog_csv(
                path,
                table,
                usecols=cols,
                nrows=nrows,
            )
    return result


def load_session_data_for_stats(session_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load only the columns needed to calculate the requested statistics."""
    return load_session_data(session_dir, usecols=STAT_COLUMNS)


def build_dataset_table(
    data_root: str | Path,
    table: str,
    *,
    subject_ids: Optional[Sequence[int]] = None,
    session_numbers: Optional[Sequence[int]] = None,
    usecols: Optional[Sequence[str]] = None,
    nrows_per_session: Optional[int] = None,
    add_session_metadata: bool = True,
) -> pd.DataFrame:
    """Concatenate one raw HMOG table across selected sessions.

    This is useful for interactive study when you want one pandas DataFrame
    for a specific modality, for example all TouchEvent rows for one user.

    Be careful with motion sensors: concatenating accelerometer/gyroscope/
    magnetometer data for the entire 100-user dataset can require a very large
    amount of RAM. Prefer selecting users/sessions or use
    `build_dataset_statistics()` for descriptive counts.
    """
    if table not in TABLE_SCHEMAS:
        raise KeyError(f"Unknown HMOG table {table!r}")

    sessions = discover_sessions(data_root)
    if subject_ids is not None:
        sessions = sessions[sessions["subject_id"].isin(list(subject_ids))]
    if session_numbers is not None:
        sessions = sessions[sessions["session_number"].isin(list(session_numbers))]

    frames: list[pd.DataFrame] = []
    for _, meta in sessions.iterrows():
        path = find_table_file(meta["session_dir"], table)
        if path is None:
            continue
        frame = read_hmog_csv(
            path,
            table,
            usecols=usecols,
            nrows=nrows_per_session,
        )
        if add_session_metadata:
            frame.insert(0, "session_number", meta.get("session_number"))
            frame.insert(0, "subject_id", meta.get("subject_id"))
            frame["session_dir"] = meta.get("session_dir")
        frames.append(frame)

    if not frames:
        base_cols = list(usecols or TABLE_SCHEMAS[table])
        if add_session_metadata:
            base_cols = ["subject_id", "session_number"] + base_cols + ["session_dir"]
        return pd.DataFrame(columns=base_cols)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Session discovery and metadata
# ---------------------------------------------------------------------------


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _first_valid(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def _infer_epoch_unit(values: pd.Series) -> Optional[str]:
    """Infer s/ms/us/ns for numeric Unix-like absolute timestamps."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    if numeric.empty:
        return None
    magnitude = float(numeric.median())
    if magnitude >= 1e17:
        return "ns"
    if magnitude >= 1e14:
        return "us"
    if magnitude >= 1e11:
        return "ms"
    if magnitude >= 1e8:
        return "s"
    return None


def to_datetime_hmog(values: pd.Series) -> pd.Series:
    """Convert HMOG absolute timestamps to pandas datetimes when possible."""
    unit = _infer_epoch_unit(values)
    if unit is not None:
        return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit=unit, errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def _duration_seconds(start: pd.Series, end: pd.Series) -> pd.Series:
    """Duration from numeric HMOG timestamps, with automatic unit inference."""
    start_num = pd.to_numeric(start, errors="coerce")
    end_num = pd.to_numeric(end, errors="coerce")
    unit = _infer_epoch_unit(pd.concat([start, end], ignore_index=True))
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(unit)
    if scale is not None:
        return (end_num - start_num) / scale

    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")
    return (end_dt - start_dt).dt.total_seconds()


def _task_from_code(code: object) -> str:
    try:
        c = int(float(code))
    except Exception:
        return "unknown"
    slot = ((c - 1) % 6) + 1
    if slot in (1, 2):
        return "reading"
    if slot in (3, 4):
        return "writing"
    if slot in (5, 6):
        return "map"
    return "unknown"


def _condition_from_activity(activity: pd.DataFrame) -> str:
    if "Gesture_scenario" in activity.columns:
        scenario = _first_valid(_numeric(activity["Gesture_scenario"]))
        if pd.notna(scenario):
            if int(scenario) == 1:
                return "sitting"
            if int(scenario) == 2:
                return "walking"

    # Fallback to the TaskID/session six-session cycle described in the data
    # documentation.
    code = None
    for col in ("TaskID", "Session_number"):
        if col in activity.columns:
            code = _first_valid(_numeric(activity[col]))
            if pd.notna(code):
                break
    if code is not None and pd.notna(code):
        slot = ((int(code) - 1) % 6) + 1
        return "sitting" if slot in (1, 3, 5) else "walking"
    return "unknown"


def _task_from_activity(activity: pd.DataFrame) -> str:
    for col in ("TaskID", "Session_number"):
        if col in activity.columns:
            code = _first_valid(_numeric(activity[col]))
            if pd.notna(code):
                return _task_from_code(code)
    return "unknown"


def discover_sessions(data_root: str | Path) -> pd.DataFrame:
    """Find HMOG sessions by recursively locating Activity.csv files.

    Returns one row per session, including the parsed subject ID and session
    number where possible.  No assumption is made about folder names.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    activity_paths = sorted(
        p for p in data_root.rglob("*")
        if p.is_file() and p.name.lower() == "activity.csv"
    )

    rows: list[dict] = []
    for path in activity_paths:
        try:
            activity = read_hmog_csv(path, "activity")
        except Exception as exc:
            print(f"WARNING: could not parse {path}: {exc}")
            continue
        if activity.empty:
            continue

        subject = _first_valid(_numeric(activity.get("SubjectID", pd.Series(dtype=float))))
        session_number = _first_valid(_numeric(activity.get("Session_number", pd.Series(dtype=float))))
        task_id = _first_valid(_numeric(activity.get("TaskID", pd.Series(dtype=float))))

        starts = to_datetime_hmog(activity["Start_time"]) if "Start_time" in activity else pd.Series(dtype="datetime64[ns]")
        ends = to_datetime_hmog(activity["End_time"]) if "End_time" in activity else pd.Series(dtype="datetime64[ns]")

        rows.append(
            {
                "subject_id": int(subject) if pd.notna(subject) else pd.NA,
                "session_number": int(session_number) if pd.notna(session_number) else pd.NA,
                "task_id": int(task_id) if pd.notna(task_id) else pd.NA,
                "task": _task_from_activity(activity),
                "condition": _condition_from_activity(activity),
                "session_dir": str(path.parent.resolve()),
                "activity_csv": str(path.resolve()),
                "session_start": starts.min() if len(starts) else pd.NaT,
                "session_end": ends.max() if len(ends) else pd.NaT,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["subject_id", "session_start", "session_number"],
            na_position="last",
        ).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Writing-answer extraction
# ---------------------------------------------------------------------------


WRITING_KEYS = {1: "Writing1answer", 2: "Writing2answer", 3: "Writing3answer"}


def _safe_read_small_text(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return None


def _extract_writing_columns_from_csv(path: Path) -> dict[int, str]:
    found: dict[int, str] = {}
    try:
        sample = pd.read_csv(path, nrows=20, encoding="utf-8-sig")
    except Exception:
        return found

    normalized = {_norm_name(c): c for c in sample.columns}
    for idx, key in WRITING_KEYS.items():
        col = normalized.get(_norm_name(key))
        if col is None:
            continue
        values = sample[col].dropna().astype(str)
        if len(values):
            found[idx] = "\n".join(values.tolist()).strip()
    return found


def load_writing_answers(session_dir: str | Path) -> dict[int, str]:
    """Find the final writing answers for ContentID 1, 2 and 3.

    The data description names the fields Writing1answer, Writing2answer and
    Writing3answer but does not prescribe their physical filename.  Therefore
    this function intentionally supports several layouts:

    1. a file named Writing1answer / Writing1answer.txt / etc.;
    2. a small CSV with columns Writing1answer..Writing3answer;
    3. a small text/CSV file containing lines such as
       `Writing1answer=<text>` or `Writing1answer,<text>`.
    """
    session_dir = Path(session_dir)
    answers: dict[int, str] = {}

    candidates = [
        p for p in session_dir.rglob("*")
        if p.is_file()
        and p.name.lower() not in RAW_EVENT_FILENAMES
        and p.stat().st_size <= 2_000_000
    ]

    # 1. Filename itself identifies the answer.
    for path in candidates:
        stem_norm = _norm_name(path.stem)
        name_norm = _norm_name(path.name)
        for idx, key in WRITING_KEYS.items():
            key_norm = _norm_name(key)
            if stem_norm == key_norm or name_norm == key_norm:
                text = _safe_read_small_text(path)
                if text is not None:
                    answers[idx] = text.strip()

    # 2. Headered CSV columns.
    for path in candidates:
        if path.suffix.lower() == ".csv":
            for idx, text in _extract_writing_columns_from_csv(path).items():
                answers.setdefault(idx, text)

    # 3. Key-value lines in small text-like files.
    pattern = re.compile(
        r"(?im)^\s*Writing([123])answer\s*[:=,\t]\s*(.*)$"
    )
    for path in candidates:
        text = _safe_read_small_text(path)
        if text is None:
            continue
        for match in pattern.finditer(text):
            idx = int(match.group(1))
            answers.setdefault(idx, match.group(2).strip())

    return answers


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------


def _count_rows_by_activity(df: pd.DataFrame) -> dict[object, int]:
    if df.empty or "ActivityID" not in df:
        return {}
    ids = df["ActivityID"].dropna()
    if ids.empty:
        return {}
    return ids.value_counts(dropna=False).astype(int).to_dict()


def _unique_gesture_pairs(df: pd.DataFrame, id_col: str) -> int:
    if df.empty or id_col not in df:
        return 0
    work = df.copy()
    ids = pd.to_numeric(work[id_col], errors="coerce")
    work = work.loc[ids >= 0]
    if work.empty:
        return 0
    if "ActivityID" in work:
        return int(work[["ActivityID", id_col]].drop_duplicates().shape[0])
    return int(work[id_col].nunique(dropna=True))


def _unique_gesture_pairs_by_activity(df: pd.DataFrame, id_col: str) -> dict[object, int]:
    if df.empty or "ActivityID" not in df or id_col not in df:
        return {}
    work = df.copy()
    ids = pd.to_numeric(work[id_col], errors="coerce")
    work = work.loc[ids >= 0, ["ActivityID", id_col]].dropna()
    if work.empty:
        return {}
    return (
        work.drop_duplicates()
        .groupby("ActivityID")
        .size()
        .astype(int)
        .to_dict()
    )


def _keyboard_counts(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty or "PressType" not in df:
        return 0, 0
    press_type = pd.to_numeric(df["PressType"], errors="coerce")
    return int((press_type == 0).sum()), int((press_type == 1).sum())


def _keyboard_counts_by_activity(df: pd.DataFrame) -> tuple[dict[object, int], dict[object, int]]:
    if df.empty or "ActivityID" not in df or "PressType" not in df:
        return {}, {}
    work = df[["ActivityID", "PressType"]].copy()
    work["PressType"] = pd.to_numeric(work["PressType"], errors="coerce")
    down = work.loc[work["PressType"] == 0].groupby("ActivityID").size().astype(int).to_dict()
    up = work.loc[work["PressType"] == 1].groupby("ActivityID").size().astype(int).to_dict()
    return down, up


def _text_stats(text: object) -> tuple[int, int, int]:
    if text is None or pd.isna(text):
        return 0, 0, 0
    value = str(text)
    return len(value), sum(not c.isspace() for c in value), len(value.split())


# ---------------------------------------------------------------------------
# Statistics for one session
# ---------------------------------------------------------------------------


def build_activity_statistics(
    raw: Mapping[str, pd.DataFrame],
    *,
    session_dir: str | Path,
    writing_answers: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    """Create one statistics row per Activity.csv row."""
    activity = raw["activity"].copy()
    if activity.empty:
        return pd.DataFrame()

    writing_answers = dict(writing_answers or {})

    # Normalize metadata to numeric where appropriate.
    for col in [
        "SubjectID", "Session_number", "Start_time", "End_time",
        "Relative_Start_time", "Relative_End_time", "Gesture_scenario",
        "TaskID", "ContentID",
    ]:
        if col in activity:
            activity[col] = pd.to_numeric(activity[col], errors="coerce")

    starts_dt = to_datetime_hmog(activity["Start_time"])
    ends_dt = to_datetime_hmog(activity["End_time"])
    durations = _duration_seconds(activity["Start_time"], activity["End_time"])

    row_counts = {
        table: _count_rows_by_activity(raw[table])
        for table in [
            "accelerometer", "gyroscope", "magnetometer", "touch_event",
            "one_finger_touch_event", "pinch_event", "scroll_event",
            "stroke_event",
        ]
    }
    key_down, key_up = _keyboard_counts_by_activity(raw["key_press_event"])
    taps = _unique_gesture_pairs_by_activity(raw["one_finger_touch_event"], "TapID")
    pinches = _unique_gesture_pairs_by_activity(raw["pinch_event"], "PinchID")
    scrolls = _unique_gesture_pairs_by_activity(raw["scroll_event"], "ScrollID")

    rows: list[dict] = []
    session_task = _task_from_activity(activity)
    session_condition = _condition_from_activity(activity)

    for pos, (_, a) in enumerate(activity.iterrows()):
        activity_id = a.get("ID")
        content_id = a.get("ContentID")
        try:
            content_int = int(content_id) if pd.notna(content_id) else None
        except Exception:
            content_int = None

        text = writing_answers.get(content_int, "") if session_task == "writing" else ""
        chars, chars_no_ws, words = _text_stats(text)

        accel_n = int(row_counts["accelerometer"].get(activity_id, 0))
        gyro_n = int(row_counts["gyroscope"].get(activity_id, 0))
        mag_n = int(row_counts["magnetometer"].get(activity_id, 0))
        tap_rows = int(row_counts["one_finger_touch_event"].get(activity_id, 0))
        pinch_rows = int(row_counts["pinch_event"].get(activity_id, 0))
        scroll_rows = int(row_counts["scroll_event"].get(activity_id, 0))
        stroke_rows = int(row_counts["stroke_event"].get(activity_id, 0))
        distinct_taps = int(taps.get(activity_id, 0))
        distinct_pinches = int(pinches.get(activity_id, 0))
        distinct_scrolls = int(scrolls.get(activity_id, 0))

        rows.append(
            {
                "subject_id": int(a["SubjectID"]) if pd.notna(a.get("SubjectID")) else pd.NA,
                "session_number": int(a["Session_number"]) if pd.notna(a.get("Session_number")) else pd.NA,
                "task_id": int(a["TaskID"]) if pd.notna(a.get("TaskID")) else pd.NA,
                "task": session_task,
                "condition": session_condition,
                "activity_id": activity_id,
                "content_id": content_int,
                "activity_start": starts_dt.iloc[pos],
                "activity_end": ends_dt.iloc[pos],
                "activity_duration_seconds": float(durations.iloc[pos]) if pd.notna(durations.iloc[pos]) else pd.NA,
                "activity_duration_minutes": float(durations.iloc[pos]) / 60.0 if pd.notna(durations.iloc[pos]) else pd.NA,
                "written_text": text,
                "written_text_chars": chars,
                "written_text_chars_no_whitespace": chars_no_ws,
                "written_text_words": words,
                # Key-down events are useful as a measure of keyboard input,
                # but they are NOT guaranteed to equal final text characters
                # because delete, shift, autocomplete, etc. may be involved.
                "key_down_events": int(key_down.get(activity_id, 0)),
                "key_up_events": int(key_up.get(activity_id, 0)),
                "raw_touch_rows": int(row_counts["touch_event"].get(activity_id, 0)),
                "one_finger_gesture_rows": tap_rows,
                "distinct_taps": distinct_taps,
                "pinch_gesture_rows": pinch_rows,
                "distinct_pinches": distinct_pinches,
                "scroll_gesture_rows": scroll_rows,
                "distinct_scrolls": distinct_scrolls,
                "stroke_gesture_rows": stroke_rows,
                "gesture_event_rows_total": tap_rows + pinch_rows + scroll_rows + stroke_rows,
                "distinct_gestures_total": distinct_taps + distinct_pinches + distinct_scrolls + stroke_rows,
                "accelerometer_rows": accel_n,
                "gyroscope_rows": gyro_n,
                "magnetometer_rows": mag_n,
                "motion_sensor_rows_total": accel_n + gyro_n + mag_n,
                "session_dir": str(Path(session_dir).resolve()),
            }
        )

    return pd.DataFrame(rows)


def build_session_statistics(
    raw: Mapping[str, pd.DataFrame],
    *,
    session_dir: str | Path,
    activities: pd.DataFrame,
    writing_answers: Optional[Mapping[int, str]] = None,
) -> dict:
    """Create one session-level summary row."""
    activity = raw["activity"]
    writing_answers = dict(writing_answers or {})

    subject = _first_valid(_numeric(activity.get("SubjectID", pd.Series(dtype=float))))
    session_number = _first_valid(_numeric(activity.get("Session_number", pd.Series(dtype=float))))
    task_id = _first_valid(_numeric(activity.get("TaskID", pd.Series(dtype=float))))

    starts = to_datetime_hmog(activity["Start_time"])
    ends = to_datetime_hmog(activity["End_time"])
    session_start = starts.min()
    session_end = ends.max()
    session_span_seconds = (
        (session_end - session_start).total_seconds()
        if pd.notna(session_start) and pd.notna(session_end)
        else pd.NA
    )

    active_duration_seconds = pd.to_numeric(
        activities.get("activity_duration_seconds", pd.Series(dtype=float)),
        errors="coerce",
    ).sum(min_count=1)

    final_texts = [writing_answers.get(i, "") for i in (1, 2, 3)]
    combined_text = "\n".join(t for t in final_texts if t)
    chars, chars_no_ws, words = _text_stats(combined_text)

    key_down, key_up = _keyboard_counts(raw["key_press_event"])
    distinct_taps = _unique_gesture_pairs(raw["one_finger_touch_event"], "TapID")
    distinct_pinches = _unique_gesture_pairs(raw["pinch_event"], "PinchID")
    distinct_scrolls = _unique_gesture_pairs(raw["scroll_event"], "ScrollID")
    stroke_rows = len(raw["stroke_event"])

    accel_n = len(raw["accelerometer"])
    gyro_n = len(raw["gyroscope"])
    mag_n = len(raw["magnetometer"])
    tap_rows = len(raw["one_finger_touch_event"])
    pinch_rows = len(raw["pinch_event"])
    scroll_rows = len(raw["scroll_event"])

    return {
        "subject_id": int(subject) if pd.notna(subject) else pd.NA,
        "session_number": int(session_number) if pd.notna(session_number) else pd.NA,
        "task_id": int(task_id) if pd.notna(task_id) else pd.NA,
        "task": _task_from_activity(activity),
        "condition": _condition_from_activity(activity),
        "session_start": session_start,
        "session_end": session_end,
        "session_span_seconds": session_span_seconds,
        "session_span_minutes": (session_span_seconds / 60.0) if pd.notna(session_span_seconds) else pd.NA,
        "active_duration_seconds": active_duration_seconds,
        "active_duration_minutes": (active_duration_seconds / 60.0) if pd.notna(active_duration_seconds) else pd.NA,
        "number_of_activities": len(activity),
        "written_text_combined": combined_text,
        "written_text_chars": chars,
        "written_text_chars_no_whitespace": chars_no_ws,
        "written_text_words": words,
        "key_down_events": key_down,
        "key_up_events": key_up,
        "raw_touch_rows": len(raw["touch_event"]),
        "one_finger_gesture_rows": tap_rows,
        "distinct_taps": distinct_taps,
        "pinch_gesture_rows": pinch_rows,
        "distinct_pinches": distinct_pinches,
        "scroll_gesture_rows": scroll_rows,
        "distinct_scrolls": distinct_scrolls,
        "stroke_gesture_rows": stroke_rows,
        "gesture_event_rows_total": tap_rows + pinch_rows + scroll_rows + stroke_rows,
        "distinct_gestures_total": distinct_taps + distinct_pinches + distinct_scrolls + stroke_rows,
        "accelerometer_rows": accel_n,
        "gyroscope_rows": gyro_n,
        "magnetometer_rows": mag_n,
        "motion_sensor_rows_total": accel_n + gyro_n + mag_n,
        "session_dir": str(Path(session_dir).resolve()),
    }


def add_inter_session_intervals(sessions: pd.DataFrame) -> pd.DataFrame:
    """Add the time from each user's previous session to the next session."""
    if sessions.empty:
        return sessions.copy()

    out = sessions.copy()
    out = out.sort_values(
        ["subject_id", "session_start", "session_number"],
        na_position="last",
    ).reset_index(drop=True)

    out["previous_session_end"] = out.groupby("subject_id", dropna=False)["session_end"].shift(1)
    delta = out["session_start"] - out["previous_session_end"]
    out["gap_from_previous_session_seconds"] = delta.dt.total_seconds()
    out["gap_from_previous_session_minutes"] = out["gap_from_previous_session_seconds"] / 60.0
    out["gap_from_previous_session_hours"] = out["gap_from_previous_session_seconds"] / 3600.0
    out["gap_from_previous_session_days"] = out["gap_from_previous_session_seconds"] / 86400.0
    return out


# ---------------------------------------------------------------------------
# Whole-dataset pass
# ---------------------------------------------------------------------------


def build_dataset_statistics(
    data_root: str | Path,
    *,
    subject_ids: Optional[Sequence[int]] = None,
    session_numbers: Optional[Sequence[int]] = None,
    max_sessions: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the main study DataFrames requested for the HMOG raw data.

    Returns a dictionary with:
      * discovered_sessions  - locations found from Activity.csv
      * sessions             - one row per session, including inter-session gaps
      * activities           - one row per sub-activity/content item
      * writing              - writing activities only, including final text
      * data_volume          - count-focused view of each session
    """
    discovered = discover_sessions(data_root)

    selected = discovered.copy()
    if subject_ids is not None:
        selected = selected[selected["subject_id"].isin(list(subject_ids))]
    if session_numbers is not None:
        selected = selected[selected["session_number"].isin(list(session_numbers))]
    if max_sessions is not None:
        selected = selected.head(int(max_sessions))
    selected = selected.reset_index(drop=True)

    session_rows: list[dict] = []
    activity_frames: list[pd.DataFrame] = []

    total = len(selected)
    for i, meta in selected.iterrows():
        session_dir = Path(meta["session_dir"])
        if verbose:
            label = f"subject={meta.get('subject_id')} session={meta.get('session_number')}"
            print(f"[{i + 1}/{total}] {label}  {session_dir}")

        try:
            raw = load_session_data_for_stats(session_dir)
            answers = load_writing_answers(session_dir)
            activities = build_activity_statistics(
                raw,
                session_dir=session_dir,
                writing_answers=answers,
            )
            session_row = build_session_statistics(
                raw,
                session_dir=session_dir,
                activities=activities,
                writing_answers=answers,
            )
        except Exception as exc:
            print(f"WARNING: failed session {session_dir}: {exc}")
            continue

        activity_frames.append(activities)
        session_rows.append(session_row)

    sessions = pd.DataFrame(session_rows)
    sessions = add_inter_session_intervals(sessions)

    activities = (
        pd.concat(activity_frames, ignore_index=True)
        if activity_frames
        else pd.DataFrame()
    )

    writing = (
        activities.loc[activities["task"] == "writing"].copy()
        if not activities.empty and "task" in activities
        else pd.DataFrame()
    )

    volume_columns = [
        "subject_id", "session_number", "task", "condition",
        "raw_touch_rows", "one_finger_gesture_rows", "distinct_taps",
        "pinch_gesture_rows", "distinct_pinches",
        "scroll_gesture_rows", "distinct_scrolls", "stroke_gesture_rows",
        "gesture_event_rows_total", "distinct_gestures_total",
        "accelerometer_rows", "gyroscope_rows", "magnetometer_rows",
        "motion_sensor_rows_total", "key_down_events", "key_up_events",
        "session_dir",
    ]
    data_volume = (
        sessions[[c for c in volume_columns if c in sessions.columns]].copy()
        if not sessions.empty
        else pd.DataFrame()
    )

    return {
        "discovered_sessions": selected,
        "sessions": sessions,
        "activities": activities,
        "writing": writing,
        "data_volume": data_volume,
    }


def save_statistics(stats: Mapping[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in stats.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"saved {path}  ({len(frame):,} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: Optional[str]) -> Optional[list[int]]:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pandas-based descriptive statistics from the raw HMOG dataset."
    )
    parser.add_argument("--data-root", required=True, help="Root of the extracted HMOG dataset")
    parser.add_argument(
        "--output-dir",
        default="hmog_raw_stats",
        help="Directory for generated CSV summaries (default: hmog_raw_stats)",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject IDs, e.g. 100669,100701",
    )
    parser.add_argument(
        "--sessions",
        default=None,
        help="Optional comma-separated session numbers, e.g. 3,4,9,10",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional limit useful for a quick test run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print one progress line per session",
    )
    args = parser.parse_args()

    stats = build_dataset_statistics(
        args.data_root,
        subject_ids=_parse_int_list(args.subjects),
        session_numbers=_parse_int_list(args.sessions),
        max_sessions=args.max_sessions,
        verbose=not args.quiet,
    )
    save_statistics(stats, args.output_dir)

    sessions = stats["sessions"]
    if not sessions.empty:
        print("\nSession duration summary (minutes):")
        print(sessions["session_span_minutes"].describe())

        gaps = sessions["gap_from_previous_session_hours"].dropna()
        if len(gaps):
            print("\nInter-session gap summary (hours):")
            print(gaps.describe())

        print("\nCounts by task/condition:")
        print(sessions.groupby(["task", "condition"]).size())

    writing = stats["writing"]
    if not writing.empty:
        print("\nWriting text length summary (characters):")
        print(writing["written_text_chars"].describe())


if __name__ == "__main__":
    main()
