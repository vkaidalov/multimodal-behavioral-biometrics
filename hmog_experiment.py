import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# ---------------------------------------------------------------------------
# Dataset layout and paper protocol
# ---------------------------------------------------------------------------

SITTING_WRITING_SESSIONS = (3, 9, 15, 21)
WALKING_WRITING_SESSIONS = (4, 10, 16, 22)
SCAN_LENGTHS_SECONDS = (20, 40, 60, 80, 100, 120, 140)

# Final Fisher thresholds reported for the best Scaled Manhattan HMOG system.
PAPER_FISHER_THRESHOLD = {
    "sitting": 0.82,
    "walking": 0.81,
}

ACTIVITY_COLUMNS = [
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
]
# Some distributed copies have 9 fields (SubjectID can be recovered from ID/session
# directory); the data-description PDF lists SubjectID separately, which yields 10.
ACTIVITY_COLUMNS_9 = [
    "ID",
    "SubjectID",
    "Session_number",
    "Start_time",
    "End_time",
    "Relative_Start_time",
    "Relative_End_time",
    "Gesture_scenario",
    "TaskID",
]

SENSOR_COLUMNS = [
    "Systime",
    "EventTime",
    "ActivityID",
    "X",
    "Y",
    "Z",
    "Phone_orientation",
]

ONE_FINGER_COLUMNS = [
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
]

META_COLUMNS = [
    "subject",
    "session",
    "condition",
    "activity_id",
    "tap_id",
    "tap_start_ms",
    "tap_end_ms",
    "session_start_ms",
    "session_end_ms",
]


@dataclass(frozen=True)
class SessionRef:
    subject: str
    session: int
    path: Path


# ---------------------------------------------------------------------------
# CSV input helpers
# ---------------------------------------------------------------------------

def _read_csv_no_header(path: Path) -> pd.DataFrame:
    """Read the HMOG CSV format without trusting a header row.

    The public dataset has appeared in copies with slightly different line endings.
    pandas' C parser handles both CRLF and LF. Malformed lines are skipped rather than
    aborting a multi-gigabyte extraction run.
    """
    return pd.read_csv(path, header=None, on_bad_lines="skip", low_memory=False)


def _assign_schema(df: pd.DataFrame, schemas: Sequence[Sequence[str]]) -> pd.DataFrame:
    for schema in schemas:
        if df.shape[1] == len(schema):
            out = df.copy()
            out.columns = list(schema)
            return out
    raise ValueError(
        f"Unexpected column count {df.shape[1]}; expected one of "
        f"{[len(s) for s in schemas]}"
    )


def read_activity(path: Path, fallback_subject: str, fallback_session: int) -> pd.DataFrame:
    df = _assign_schema(_read_csv_no_header(path), [ACTIVITY_COLUMNS, ACTIVITY_COLUMNS_9])
    # A textual header, if present, becomes NaN under coercion and is dropped.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ID", "Start_time", "End_time"]).copy()
    if "SubjectID" not in df or df["SubjectID"].isna().all():
        df["SubjectID"] = int(fallback_subject)
    if "Session_number" not in df or df["Session_number"].isna().all():
        df["Session_number"] = fallback_session
    if "ContentID" not in df:
        df["ContentID"] = np.nan
    return df


def read_sensor(path: Path) -> pd.DataFrame:
    df = _assign_schema(_read_csv_no_header(path), [SENSOR_COLUMNS])
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Systime", "ActivityID", "X", "Y", "Z"]).copy()


def read_one_finger(path: Path) -> pd.DataFrame:
    df = _assign_schema(_read_csv_no_header(path), [ONE_FINGER_COLUMNS])
    for col in ONE_FINGER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Systime", "ActivityID", "TapID", "Action_type"]).copy()


def find_file(session_path: Path, *names: str) -> Path:
    for name in names:
        p = session_path / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {names!r} found in {session_path}")


def iter_sessions(data_root: Path, wanted_sessions: set[int]) -> Iterator[SessionRef]:
    """Yield sessions from the official layout: root/<subject>/<subject>_session_N/.

    Subject id is taken from the session directory name rather than the parent folder.
    This also tolerates the known public-copy folder-name inconsistency for one subject.
    """
    session_re = re.compile(r"(?P<subject>\d{6})_session_(?P<session>\d+)$")
    for p in sorted(data_root.rglob("*_session_*")):
        if not p.is_dir():
            continue
        m = session_re.search(p.name)
        if not m:
            continue
        session = int(m.group("session"))
        if session not in wanted_sessions:
            continue
        yield SessionRef(m.group("subject"), session, p)


# ---------------------------------------------------------------------------
# HMOG feature extraction (paper Section II)
# ---------------------------------------------------------------------------

def _magnitude(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(xyz).sum(axis=1))


def _window(
    arr_t: np.ndarray,
    arr_v: np.ndarray,
    lo: float,
    hi: float,
    *,
    include_lo: bool = True,
    include_hi: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(arr_t, lo, side="left" if include_lo else "right")
    right = np.searchsorted(arr_t, hi, side="right" if include_hi else "left")
    return arr_t[left:right], arr_v[left:right]


def _hmog_for_dimension(
    t: np.ndarray,
    v: np.ndarray,
    tap_start: float,
    tap_end: float,
    std_ddof: int = 1,
) -> np.ndarray:
    """Eight HMOG values for one sensor dimension (X/Y/Z/M).

    Order: 5 resistance features, then 3 stability features.
    Time is measured in milliseconds, matching the 100 ms / 200 ms paper windows.
    """
    # "before" and "after" windows exclude the tap boundary itself.
    t_before, v_before = _window(t, v, tap_start - 100.0, tap_start, include_hi=False)
    t_tap, v_tap = _window(t, v, tap_start, tap_end)
    t_after100, v_after100 = _window(t, v, tap_end, tap_end + 100.0, include_lo=False)
    t_after200, v_after200 = _window(t, v, tap_end, tap_end + 200.0, include_lo=False)

    if min(len(v_before), len(v_tap), len(v_after100), len(v_after200)) == 0:
        return np.full(8, np.nan, dtype=float)

    avg_before = float(np.mean(v_before))
    avg_after = float(np.mean(v_after100))
    avg_tap = float(np.mean(v_tap))
    tap_std = float(np.std(v_tap, ddof=std_ddof)) if len(v_tap) > std_ddof else 0.0
    max_idx = int(np.argmax(v_tap))
    max_tap = float(v_tap[max_idx])
    t_max_in_tap = float(t_tap[max_idx])

    # Resistance 1-5.
    r1 = avg_tap
    r2 = tap_std
    r3 = avg_after - avg_before
    r4 = avg_tap - avg_before
    r5 = max_tap - avg_before

    # Stability 1: Algorithm 1 from the paper.
    diffs = np.abs(v_after200 - avg_before)
    # suffix_mean[i] = mean(diffs[i:]) in O(n), mathematically identical to Algorithm 1.
    suffix_sum = np.cumsum(diffs[::-1])[::-1]
    suffix_count = np.arange(len(diffs), 0, -1)
    suffix_mean = suffix_sum / suffix_count
    i_min = int(np.argmin(suffix_mean))
    t_min = float(t_after200[i_min])
    s1 = t_min - tap_end

    # Stability 2-3. Centers of the 100 ms windows are fixed by definition.
    t_before_center = tap_start - 50.0
    t_after_center = tap_end + 50.0
    den2 = avg_after - avg_before
    den3 = avg_after - max_tap
    s2 = (t_after_center - t_before_center) / den2 if not np.isclose(den2, 0.0) else np.nan
    s3 = (t_after_center - t_max_in_tap) / den3 if not np.isclose(den3, 0.0) else np.nan

    return np.array([r1, r2, r3, r4, r5, s1, s2, s3], dtype=float)


def hmog_feature_names(sensor_names: Sequence[str] = ("acc", "gyro")) -> list[str]:
    dims = ("x", "y", "z", "m")
    feats = (
        "res_mean",
        "res_std",
        "res_after_minus_before",
        "res_tap_minus_before",
        "res_max_minus_before",
        "stab_tmin_minus_end",
        "stab_duration_norm",
        "stab_max_to_avg_norm",
    )
    return [f"{s}_{d}_{f}" for s in sensor_names for d in dims for f in feats]


def _prepare_sensor_by_activity(df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # Paper authentication experiments used portrait data only.
    df = df[df["Phone_orientation"] == 0].copy()
    for activity_id, g in df.groupby("ActivityID", sort=False):
        g = g.sort_values("Systime")
        t = g["Systime"].to_numpy(dtype=float)
        xyz = g[["X", "Y", "Z"]].to_numpy(dtype=float)
        m = _magnitude(xyz)[:, None]
        values = np.hstack([xyz, m])
        out[int(activity_id)] = (t, values)
    return out


def _extract_tap_intervals(touch: pd.DataFrame, valid_activity_ids: set[int]) -> pd.DataFrame:
    # TapID -1/-2 are special UI regions according to the data description, not normal tap ids.
    t = touch[
        (touch["ActivityID"].isin(valid_activity_ids))
        & (touch["TapID"] >= 0)
        & (touch["Phone_orientation"] == 0)
    ].copy()
    records = []
    for (activity_id, tap_id), g in t.groupby(["ActivityID", "TapID"], sort=False):
        down = g[g["Action_type"] == 0]
        up = g[g["Action_type"] == 1]
        if down.empty or up.empty:
            continue
        start = float(down["Systime"].min())
        end = float(up["Systime"].max())
        if end < start:
            continue
        records.append((int(activity_id), int(tap_id), start, end))
    return pd.DataFrame(records, columns=["activity_id", "tap_id", "tap_start_ms", "tap_end_ms"])


def extract_session_hmog(
    ref: SessionRef,
    include_magnetometer: bool = False,
    std_ddof: int = 1,
) -> pd.DataFrame:
    activity = read_activity(find_file(ref.path, "Activity.csv"), ref.subject, ref.session)

    # The paper's writing sessions are 3/9/15/21 (sitting) and 4/10/16/22 (walking).
    task_rows = activity[activity["TaskID"] == ref.session]
    if task_rows.empty:
        # Some copies have imperfect Activity metadata; session directory remains authoritative.
        task_rows = activity
    valid_activity_ids = set(task_rows["ID"].astype(np.int64).tolist())
    if not valid_activity_ids:
        return pd.DataFrame()

    session_start = float(task_rows["Start_time"].min())
    session_end = float(task_rows["End_time"].max())

    touch = read_one_finger(find_file(ref.path, "OneFingerTouchEvent.csv"))
    taps = _extract_tap_intervals(touch, valid_activity_ids)
    if taps.empty:
        return pd.DataFrame()

    sensor_specs = [
        ("acc", find_file(ref.path, "Accelerometer.csv")),
        ("gyro", find_file(ref.path, "Gyroscope.csv", "Gyroscrope.csv")),
    ]
    if include_magnetometer:
        sensor_specs.append(("mag", find_file(ref.path, "Magnetometer.csv")))

    sensor_data = {name: _prepare_sensor_by_activity(read_sensor(path)) for name, path in sensor_specs}
    names = hmog_feature_names([s[0] for s in sensor_specs])
    condition = "sitting" if ref.session in SITTING_WRITING_SESSIONS else "walking"

    rows: list[dict[str, float | int | str]] = []
    for tap in taps.itertuples(index=False):
        values: list[float] = []
        missing_sensor = False
        for sensor_name, _ in sensor_specs:
            prepared = sensor_data[sensor_name].get(int(tap.activity_id))
            if prepared is None:
                missing_sensor = True
                values.extend([np.nan] * 32)
                continue
            st, sv = prepared
            for dim_idx in range(4):
                values.extend(
                    _hmog_for_dimension(
                        st,
                        sv[:, dim_idx],
                        float(tap.tap_start_ms),
                        float(tap.tap_end_ms),
                        std_ddof=std_ddof,
                    ).tolist()
                )
        row: dict[str, float | int | str] = {
            "subject": ref.subject,
            "session": ref.session,
            "condition": condition,
            "activity_id": int(tap.activity_id),
            "tap_id": int(tap.tap_id),
            "tap_start_ms": float(tap.tap_start_ms),
            "tap_end_ms": float(tap.tap_end_ms),
            "session_start_ms": session_start,
            "session_end_ms": session_end,
        }
        row.update(dict(zip(names, values)))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature selection and Scaled Manhattan verifier
# ---------------------------------------------------------------------------

def fisher_scores(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    """Population Fisher score: between-user variance / within-user variance.

    This is the standard weighted ANOVA form consistent with the paper's description.
    All computation uses enrollment/training data only.
    """
    scores: dict[str, float] = {}
    for f in feature_cols:
        work = df[["subject", f]].replace([np.inf, -np.inf], np.nan).dropna()
        if work.empty:
            scores[f] = 0.0
            continue
        global_mean = float(work[f].mean())
        between = 0.0
        within = 0.0
        for _, g in work.groupby("subject"):
            x = g[f].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            mu = float(np.mean(x))
            between += len(x) * (mu - global_mean) ** 2
            within += float(np.square(x - mu).sum())
        scores[f] = between / within if within > 0 else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def select_by_cumulative_fisher(scores: pd.Series, threshold: float) -> list[str]:
    if not 0 < threshold <= 1:
        raise ValueError("Fisher threshold must be in (0, 1].")
    finite = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(finite.sum())
    if total <= 0:
        return list(finite.index)
    cumulative = finite.cumsum() / total
    k = int(np.searchsorted(cumulative.to_numpy(), threshold, side="left") + 1)
    return list(finite.index[:k])


@dataclass
class ScaledManhattanTemplate:
    mean: np.ndarray
    std: np.ndarray


def fit_scaled_manhattan_templates(
    train: pd.DataFrame, feature_cols: Sequence[str], owner_subjects: Sequence[str]
) -> dict[str, ScaledManhattanTemplate]:
    templates: dict[str, ScaledManhattanTemplate] = {}
    for subject in owner_subjects:
        x = train.loc[train["subject"] == subject, feature_cols].replace([np.inf, -np.inf], np.nan)
        arr = x.to_numpy(dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        std[(~np.isfinite(std)) | (std <= 1e-12)] = np.nan
        templates[str(subject)] = ScaledManhattanTemplate(mean=mean, std=std)
    return templates


def scaled_manhattan_distance(vector: np.ndarray, template: ScaledManhattanTemplate) -> float:
    valid = np.isfinite(vector) & np.isfinite(template.mean) & np.isfinite(template.std)
    if not np.any(valid):
        return float("nan")
    d = np.abs(vector[valid] - template.mean[valid]) / template.std[valid]
    # Mean and sum are equivalent for threshold ranking when dimensionality is fixed.
    return float(np.mean(d))


# ---------------------------------------------------------------------------
# t-second authentication scans and population EER
# ---------------------------------------------------------------------------

def make_scan_vectors(
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    scan_seconds: int,
    min_taps: int = 1,
) -> pd.DataFrame:
    scan_ms = scan_seconds * 1000.0
    rows: list[dict[str, object]] = []

    for (subject, session), g in test.groupby(["subject", "session"], sort=False):
        g = g.sort_values("tap_start_ms")
        if g.empty:
            continue
        # Align windows to the recorded session start, not the first tap.
        start = float(g["session_start_ms"].iloc[0])
        end = float(g["session_end_ms"].iloc[0])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            start = float(g["tap_start_ms"].min())
            end = float(g["tap_end_ms"].max())
        n_windows = max(1, int(math.ceil((end - start) / scan_ms)))
        tap_times = g["tap_start_ms"].to_numpy(dtype=float)
        feat = g[list(feature_cols)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

        for w in range(n_windows):
            lo = start + w * scan_ms
            hi = min(end, lo + scan_ms)
            mask = (tap_times >= lo) & (tap_times < hi)
            n = int(mask.sum())
            if n < min_taps:
                continue
            v = np.nanmean(feat[mask], axis=0)
            rows.append(
                {
                    "subject": str(subject),
                    "session": int(session),
                    "window": w,
                    "window_start_ms": lo,
                    "window_end_ms": hi,
                    "n_taps": n,
                    "vector": v,
                }
            )
    return pd.DataFrame(rows)


def score_population(
    scans: pd.DataFrame,
    templates: dict[str, ScaledManhattanTemplate],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scan in scans.itertuples(index=False):
        vector = np.asarray(scan.vector, dtype=float)
        probe = str(scan.subject)
        for owner, template in templates.items():
            distance = scaled_manhattan_distance(vector, template)
            if not np.isfinite(distance):
                continue
            rows.append(
                {
                    "probe_subject": probe,
                    "owner_subject": owner,
                    "session": int(scan.session),
                    "window": int(scan.window),
                    "distance": distance,
                    "is_genuine": int(probe == owner),
                }
            )
    return pd.DataFrame(rows)


def population_eer(scores: pd.DataFrame) -> tuple[float, float, int, int]:
    if scores.empty or scores["is_genuine"].nunique() < 2:
        return float("nan"), float("nan"), 0, 0
    y = scores["is_genuine"].to_numpy(dtype=int)
    similarity = -scores["distance"].to_numpy(dtype=float)  # lower distance => more genuine
    fpr, tpr, thresholds = roc_curve(y, similarity, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    crossing = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(crossing):
        i = int(crossing[0])
        x0, x1 = diff[i], diff[i + 1]
        alpha = 0.0 if np.isclose(x0, x1) else float(-x0 / (x1 - x0))
        eer = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
        threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    else:
        i = int(np.argmin(np.abs(diff)))
        eer = float((fpr[i] + fnr[i]) / 2.0)
        threshold = float(thresholds[i])

    n_genuine = int((y == 1).sum())
    n_impostor = int((y == 0).sum())
    return eer, threshold, n_genuine, n_impostor


# ---------------------------------------------------------------------------
# Cache extraction and full paper-best HMOG/SM experiment
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, ref: SessionRef, include_magnetometer: bool) -> Path:
    suffix = "hmog96" if include_magnetometer else "hmog64"
    return cache_dir / ref.subject / f"session_{ref.session}_{suffix}.pkl.gz"


def load_or_extract(
    ref: SessionRef,
    cache_dir: Path,
    include_magnetometer: bool,
    force: bool,
    std_ddof: int,
) -> pd.DataFrame:
    p = cache_path(cache_dir, ref, include_magnetometer)
    if p.exists() and not force:
        out = pd.read_pickle(p, compression="gzip")
        print(f"  cache hit: {len(out):,} feature rows")
        return out
    print("  extracting features from source CSV files...")
    out = extract_session_hmog(ref, include_magnetometer=include_magnetometer, std_ddof=std_ddof)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(p, compression="gzip")
    print(f"  extracted and cached: {len(out):,} feature rows")
    return out


def extract_dataset(
    data_root: Path,
    cache_dir: Path,
    include_magnetometer: bool,
    force: bool,
    std_ddof: int,
) -> pd.DataFrame:
    wanted = set(SITTING_WRITING_SESSIONS + WALKING_WRITING_SESSIONS)
    refs = list(iter_sessions(data_root, wanted))
    if not refs:
        raise FileNotFoundError(
            "No HMOG session directories found. Expected layout like "
            "<root>/100669/100669_session_3/Activity.csv"
        )
    print(f"Found {len(refs)} writing sessions; preparing HMOG feature dataset.")
    frames = []
    for i, ref in enumerate(refs, 1):
        print(f"[{i}/{len(refs)}] {ref.subject} session {ref.session}")
        try:
            f = load_or_extract(ref, cache_dir, include_magnetometer, force, std_ddof)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skipped: {exc}")
            continue
        if not f.empty:
            frames.append(f)
    if not frames:
        raise RuntimeError("No HMOG feature rows could be extracted.")
    dataset = pd.concat(frames, ignore_index=True)
    print(
        f"Feature dataset ready: {len(dataset):,} rows from "
        f"{dataset['subject'].nunique():,} subjects."
    )
    return dataset


def run_condition_sm(
    all_features: pd.DataFrame,
    condition: str,
    fisher_threshold: float,
    min_enrollment_vectors: int,
    min_taps_per_scan: int,
    scan_lengths: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = SITTING_WRITING_SESSIONS if condition == "sitting" else WALKING_WRITING_SESSIONS
    train_sessions = set(sessions[:2])
    test_sessions = set(sessions[2:])

    data = all_features[all_features["condition"] == condition].copy()
    train = data[data["session"].isin(train_sessions)].copy()
    test = data[data["session"].isin(test_sessions)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Missing train/test data for {condition}.")
    print(
        f"\nEvaluating {condition}: {len(train):,} enrollment rows "
        f"(sessions {sorted(train_sessions)}), {len(test):,} test rows "
        f"(sessions {sorted(test_sessions)})."
    )

    # The paper's best HMOG-only SM system uses accelerometer + gyroscope.
    feature_cols = [c for c in hmog_feature_names(("acc", "gyro")) if c in train.columns]

    counts = train.groupby("subject").size()
    owners = sorted(counts[counts >= min_enrollment_vectors].index.astype(str).tolist())
    if not owners:
        raise RuntimeError(f"No users have >= {min_enrollment_vectors} training vectors.")
    train_owners = train[train["subject"].astype(str).isin(owners)].copy()
    print(
        f"Eligible owners: {len(owners):,}/{len(counts):,} "
        f"with at least {min_enrollment_vectors} enrollment vectors."
    )

    print(f"Calculating Fisher scores for {len(feature_cols)} features...")
    fisher = fisher_scores(train_owners, feature_cols)
    selected = select_by_cumulative_fisher(fisher, fisher_threshold)
    print(
        f"Selected {len(selected)} features at cumulative Fisher threshold "
        f"{fisher_threshold:.2f}; fitting {len(owners)} templates..."
    )
    templates = fit_scaled_manhattan_templates(train_owners, selected, owners)

    selection_table = pd.DataFrame(
        {
            "feature": fisher.index,
            "fisher_score": fisher.values,
            "selected": [f in set(selected) for f in fisher.index],
        }
    )

    result_rows = []
    for t in scan_lengths:
        print(f"  Scan length {t}s: creating scan vectors and scoring population...")
        scans = make_scan_vectors(test, selected, t, min_taps=min_taps_per_scan)
        scores = score_population(scans, templates)
        eer, threshold, n_genuine, n_impostor = population_eer(scores)
        eer_text = f"{eer * 100.0:.3f}%" if np.isfinite(eer) else "unavailable"
        print(
            f"    {len(scans):,} scans, {len(scores):,} scores "
            f"({n_genuine:,} genuine / {n_impostor:,} impostor), EER: {eer_text}."
        )
        result_rows.append(
            {
                "condition": condition,
                "verifier": "scaled_manhattan",
                "scan_seconds": t,
                "eer": eer,
                "eer_percent": eer * 100.0 if np.isfinite(eer) else np.nan,
                "similarity_threshold_at_eer": threshold,
                "n_owner_templates": len(owners),
                "n_selected_features": len(selected),
                "n_genuine_scores": n_genuine,
                "n_impostor_scores": n_impostor,
                "fisher_threshold": fisher_threshold,
            }
        )
    return pd.DataFrame(result_rows), selection_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the HMOG paper's best HMOG-only Scaled Manhattan authentication experiment."
    )
    parser.add_argument("--data-root", type=Path, default=Path("./input_data/hmog_unarchived"), help="Path to extracted hmog_dataset directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("./output_data/hmog_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output_data/hmog_results"))
    parser.add_argument("--condition", choices=["sitting", "walking", "both"], default="both")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--include-magnetometer", action="store_true", help="Extract 96 features instead of the paper-best 64; final SM selection still uses acc+gyro")
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1, help="Standard-deviation convention for HMOG feature 2; paper does not specify ddof")
    parser.add_argument("--min-enrollment-vectors", type=int, default=80)
    parser.add_argument("--min-taps-per-scan", type=int, default=1)
    parser.add_argument("--scan-lengths", type=int, nargs="+", default=list(SCAN_LENGTHS_SECONDS))
    parser.add_argument("--sitting-fisher", type=float, default=PAPER_FISHER_THRESHOLD["sitting"])
    parser.add_argument("--walking-fisher", type=float, default=PAPER_FISHER_THRESHOLD["walking"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Starting HMOG Scaled Manhattan evaluation: "
        f"condition={args.condition}, scan lengths={args.scan_lengths}, "
        f"force_extract={args.force_extract}."
    )
    print(f"Data root: {args.data_root.resolve()}")
    print(f"Cache directory: {args.cache_dir.resolve()}")
    all_features = extract_dataset(
        args.data_root,
        args.cache_dir,
        include_magnetometer=args.include_magnetometer,
        force=args.force_extract,
        std_ddof=args.std_ddof,
    )

    conditions = ["sitting", "walking"] if args.condition == "both" else [args.condition]
    all_results = []
    manifest = {
        "paper_protocol": {
            "sitting_train_sessions": list(SITTING_WRITING_SESSIONS[:2]),
            "sitting_test_sessions": list(SITTING_WRITING_SESSIONS[2:]),
            "walking_train_sessions": list(WALKING_WRITING_SESSIONS[:2]),
            "walking_test_sessions": list(WALKING_WRITING_SESSIONS[2:]),
            "portrait_only": True,
            "min_enrollment_vectors": args.min_enrollment_vectors,
            "scan_lengths_seconds": args.scan_lengths,
        },
        "implementation_choices_not_fully_specified_in_paper": {
            "scan_windows": "non-overlapping; aligned to session Start_time",
            "hmog_std_ddof": args.std_ddof,
            "fisher_formula": "weighted ANOVA between-user sum of squares / within-user sum of squares",
            "eer": "population EER from all genuine and zero-effort impostor scores, linearly interpolated",
        },
    }

    for condition in conditions:
        threshold = args.sitting_fisher if condition == "sitting" else args.walking_fisher
        results, selection = run_condition_sm(
            all_features,
            condition,
            fisher_threshold=threshold,
            min_enrollment_vectors=args.min_enrollment_vectors,
            min_taps_per_scan=args.min_taps_per_scan,
            scan_lengths=args.scan_lengths,
        )
        all_results.append(results)
        selection.to_csv(args.output_dir / f"fisher_{condition}.csv", index=False)
        print("\n", condition.upper())
        print(results[["scan_seconds", "eer_percent", "n_owner_templates", "n_selected_features"]].to_string(index=False))

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(args.output_dir / "eer_results.csv", index=False)
    with open(args.output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
