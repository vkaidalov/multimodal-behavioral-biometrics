#!/usr/bin/env python3
"""
From-paper Python reconstruction of DeFFusion on the public HMOG dataset.

Paper:
Y. Li, P. Tao, S. Deng, G. Zhou,
"DeFFusion: CNN-based Continuous Authentication Using Deep Feature Fusion",
ACM Transactions on Sensor Networks, DOI 10.1145/3485060.

The paper leaves several implementation details unspecified or internally
inconsistent. This program exposes the consequential choices as command-line
options and prints them at runtime so that experiments remain auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Reproducibility / utility helpers
# ---------------------------------------------------------------------------


def p(msg: str = "") -> None:
    """Console progress output required by the reproduction request."""
    print(msg, flush=True)


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time for pipeline logs."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


@contextmanager
def stage(name: str):
    """Log the start, completion/failure, and wall-clock duration of a pipeline stage."""
    started = time.perf_counter()
    p(f"\n[stage] START  {name}")
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        p(f"[stage] FAILED {name} | elapsed={format_duration(elapsed)}")
        raise
    else:
        elapsed = time.perf_counter() - started
        p(f"[stage] DONE   {name} | elapsed={format_duration(elapsed)}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be much slower; still request determinism
    # where PyTorch can provide it without failing.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Union[str, Path]):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def summarize(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "sd": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


# ---------------------------------------------------------------------------
# HMOG discovery and raw CSV parsing
# ---------------------------------------------------------------------------

SENSOR_COLUMNS = [
    "Systime",
    "EventTime",
    "ActivityID",
    "X",
    "Y",
    "Z",
    "Phone_orientation",
]


@dataclass
class SessionFiles:
    user_id: str
    session_no: int
    directory: str
    accelerometer: str
    gyroscope: str


def _case_insensitive_file(directory: Path, wanted: str) -> Optional[Path]:
    wanted_lower = wanted.lower()
    try:
        for child in directory.iterdir():
            if child.is_file() and child.name.lower() == wanted_lower:
                return child
    except PermissionError:
        return None
    return None


def _infer_user_id(path: Path) -> Optional[str]:
    # Prefer an ancestor whose entire directory name is exactly six digits.
    for parent in [path] + list(path.parents):
        if re.fullmatch(r"\d{6}", parent.name):
            return parent.name
    # Common HMOG session naming: 100669_session_3, etc.
    for parent in [path] + list(path.parents):
        m = re.search(r"(?<!\d)(\d{6})(?!\d)", parent.name)
        if m:
            return m.group(1)
    return None


def _infer_session_no(path: Path) -> int:
    candidates = [path.name] + [x.name for x in path.parents[:3]]
    patterns = [r"session[_\- ]*(\d+)", r"(?:^|[_\- ])s(\d+)(?:$|[_\- ])"]
    for name in candidates:
        for pat in patterns:
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                return int(m.group(1))
    return 10**9


def discover_sessions(data_root: Union[str, Path]) -> List[SessionFiles]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"HMOG root does not exist: {root}")

    p(f"[inspect] Recursively scanning for HMOG sensor CSVs under: {root}")
    sessions: List[SessionFiles] = []
    seen_dirs = set()

    for acc in root.rglob("*"):
        if not acc.is_file() or acc.name.lower() != "accelerometer.csv":
            continue
        d = acc.parent.resolve()
        if d in seen_dirs:
            continue
        gyro = _case_insensitive_file(d, "Gyroscope.csv")
        if gyro is None:
            # Defensive support for the misspelling used in some documentation.
            gyro = _case_insensitive_file(d, "Gyroscrope.csv")
        if gyro is None:
            continue
        uid = _infer_user_id(d)
        if uid is None:
            p(f"[inspect][WARN] Could not infer six-digit user ID from: {d}")
            continue
        sessions.append(
            SessionFiles(
                user_id=uid,
                session_no=_infer_session_no(d),
                directory=str(d),
                accelerometer=str(acc),
                gyroscope=str(gyro),
            )
        )
        seen_dirs.add(d)

    sessions.sort(key=lambda s: (s.user_id, s.session_no, s.directory))
    p(f"[inspect] Found {len(sessions)} session directories with both sensors.")
    return sessions


def _normalize_colname(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def read_sensor_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read HMOG Accelerometer/Gyroscope file with or without a header."""
    path = Path(path)
    # First try a normal header.
    df = pd.read_csv(path, low_memory=False)
    norm = {_normalize_colname(c): c for c in df.columns}
    required = {_normalize_colname(x) for x in ["EventTime", "X", "Y", "Z"]}
    if not required.issubset(norm.keys()):
        # HMOG releases are often headerless.
        df = pd.read_csv(path, header=None, names=SENSOR_COLUMNS, low_memory=False)
    else:
        rename = {}
        for expected in SENSOR_COLUMNS:
            k = _normalize_colname(expected)
            if k in norm:
                rename[norm[k]] = expected
        df = df.rename(columns=rename)

    needed = ["EventTime", "X", "Y", "Z"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"{path}: missing expected column {col}; columns={list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed)
    return df[needed].copy()


def timestamp_to_seconds(t: np.ndarray) -> Tuple[np.ndarray, str]:
    """Infer timestamp scale from positive sample deltas and convert to seconds."""
    t = np.asarray(t, dtype=np.float64)
    t = t[np.isfinite(t)] if t.ndim == 1 else t
    if t.size < 2:
        return t, "seconds"
    s = np.sort(t)
    d = np.diff(s)
    d = d[d > 0]
    if d.size == 0:
        return t - np.nanmin(t), "unknown"
    med = float(np.median(d))
    if med >= 1e6:
        scale, unit = 1e9, "nanoseconds"
    elif med >= 1e3:
        scale, unit = 1e6, "microseconds"
    elif med >= 1.0:
        scale, unit = 1e3, "milliseconds"
    else:
        scale, unit = 1.0, "seconds"
    # EventTime is relative to the same Android system-boot clock for all
    # sensors in a session. Preserve that common origin so accelerometer and
    # gyroscope offsets are not accidentally erased before synchronization.
    return t / scale, unit


def _clean_time_xyz(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    t_raw = df["EventTime"].to_numpy(dtype=np.float64)
    t, unit = timestamp_to_seconds(t_raw)
    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    xyz = xyz[order]
    # np.interp requires monotonically increasing x. Keep the first of duplicates.
    unique_t, unique_idx = np.unique(t, return_index=True)
    xyz = xyz[unique_idx]
    return unique_t, xyz, unit


def synchronize_session(
    acc_path: Union[str, Path],
    gyro_path: Union[str, Path],
    fs: int = 100,
    max_interpolation_gap: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, Union[str, float, int]]]:
    """
    Synchronize accelerometer and gyroscope on a shared regular fs-Hz grid.

    The paper does not state its synchronization method. We use linear interpolation
    over the overlap and remove points that lie in large raw-sensor gaps.
    """
    adf = read_sensor_csv(acc_path)
    gdf = read_sensor_csv(gyro_path)
    at, av, aunit = _clean_time_xyz(adf)
    gt, gv, gunit = _clean_time_xyz(gdf)
    if len(at) < 3 or len(gt) < 3:
        return np.empty((0, 6), dtype=np.float32), {
            "acc_unit": aunit,
            "gyro_unit": gunit,
            "n": 0,
        }

    t0 = max(float(at[0]), float(gt[0]))
    t1 = min(float(at[-1]), float(gt[-1]))
    if t1 <= t0:
        return np.empty((0, 6), dtype=np.float32), {
            "acc_unit": aunit,
            "gyro_unit": gunit,
            "n": 0,
        }
    dt = 1.0 / fs
    grid = np.arange(t0, t1, dt, dtype=np.float64)
    if grid.size == 0:
        return np.empty((0, 6), dtype=np.float32), {
            "acc_unit": aunit,
            "gyro_unit": gunit,
            "n": 0,
        }

    # Reject target times falling into a large gap in either stream.
    def gap_mask(raw_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(raw_t, target_t, side="left")
        idx = np.clip(idx, 1, len(raw_t) - 1)
        left = raw_t[idx - 1]
        right = raw_t[idx]
        return (right - left) <= max_interpolation_gap

    valid = gap_mask(at, grid) & gap_mask(gt, grid)
    grid = grid[valid]
    if grid.size == 0:
        return np.empty((0, 6), dtype=np.float32), {
            "acc_unit": aunit,
            "gyro_unit": gunit,
            "n": 0,
        }

    cols = []
    for j in range(3):
        cols.append(np.interp(grid, at, av[:, j]))
    for j in range(3):
        cols.append(np.interp(grid, gt, gv[:, j]))
    out = np.column_stack(cols).astype(np.float32)
    meta = {
        "acc_unit": aunit,
        "gyro_unit": gunit,
        "n": int(out.shape[0]),
        "seconds": float(out.shape[0] / fs),
        "fs": int(fs),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Prepared synchronized-series storage
# ---------------------------------------------------------------------------


def prepare_dataset(args) -> None:
    prepare_t0 = time.perf_counter()
    seed_everything(args.seed)
    with stage("Discover HMOG sensor sessions"):
        sessions = discover_sessions(args.data_root)
    by_user: Dict[str, List[SessionFiles]] = defaultdict(list)
    for s in sessions:
        by_user[s.user_id].append(s)

    requested_users = None
    if args.users_file:
        requested_users = [x.strip() for x in Path(args.users_file).read_text().splitlines() if x.strip()]
        p(f"[prepare] Using explicit users file with {len(requested_users)} IDs.")

    work = ensure_dir(args.work_dir)
    series_dir = ensure_dir(work / "series")
    target_samples = int(args.minutes * 60 * args.fs)
    p(f"[prepare] Target per user: {args.minutes:.2f} minutes = {target_samples:,} samples at {args.fs} Hz")

    candidates = requested_users if requested_users is not None else sorted(by_user.keys())
    selected = []
    manifest_users = {}

    for candidate_i, uid in enumerate(candidates, 1):
        if uid not in by_user:
            p(f"[prepare][WARN] User {uid} has no discovered sensor sessions; skipping.")
            continue
        if args.max_users and len(selected) >= args.max_users:
            break
        p(f"\n[prepare] User candidate {candidate_i}/{len(candidates)}: {uid}")
        user_t0 = time.perf_counter()
        user_chunks = []
        segment_lengths = []
        seconds_total = 0.0
        session_records = []
        for si, session in enumerate(sorted(by_user[uid], key=lambda x: (x.session_no, x.directory)), 1):
            if sum(segment_lengths) >= target_samples:
                break
            p(f"[prepare]   session {si}/{len(by_user[uid])}: {Path(session.directory).name}")
            try:
                arr, meta = synchronize_session(
                    session.accelerometer,
                    session.gyroscope,
                    fs=args.fs,
                    max_interpolation_gap=args.max_interpolation_gap,
                )
            except Exception as e:
                p(f"[prepare][WARN]   failed session: {type(e).__name__}: {e}")
                continue
            if arr.shape[0] == 0:
                p("[prepare][WARN]   no usable synchronized samples")
                continue
            remaining = target_samples - sum(segment_lengths)
            if arr.shape[0] > remaining:
                arr = arr[:remaining]
            user_chunks.append(arr)
            segment_lengths.append(int(arr.shape[0]))
            seconds_total += arr.shape[0] / args.fs
            session_records.append({
                "directory": session.directory,
                "session_no": session.session_no,
                "samples_kept": int(arr.shape[0]),
                "sync_meta": meta,
            })
            p(f"[prepare]   kept {arr.shape[0]:,} samples; cumulative {sum(segment_lengths):,}/{target_samples:,}")

        n_kept = sum(segment_lengths)
        if n_kept < target_samples and not args.allow_short_users:
            p(f"[prepare][SKIP] {uid}: only {n_kept / args.fs / 60:.2f} usable minutes (< {args.minutes}).")
            continue
        if n_kept == 0:
            continue

        out = np.concatenate(user_chunks, axis=0)
        out_path = series_dir / f"{uid}.npy"
        np.save(out_path, out, allow_pickle=False)
        starts = []
        pos = 0
        for n in segment_lengths:
            starts.append([pos, pos + n])
            pos += n
        manifest_users[uid] = {
            "path": str(out_path.resolve()),
            "n_samples": int(out.shape[0]),
            "minutes": float(out.shape[0] / args.fs / 60),
            "segments": starts,
            "sessions": session_records,
        }
        selected.append(uid)
        p(f"[prepare][OK] User {uid}: saved {out.shape[0]:,} x 6 samples -> {out_path}")
        p(f"[prepare][TIME] User {uid}: {format_duration(time.perf_counter()-user_t0)}")

    manifest = {
        "fs": args.fs,
        "target_minutes": args.minutes,
        "selected_users": selected,
        "n_users": len(selected),
        "users": manifest_users,
        "preparation": {
            "sync": "linear interpolation to common regular grid over sensor overlap",
            "max_interpolation_gap": args.max_interpolation_gap,
            "selection": "explicit users file" if requested_users is not None else "sorted eligible user IDs",
        },
    }
    save_json(manifest, work / "manifest.json")
    Path(work / "selected_users.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    p(f"\n[prepare] DONE. Prepared {len(selected)} users.")
    if len(selected) != args.max_users and args.max_users:
        p(f"[prepare][WARN] Requested max {args.max_users}, got {len(selected)} users.")
    p(f"[prepare] Manifest: {work / 'manifest.json'}")
    p(f"[prepare] Total preparation time: {format_duration(time.perf_counter()-prepare_t0)}")


# ---------------------------------------------------------------------------
# Window indexing and paper preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRef:
    user_index: int
    user_id: str
    start: int
    length: int


def load_manifest(work_dir: Union[str, Path]) -> dict:
    path = Path(work_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the 'prepare' command first.")
    return load_json(path)


def build_window_refs(
    manifest: dict,
    window_seconds: float,
    max_windows_per_user: Optional[int] = None,
) -> Tuple[List[WindowRef], Dict[str, int]]:
    fs = int(manifest["fs"])
    length = int(round(window_seconds * fs))
    refs: List[WindowRef] = []
    counts = {}
    for ui, uid in enumerate(manifest["selected_users"]):
        u = manifest["users"][uid]
        user_refs = []
        for a, b in u["segments"]:
            pos = int(a)
            b = int(b)
            while pos + length <= b:
                user_refs.append(WindowRef(ui, uid, pos, length))
                pos += length
        if max_windows_per_user is not None:
            user_refs = user_refs[:max_windows_per_user]
        refs.extend(user_refs)
        counts[uid] = len(user_refs)
    return refs, counts


def row_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def preprocess_window(raw: np.ndarray, domain: str) -> np.ndarray:
    """
    raw: [n, 6] in x/y/z accelerometer then x/y/z gyroscope order.
    Returns [1, 20, width] for CNN input.
    """
    t = raw.T.astype(np.float32, copy=False)  # 6 x n
    if domain == "time":
        z = row_standardize(t)
    elif domain == "frequency":
        # Paper defines Ft as d x 2n (d=6): for each sensor-axis row,
        # concatenate its FFT real and imaginary parts, then StandardScaler
        # is applied to that complete row. For a 5 s / 100 Hz window this is
        # 6 x 1000, not 12 x 500.
        f = np.fft.fft(t, axis=1)
        z = np.concatenate([f.real, f.imag], axis=1).astype(np.float32)  # 6 x (2n)
        z = row_standardize(z)
    else:
        raise ValueError(domain)
    flat = z.reshape(-1)
    if flat.size % 20 != 0:
        raise ValueError(f"Cannot reshape {flat.size} values into 20 rows")
    return flat.reshape(1, 20, flat.size // 20).astype(np.float32)


class HMOGWindowDataset(Dataset):
    def __init__(self, manifest: dict, refs: Sequence[WindowRef], domain: str):
        self.manifest = manifest
        self.refs = list(refs)
        self.domain = domain
        self._maps: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.refs)

    def _memmap(self, uid: str) -> np.ndarray:
        if uid not in self._maps:
            path = self.manifest["users"][uid]["path"]
            self._maps[uid] = np.load(path, mmap_mode="r")
        return self._maps[uid]

    def __getitem__(self, idx: int):
        r = self.refs[idx]
        mm = self._memmap(r.user_id)
        raw = np.asarray(mm[r.start:r.start + r.length], dtype=np.float32)
        x = preprocess_window(raw, self.domain)
        return torch.from_numpy(x), torch.tensor(r.user_index, dtype=torch.long)


# ---------------------------------------------------------------------------
# CNN reconstructed from paper
# ---------------------------------------------------------------------------


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError(f"channels {c} not divisible by groups {groups}")
    x = x.view(b, groups, c // groups, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(b, c, h, w)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels % 2:
            raise ValueError("BasicBlock channels must be even")
        half = channels // 2
        self.branch = nn.Sequential(
            ConvBNReLU(half, half, 1),
            nn.Conv2d(half, half, 3, stride=1, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            ConvBNReLU(half, half, 1),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y2 = self.branch(x2)
        return channel_shuffle(torch.cat([x1, y2], dim=1), 2)


class DownBlock(nn.Module):
    """ShuffleNet-like two-branch down block, spatial stride (1,2)."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        if out_c % 2:
            raise ValueError("DownBlock output channels must be even")
        half = out_c // 2
        # The text/figure are not perfectly explicit about branch channel counts.
        # This implementation preserves the published output C=32.
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=(1, 6), stride=(1, 2), padding=0, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            ConvBNReLU(in_c, half, 1),
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_c, half, 1),
            nn.Conv2d(half, half, kernel_size=(1, 6), stride=(1, 2), padding=0, groups=half, bias=False),
            nn.BatchNorm2d(half),
            ConvBNReLU(half, half, 1),
        )

    def forward(self, x):
        return channel_shuffle(torch.cat([self.branch1(x), self.branch2(x)], dim=1), 2)


class DeFFusionCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        domain: str,
        conv2_mode: str = "table_shape",
    ):
        super().__init__()
        self.domain = domain
        self.conv2_mode = conv2_mode
        self.conv1 = ConvBNReLU(1, 24, 3, stride=1, padding=1)
        self.down1 = DownBlock(24, 32)
        self.basic1 = BasicBlock(32)
        self.down2 = DownBlock(32, 32)
        self.basic2 = BasicBlock(32)
        self.basic3 = BasicBlock(32)

        # Published Table 1 says: input height 20 -> Conv2 output height 6,
        # but also states KSize=(1,1), which cannot change height. Default matches
        # the published tensor shape by using a (15,1) kernel. Alternative follows
        # the stated kernel literally.
        if conv2_mode == "table_shape":
            self.conv2 = ConvBNReLU(32, 64, kernel_size=(15, 1), stride=1, padding=0)
            pool_h = 6
        elif conv2_mode == "one_by_one":
            self.conv2 = ConvBNReLU(32, 64, kernel_size=1, stride=1, padding=0)
            pool_h = 20
        else:
            raise ValueError(conv2_mode)

        # The paper explicitly specifies AvgPool (1,5) for time-domain data
        # and (1,12) for frequency-domain data. Use those exact kernels for
        # the principal 5-second experiment rather than adaptive pooling.
        # For other window sizes, adaptive pooling preserves a six-column
        # output so the window-size sweep remains runnable.
        if domain == "time":
            self.avgpool = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
        elif domain == "frequency":
            self.avgpool = nn.AvgPool2d(kernel_size=(1, 12), stride=(1, 12))
        else:
            raise ValueError(domain)
        flat = 64 * pool_h * 6
        self.fc1 = nn.Linear(flat, 3072)
        self.fc2 = nn.Linear(3072, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, return_features: bool = False, feature_source: str = "logits"):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.basic1(x)
        x = self.down2(x)
        x = self.basic2(x)
        x = self.basic3(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        fc2 = self.relu(self.fc2(x))
        logits = self.classifier(fc2)
        if return_features:
            if feature_source == "logits":
                return logits
            if feature_source == "fc2":
                return fc2
            raise ValueError(feature_source)
        return logits


# ---------------------------------------------------------------------------
# CNN training / feature extraction / FA fusion
# ---------------------------------------------------------------------------


def model_paths(work: Path, window: float) -> Tuple[Path, Path, Path]:
    tag = f"w{window:g}s"
    root = ensure_dir(work / "models" / tag)
    return root / "time.pt", root / "frequency.pt", root


def feature_dir(work: Path, window: float) -> Path:
    return ensure_dir(work / "features" / f"w{window:g}s")


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        persistent_workers=(workers > 0),
    )


def train_one_cnn(
    manifest: dict,
    refs: Sequence[WindowRef],
    domain: str,
    output_path: Path,
    args,
) -> None:
    if output_path.exists() and not args.force:
        p(f"[train:{domain}] Existing model found, skipping: {output_path}")
        return
    device = parse_device(args.device)
    p(f"[train:{domain}] Device: {device}")
    dataset = HMOGWindowDataset(manifest, refs, domain)
    loader = _make_loader(dataset, args.batch_size, True, args.workers, args.seed)
    model = DeFFusionCNN(len(manifest["selected_users"]), domain, args.conv2_mode).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    p(f"[train:{domain}] Windows: {len(dataset):,}; batch={args.batch_size}; epochs={args.epochs}")
    p(f"[train:{domain}] LR={args.lr}, StepLR step={args.lr_step}, gamma=0.1")
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for bi, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += y.size(0)
            if args.print_every and bi % args.print_every == 0:
                p(
                    f"[train:{domain}] epoch {epoch:03d}/{args.epochs} "
                    f"batch {bi:04d}/{len(loader):04d} "
                    f"loss={total_loss/max(seen,1):.5f} acc={100*correct/max(seen,1):.2f}%"
                )
        scheduler.step()
        p(
            f"[train:{domain}] epoch {epoch:03d}/{args.epochs} DONE "
            f"loss={total_loss/max(seen,1):.5f} acc={100*correct/max(seen,1):.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.3g} "
            f"epoch_time={format_duration(time.perf_counter()-epoch_t0)} "
            f"total_time={format_duration(time.perf_counter()-t0)}"
        )

    payload = {
        "state_dict": model.state_dict(),
        "num_classes": len(manifest["selected_users"]),
        "domain": domain,
        "conv2_mode": args.conv2_mode,
        "feature_source": args.feature_source,
        "window": args.window,
        "users": manifest["selected_users"],
    }
    torch.save(payload, output_path)
    p(f"[train:{domain}] Saved: {output_path}")
    p(f"[train:{domain}] Total training time: {format_duration(time.perf_counter()-t0)}")


def load_model(path: Path, device: torch.device) -> Tuple[DeFFusionCNN, dict]:
    payload = torch.load(path, map_location=device)
    model = DeFFusionCNN(payload["num_classes"], payload["domain"], payload["conv2_mode"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def extract_features(
    manifest: dict,
    refs: Sequence[WindowRef],
    domain: str,
    model_path: Path,
    out_path: Path,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    if out_path.exists() and not args.force:
        p(f"[features:{domain}] Loading cached features: {out_path}")
        d = np.load(out_path)
        return d["X"], d["y"]
    device = parse_device(args.device)
    model, payload = load_model(model_path, device)
    dataset = HMOGWindowDataset(manifest, refs, domain)
    loader = _make_loader(dataset, args.batch_size, False, args.workers, args.seed)
    xs, ys = [], []
    extract_t0 = time.perf_counter()
    p(f"[features:{domain}] Extracting from {len(dataset):,} windows")
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            feat = model(x, return_features=True, feature_source=args.feature_source)
            xs.append(feat.detach().cpu().numpy().astype(np.float32))
            ys.append(y.numpy().astype(np.int16))
            if args.print_every and bi % args.print_every == 0:
                p(f"[features:{domain}] batch {bi}/{len(loader)}")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.savez_compressed(out_path, X=X, y=y)
    p(f"[features:{domain}] Saved {X.shape}: {out_path}")
    p(f"[features:{domain}] Total extraction time: {format_duration(time.perf_counter()-extract_t0)}")
    return X, y


def fit_fa_fusion(
    Xtime: np.ndarray,
    Xfreq: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    args,
) -> np.ndarray:
    fused_path = out_dir / "fused.npz"
    if fused_path.exists() and not args.force:
        p(f"[fusion] Loading cached fused features: {fused_path}")
        return np.load(fused_path)["X"]

    nt, nf = args.time_factors, args.freq_factors
    if args.feature_source == "fc2" and max(nt, nf) > 512:
        raise ValueError("Too many FA factors")
    p(f"[fusion] FactorAnalysis time={nt}, frequency={nf}")
    p("[fusion][NOTE] Paper does not specify FA implementation; using sklearn FactorAnalysis.")

    fa_t = FactorAnalysis(n_components=nt, random_state=args.seed)
    fa_f = FactorAnalysis(n_components=nf, random_state=args.seed)
    if args.fa_fit_max and Xtime.shape[0] > args.fa_fit_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(Xtime.shape[0], args.fa_fit_max, replace=False)
        fit_t, fit_f = Xtime[idx], Xfreq[idx]
        p(f"[fusion] Fitting FA on random {args.fa_fit_max:,}/{Xtime.shape[0]:,} windows (speed option).")
    else:
        fit_t, fit_f = Xtime, Xfreq
    t0 = time.perf_counter()
    fa_t.fit(fit_t)
    p(f"[fusion] Time-domain FA fit done in {format_duration(time.perf_counter()-t0)}")
    t0 = time.perf_counter()
    fa_f.fit(fit_f)
    p(f"[fusion] Frequency-domain FA fit done in {format_duration(time.perf_counter()-t0)}")
    t0 = time.perf_counter()
    Ft = fa_t.transform(Xtime).astype(np.float32)
    Ff = fa_f.transform(Xfreq).astype(np.float32)
    p(f"[fusion] FA transforms done in {format_duration(time.perf_counter()-t0)}")
    fused = np.concatenate([Ft, Ff], axis=1).astype(np.float32)
    joblib.dump(fa_t, out_dir / "fa_time.joblib")
    joblib.dump(fa_f, out_dir / "fa_frequency.joblib")
    np.savez_compressed(fused_path, X=fused, y=y)
    p(f"[fusion] Saved fused features {fused.shape}: {fused_path}")
    return fused


def train_pipeline(args) -> None:
    pipeline_t0 = time.perf_counter()
    seed_everything(args.seed)
    work = Path(args.work_dir)

    with stage("Load prepared dataset manifest"):
        manifest = load_manifest(work)
        if len(manifest["selected_users"]) < 2:
            raise RuntimeError("Need at least two users.")
        if len(manifest["selected_users"]) != 95:
            p(f"[train][WARN] Paper used 95 users; manifest contains {len(manifest['selected_users'])}.")

    with stage(f"Build {args.window:g}s authentication-window index"):
        refs, counts = build_window_refs(manifest, args.window, args.max_windows_per_user)
        p(f"[train] Window size={args.window:g}s. Total windows={len(refs):,}")
        p(f"[train] Per-user windows: min={min(counts.values())}, median={np.median(list(counts.values())):.0f}, max={max(counts.values())}")

    time_model, freq_model, model_root = model_paths(work, args.window)
    with stage("Train time-domain CNN"):
        train_one_cnn(manifest, refs, "time", time_model, args)
    with stage("Train frequency-domain CNN"):
        train_one_cnn(manifest, refs, "frequency", freq_model, args)

    fdir = feature_dir(work, args.window)
    with stage("Extract time-domain deep features"):
        Xt, yt = extract_features(manifest, refs, "time", time_model, fdir / "time_deep.npz", args)
    with stage("Extract frequency-domain deep features"):
        Xf, yf = extract_features(manifest, refs, "frequency", freq_model, fdir / "frequency_deep.npz", args)
    if not np.array_equal(yt, yf):
        raise RuntimeError("Time/frequency labels do not align")
    with stage("Factor Analysis and time/frequency feature fusion"):
        fused = fit_fa_fusion(Xt, Xf, yt, fdir, args)
    with stage("Save feature-pipeline metadata"):
        save_json(
        {
            "window": args.window,
            "feature_source": args.feature_source,
            "time_factors": args.time_factors,
            "freq_factors": args.freq_factors,
            "shape": list(fused.shape),
            "users": manifest["selected_users"],
            "conv2_mode": args.conv2_mode,
            "paper_gap_notes": [
                "Conv2 kernel/shape inconsistency resolved by chosen conv2_mode",
                "95-deep-feature layer is not explicit; chosen feature_source recorded above",
                "FactorAnalysis implementation is not specified in paper",
            ],
        },
            fdir / "feature_config.json",
        )
    p(f"[train] DONE: CNNs trained and fused features created. Total pipeline time: {format_duration(time.perf_counter()-pipeline_t0)}")


# ---------------------------------------------------------------------------
# Authentication evaluation
# ---------------------------------------------------------------------------


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    # Linear interpolation at a crossing if available.
    crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(crossings):
        i = crossings[0]
        x0, x1 = diff[i], diff[i + 1]
        if abs(x1 - x0) > 1e-12:
            a = -x0 / (x1 - x0)
            return float(fpr[i] + a * (fpr[i + 1] - fpr[i]))
    i = int(np.argmin(np.abs(diff)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _gamma_arg(value: str) -> Union[str, float]:
    if value in {"scale", "auto"}:
        return value
    return float(value)


def evaluate_ocsvm_user(
    X_pos: np.ndarray,
    X_neg_pool: np.ndarray,
    folds: int,
    repeats: int,
    positive_limit: Optional[int],
    nu: float,
    gamma: Union[str, float],
    seed: int,
    print_prefix: str,
) -> Dict[str, float]:
    rng_master = np.random.default_rng(seed)
    repeat_metrics = []
    n_pos_all = X_pos.shape[0]
    for rep in range(repeats):
        rep_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(rep_seed)
        npos = min(n_pos_all, positive_limit) if positive_limit else n_pos_all
        if npos < folds:
            raise ValueError(f"Only {npos} positives, less than folds={folds}")
        pos_idx = rng.choice(n_pos_all, npos, replace=False) if npos < n_pos_all else rng.permutation(n_pos_all)
        Xp = X_pos[pos_idx]
        neg_idx = rng.choice(X_neg_pool.shape[0], npos, replace=(X_neg_pool.shape[0] < npos))
        Xn = X_neg_pool[neg_idx]
        kfp = KFold(n_splits=folds, shuffle=True, random_state=rep_seed)
        kfn = KFold(n_splits=folds, shuffle=True, random_state=rep_seed + 1)
        neg_splits = list(kfn.split(Xn))
        all_y, all_scores, all_pred = [], [], []
        for fi, ((p_train, p_test), (_, n_test)) in enumerate(zip(kfp.split(Xp), neg_splits), 1):
            # The paper standardizes the raw time/frequency rows before the
            # CNN, but does not report an additional normalization of the fused
            # deep features before OC-SVM. Keep the fused features unchanged.
            train = Xp[p_train]
            testp = Xp[p_test]
            testn = Xn[n_test]
            clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            clf.fit(train)
            sp = clf.decision_function(testp).ravel()
            sn = clf.decision_function(testn).ravel()
            pp = (sp >= 0).astype(int)
            pn = (sn >= 0).astype(int)
            all_y.append(np.concatenate([np.ones_like(sp, dtype=int), np.zeros_like(sn, dtype=int)]))
            all_scores.append(np.concatenate([sp, sn]))
            all_pred.append(np.concatenate([pp, pn]))
        y = np.concatenate(all_y)
        scores = np.concatenate(all_scores)
        pred = np.concatenate(all_pred)
        eer = eer_from_scores(y, scores)
        neg_mask = y == 0
        pos_mask = y == 1
        far = float(np.mean(pred[neg_mask] == 1))
        frr = float(np.mean(pred[pos_mask] == 0))
        repeat_metrics.append((eer, far, frr))
        p(f"{print_prefix} repeat {rep+1:02d}/{repeats}: EER={100*eer:.3f}% FAR={100*far:.3f}% FRR={100*frr:.3f}%")
    a = np.asarray(repeat_metrics)
    return {
        "eer": float(a[:, 0].mean()),
        "far": float(a[:, 1].mean()),
        "frr": float(a[:, 2].mean()),
        "eer_repeat_sd": float(a[:, 0].std(ddof=0)),
        "n_positive_available": int(n_pos_all),
        "n_positive_used": int(min(n_pos_all, positive_limit) if positive_limit else n_pos_all),
    }


def load_fused_features(work: Path, window: float) -> Tuple[np.ndarray, np.ndarray]:
    path = feature_dir(work, window) / "fused.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 'train' for this window first.")
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"].astype(int)


def evaluate_features_ocsvm(
    X: np.ndarray,
    y: np.ndarray,
    users: Sequence[str],
    folds: int,
    repeats: int,
    positive_limit: Optional[int],
    nu: float,
    gamma: Union[str, float],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for ui, uid in enumerate(users):
        Xp = X[y == ui]
        Xn = X[y != ui]
        p(f"\n[evaluate] User {ui+1}/{len(users)} {uid}: positives={len(Xp):,}, impostor pool={len(Xn):,}")
        user_t0 = time.perf_counter()
        m = evaluate_ocsvm_user(
            Xp, Xn, folds, repeats, positive_limit, nu, gamma,
            seed + ui * 100003,
            print_prefix=f"[evaluate:{uid}]",
        )
        rows.append({"user_id": uid, **m})
        p(f"[evaluate:{uid}] user total: {format_duration(time.perf_counter()-user_t0)}")
    return pd.DataFrame(rows)


def print_metric_summary(df: pd.DataFrame, label: str) -> Dict[str, dict]:
    p(f"\n========== {label} ==========")
    out = {}
    for col in ["eer", "far", "frr"]:
        s = summarize(100.0 * df[col].to_numpy())
        out[col] = s
        p(f"{col.upper():3s}: mean={s['mean']:.3f}% median={s['median']:.3f}% SD={s['sd']:.3f}% min={s['min']:.3f}% max={s['max']:.3f}%")
    return out


def evaluate_command(args) -> None:
    eval_t0 = time.perf_counter()
    seed_everything(args.seed)
    work = Path(args.work_dir)
    with stage("Load manifest and fused features"):
        manifest = load_manifest(work)
        X, y = load_fused_features(work, args.window)
    gamma = _gamma_arg(args.ocsvm_gamma)
    p("[evaluate] Paper protocol: 10-fold CV, balanced negative test samples, repeat 20 times.")
    p(f"[evaluate] Current: folds={args.folds}, repeats={args.repeats}, positive_limit={args.positive_limit}")
    p(f"[evaluate][GAP] OC-SVM nu/gamma are not reported by paper. Using nu={args.ocsvm_nu}, gamma={gamma}.")
    with stage("Per-user repeated OC-SVM cross-validation"):
        df = evaluate_features_ocsvm(
            X, y, manifest["selected_users"], args.folds, args.repeats,
            args.positive_limit, args.ocsvm_nu, gamma, args.seed,
        )
    with stage("Aggregate authentication metrics"):
        summary = print_metric_summary(df, f"DeFFusion {args.window:g}s")
    outdir = ensure_dir(work / "results")
    tag = f"deffusion_w{args.window:g}s"
    with stage("Save evaluation results"):
        df.to_csv(outdir / f"{tag}_per_user.csv", index=False)
        save_json({
        "summary_percent": summary,
        "settings": {
            "window": args.window,
            "folds": args.folds,
            "repeats": args.repeats,
            "positive_limit": args.positive_limit,
            "ocsvm_nu": args.ocsvm_nu,
            "ocsvm_gamma": args.ocsvm_gamma,
            "seed": args.seed,
        },
        "paper_5s_target": {
            "table5_eer_mean": 1.29,
            "table5_eer_median": 1.00,
            "table5_eer_sd": 1.03,
            "table7_eer": 1.00,
            "table7_far": 1.42,
            "table7_frr": 0.75,
        },
        }, outdir / f"{tag}_summary.json")
    p(f"[evaluate] Saved results under {outdir}")
    p(f"[evaluate] Total evaluation time: {format_duration(time.perf_counter()-eval_t0)}")


# ---------------------------------------------------------------------------
# Training-size sweep
# ---------------------------------------------------------------------------

PAPER_TRAINING_SIZE = {
    100: (2.13, 1.25, 2.35), 200: (2.34, 1.75, 2.19),
    300: (1.87, 1.75, 1.46), 400: (2.45, 2.43, 1.75),
    500: (2.20, 2.20, 1.42), 600: (1.89, 1.71, 1.27),
    700: (2.08, 1.75, 1.46), 800: (2.03, 1.72, 1.44),
    900: (2.01, 1.78, 1.29), 1000: (1.86, 1.60, 1.29),
}


def training_size_sweep(args) -> None:
    work = Path(args.work_dir)
    manifest = load_manifest(work)
    X, y = load_fused_features(work, args.window)
    gamma = _gamma_arg(args.ocsvm_gamma)
    rows = []
    for n in args.sizes:
        p(f"\n######## TRAINING-SIZE SWEEP: n={n} ########")
        df = evaluate_features_ocsvm(
            X, y, manifest["selected_users"], args.folds, args.repeats,
            n, args.ocsvm_nu, gamma, args.seed + n,
        )
        s = summarize(100 * df["eer"].to_numpy())
        target = PAPER_TRAINING_SIZE.get(n)
        row = {"size": n, "mean": s["mean"], "median": s["median"], "sd": s["sd"]}
        if target:
            row.update({"paper_mean": target[0], "paper_median": target[1], "paper_sd": target[2]})
        rows.append(row)
        p(f"[training-size] n={n}: mean={s['mean']:.3f}, median={s['median']:.3f}, sd={s['sd']:.3f}")
    out = pd.DataFrame(rows)
    outdir = ensure_dir(work / "results")
    out.to_csv(outdir / f"training_size_w{args.window:g}s.csv", index=False)
    p("\n" + out.to_string(index=False))


# ---------------------------------------------------------------------------
# Designed 90-feature baseline
# ---------------------------------------------------------------------------


def spectral_features(x: np.ndarray, fs: int) -> Tuple[float, float, float, float, float]:
    # Standard reconstruction because the paper gives feature names but not exact formulas.
    x = np.asarray(x, dtype=float)
    energy = float(np.sum(x * x))
    mag = np.abs(np.fft.rfft(x))
    power = mag * mag
    psum = power.sum()
    if psum > 0:
        prob = power / psum
        nz = prob > 0
        entropy = float(-np.sum(prob[nz] * np.log(prob[nz])))
    else:
        entropy = 0.0
    # Exclude DC when identifying motion-frequency peaks.
    if len(mag) <= 2:
        return energy, entropy, 0.0, 0.0, 0.0
    mm = mag.copy()
    mm[0] = 0.0
    order = np.argsort(mm)[::-1]
    p1_i = int(order[0])
    p2_i = int(order[1]) if len(order) > 1 else p1_i
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return energy, entropy, float(mm[p1_i]), float(freqs[p2_i]), float(mm[p2_i])


def designed_axis_features(x: np.ndarray, fs: int) -> np.ndarray:
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    e, ent, p1, p2f, p2 = spectral_features(x, fs)
    vals = [
        np.mean(x), np.std(x), np.max(x), np.min(x), np.max(x) - np.min(x),
        kurtosis(x, fisher=True, bias=False) if len(x) > 3 else 0.0,
        skew(x, bias=False) if len(x) > 2 else 0.0,
        q25, q50, q75,
        e, ent, p1, p2f, p2,
    ]
    return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_designed_features(manifest: dict, refs: Sequence[WindowRef], fs: int, out_path: Path, force: bool) -> Tuple[np.ndarray, np.ndarray]:
    if out_path.exists() and not force:
        p(f"[designed] Loading cache: {out_path}")
        d = np.load(out_path)
        return d["X"], d["y"]
    maps = {}
    rows, labels = [], []
    p(f"[designed] Computing 90 features for {len(refs):,} windows.")
    for i, r in enumerate(refs, 1):
        if r.user_id not in maps:
            maps[r.user_id] = np.load(manifest["users"][r.user_id]["path"], mmap_mode="r")
        raw = np.asarray(maps[r.user_id][r.start:r.start+r.length], dtype=np.float32)
        feats = np.concatenate([designed_axis_features(raw[:, j], fs) for j in range(6)])
        rows.append(feats)
        labels.append(r.user_index)
        if i % 10000 == 0:
            p(f"[designed] {i:,}/{len(refs):,}")
    X = np.vstack(rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int16)
    np.savez_compressed(out_path, X=X, y=y)
    p(f"[designed] Saved {X.shape}: {out_path}")
    return X, y


def designed_baseline(args) -> None:
    work = Path(args.work_dir)
    manifest = load_manifest(work)
    refs, _ = build_window_refs(manifest, args.window, args.max_windows_per_user)
    fdir = feature_dir(work, args.window)
    X90, y = compute_designed_features(manifest, refs, int(manifest["fs"]), fdir / "designed90.npz", args.force)

    # Paper says feature selection and fusion remain, without saying exactly how 90
    # hand-designed features are split for FA. We apply FA to 50 latent factors.
    fa_path = fdir / "designed_fa50.joblib"
    transformed_path = fdir / "designed_fused50.npz"
    if transformed_path.exists() and not args.force:
        d = np.load(transformed_path)
        X = d["X"]
    else:
        p("[designed][GAP] Paper does not specify exact FA/fusion handling for the 90 designed features; using FA(50).")
        fa = FactorAnalysis(n_components=min(50, X90.shape[1]), random_state=args.seed)
        X = fa.fit_transform(X90).astype(np.float32)
        joblib.dump(fa, fa_path)
        np.savez_compressed(transformed_path, X=X, y=y)

    gamma = _gamma_arg(args.ocsvm_gamma)
    df = evaluate_features_ocsvm(X, y, manifest["selected_users"], args.folds, args.repeats,
                                 args.positive_limit, args.ocsvm_nu, gamma, args.seed)
    summary = print_metric_summary(df, "Designed 90-feature + OC-SVM baseline")
    outdir = ensure_dir(work / "results")
    df.to_csv(outdir / f"designed_baseline_w{args.window:g}s_per_user.csv", index=False)
    save_json({"summary_percent": summary, "paper_target": {"eer": 9.00, "far": 8.96, "frr": 9.04}},
              outdir / f"designed_baseline_w{args.window:g}s_summary.json")


# ---------------------------------------------------------------------------
# Binary classifier comparison on CNN features
# ---------------------------------------------------------------------------


def evaluate_binary_classifier_user(
    X_pos: np.ndarray,
    X_neg_pool: np.ndarray,
    kind: str,
    folds: int,
    repeats: int,
    positive_limit: Optional[int],
    seed: int,
    prefix: str,
) -> Dict[str, float]:
    master = np.random.default_rng(seed)
    rms = []
    for rep in range(repeats):
        rs = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(rs)
        n = min(len(X_pos), positive_limit) if positive_limit else len(X_pos)
        pos_idx = rng.choice(len(X_pos), n, replace=False) if n < len(X_pos) else rng.permutation(len(X_pos))
        neg_idx = rng.choice(len(X_neg_pool), n, replace=(len(X_neg_pool) < n))
        Xp, Xn = X_pos[pos_idx], X_neg_pool[neg_idx]
        kfp = list(KFold(folds, shuffle=True, random_state=rs).split(Xp))
        kfn = list(KFold(folds, shuffle=True, random_state=rs + 1).split(Xn))
        ys, scores, preds = [], [], []
        for (ptr, pte), (ntr, nte) in zip(kfp, kfn):
            Xtr = np.concatenate([Xp[ptr], Xn[ntr]], axis=0)
            ytr = np.concatenate([np.ones(len(ptr), dtype=int), np.zeros(len(ntr), dtype=int)])
            Xte = np.concatenate([Xp[pte], Xn[nte]], axis=0)
            yte = np.concatenate([np.ones(len(pte), dtype=int), np.zeros(len(nte), dtype=int)])
            # No post-fusion standardization is reported in the paper.
            if kind == "knn":
                clf = KNeighborsClassifier(n_neighbors=5)
            elif kind == "rf":
                clf = RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)
            elif kind == "dt":
                clf = DecisionTreeClassifier(random_state=rs)
            else:
                raise ValueError(kind)
            clf.fit(Xtr, ytr)
            if hasattr(clf, "predict_proba"):
                sc = clf.predict_proba(Xte)[:, 1]
            else:
                sc = clf.predict(Xte).astype(float)
            pr = clf.predict(Xte)
            ys.append(yte); scores.append(sc); preds.append(pr)
        yy = np.concatenate(ys); ss = np.concatenate(scores); ppred = np.concatenate(preds)
        eer = eer_from_scores(yy, ss)
        far = float(np.mean(ppred[yy == 0] == 1))
        frr = float(np.mean(ppred[yy == 1] == 0))
        rms.append((eer, far, frr))
        p(f"{prefix} repeat {rep+1:02d}/{repeats}: EER={100*eer:.3f}% FAR={100*far:.3f}% FRR={100*frr:.3f}%")
    a = np.asarray(rms)
    return {"eer": float(a[:,0].mean()), "far": float(a[:,1].mean()), "frr": float(a[:,2].mean())}


def classifier_baselines(args) -> None:
    work = Path(args.work_dir)
    manifest = load_manifest(work)
    X, y = load_fused_features(work, args.window)
    outdir = ensure_dir(work / "results")
    targets = {
        "knn": {"eer": 1.29, "far": 1.71, "frr": 0.79},
        "rf": {"eer": 1.02, "far": 1.17, "frr": 0.83},
        "dt": {"eer": 3.65, "far": 4.33, "frr": 3.29},
    }
    for kind in ["knn", "rf", "dt"]:
        p(f"\n######## CNN-{kind.upper()} ########")
        rows = []
        for ui, uid in enumerate(manifest["selected_users"]):
            Xp, Xn = X[y == ui], X[y != ui]
            p(f"[classifier:{kind}] User {ui+1}/{len(manifest['selected_users'])} {uid}")
            m = evaluate_binary_classifier_user(Xp, Xn, kind, args.folds, args.repeats,
                                                args.positive_limit, args.seed + ui*1009,
                                                f"[classifier:{kind}:{uid}]")
            rows.append({"user_id": uid, **m})
        df = pd.DataFrame(rows)
        summary = print_metric_summary(df, f"CNN-{kind.upper()}")
        df.to_csv(outdir / f"cnn_{kind}_w{args.window:g}s_per_user.csv", index=False)
        save_json({"summary_percent": summary, "paper_target": targets[kind]},
                  outdir / f"cnn_{kind}_w{args.window:g}s_summary.json")


# ---------------------------------------------------------------------------
# Window sweep orchestration
# ---------------------------------------------------------------------------

PAPER_WINDOW = {
    1: (2.74, 2.53, 1.68), 2: (1.85, 1.67, 1.18), 3: (1.68, 1.55, 1.08),
    4: (1.64, 1.47, 1.05), 5: (1.29, 1.00, 1.03), 6: (1.35, 1.23, 0.87),
    7: (1.11, 0.95, 0.87), 8: (1.23, 1.18, 0.77), 9: (1.08, 0.93, 0.84),
    10: (1.11, 1.00, 0.79),
}


def window_sweep(args) -> None:
    rows = []
    for w in args.windows:
        p(f"\n================ WINDOW {w}s ================")
        # Reuse train_pipeline by setting window and paper-reported balance.
        args.window = float(w)
        if int(w) == 2:
            args.time_factors, args.freq_factors = 29, 21
        else:
            args.time_factors, args.freq_factors = 28, 22
        train_pipeline(args)
        work = Path(args.work_dir)
        manifest = load_manifest(work)
        X, y = load_fused_features(work, args.window)
        df = evaluate_features_ocsvm(X, y, manifest["selected_users"], args.folds, args.repeats,
                                     args.positive_limit, args.ocsvm_nu, _gamma_arg(args.ocsvm_gamma),
                                     args.seed + int(w)*10000)
        s = summarize(100 * df["eer"].to_numpy())
        target = PAPER_WINDOW.get(int(w))
        row = {"window": w, "mean": s["mean"], "median": s["median"], "sd": s["sd"]}
        if target:
            row.update({"paper_mean": target[0], "paper_median": target[1], "paper_sd": target[2]})
        rows.append(row)
        p(f"[window-sweep] {w}s -> mean={s['mean']:.3f} median={s['median']:.3f} SD={s['sd']:.3f}")
    out = pd.DataFrame(rows)
    outdir = ensure_dir(Path(args.work_dir) / "results")
    out.to_csv(outdir / "window_sweep.csv", index=False)
    p("\n" + out.to_string(index=False))


# ---------------------------------------------------------------------------
# Dataset inspect command
# ---------------------------------------------------------------------------


def inspect_command(args) -> None:
    sessions = discover_sessions(args.data_root)
    by_user = defaultdict(list)
    for s in sessions:
        by_user[s.user_id].append(s)
    p(f"[inspect] Distinct inferred users: {len(by_user)}")
    if by_user:
        counts = [len(v) for v in by_user.values()]
        p(f"[inspect] Sessions/user: min={min(counts)}, median={np.median(counts):.1f}, max={max(counts)}")
        p("[inspect] First users:")
        for uid in sorted(by_user)[: min(20, len(by_user))]:
            nums = [s.session_no for s in by_user[uid]]
            p(f"  {uid}: {len(nums)} sessions; parsed session numbers={nums[:24]}")
    if args.probe and sessions:
        p("\n[inspect] Probing first sensor pair...")
        s = sessions[0]
        a = read_sensor_csv(s.accelerometer)
        g = read_sensor_csv(s.gyroscope)
        at, _, au = _clean_time_xyz(a)
        gt, _, gu = _clean_time_xyz(g)
        p(f"[inspect] {s.user_id}: accel rows={len(a):,}, gyro rows={len(g):,}, inferred units={au}/{gu}")
        arr, meta = synchronize_session(s.accelerometer, s.gyroscope, fs=args.fs)
        p(f"[inspect] synchronized: {arr.shape}, {meta}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def add_common_eval(sp):
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--window", type=float, default=5.0)
    sp.add_argument("--folds", type=int, default=10)
    sp.add_argument("--repeats", type=int, default=20)
    sp.add_argument("--positive-limit", type=int, default=1000,
                    help="Maximum genuine samples per user. Paper training-size study goes to 1000.")
    sp.add_argument("--ocsvm-nu", type=float, default=0.01,
                    help="NOT reported in paper; reconstruction default 0.01.")
    sp.add_argument("--ocsvm-gamma", default="scale",
                    help="NOT reported in paper; sklearn 'scale', 'auto', or numeric value.")
    sp.add_argument("--seed", type=int, default=2021)


def add_train_args(sp, include_work=True):
    if include_work:
        sp.add_argument("--work-dir", required=True)
    sp.add_argument("--window", type=float, default=5.0)
    sp.add_argument("--epochs", type=int, default=150)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--lr", type=float, default=0.001)
    sp.add_argument("--lr-step", type=int, default=50,
                    help="Paper prose says 50; displayed equation says step_size=30.")
    sp.add_argument("--time-factors", type=int, default=28)
    sp.add_argument("--freq-factors", type=int, default=22)
    sp.add_argument("--feature-source", choices=["logits", "fc2"], default="logits")
    sp.add_argument("--conv2-mode", choices=["table_shape", "one_by_one"], default="one_by_one")
    sp.add_argument("--fa-fit-max", type=int, default=None,
                    help="Optional cap on samples used to fit FA; omit for full paper-like dataset.")
    sp.add_argument("--max-windows-per-user", type=int, default=None,
                    help="Speed/debug cap; omit for full data.")
    sp.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. 0 is safest on Windows laptops.")
    sp.add_argument("--device", default=None,
                    help="e.g. cuda, cpu, mps; default auto-detect.")
    sp.add_argument("--print-every", type=int, default=50)
    sp.add_argument("--force", action="store_true", help="Recompute/overwrite cached artifacts.")
    sp.add_argument("--seed", type=int, default=2021)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproduce DeFFusion authentication experiments on the public HMOG dataset."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="Find HMOG sessions and probe CSV format.")
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--probe", action="store_true")
    sp.add_argument("--fs", type=int, default=100)
    sp.set_defaults(func=inspect_command)

    sp = sub.add_parser("prepare", help="Synchronize and cache first N minutes per user.")
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--minutes", type=float, default=100.0)
    sp.add_argument("--fs", type=int, default=100)
    sp.add_argument("--max-users", type=int, default=95)
    sp.add_argument("--users-file", default=None)
    sp.add_argument("--allow-short-users", action="store_true")
    sp.add_argument("--max-interpolation-gap", type=float, default=0.25)
    sp.add_argument("--seed", type=int, default=2021)
    sp.set_defaults(func=prepare_dataset)

    sp = sub.add_parser("train", help="Train two CNNs, extract 95-d features, FA-select and fuse.")
    add_train_args(sp)
    sp.set_defaults(func=train_pipeline)

    sp = sub.add_parser("evaluate", help="Run repeated 10-fold OC-SVM authentication protocol.")
    add_common_eval(sp)
    sp.set_defaults(func=evaluate_command)

    sp = sub.add_parser("training-size-sweep", help="Table-4-style 100..1000 positive-sample sweep.")
    add_common_eval(sp)
    sp.add_argument("--sizes", type=int, nargs="+", default=list(range(100, 1001, 100)))
    sp.set_defaults(func=training_size_sweep)

    sp = sub.add_parser("designed-baseline", help="Table-6/7-style 90 hand-designed features + OC-SVM.")
    add_common_eval(sp)
    sp.add_argument("--max-windows-per-user", type=int, default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=designed_baseline)

    sp = sub.add_parser("classifier-baselines", help="kNN/RF/DT on same fused CNN features.")
    add_common_eval(sp)
    sp.set_defaults(func=classifier_baselines)

    sp = sub.add_parser("window-sweep", help="Table-5-style window-size sweep; retrains CNNs.")
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--windows", type=int, nargs="+", default=list(range(1, 11)))
    sp.add_argument("--folds", type=int, default=10)
    sp.add_argument("--repeats", type=int, default=20)
    sp.add_argument("--positive-limit", type=int, default=1000)
    sp.add_argument("--ocsvm-nu", type=float, default=0.01)
    sp.add_argument("--ocsvm-gamma", default="scale")
    # Training options, without duplicate --work-dir / --window.
    sp.add_argument("--epochs", type=int, default=150)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--lr", type=float, default=0.001)
    sp.add_argument("--lr-step", type=int, default=50)
    sp.add_argument("--feature-source", choices=["logits", "fc2"], default="logits")
    sp.add_argument("--conv2-mode", choices=["table_shape", "one_by_one"], default="one_by_one")
    sp.add_argument("--fa-fit-max", type=int, default=None)
    sp.add_argument("--max-windows-per-user", type=int, default=None)
    sp.add_argument("--workers", type=int, default=0)
    sp.add_argument("--device", default=None)
    sp.add_argument("--print-every", type=int, default=50)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--seed", type=int, default=2021)
    sp.set_defaults(func=window_sweep)

    return ap


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    p("=" * 78)
    p("DeFFusion / HMOG reproduction")
    p(f"Command: {args.command}")
    p("=" * 78)
    command_t0 = time.perf_counter()
    try:
        args.func(args)
    except KeyboardInterrupt:
        p(f"\n[STOP] Interrupted by user after {format_duration(time.perf_counter()-command_t0)}.")
        return 130
    except Exception as e:
        p(f"\n[ERROR] {type(e).__name__}: {e}")
        p(f"[ERROR] Command runtime before failure: {format_duration(time.perf_counter()-command_t0)}")
        if os.environ.get("DEFFUSION_DEBUG") == "1":
            raise
        p("Set DEFFUSION_DEBUG=1 to show a full traceback.")
        return 1
    p(f"\n[total] Command '{args.command}' completed in {format_duration(time.perf_counter()-command_t0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
