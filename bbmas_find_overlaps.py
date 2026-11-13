from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_ROOT = Path("input_data/BB-MAS_Dataset/BB-MAS_Dataset")

# If there is a gap larger than this between consecutive timestamps,
# treat them as separate recording intervals.
#
# BB-MAS accelerometer/gyro data is sampled at roughly 100 Hz, so
# 30 seconds is very conservative.
INACTIVITY_GAP_SECONDS = 30.0

# Print a few source-to-parsed timestamp examples early in the run.  This is
# enough to spot a shifted/mislabelled CSV column without flooding the output.
TIMESTAMP_VALIDATION_FILES_TO_PRINT = 5
_timestamp_validation_files_printed = 0


@dataclass(frozen=True)
class Interval:
    user_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def read_timestamps(csv_path: Path) -> pd.Series:
    """Read and parse the 'time' column from a BB-MAS CSV file."""
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["time"],
            dtype={"time": "string"},
        )
    except ValueError:
        # File does not contain a 'time' column.
        print(f"Timestamp check: {csv_path} has no 'time' column; skipped.")
        return pd.Series([], dtype="datetime64[ns]")

    parsed_timestamps = pd.to_datetime(
        df["time"],
        errors="coerce",
        format="mixed",
    )

    _print_timestamp_validation(
        csv_path=csv_path,
        raw_timestamps=df["time"],
        parsed_timestamps=parsed_timestamps,
    )

    timestamps = parsed_timestamps.dropna().drop_duplicates().sort_values()

    return timestamps


def _print_timestamp_validation(
    csv_path: Path,
    raw_timestamps: pd.Series,
    parsed_timestamps: pd.Series,
) -> None:
    """Print a small audit trail for the CSV timestamp column."""
    global _timestamp_validation_files_printed

    invalid_count = int(parsed_timestamps.isna().sum())
    valid = parsed_timestamps.dropna()

    if invalid_count or valid.empty:
        print(
            f"Timestamp check: {csv_path} | rows={len(raw_timestamps)}, "
            f"parsed={len(valid)}, invalid={invalid_count}"
        )

    if _timestamp_validation_files_printed >= TIMESTAMP_VALIDATION_FILES_TO_PRINT:
        return

    examples = pd.DataFrame(
        {
            "CSV value": raw_timestamps[parsed_timestamps.notna()],
            "parsed timestamp": parsed_timestamps.dropna(),
        }
    ).head(3)

    print(
        f"Timestamp check: {csv_path} | rows={len(raw_timestamps)}, "
        f"parsed={len(valid)}, invalid={invalid_count}, "
        f"range={valid.min()} to {valid.max()}"
    )
    print(examples.to_string(index=False))
    _timestamp_validation_files_printed += 1


def timestamps_to_intervals(
    timestamps: pd.Series,
    user_id: str,
    gap_seconds: float,
) -> list[Interval]:
    """
    Convert sorted timestamps to active intervals.

    A new interval starts when the gap between consecutive timestamps
    exceeds gap_seconds.
    """
    if timestamps.empty:
        return []

    gap = pd.Timedelta(seconds=gap_seconds)

    values = timestamps.tolist()

    intervals: list[Interval] = []

    start = values[0]
    previous = values[0]

    for current in values[1:]:
        if current - previous > gap:
            intervals.append(
                Interval(
                    user_id=user_id,
                    start=start,
                    end=previous,
                )
            )
            start = current

        previous = current

    intervals.append(
        Interval(
            user_id=user_id,
            start=start,
            end=previous,
        )
    )

    return intervals


def merge_intervals(
    intervals: Iterable[Interval],
    merge_gap_seconds: float,
) -> list[Interval]:
    """
    Merge overlapping or nearby intervals for one user.

    This is useful because accelerometer, gyroscope, touch, etc.
    may describe the same physical period independently.
    """
    intervals = sorted(intervals, key=lambda x: x.start)

    if not intervals:
        return []

    gap = pd.Timedelta(seconds=merge_gap_seconds)

    merged: list[Interval] = []

    current_start = intervals[0].start
    current_end = intervals[0].end
    user_id = intervals[0].user_id

    for interval in intervals[1:]:
        if interval.start <= current_end + gap:
            current_end = max(current_end, interval.end)
        else:
            merged.append(
                Interval(
                    user_id=user_id,
                    start=current_start,
                    end=current_end,
                )
            )

            current_start = interval.start
            current_end = interval.end

    merged.append(
        Interval(
            user_id=user_id,
            start=current_start,
            end=current_end,
        )
    )

    return merged


def get_device_intervals(
    dataset_root: Path,
    device_name: str,
) -> list[Interval]:
    """
    Extract active intervals for all users for one BB-MAS device type.

    device_name examples:
        "HandPhone"
        "PocketPhone"

    All raw files matching:
        *_HandPhone_*.csv
        *_PocketPhone_*.csv

    are considered.
    """

    all_intervals: list[Interval] = []

    # User directories in BB-MAS are named 1 ... 117.
    user_dirs = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda p: int(p.name),
    )

    for user_dir in user_dirs:
        user_id = user_dir.name

        files = sorted(user_dir.glob(f"*_{device_name}_*.csv"))

        if not files:
            continue

        user_intervals: list[Interval] = []

        for csv_path in files:
            timestamps = read_timestamps(csv_path)

            if timestamps.empty:
                continue

            file_intervals = timestamps_to_intervals(
                timestamps=timestamps,
                user_id=user_id,
                gap_seconds=INACTIVITY_GAP_SECONDS,
            )

            user_intervals.extend(file_intervals)

        # Accelerator, gyro, keyboard and touch can overlap heavily.
        # Merge them to reconstruct periods during which this user
        # was actually using this device.
        user_intervals = merge_intervals(
            user_intervals,
            merge_gap_seconds=INACTIVITY_GAP_SECONDS,
        )

        all_intervals.extend(user_intervals)

    return sorted(all_intervals, key=lambda x: x.start)


def find_overlaps(intervals: list[Interval]):
    """
    Find overlapping active intervals belonging to different users.
    """
    overlaps = []

    for i, a in enumerate(intervals):
        for b in intervals[i + 1 :]:

            # Since intervals are sorted by start time, once B starts
            # after A ends there cannot be any later overlaps with A.
            if b.start > a.end:
                break

            if a.user_id == b.user_id:
                continue

            overlap_start = max(a.start, b.start)
            overlap_end = min(a.end, b.end)

            if overlap_start <= overlap_end:
                overlaps.append(
                    (
                        a.user_id,
                        b.user_id,
                        overlap_start,
                        overlap_end,
                    )
                )

    return overlaps


def print_overlaps(
    device_name: str,
    intervals: list[Interval],
) -> None:

    overlaps = find_overlaps(intervals)

    print()
    print("=" * 80)
    print(device_name)
    print("=" * 80)

    print(f"Active intervals found: {len(intervals)}")
    print(f"Cross-user overlaps found: {len(overlaps)}")
    print()

    if not overlaps:
        print(
            f"No evidence that {device_name} was used by "
            f"multiple participants simultaneously."
        )
        return

    for n, (
        user_a,
        user_b,
        overlap_start,
        overlap_end,
    ) in enumerate(overlaps, start=1):

        duration = overlap_end - overlap_start

        print(f"Overlap #{n}")
        print(f"  User A:   {user_a}")
        print(f"  User B:   {user_b}")
        print(f"  Start:    {overlap_start}")
        print(f"  End:      {overlap_end}")
        print(f"  Duration: {duration}")
        print()


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_ROOT.resolve()}"
        )

    for device_name in (
        "HandPhone",
        "PocketPhone",
    ):
        intervals = get_device_intervals(
            dataset_root=DATASET_ROOT,
            device_name=device_name,
        )

        print_overlaps(
            device_name=device_name,
            intervals=intervals,
        )


if __name__ == "__main__":
    main()
