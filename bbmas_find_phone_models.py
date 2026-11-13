from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import pandas as pd


# The archive extracts into an outer folder containing the actual dataset
# directory (and ``__MACOSX`` metadata).  Anchor the path to this script so
# it works regardless of the IDE's selected working directory.
DATASET_ROOT = (
    Path(__file__).resolve().parent
    / "input_data/BB-MAS_Dataset/BB-MAS_Dataset"
)

# Accelerometer / gyroscope are sampled at roughly 100 Hz in BB-MAS.
# A gap longer than this is considered a break between recording periods.
RECORDING_GAP_SECONDS = 30.0

# Merge intervals from accelerometer and gyroscope if they overlap or
# are separated by no more than this amount.
MERGE_GAP_SECONDS = 2.0


# Screen resolutions stated in the BB-MAS README.
SAMSUNG_S6_WIDTH = 1440
SAMSUNG_S6_HEIGHT = 2560

HTC_ONE_WIDTH = 1080
HTC_ONE_HEIGHT = 1920


@dataclass(frozen=True)
class Interval:
    user_id: str
    start: pd.Timestamp
    end: pd.Timestamp


# ---------------------------------------------------------------------------
# Timestamp / interval analysis
# ---------------------------------------------------------------------------

def read_timestamps(csv_path: Path) -> pd.Series:
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["time"],
            dtype={"time": "string"},
        )
    except (ValueError, pd.errors.EmptyDataError):
        return pd.Series([], dtype="datetime64[ns]")

    timestamps = pd.to_datetime(
        df["time"],
        errors="coerce",
        format="mixed",
    )

    return (
        timestamps
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def timestamps_to_intervals(
    timestamps: pd.Series,
    user_id: str,
    gap_seconds: float,
) -> list[Interval]:
    if timestamps.empty:
        return []

    gap = pd.Timedelta(seconds=gap_seconds)

    values = timestamps.tolist()

    result: list[Interval] = []

    start = values[0]
    previous = values[0]

    for current in values[1:]:
        if current - previous > gap:
            result.append(
                Interval(
                    user_id=user_id,
                    start=start,
                    end=previous,
                )
            )
            start = current

        previous = current

    result.append(
        Interval(
            user_id=user_id,
            start=start,
            end=previous,
        )
    )

    return result


def merge_user_intervals(
    intervals: list[Interval],
    merge_gap_seconds: float,
) -> list[Interval]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda i: i.start)

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


def user_directories(dataset_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=lambda p: int(p.name),
    )


def find_device_motion_files(
    user_dir: Path,
    device_name: str,
) -> list[Path]:
    """
    Use only accelerometer and gyroscope files.

    These sensors are continuously sampled, making them much better
    for reconstructing actual recording intervals than keyboard or
    touch events.
    """

    files = []

    for sensor in ("Accelerometer", "Gyroscope"):
        files.extend(
            user_dir.glob(f"*_{device_name}_{sensor}_*.csv")
        )

    return sorted(files)


def get_device_intervals(
    dataset_root: Path,
    device_name: str,
) -> list[Interval]:
    all_intervals: list[Interval] = []

    for user_dir in user_directories(dataset_root):
        user_id = user_dir.name

        sensor_files = find_device_motion_files(
            user_dir,
            device_name,
        )

        if not sensor_files:
            continue

        user_intervals: list[Interval] = []

        for csv_path in sensor_files:
            timestamps = read_timestamps(csv_path)

            user_intervals.extend(
                timestamps_to_intervals(
                    timestamps=timestamps,
                    user_id=user_id,
                    gap_seconds=RECORDING_GAP_SECONDS,
                )
            )

        # Accelerometer and gyroscope represent the same recording periods,
        # so combine overlapping intervals.
        user_intervals = merge_user_intervals(
            user_intervals,
            merge_gap_seconds=MERGE_GAP_SECONDS,
        )

        all_intervals.extend(user_intervals)

    return sorted(
        all_intervals,
        key=lambda i: (i.start, i.end),
    )


# ---------------------------------------------------------------------------
# Maximum simultaneous-device usage
# ---------------------------------------------------------------------------

def find_concurrency_segments(
    intervals: list[Interval],
):
    """
    Return periods during which the set of active users was constant.

    End events are processed before start events at an identical timestamp,
    preventing a zero-duration boundary from being counted as simultaneous
    usage.
    """

    events = defaultdict(lambda: {"start": [], "end": []})

    for interval in intervals:
        events[interval.start]["start"].append(interval.user_id)
        events[interval.end]["end"].append(interval.user_id)

    timestamps = sorted(events)

    active_users: set[str] = set()
    segments = []

    previous_time = None

    for current_time in timestamps:
        # Between previous_time and current_time, this was the active set.
        if (
            previous_time is not None
            and current_time > previous_time
            and active_users
        ):
            segments.append(
                (
                    previous_time,
                    current_time,
                    frozenset(active_users),
                )
            )

        # Recording ending exactly here is no longer active afterwards.
        for user_id in events[current_time]["end"]:
            active_users.discard(user_id)

        # Recording starting exactly here becomes active afterwards.
        for user_id in events[current_time]["start"]:
            active_users.add(user_id)

        previous_time = current_time

    return segments


def print_concurrency_analysis(
    dataset_root: Path,
    device_name: str,
):
    print()
    print("=" * 90)
    print(f"{device_name}: simultaneous-use analysis")
    print("=" * 90)

    intervals = get_device_intervals(
        dataset_root,
        device_name,
    )

    print(f"Recording intervals: {len(intervals)}")

    segments = find_concurrency_segments(intervals)

    if not segments:
        print("No recording intervals found.")
        return

    max_concurrency = max(
        len(users)
        for _, _, users in segments
    )

    print()
    print(
        f"Maximum number of simultaneous users: "
        f"{max_concurrency}"
    )

    print(
        f"Therefore the dataset required AT LEAST "
        f"{max_concurrency} physical {device_name} device(s)."
    )

    max_segments = [
        segment
        for segment in segments
        if len(segment[2]) == max_concurrency
    ]

    print()
    print(
        f"Periods reaching maximum concurrency: "
        f"{len(max_segments)}"
    )

    for n, (start, end, users) in enumerate(
        max_segments,
        start=1,
    ):
        duration = end - start

        users_sorted = sorted(
            users,
            key=lambda x: int(x),
        )

        print()
        print(f"Maximum-concurrency period #{n}")
        print(f"  Start:    {start}")
        print(f"  End:      {end}")
        print(f"  Duration: {duration}")
        print(
            "  Users:    "
            + ", ".join(users_sorted)
        )


# ---------------------------------------------------------------------------
# HandPhone model inference from touch coordinates
# ---------------------------------------------------------------------------

def find_handphone_touch_files(
    dataset_root: Path,
) -> list[tuple[str, Path]]:
    result = []

    for user_dir in user_directories(dataset_root):
        user_id = user_dir.name

        for path in user_dir.glob(
            "*_HandPhone_TouchEvent_*.csv"
        ):
            result.append((user_id, path))

    return result


def analyze_handphone_touch_coordinates(
    dataset_root: Path,
):
    print()
    print("=" * 90)
    print("HandPhone model inference from touch coordinates")
    print("=" * 90)

    files = find_handphone_touch_files(dataset_root)

    if not files:
        print(
            "No HandPhone TouchEvent files were found."
        )
        return

    chunks = []

    per_user_stats = []

    for user_id, path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["Xvalue", "Yvalue"],
            )
        except (
            ValueError,
            pd.errors.EmptyDataError,
        ):
            continue

        df["Xvalue"] = pd.to_numeric(
            df["Xvalue"],
            errors="coerce",
        )

        df["Yvalue"] = pd.to_numeric(
            df["Yvalue"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["Xvalue", "Yvalue"]
        )

        if df.empty:
            continue

        df["user_id"] = user_id

        chunks.append(df)

        per_user_stats.append(
            {
                "user_id": user_id,
                "n_events": len(df),
                "max_x": df["Xvalue"].max(),
                "max_y": df["Yvalue"].max(),
            }
        )

    if not chunks:
        print(
            "No valid touch coordinates were found."
        )
        return

    all_touch = pd.concat(
        chunks,
        ignore_index=True,
    )

    x = all_touch["Xvalue"]
    y = all_touch["Yvalue"]

    print(f"Touch files analyzed: {len(files)}")
    print(
        f"Touch datapoints analyzed: "
        f"{len(all_touch):,}"
    )

    print()
    print("Observed coordinate ranges:")
    print(
        f"  X: {x.min():.3f} .. {x.max():.3f}"
    )
    print(
        f"  Y: {y.min():.3f} .. {y.max():.3f}"
    )

    print()
    print("High coordinate quantiles:")

    for q in (
        0.90,
        0.95,
        0.99,
        0.999,
        0.9999,
    ):
        print(
            f"  {q:>7.4f}: "
            f"X={x.quantile(q):9.3f}, "
            f"Y={y.quantile(q):9.3f}"
        )

    # ------------------------------------------------------------------
    # Evidence incompatible with HTC One resolution
    # ------------------------------------------------------------------

    x_above_htc = all_touch[
        all_touch["Xvalue"] > HTC_ONE_WIDTH
    ]

    y_above_htc = all_touch[
        all_touch["Yvalue"] > HTC_ONE_HEIGHT
    ]

    either_above_htc = all_touch[
        (all_touch["Xvalue"] > HTC_ONE_WIDTH)
        |
        (all_touch["Yvalue"] > HTC_ONE_HEIGHT)
    ]

    print()
    print("Known portrait resolutions:")
    print(
        f"  HTC One:          "
        f"{HTC_ONE_WIDTH} x {HTC_ONE_HEIGHT}"
    )
    print(
        f"  Samsung Galaxy S6:"
        f" {SAMSUNG_S6_WIDTH} x {SAMSUNG_S6_HEIGHT}"
    )

    print()
    print("Coordinates beyond HTC One bounds:")
    print(
        f"  X > {HTC_ONE_WIDTH}: "
        f"{len(x_above_htc):,}"
    )
    print(
        f"  Y > {HTC_ONE_HEIGHT}: "
        f"{len(y_above_htc):,}"
    )
    print(
        f"  Either axis beyond HTC bounds: "
        f"{len(either_above_htc):,}"
    )

    proportion = (
        len(either_above_htc)
        / len(all_touch)
    )

    print(
        f"  Fraction of all touch samples: "
        f"{proportion:.6%}"
    )

    affected_users = sorted(
        either_above_htc["user_id"]
        .astype(str)
        .unique(),
        key=int,
    )

    print(
        f"  Users containing such samples: "
        f"{len(affected_users)}"
    )

    if affected_users:
        print(
            "  User IDs: "
            + ", ".join(affected_users)
        )

    print()
    print("-" * 90)

    if len(either_above_htc) > 0:
        print(
            "RESULT: HandPhone is strongly indicated to be "
            "the Samsung Galaxy S6."
        )

        print(
            "Reason: the dataset contains HandPhone touch "
            "coordinates that exceed the stated portrait "
            "resolution bounds of the HTC One (1080 x 1920)."
        )

        print(
            "Such coordinates are compatible with the "
            "Samsung Galaxy S6 resolution of 1440 x 2560."
        )

    else:
        print(
            "RESULT: The touch coordinates do NOT uniquely "
            "identify the HandPhone model."
        )

        print(
            "All observed touches fit inside the HTC One "
            "bounds. This does not prove that HandPhone was "
            "the HTC One, because users may simply never "
            "touch near the edges of a larger Galaxy S6 screen."
        )

    # ------------------------------------------------------------------
    # Print users with the largest coordinate maxima.
    # Useful for checking whether the evidence is systematic rather than
    # caused by a single malformed record.
    # ------------------------------------------------------------------

    stats_df = pd.DataFrame(per_user_stats)

    if not stats_df.empty:
        print()
        print("Users with largest observed X coordinates:")

        print(
            stats_df
            .sort_values(
                "max_x",
                ascending=False,
            )
            .head(10)
            .to_string(
                index=False,
            )
        )

        print()
        print("Users with largest observed Y coordinates:")

        print(
            stats_df
            .sort_values(
                "max_y",
                ascending=False,
            )
            .head(10)
            .to_string(
                index=False,
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{DATASET_ROOT.resolve()}"
        )

    # 1. Hard lower bound on physical HandPhones.
    print_concurrency_analysis(
        DATASET_ROOT,
        "HandPhone",
    )

    # Also run PocketPhone for comparison.
    print_concurrency_analysis(
        DATASET_ROOT,
        "PocketPhone",
    )

    # 2. Infer which model HandPhone was from touch-screen coordinates.
    analyze_handphone_touch_coordinates(
        DATASET_ROOT,
    )


if __name__ == "__main__":
    main()
