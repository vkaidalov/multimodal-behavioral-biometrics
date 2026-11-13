from collections import defaultdict
from bisect import bisect_right
from datetime import datetime
import re

import pandas as pd
from pymongo import MongoClient


# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "brainrun"

USERS_COLLECTION = "users"
GAMES_COLLECTION = "games"
GESTURES_COLLECTION = "gestures"
SENSOR_FILES_COLLECTION = "sensor_files"

# BrainRun nominal motion-sensor sampling interval is 100 ms.
SENSOR_RATE_HZ = 10.0

# IMPORTANT:
# The BrainRun paper does not unambiguously state whether the
# timestamp in <player_id>_<timestamp>.json is the beginning
# or end of that recording.
#
# Try "start" first. You can later rerun with "end" and compare.
SENSOR_TIMESTAMP_MODE = "start"

OUTPUT_CSV = "output_data/brainrun_motion_covered_gestures.csv"


# ============================================================
# TIMESTAMP UTILITIES
# ============================================================

SOURCE_TIMESTAMP_RE = re.compile(
    r"_(\d{10,16})\.json$",
    re.IGNORECASE
)


def to_epoch_ms(value):
    """
    Convert common MongoDB time representations to Unix ms.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.timestamp() * 1000.0

    if isinstance(value, (int, float)):
        value = float(value)

        # Values around 1e9 are normally Unix seconds.
        # Values around 1e12 are normally Unix milliseconds.
        if value < 100_000_000_000:
            return value * 1000.0

        return value

    if isinstance(value, str):
        value = value.strip()

        try:
            return to_epoch_ms(float(value))
        except ValueError:
            pass

        try:
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
            return parsed.timestamp() * 1000.0
        except ValueError:
            return None

    return None


def extract_sensor_timestamp(doc):
    """
    Prefer sensor_files.timestamp if present.
    Otherwise extract timestamp from the source filename.
    """

    timestamp = to_epoch_ms(doc.get("timestamp"))

    if timestamp is not None:
        return timestamp

    source_file = (
        doc.get("source_file")
        or doc.get("_source_file")
    )

    if not source_file:
        return None

    filename = source_file.replace("\\", "/").rsplit("/", 1)[-1]

    match = SOURCE_TIMESTAMP_RE.search(filename)

    if not match:
        return None

    return to_epoch_ms(int(match.group(1)))


# ============================================================
# INTERVAL UTILITIES
# ============================================================

def merge_intervals(intervals):
    """
    Merge overlapping or directly adjacent time intervals.
    """

    if not intervals:
        return []

    intervals.sort()

    merged = [list(intervals[0])]

    for start, stop in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], stop)
        else:
            merged.append([start, stop])

    return [
        (start, stop)
        for start, stop in merged
    ]


def make_sensor_interval(timestamp_ms, duration_ms):
    if SENSOR_TIMESTAMP_MODE == "start":
        return (
            timestamp_ms,
            timestamp_ms + duration_ms
        )

    if SENSOR_TIMESTAMP_MODE == "end":
        return (
            timestamp_ms - duration_ms,
            timestamp_ms
        )

    raise ValueError(
        "SENSOR_TIMESTAMP_MODE must be 'start' or 'end'"
    )


def prepare_interval_index(intervals):
    """
    Store merged intervals plus their start times for binary search.
    """

    merged = merge_intervals(intervals)

    return {
        "intervals": merged,
        "starts": [x[0] for x in merged]
    }


def is_fully_covered(start, stop, interval_index):
    """
    Return True only when the complete [start, stop] interval
    lies inside one merged sensor interval.
    """

    intervals = interval_index["intervals"]
    starts = interval_index["starts"]

    if not intervals:
        return False

    i = bisect_right(starts, start) - 1

    if i < 0:
        return False

    sensor_start, sensor_stop = intervals[i]

    return (
        sensor_start <= start
        and stop <= sensor_stop
    )


def has_any_overlap(start, stop, interval_index):
    """
    Diagnostic only: gesture overlaps a sensor interval at all.
    """

    intervals = interval_index["intervals"]
    starts = interval_index["starts"]

    if not intervals:
        return False

    i = bisect_right(starts, start) - 1

    if i >= 0:
        s, e = intervals[i]

        if e > start and s < stop:
            return True

    i += 1

    if i < len(intervals):
        s, e = intervals[i]

        if e > start and s < stop:
            return True

    return False


# ============================================================
# CONNECT
# ============================================================

client = MongoClient(MONGO_URI)

db = client[DB_NAME]

users = db[USERS_COLLECTION]
games = db[GAMES_COLLECTION]
gestures = db[GESTURES_COLLECTION]
sensor_files = db[SENSOR_FILES_COLLECTION]


# ============================================================
# 1. LOAD USER MAPPING
#
# users._id <-> users.player_id
# ============================================================

user_id_to_player_id = {}
player_id_to_user_id = {}

for doc in users.find(
    {},
    {
        "_id": 1,
        "player_id": 1
    }
):
    player_id = doc.get("player_id")

    if not player_id:
        continue

    user_id_to_player_id[doc["_id"]] = player_id
    player_id_to_user_id[player_id] = doc["_id"]


print(f"Users with player_id: {len(player_id_to_user_id):,}")


# ============================================================
# 2. BUILD ACCELEROMETER + GYROSCOPE COVERAGE INTERVALS
# ============================================================

sensor_intervals = defaultdict(list)

sensor_documents = 0
usable_sensor_documents = 0
missing_sensor_timestamp = 0
missing_sample_counts = 0


sensor_projection = {
    "_id": 0,

    "player_id": 1,

    "timestamp": 1,
    "source_file": 1,
    "_source_file": 1,

    "accelerometer_samples": 1,
    "gyroscope_samples": 1,

    # Support old full documents as a fallback.
    "accelerometer": 1,
    "gyroscope": 1
}


for doc in sensor_files.find(
    {
        "has_accelerometer": True,
        "has_gyroscope": True
    },
    sensor_projection
):
    sensor_documents += 1

    player_id = doc.get("player_id")

    if not player_id:
        continue

    timestamp = extract_sensor_timestamp(doc)

    if timestamp is None:
        missing_sensor_timestamp += 1
        continue

    acc_count = doc.get("accelerometer_samples")
    gyro_count = doc.get("gyroscope_samples")

    # Fallback for your originally imported documents
    # that may still contain the arrays themselves.
    if acc_count is None:
        acc = doc.get("accelerometer")

        if isinstance(acc, list):
            acc_count = len(acc)

    if gyro_count is None:
        gyro = doc.get("gyroscope")

        if isinstance(gyro, list):
            gyro_count = len(gyro)

    if acc_count is None or gyro_count is None:
        missing_sample_counts += 1
        continue

    # Conservative estimate of time for which both streams exist.
    simultaneous_samples = min(
        int(acc_count),
        int(gyro_count)
    )

    if simultaneous_samples <= 0:
        continue

    duration_ms = (
        simultaneous_samples
        / SENSOR_RATE_HZ
        * 1000.0
    )

    sensor_start, sensor_stop = make_sensor_interval(
        timestamp,
        duration_ms
    )

    sensor_intervals[player_id].append(
        (sensor_start, sensor_stop)
    )

    usable_sensor_documents += 1


# Merge sensor intervals once per user.
sensor_indexes = {}

for player_id, intervals in sensor_intervals.items():
    sensor_indexes[player_id] = prepare_interval_index(
        intervals
    )


print("\n=== SENSOR DATA ===")
print(
    f"Sensor documents with accel+gyro flags: "
    f"{sensor_documents:,}"
)
print(
    f"Usable accel+gyro documents: "
    f"{usable_sensor_documents:,}"
)
print(
    f"Players with accel+gyro intervals: "
    f"{len(sensor_indexes):,}"
)
print(
    f"Missing timestamps: "
    f"{missing_sensor_timestamp:,}"
)
print(
    f"Missing sample counts: "
    f"{missing_sample_counts:,}"
)


# ============================================================
# 3. LOAD GAMES AND INDEX THEM BY DEVICE
#
# A gesture itself does not contain user_id or game_type.
# Games provide both.
# ============================================================

games_by_device = defaultdict(list)

valid_games = 0


for game_doc in games.find(
    {},
    {
        "_id": 1,
        "user_id": 1,
        "device_id": 1,
        "game_type": 1,
        "t_start": 1,
        "t_stop": 1
    }
):
    user_id = game_doc.get("user_id")

    player_id = user_id_to_player_id.get(user_id)

    if player_id is None:
        continue

    device_id = game_doc.get("device_id")
    game_type = game_doc.get("game_type")

    if device_id is None or game_type is None:
        continue

    game_start = to_epoch_ms(game_doc.get("t_start"))
    game_stop = to_epoch_ms(game_doc.get("t_stop"))

    if (
        game_start is None
        or game_stop is None
        or game_stop <= game_start
    ):
        continue

    games_by_device[device_id].append(
        {
            "_id": game_doc["_id"],
            "player_id": player_id,
            "user_id": user_id,
            "device_id": device_id,
            "game_type": game_type,
            "start": game_start,
            "stop": game_stop
        }
    )

    valid_games += 1


# Sort each device's games chronologically.
game_starts_by_device = {}

for device_id, device_games in games_by_device.items():
    device_games.sort(
        key=lambda x: x["start"]
    )

    game_starts_by_device[device_id] = [
        x["start"]
        for x in device_games
    ]


print("\n=== GAMES ===")
print(f"Valid games loaded: {valid_games:,}")
print(f"Devices represented: {len(games_by_device):,}")


# ============================================================
# 4. FIND THE GAME CONTAINING A GESTURE
# ============================================================

def find_game_for_gesture(
    device_id,
    gesture_start,
    gesture_stop
):
    """
    Find a game on this device that completely contains
    the gesture.

    Normally games on a device should not overlap.
    """

    device_games = games_by_device.get(device_id)

    if not device_games:
        return None

    starts = game_starts_by_device[device_id]

    i = bisect_right(
        starts,
        gesture_start
    ) - 1

    if i < 0:
        return None

    game = device_games[i]

    if (
        game["start"] <= gesture_start
        and gesture_stop <= game["stop"]
    ):
        return game

    return None


# ============================================================
# 5. STREAM THROUGH ALL GESTURES ONCE
#
# Do NOT load 3.1 million gestures into RAM.
# ============================================================

stats = defaultdict(
    lambda: {
        "total_gestures": 0,
        "motion_covered_gestures": 0,
        "partially_motion_overlapping_gestures": 0
    }
)


gestures_seen = 0
gestures_assigned_to_games = 0
gestures_without_game = 0
gestures_invalid_time = 0


gesture_cursor = gestures.find(
    {},
    {
        "_id": 0,
        "device_id": 1,
        "t_start": 1,
        "t_stop": 1,
        "screen": 1,
        "type": 1
    },
    no_cursor_timeout=True
).batch_size(10_000)


try:

    for gesture in gesture_cursor:

        gestures_seen += 1

        if gestures_seen % 100_000 == 0:
            print(
                f"Processed {gestures_seen:,} gestures..."
            )

        device_id = gesture.get("device_id")

        if device_id is None:
            gestures_without_game += 1
            continue

        gesture_start = to_epoch_ms(
            gesture.get("t_start")
        )

        gesture_stop = to_epoch_ms(
            gesture.get("t_stop")
        )

        # For safety, treat a zero-duration tap as a point
        # rather than discarding it.
        if gesture_start is None:
            gestures_invalid_time += 1
            continue

        if gesture_stop is None:
            gesture_stop = gesture_start

        if gesture_stop < gesture_start:
            gestures_invalid_time += 1
            continue

        game = find_game_for_gesture(
            device_id,
            gesture_start,
            gesture_stop
        )

        if game is None:
            gestures_without_game += 1
            continue

        gestures_assigned_to_games += 1

        player_id = game["player_id"]
        game_type = game["game_type"]

        key = (
            player_id,
            game_type
        )

        stats[key]["total_gestures"] += 1

        sensor_index = sensor_indexes.get(player_id)

        if sensor_index is None:
            continue

        # Full coverage is the metric requested.
        if is_fully_covered(
            gesture_start,
            gesture_stop,
            sensor_index
        ):
            stats[key][
                "motion_covered_gestures"
            ] += 1

        # Diagnostic: some motion overlap but not full coverage.
        elif has_any_overlap(
            gesture_start,
            gesture_stop,
            sensor_index
        ):
            stats[key][
                "partially_motion_overlapping_gestures"
            ] += 1

finally:
    gesture_cursor.close()


# ============================================================
# 6. CREATE RESULT TABLE
# ============================================================

rows = []

for (
    player_id,
    game_type
), values in stats.items():

    total = values["total_gestures"]

    covered = values[
        "motion_covered_gestures"
    ]

    partial = values[
        "partially_motion_overlapping_gestures"
    ]

    percentage = (
        covered / total * 100.0
        if total
        else 0.0
    )

    rows.append(
        {
            "player_id": player_id,
            "game_type": game_type,

            "total_gestures": total,

            "motion_covered_gestures": covered,

            "motion_covered_percent": percentage,

            # Mainly useful for diagnosing boundaries between
            # sensor recording intervals.
            "partial_motion_overlap_gestures": partial
        }
    )


result = pd.DataFrame(rows)

if not result.empty:

    result = result.sort_values(
        [
            "player_id",
            "game_type"
        ]
    )

result.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 7. PRINT DIAGNOSTICS AND GLOBAL SUMMARY
# ============================================================

print("\n=== GESTURE MATCHING ===")

print(
    f"Gestures scanned: "
    f"{gestures_seen:,}"
)

print(
    f"Gestures assigned to a game: "
    f"{gestures_assigned_to_games:,}"
)

print(
    f"Gestures not assigned to a game: "
    f"{gestures_without_game:,}"
)

print(
    f"Gestures with invalid timestamps: "
    f"{gestures_invalid_time:,}"
)


if not result.empty:

    total = result["total_gestures"].sum()

    covered = result[
        "motion_covered_gestures"
    ].sum()

    print("\n=== OVERALL ===")

    print(
        f"Total game gestures: "
        f"{total:,}"
    )

    print(
        f"Fully accel+gyro-covered gestures: "
        f"{covered:,}"
    )

    print(
        f"Overall covered percentage: "
        f"{covered / total * 100:.2f}%"
    )


    print("\n=== BY GAME TYPE ===")

    summary = (
        result
        .groupby(
            "game_type",
            as_index=False
        )
        .agg(
            users=("player_id", "nunique"),

            total_gestures=(
                "total_gestures",
                "sum"
            ),

            motion_covered_gestures=(
                "motion_covered_gestures",
                "sum"
            )
        )
    )

    summary["motion_covered_percent"] = (
        summary["motion_covered_gestures"]
        / summary["total_gestures"]
        * 100
    )

    print(
        summary
        .sort_values("game_type")
        .to_string(index=False)
    )


print(
    f"\nSaved result to: {OUTPUT_CSV}"
)