from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from pymongo import MongoClient


# ============================================================
# Configuration
# ============================================================

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "brainrun"

SENSOR_COLLECTION = "sensor_files"
USERS_COLLECTION = "users"
GAMES_COLLECTION = "games"

# BrainRun paper: sensors sampled every 100 ms.
SENSOR_RATE_HZ = 10.0

# "auto" = test whether the filename timestamp behaves more like
# the start or the end of a sensor recording.
#
# You can override this with "start" or "end" after inspecting
# the diagnostics printed by the script.
TIMESTAMP_MODE = "auto"

OUTPUT_CSV = Path("output_data/brainrun_accel_gyro_games_by_user_device.csv")
GAME_DETAILS_CSV = Path("output_data/brainrun_accel_gyro_game_details.csv")


# ============================================================
# Timestamp helpers
# ============================================================

SOURCE_TIMESTAMP_RE = re.compile(r"_(\d{10,16})\.json$", re.IGNORECASE)


def to_epoch_ms(value):
    """
    Convert common MongoDB timestamp representations to epoch ms.

    Supports:
      - BSON/Python datetime
      - integers/floats in seconds or milliseconds
      - numeric strings
      - ISO datetime strings
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.timestamp() * 1000.0

    if isinstance(value, (int, float)):
        value = float(value)

        # ~1e9 = Unix seconds around recent decades.
        # ~1e12 = Unix milliseconds.
        if value < 100_000_000_000:
            return value * 1000.0

        return value

    if isinstance(value, str):
        s = value.strip()

        try:
            return to_epoch_ms(float(s))
        except ValueError:
            pass

        try:
            # Handle trailing Z.
            s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.timestamp() * 1000.0
        except ValueError:
            return None

    return None


def get_sensor_timestamp_ms(doc):
    """
    Prefer the timestamp field produced by the sensor metadata importer.
    Fall back to extracting it from _source_file/source_file.
    """
    ts = to_epoch_ms(doc.get("timestamp"))
    if ts is not None:
        return ts

    source = doc.get("source_file") or doc.get("_source_file")

    if not source:
        return None

    filename = source.replace("\\", "/").rsplit("/", 1)[-1]

    m = SOURCE_TIMESTAMP_RE.search(filename)
    if not m:
        return None

    return to_epoch_ms(int(m.group(1)))


# ============================================================
# Interval helpers
# ============================================================

def make_sensor_interval(timestamp_ms, duration_ms, mode):
    if mode == "start":
        return timestamp_ms, timestamp_ms + duration_ms

    if mode == "end":
        return timestamp_ms - duration_ms, timestamp_ms

    raise ValueError(f"Unknown timestamp mode: {mode}")


def merge_intervals(intervals):
    """
    Merge overlapping sensor intervals.

    This is important so that overlapping sensor files do not cause
    the same second of a game to be counted twice.
    """
    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [list(intervals[0])]

    for start, stop in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], stop)
        else:
            merged.append([start, stop])

    return [(a, b) for a, b in merged]


def overlap_ms(start, stop, intervals):
    """
    Total overlap of [start, stop] with already-merged intervals.
    """
    total = 0.0

    for sensor_start, sensor_stop in intervals:
        if sensor_stop <= start:
            continue

        if sensor_start >= stop:
            break

        left = max(start, sensor_start)
        right = min(stop, sensor_stop)

        if right > left:
            total += right - left

    return total


# ============================================================
# MongoDB
# ============================================================

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

sensor_col = db[SENSOR_COLLECTION]
users_col = db[USERS_COLLECTION]
games_col = db[GAMES_COLLECTION]


# ============================================================
# 1. Load only sensor metadata having BOTH accel + gyro
# ============================================================

sensor_query = {
    "has_accelerometer": True,
    "has_gyroscope": True,
}

sensor_projection = {
    "_id": 0,
    "player_id": 1,
    "timestamp": 1,
    "source_file": 1,
    "_source_file": 1,
    "accelerometer_samples": 1,
    "gyroscope_samples": 1,
}

sensor_records = defaultdict(list)

n_sensor_docs = 0
n_sensor_docs_usable = 0
n_missing_timestamp = 0
n_missing_counts = 0

for doc in sensor_col.find(sensor_query, sensor_projection):
    n_sensor_docs += 1

    player_id = doc.get("player_id")
    if not player_id:
        continue

    ts_ms = get_sensor_timestamp_ms(doc)
    if ts_ms is None:
        n_missing_timestamp += 1
        continue

    acc_n = doc.get("accelerometer_samples")
    gyro_n = doc.get("gyroscope_samples")

    # Support the original fully imported documents too,
    # if some of them still contain raw arrays.
    if acc_n is None and isinstance(doc.get("accelerometer"), list):
        acc_n = len(doc["accelerometer"])

    if gyro_n is None and isinstance(doc.get("gyroscope"), list):
        gyro_n = len(doc["gyroscope"])

    if acc_n is None or gyro_n is None:
        n_missing_counts += 1
        continue

    paired_samples = min(int(acc_n), int(gyro_n))

    if paired_samples <= 0:
        continue

    duration_ms = paired_samples / SENSOR_RATE_HZ * 1000.0

    sensor_records[player_id].append(
        {
            "timestamp_ms": ts_ms,
            "duration_ms": duration_ms,
            "paired_samples": paired_samples,
        }
    )

    n_sensor_docs_usable += 1


print("\n=== Sensor metadata ===")
print(f"Documents with accel + gyro flags: {n_sensor_docs:,}")
print(f"Usable sensor documents:           {n_sensor_docs_usable:,}")
print(f"Missing timestamp:                 {n_missing_timestamp:,}")
print(f"Missing sample counts:             {n_missing_counts:,}")
print(f"Players with usable accel+gyro:    {len(sensor_records):,}")


# ============================================================
# 2. player_id -> MongoDB users._id
# ============================================================

player_ids = list(sensor_records.keys())

player_to_user_id = {}

for user in users_col.find(
    {"player_id": {"$in": player_ids}},
    {"_id": 1, "player_id": 1},
):
    player_to_user_id[user["player_id"]] = user["_id"]


user_id_to_player = {
    user_id: player_id
    for player_id, user_id in player_to_user_id.items()
}

print(f"Players matched to users collection: {len(player_to_user_id):,}")


# ============================================================
# 3. Load games for those users
# ============================================================

games_by_player = defaultdict(list)

game_projection = {
    "_id": 1,
    "user_id": 1,
    "device_id": 1,
    "game_type": 1,
    "game": 1,
    "stage": 1,
    "base": 1,
    "t_start": 1,
    "t_stop": 1,
}

user_ids = list(user_id_to_player.keys())

for game in games_col.find(
    {"user_id": {"$in": user_ids}},
    game_projection,
):
    player_id = user_id_to_player.get(game.get("user_id"))
    if player_id is None:
        continue

    start = to_epoch_ms(game.get("t_start"))
    stop = to_epoch_ms(game.get("t_stop"))

    if start is None or stop is None or stop <= start:
        continue

    game["_start_ms"] = start
    game["_stop_ms"] = stop

    games_by_player[player_id].append(game)


for games in games_by_player.values():
    games.sort(key=lambda g: g["_start_ms"])


print(
    "Valid games belonging to accel+gyro users:",
    f"{sum(len(x) for x in games_by_player.values()):,}",
)


# ============================================================
# 4. Determine whether file timestamp is START or END
# ============================================================

def build_intervals(mode):
    result = {}

    for player_id, recordings in sensor_records.items():
        intervals = []

        for recording in recordings:
            interval = make_sensor_interval(
                recording["timestamp_ms"],
                recording["duration_ms"],
                mode,
            )
            intervals.append(interval)

        result[player_id] = merge_intervals(intervals)

    return result


def score_timestamp_mode(mode):
    intervals_by_player = build_intervals(mode)

    total_overlap = 0.0
    games_touched = 0

    for player_id, games in games_by_player.items():
        intervals = intervals_by_player.get(player_id, [])

        for game in games:
            overlap = overlap_ms(
                game["_start_ms"],
                game["_stop_ms"],
                intervals,
            )

            if overlap > 0:
                games_touched += 1
                total_overlap += overlap

    return total_overlap, games_touched, intervals_by_player


if TIMESTAMP_MODE == "auto":
    start_overlap, start_games, start_intervals = score_timestamp_mode("start")
    end_overlap, end_games, end_intervals = score_timestamp_mode("end")

    print("\n=== Filename timestamp interpretation ===")
    print(
        "Assuming timestamp = START:",
        f"{start_games:,} games touched, "
        f"{start_overlap / 3_600_000:.2f} overlap hours",
    )
    print(
        "Assuming timestamp = END:  ",
        f"{end_games:,} games touched, "
        f"{end_overlap / 3_600_000:.2f} overlap hours",
    )

    if start_overlap >= end_overlap:
        timestamp_mode = "start"
        intervals_by_player = start_intervals
    else:
        timestamp_mode = "end"
        intervals_by_player = end_intervals

    print(f"\nAutomatically selected: {timestamp_mode.upper()}")

else:
    timestamp_mode = TIMESTAMP_MODE
    _, _, intervals_by_player = score_timestamp_mode(timestamp_mode)


# ============================================================
# 5. Match games against accel+gyro intervals
# ============================================================

# Aggregate key:
# player_id + device_id + game_type
stats = defaultdict(
    lambda: {
        "games_with_any_sensor_overlap": 0,
        "games_fully_sensor_covered": 0,
        "game_duration_ms": 0.0,
        "sensor_covered_ms": 0.0,
    }
)

game_detail_rows = []

for player_id, games in games_by_player.items():
    intervals = intervals_by_player.get(player_id, [])

    if not intervals:
        continue

    for game in games:
        game_start = game["_start_ms"]
        game_stop = game["_stop_ms"]

        duration = game_stop - game_start

        covered = overlap_ms(
            game_start,
            game_stop,
            intervals,
        )

        if covered <= 0:
            continue

        device_id = game.get("device_id")
        game_type = game.get("game_type")

        key = (
            player_id,
            str(device_id),
            game_type,
        )

        s = stats[key]

        s["games_with_any_sensor_overlap"] += 1
        s["game_duration_ms"] += duration
        s["sensor_covered_ms"] += covered

        coverage = covered / duration

        # Allow a tiny numerical tolerance.
        if coverage >= 0.999:
            s["games_fully_sensor_covered"] += 1

        game_detail_rows.append(
            {
                "player_id": player_id,
                "device_id": str(device_id),
                "game_type": game_type,
                "game_id": str(game.get("_id")),
                "stage": game.get("stage"),
                "base": game.get("base"),
                "game": game.get("game"),
                "game_start_ms": game_start,
                "game_stop_ms": game_stop,
                "game_duration_seconds": duration / 1000.0,
                "accel_gyro_covered_seconds": covered / 1000.0,
                "coverage_percent": coverage * 100.0,
            }
        )


# ============================================================
# 6. Produce per-user / per-device / per-game-type table
# ============================================================

rows = []

for (
    player_id,
    device_id,
    game_type,
), s in stats.items():

    game_hours = s["game_duration_ms"] / 3_600_000.0
    covered_hours = s["sensor_covered_ms"] / 3_600_000.0

    coverage_pct = (
        100.0 * s["sensor_covered_ms"] / s["game_duration_ms"]
        if s["game_duration_ms"] > 0
        else 0.0
    )

    rows.append(
        {
            "player_id": player_id,
            "device_id": device_id,
            "game_type": game_type,

            # Number of games where at least some portion coincided
            # with both accel + gyro availability.
            "games_with_accel_gyro": s[
                "games_with_any_sensor_overlap"
            ],

            # Much stricter criterion.
            "games_fully_covered": s[
                "games_fully_sensor_covered"
            ],

            # Entire duration of qualifying games.
            "qualifying_game_hours": game_hours,

            # Actual time inside those games for which we estimate
            # both accel + gyro to have been available.
            "accel_gyro_covered_hours": covered_hours,

            "coverage_percent": coverage_pct,
        }
    )


result = pd.DataFrame(rows)

if not result.empty:
    result = result.sort_values(
        [
            "player_id",
            "device_id",
            "game_type",
        ]
    )

result.to_csv(OUTPUT_CSV, index=False)


details = pd.DataFrame(game_detail_rows)

if not details.empty:
    details = details.sort_values(
        [
            "player_id",
            "device_id",
            "game_start_ms",
        ]
    )

details.to_csv(GAME_DETAILS_CSV, index=False)


# ============================================================
# 7. Print results
# ============================================================

print("\n=== Results ===")
print(f"Timestamp interpretation: {timestamp_mode}")

print(
    "Games with some accel+gyro coverage:",
    f"{len(game_detail_rows):,}",
)

if not result.empty:
    print(
        "Unique player-device pairs:",
        result[["player_id", "device_id"]]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Users represented:",
        result["player_id"].nunique(),
    )

    print(
        "Devices represented:",
        result["device_id"].nunique(),
    )

    print("\nTotals by game type:")

    summary = (
        result
        .groupby("game_type", as_index=False)
        .agg(
            users=("player_id", "nunique"),
            user_device_pairs=("device_id", "count"),
            games=("games_with_accel_gyro", "sum"),
            qualifying_game_hours=("qualifying_game_hours", "sum"),
            accel_gyro_covered_hours=("accel_gyro_covered_hours", "sum"),
        )
        .sort_values("games", ascending=False)
    )

    print(summary.to_string(index=False))

print(f"\nSaved: {OUTPUT_CSV.resolve()}")
print(f"Saved: {GAME_DETAILS_CSV.resolve()}")
