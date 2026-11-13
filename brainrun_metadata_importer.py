import json
import re
import zipfile

from pymongo import MongoClient, UpdateOne


ZIP_PATH = r"input_data/sensors_data.zip"

client = MongoClient("mongodb://localhost:27017")
db = client["brainrun"]
collection = db["sensor_files"]


# Filename expected to end in:
# <player_id>_<unix_timestamp_ms>.json
filename_re = re.compile(r"([^/]+)_(\d+)\.json$")

BATCH_SIZE = 1000
batch = []


def flush():
    global batch

    if not batch:
        return

    collection.bulk_write(batch, ordered=False)

    print(f"Written {len(batch)} metadata records")

    batch = []


with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    json_files = [
        name
        for name in archive.namelist()
        if name.lower().endswith(".json")
    ]

    print(f"JSON files in archive: {len(json_files)}")

    for i, name in enumerate(json_files, start=1):

        with archive.open(name) as f:
            data = json.load(f)

        filename = name.rsplit("/", 1)[-1]

        match = filename_re.search(filename)

        if match:
            filename_player_id = match.group(1)
            timestamp = int(match.group(2))
        else:
            filename_player_id = None
            timestamp = None

        player_id = data.get("player_id")

        accelerometer = data.get("accelerometer") or []
        gyroscope = data.get("gyroscope") or []
        magnetometer = data.get("magnetometer") or []
        device_motion = data.get("deviceMotion") or []

        document = {
            "source_file": name,
            "player_id": player_id,
            "timestamp": timestamp,

            "accelerometer_samples": len(accelerometer),
            "gyroscope_samples": len(gyroscope),
            "magnetometer_samples": len(magnetometer),
            "deviceMotion_samples": len(device_motion),

            "has_accelerometer": len(accelerometer) > 0,
            "has_gyroscope": len(gyroscope) > 0,
            "has_magnetometer": len(magnetometer) > 0,
            "has_deviceMotion": len(device_motion) > 0,
        }

        # Useful integrity check
        if (
            filename_player_id is not None
            and player_id != filename_player_id
        ):
            document["player_id_mismatch"] = True
            document["filename_player_id"] = filename_player_id

        # Upsert makes the script safe to restart.
        batch.append(
            UpdateOne(
                {"source_file": name},
                {"$set": document},
                upsert=True
            )
        )

        if len(batch) >= BATCH_SIZE:
            flush()

        if i % 1000 == 0:
            print(f"{i}/{len(json_files)} processed")

flush()

collection.create_index("source_file", unique=True)
collection.create_index("player_id")
collection.create_index("timestamp")
collection.create_index([
    ("has_accelerometer", 1),
    ("has_gyroscope", 1)
])

print("Finished.")
