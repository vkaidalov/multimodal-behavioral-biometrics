import zipfile
import json
import logging
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ZIP_PATH = "./input_data/sensors_data.zip"

client = MongoClient("mongodb://localhost:27017")
db = client["brainrun"]
collection = db["sensors"]

with zipfile.ZipFile(ZIP_PATH) as zf:
    processed_files = 0
    for name in zf.namelist():
        if not name.endswith(".json"):
            continue

        with zf.open(name) as f:
            data = json.load(f)

        # Useful for tracing the original dataset file
        data["_source_file"] = name

        collection.insert_one(data)
        processed_files += 1
        if processed_files % 1000 == 0:
            logging.info("Processed %s files", processed_files)
