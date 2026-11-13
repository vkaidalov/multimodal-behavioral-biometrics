
from pathlib import Path
import pandas as pd

root = Path("input_data/BB-MAS_Dataset/BB-MAS_Dataset")

for user_id in (1, 9):
    print(f"\n{'=' * 50}\nUSER {user_id}")

    for kind in ("Accelerometer", "Gyroscope", "TouchEvent", "Keyboard"):
        files = list(root.rglob(
            f"{user_id}_HandPhone_{kind}_*.csv"
        ))

        if not files:
            print(f"{kind}: NOT FOUND")
            continue

        path = files[0]
        columns = ["time"]
        if kind == "TouchEvent":
            columns.append("actionType")

        df = pd.read_csv(path, usecols=columns, nrows=50000)

        print(f"\n{path.name}")
        print("Timestamp dtype:", df["time"].dtype)
        print("First timestamps:",
              [repr(v) for v in df["time"].head(5)])
        print("Last sampled timestamps:",
              [repr(v) for v in df["time"].tail(3)])

        if kind == "TouchEvent":
            print("Action types:")
            print(
                df["actionType"].value_counts(
                    dropna=False
                ).to_string()
            )
