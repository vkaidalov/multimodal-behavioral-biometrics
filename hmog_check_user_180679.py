import pandas as pd

sessions = pd.read_csv(
    "output_data/hmog_raw_feta_audit/sessions.csv"
)

session = sessions[
    (sessions["user_id"].astype(str) == "180679") &
    (sessions["session_number"].astype(str) == "12")
].iloc[0]

gestures = pd.read_csv(session["recovered_gestures_file"])

swipes = gestures[
    (gestures["type"] == "swipe") &
    (gestures["portrait_single_valid"] == 1)
]

failed = swipes[swipes["both_strict_paired"] == 0]

reasons = {
    "Fewer than 3 sensor samples": (
        (failed["accel_samples"] < 3) |
        (failed["gyro_samples"] < 3)
    ),
    "Sensor edge gap > 40 ms": (
        (failed["accel_edge_max_ms"] > 40) |
        (failed["gyro_edge_max_ms"] > 40)
    ),
    "Internal sensor gap > 120 ms": (
        (failed["accel_max_gap_ms"] > 120) |
        (failed["gyro_max_gap_ms"] > 120)
    ),
    "No accel/gyro pair within 25 ms": (
        failed["accel_gyro_pairs"] < 1
    ),
}

print(f"Portrait swipes: {len(swipes)}")
print(f"Failed swipes: {len(failed)}")

for name, condition in reasons.items():
    print(f"{name}: {condition.sum()}")

print(
    failed[[
        "duration_ms",
        "accel_samples",
        "gyro_samples",
        "accel_edge_max_ms",
        "gyro_edge_max_ms",
        "accel_max_gap_ms",
        "gyro_max_gap_ms",
        "accel_gyro_pairs",
    ]]
)