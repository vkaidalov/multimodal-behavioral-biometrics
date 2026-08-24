import re
from pathlib import Path


def validate_hmog_dataset():
    base_dir = Path("input_data/hmog_unarchived")

    # Common CSV files required in every single session
    base_csv_files = [
        "Accelerometer.csv", "Activity.csv", "Gyroscope.csv",
        "KeyPressEvent.csv", "Magnetometer.csv", "OneFingerTouchEvent.csv",
        "PinchEvent.csv", "ScrollEvent.csv", "StrokeEvent.csv", "TouchEvent.csv"
    ]

    # Specific task-dependent answer files
    reading_answers = ["Reading1answer", "Reading2answer", "Reading3answer"]
    writing_answers = ["Writing1answer", "Writing2answer", "Writing3answer"]

    # Define session groups
    reading_sessions = {1, 7, 13, 19}
    writing_sessions = {3, 9, 15, 21}

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"❌ Error: Base directory '{base_dir}' not found.")
        return

    subject_dir_pattern = re.compile(r"^\d{6}$")

    # 1. Find and count 6-digit folders
    all_items = list(base_dir.iterdir())
    subject_folders = [d for d in all_items if d.is_dir() and subject_dir_pattern.match(d.name)]

    print(f"Found {len(subject_folders)} valid 6-digit subject folders.")
    if len(subject_folders) != 100:
        print(f"⚠️ Warning: Expected exactly 100 subject folders, but found {len(subject_folders)}.")
    else:
        print("✅ Subject folder count is exactly 100.")

    missing_files_report = []

    # 2. Iterate through each valid subject folder
    for subject_dir in subject_folders:
        subject_id = subject_dir.name
        session_pattern = re.compile(rf"^{subject_id}_session_([1-9]|1[0-9]|2[0-4])$")

        session_folders = [d for d in subject_dir.iterdir() if d.is_dir()]

        for session_dir in session_folders:
            match = session_pattern.match(session_dir.name)
            if not match:
                print(f"⚠️ Warning: Unexpected folder format found and skipped: '{session_dir}'")
                continue

            session_num = int(match.group(1))

            # Determine expected files for this specific session
            expected_files = list(base_csv_files)

            if session_num in reading_sessions:
                expected_files.extend(reading_answers)
            elif session_num in writing_sessions:
                expected_files.extend(writing_answers)
            # Map navigation sessions (5, 6, 11, 12, 17, 18, 23, 24) don't have answer files

            # 3. Check the presence of mapped files
            for file_name in expected_files:
                file_path = session_dir / file_name
                if not file_path.exists():
                    missing_files_report.append((session_dir.relative_to(base_dir), file_name))

    # 4. Output Results
    print("\n--- Validation Report ---")
    if not missing_files_report:
        print("✅ All required task-specific files are present in all session folders!")
    else:
        print(f"❌ Found {len(missing_files_report)} missing files:")
        for folder, file in missing_files_report:
            print(f"  - Missing '{file}' in '{folder}'")


if __name__ == "__main__":
    validate_hmog_dataset()
