from pathlib import Path
import shutil
import zipfile


# Paths
input_dir = Path("input_data")

main_archive = input_dir / "hmog_dataset.zip"
public_dataset_dir = input_dir / "public_dataset"
output_dir = input_dir / "hmog_unarchived"


def extract_zip(zip_path: Path, output_path: Path) -> None:
    """Extract a ZIP archive into the specified directory."""
    output_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_path)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Extract the main HMOG archive
    # ---------------------------------------------------------
    print(f"Extracting main archive: {main_archive}")

    if not main_archive.exists():
        raise FileNotFoundError(
            f"Main archive not found: {main_archive}"
        )

    if public_dataset_dir.exists():
        print(f"Removing existing directory: {public_dataset_dir}")
        shutil.rmtree(public_dataset_dir)

    extract_zip(main_archive, input_dir)

    if not public_dataset_dir.exists():
        raise RuntimeError(
            f"Expected extracted directory not found: {public_dataset_dir}"
        )

    print(f"Main archive extracted to: {public_dataset_dir}")

    # ---------------------------------------------------------
    # 2. Find user ZIP archives
    # ---------------------------------------------------------
    user_archives = sorted(public_dataset_dir.glob("*.zip"))

    print(f"Found {len(user_archives)} user archives.")

    if not user_archives:
        raise RuntimeError(
            f"No ZIP archives found in {public_dataset_dir}"
        )

    # ---------------------------------------------------------
    # 3. Extract each user archive
    # ---------------------------------------------------------
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    for index, user_archive in enumerate(user_archives, start=1):
        user_output_dir = output_dir

        print(
            f"[{index}/{len(user_archives)}] "
            f"Extracting {user_archive.name} → {user_output_dir}"
        )

        extract_zip(user_archive, user_output_dir)

    # ---------------------------------------------------------
    # 4. Delete intermediary public_dataset directory
    # ---------------------------------------------------------
    print(f"Removing intermediary directory: {public_dataset_dir}")
    shutil.rmtree(public_dataset_dir)

    print("HMOG extraction completed successfully.")
    print(f"Extracted data: {output_dir}")


if __name__ == "__main__":
    main()
