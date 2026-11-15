
#!/usr/bin/env python3
"""Final test evaluation for the seven baseline-CNN modality ablations.

Additive script: does not alter existing training, extraction or evaluation code.
Commands: preflight, extract, evaluate, summarize.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ABLATION = Path("experiments/modality_ablation")
ARCH = Path("experiments/architecture_study")
TEST_CACHE = Path("output_data/gesture_cache_test_first_four")
SPLITS = Path("experiments/splits/hmog_feta_splits_seed_20261112.json")

SEED = 20261112
WINDOWS = list(range(1, 21))
WINDOW_TEXT = ",".join(map(str, WINDOWS))
TARGET_FAR = 0.05
IMPOSTOR_PER_SOURCE = 100

MODALITIES = {
    "touch": "Touch",
    "accel": "Accelerometer",
    "gyro": "Gyroscope",
    "touch_accel": "Touch + Accelerometer",
    "touch_gyro": "Touch + Gyroscope",
    "accel_gyro": "Accelerometer + Gyroscope",
    "full": "Touch + Accelerometer + Gyroscope",
}

SCENARIOS = (
    "feta_6s_scroll",
    "feta_6s_swipe",
    "feta_7_scroll",
    "feta_7_swipe",
    "feta_x_scroll",
    "feta_x_swipe",
    "hmog_sitting",
    "hmog_walking",
)

FREEZE = ABLATION / "_test_protocol_frozen_v1.json"
SUMMARY_DIR = ABLATION / "_publication_test_v1"


def absolute(path):
    return BASE / path


def require(path):
    path = absolute(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_root(name):
    if name == "full":
        return ARCH / f"baseline_supcon_seed{SEED}"
    return ABLATION / f"{name}_supcon_seed{SEED}"


def embedding_path(name, role):
    return run_root(name) / "global" / "embeddings" / f"{role}.npz"


def calibration_path(name):
    return (
        run_root(name)
        / "global/evaluation/method_study/validation"
        / "centroid/thresholds_validation.json"
    )


def metrics_path(name, role):
    return (
        run_root(name)
        / "global/evaluation/method_study"
        / role / "centroid/scenario_metrics.csv"
    )


def evaluation_dir(name):
    return (
        run_root(name)
        / "global/evaluation/method_study/test/centroid"
    )


def load_calibration(name):
    path = require(calibration_path(name))
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("model_id") != "global":
        raise ValueError(f"{name}: expected global model")

    if data.get("windows") != WINDOWS:
        raise ValueError(
            f"{name}: validation windows differ from {WINDOWS}"
        )

    if int(data.get("seed", -1)) != SEED:
        raise ValueError(f"{name}: impostor sampling seed mismatch")

    if int(data.get("impostor_per_source", -1)) != IMPOSTOR_PER_SOURCE:
        raise ValueError(f"{name}: impostor sampling policy mismatch")

    if not math.isclose(
        float(data.get("target_far", -1)), TARGET_FAR
    ):
        raise ValueError(f"{name}: validation FAR target mismatch")

    expected_keys = {
        f"{scenario}/k{k}"
        for scenario in SCENARIOS
        for k in WINDOWS
    }

    if not expected_keys.issubset(data.get("thresholds", {})):
        raise ValueError(f"{name}: missing calibration entries")

    if data.get("splits_sha256") != sha256(require(SPLITS)):
        raise ValueError(f"{name}: frozen split SHA mismatch")

    validation_npz = require(embedding_path(name, "validation"))
    origin_hash = data.get("source_validation_embedding_sha256")
    if origin_hash and origin_hash != sha256(validation_npz):
        raise ValueError(
            f"{name}: calibration was fitted to different validation embeddings"
        )

    return data


def make_manifest():
    manifest = {
        "experiment": "baseline_cnn_modality_ablation",
        "role": "test",
        "selection_role": "validation",
        "model_id": "global",
        "method": "centroid",
        "windows": WINDOWS,
        "seed": SEED,
        "impostor_per_source": IMPOSTOR_PER_SOURCE,
        "target_far": TARGET_FAR,
        "splits_sha256": sha256(require(SPLITS)),
        "runs": {},
        "source_code_sha256": {},
    }

    for script in (
        "modality_ablation.py",
        "extract_architecture_embeddings.py",
        "evaluate_authentication.py",
    ):
        manifest["source_code_sha256"][script] = sha256(require(script))

    for name in MODALITIES:
        root = run_root(name)
        ckpt = require(root / "global/model_best.pt")
        val_results = require(metrics_path(name, "validation"))
        cal = load_calibration(name)

        manifest["runs"][name] = {
            "run_root": root.as_posix(),
            "checkpoint_sha256": sha256(ckpt),
            "calibration_sha256": sha256(calibration_path(name)),
            "validation_metrics_sha256": sha256(val_results),
            "checkpoint_epoch": cal.get("checkpoint_epoch"),
            "normalizer_sha256": cal.get("normalizer_sha256"),
        }

    return manifest


def preflight():
    if Path.cwd().resolve() != BASE:
        raise RuntimeError("Run commands from the repository root")

    # Confirm the installed evaluator supports method-specific TEST.
    help_result = subprocess.run(
        [sys.executable, "evaluate_authentication.py", "--help"],
        cwd=BASE,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )

    for flag in ("--methods", "--calibration", "--role"):
        if flag not in help_result.stdout:
            raise RuntimeError(
                f"Evaluator lacks {flag}. Install Enrollment Methods v2."
            )

    expected = make_manifest()
    path = absolute(FREEZE)

    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen != expected:
            raise RuntimeError(
                "Frozen protocol or source/checkpoint/calibration changed. "
                "Refusing to continue."
            )
        print("[OK] Existing frozen protocol matches", flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)
        print(f"[FROZEN] {path}", flush=True)

    print("[OK] Seven validation calibrations and checkpoints verified")
    print("[OK] Test users were not used in this preflight")


def execute(command):
    print("\n[COMMAND]", " ".join(map(str, command)), flush=True)
    subprocess.run(
        list(map(str, command)),
        cwd=BASE,
        check=True,
    )


def check_embedding(name):
    """Verify provenance before reusing or evaluating an existing TEST file."""
    from gesture_evaluation_core import load_embedding_table

    path = require(embedding_path(name, "test"))
    embeddings, meta, extra = load_embedding_table(path)
    cal = load_calibration(name)

    if extra.get("role") != "test":
        raise ValueError(f"{name}: embeddings are not TEST data")

    if extra.get("model_id") != "global":
        raise ValueError(f"{name}: incorrect model ID")

    for key in (
        "splits_sha256",
        "normalizer_sha256",
        "checkpoint_epoch",
    ):
        if extra.get(key) != cal.get(key):
            raise ValueError(
                f"{name}: test embeddings/calibration mismatch: {key}"
            )

    if len(embeddings) == 0:
        raise ValueError(f"{name}: empty test embeddings")

    print(
        f"[VERIFIED] {name}: {len(embeddings)} test embeddings",
        flush=True,
    )


def extract(names):
    preflight()
    require(TEST_CACHE / "index.json")

    for name in names:
        target = absolute(embedding_path(name, "test"))

        if target.exists():
            check_embedding(name)
            print(f"[SKIP] Existing test embeddings: {name}")
            continue

        execute([
            sys.executable, "-u",
            "modality_ablation.py",
            "invoke", "extract", name,
            "--model-id", "global",
            "--role", "test",
            "--cache", TEST_CACHE,
            "--results", run_root(name),
        ])

        check_embedding(name)


def evaluate(names):
    preflight()

    for name in names:
        check_embedding(name)

        result = absolute(metrics_path(name, "test"))
        output_dir = absolute(evaluation_dir(name))

        if result.is_file():
            print(f"[SKIP] Existing test evaluation: {name}")
            continue

        if output_dir.exists():
            raise FileExistsError(
                f"Incomplete evaluation directory exists: {output_dir}. "
                "Inspect it before retrying; no automatic overwrites."
            )

        execute([
            sys.executable, "-u",
            "evaluate_authentication.py",
            "--model-id", "global",
            "--role", "test",
            "--results", run_root(name),
            "--methods", "centroid",
            "--calibration", calibration_path(name),
            "--splits", SPLITS,
            "--windows", WINDOW_TEXT,
            "--impostor-per-source", str(IMPOSTOR_PER_SOURCE),
            "--target-far", str(TARGET_FAR),
            "--seed", str(SEED),
            "--workers", "4",
        ])

        require(metrics_path(name, "test"))
        print(f"[DONE] {name}: test evaluation complete")


def read_metrics(name, role):
    path = require(metrics_path(name, role))

    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    result = {}
    for row in rows:
        key = (row["scenario"], int(float(row["k"])))
        if key in result:
            raise ValueError(f"Duplicate row: {name}/{role}/{key}")
        result[key] = row

    expected = {
        (scenario, k)
        for scenario in SCENARIOS
        for k in WINDOWS
    }
    if set(result) != expected:
        raise ValueError(
            f"{name}/{role}: incomplete scenario/window metrics"
        )

    return result


def number(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "None", "nan", "NaN"):
            return float(value)
    return float("nan")


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save empty table")

    columns = list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    preflight()

    validation = {
        name: read_metrics(name, "validation")
        for name in MODALITIES
    }
    testing = {
        name: read_metrics(name, "test")
        for name in MODALITIES
    }

    per_scenario = []
    for name in MODALITIES:
        for scenario in SCENARIOS:
            for k in WINDOWS:
                val = validation[name][(scenario, k)]
                test = testing[name][(scenario, k)]

                per_scenario.append({
                    "configuration": name,
                    "modalities": MODALITIES[name],
                    "scenario": scenario,
                    "k": k,
                    "validation_auc": number(val, "macro_auc"),
                    "validation_eer": number(val, "macro_eer"),
                    "test_auc": number(test, "macro_auc"),
                    "test_eer": number(test, "macro_eer"),
                    "test_far": number(
                        test, "macro_far", "macro_operating_far"
                    ),
                    "test_frr": number(
                        test, "macro_frr", "macro_operating_frr"
                    ),
                    "test_coverage": number(test, "coverage"),
                    "test_evaluable_users": number(
                        test, "evaluable_users"
                    ),
                    "test_status": test.get("status", ""),
                })

    by_config_k = []
    for name in MODALITIES:
        for k in WINDOWS:
            rows = [
                row for row in per_scenario
                if row["configuration"] == name and row["k"] == k
            ]

            # Require complete coverage of scenario-level metric values
            # for the primary k=1 publication comparison.
            if k == 1 and (
                len(rows) != 8
                or not all(math.isfinite(r["test_auc"]) for r in rows)
                or not all(math.isfinite(r["test_eer"]) for r in rows)
            ):
                raise ValueError(
                    f"{name}: missing primary k=1 test metrics"
                )

            def mean_finite(field):
                values = [
                    row[field]
                    for row in rows if math.isfinite(row[field])
                ]
                return (
                    statistics.mean(values)
                    if values else float("nan")
                )

            by_config_k.append({
                "configuration": name,
                "modalities": MODALITIES[name],
                "k": k,
                "validation_macro_auc": mean_finite("validation_auc"),
                "validation_macro_eer": mean_finite("validation_eer"),
                "test_macro_auc": mean_finite("test_auc"),
                "test_macro_eer": mean_finite("test_eer"),
                "test_macro_far": mean_finite("test_far"),
                "test_macro_frr": mean_finite("test_frr"),
                "test_mean_coverage": mean_finite("test_coverage"),
                "valid_test_scenarios": sum(
                    math.isfinite(row["test_auc"])
                    for row in rows
                ),
            })

    full = {
        row["scenario"]: row
        for row in per_scenario
        if row["configuration"] == "full" and row["k"] == 1
    }

    deltas = []
    for row in per_scenario:
        if row["k"] != 1 or row["configuration"] == "full":
            continue
        reference = full[row["scenario"]]
        deltas.append({
            "configuration": row["configuration"],
            "scenario": row["scenario"],
            "test_delta_auc_vs_full": (
                row["test_auc"] - reference["test_auc"]
            ),
            "test_delta_eer_vs_full": (
                row["test_eer"] - reference["test_eer"]
            ),
        })

    output = absolute(SUMMARY_DIR)
    output.mkdir(parents=True, exist_ok=False)

    save_csv(output / "test_all_scenarios_windows.csv", per_scenario)
    save_csv(output / "test_summary_all_windows.csv", by_config_k)
    save_csv(
        output / "test_summary_k1.csv",
        [row for row in by_config_k if row["k"] == 1],
    )
    save_csv(output / "test_deltas_vs_full_k1.csv", deltas)

    print("\nFINAL TEST: k=1", flush=True)
    print("-" * 70)
    for row in by_config_k:
        if row["k"] == 1:
            print(
                f"{row['configuration']:15} "
                f"AUC={row['test_macro_auc']:.5f} "
                f"EER={row['test_macro_eer']:.2%} "
                f"FAR={row['test_macro_far']:.2%} "
                f"FRR={row['test_macro_frr']:.2%}",
                flush=True,
            )

    print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "extract", "evaluate", "summarize"),
    )
    parser.add_argument(
        "--only",
        choices=list(MODALITIES),
        default=None,
        help="Run extraction/evaluation for one configuration only",
    )
    args = parser.parse_args()

    names = [args.only] if args.only else list(MODALITIES)

    if args.action == "preflight":
        preflight()
    elif args.action == "extract":
        extract(names)
    elif args.action == "evaluate":
        evaluate(names)
    else:
        if args.only is not None:
            parser.error("--only is not available for summarize")
        summarize()


if __name__ == "__main__":
    main()
