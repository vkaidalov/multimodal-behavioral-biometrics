
import argparse
import csv
import dataclasses
import inspect
import json
import runpy
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "modality_ablation"
BASELINE = (
    ROOT / "experiments" / "architecture_study"
    / "baseline_supcon_seed20261112"
)

CONFIGS = OrderedDict([
    ("touch", ("touch",)),
    ("accel", ("accel",)),
    ("gyro", ("gyro",)),
    ("touch_accel", ("touch", "accel")),
    ("touch_gyro", ("touch", "gyro")),
    ("accel_gyro", ("accel", "gyro")),
    ("full", ("touch", "accel", "gyro")),
])

SCENARIOS = (
    "feta_6s_scroll", "feta_6s_swipe",
    "feta_7_scroll", "feta_7_swipe",
    "feta_x_scroll", "feta_x_swipe",
    "hmog_sitting", "hmog_walking",
)

SEED = 20261112
TEMPERATURE = 0.05
EPOCHS = 30


def scenario_file(root):
    return (
        root / "global" / "evaluation" / "method_study"
        / "validation" / "centroid" / "scenario_metrics.csv"
    )


def neutralize_model_inputs(enabled):
    """Patch the original baseline model without modifying source files."""
    import torch
    from gesture_model import GestureEncoder

    original_forward = GestureEncoder.forward
    signature = inspect.signature(original_forward)
    disabled = {"touch", "accel", "gyro"} - set(enabled)

    aliases = {
        "touch": ("touch", "touchscreen"),
        "accel": ("accel", "accelerometer"),
        "gyro": ("gyro", "gyroscope"),
    }

    def modality_of(path):
        name = path.lower()
        for modality, names in aliases.items():
            if any(alias in name for alias in names):
                return modality
        return None

    def neutralize(value, path, hits):
        modality = modality_of(path)

        if isinstance(value, torch.Tensor):
            if modality not in disabled:
                return value

            hits.add(modality)
            result = torch.zeros_like(value)

            # A fixed one-sample dummy mask avoids all-masked pooling.
            if "mask" in path.lower() or "valid" in path.lower():
                if result.ndim >= 2:
                    result[:, 0, ...] = 1
                elif result.numel():
                    result[...] = 1
            elif any(s in path.lower() for s in ("length", "_len")):
                result[...] = 1

            return result

        if isinstance(value, dict):
            return {
                key: neutralize(item, f"{path}.{key}", hits)
                for key, item in value.items()
            }

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            updates = {
                field.name: neutralize(
                    getattr(value, field.name),
                    f"{path}.{field.name}",
                    hits,
                )
                for field in dataclasses.fields(value)
            }
            return dataclasses.replace(value, **updates)

        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*[
                neutralize(item, f"{path}.{key}", hits)
                for key, item in zip(value._fields, value)
            ])

        if isinstance(value, tuple):
            return tuple(
                neutralize(item, f"{path}[{i}]", hits)
                for i, item in enumerate(value)
            )

        if isinstance(value, list):
            return [
                neutralize(item, f"{path}[{i}]", hits)
                for i, item in enumerate(value)
            ]

        if modality in disabled and isinstance(value, (int, float)):
            hits.add(modality)
            return 1 if any(s in path.lower() for s in ("length", "_len")) else 0

        return value

    counter = [0]

    def patched_forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        hits = set()

        for name, value in list(bound.arguments.items()):
            if name != "self":
                bound.arguments[name] = neutralize(value, name, hits)

        counter[0] += 1

        if counter[0] == 1:
            missing = disabled - hits
            if missing:
                raise RuntimeError(
                    f"Modality masking was not verified for: {sorted(missing)}. "
                    "Inspect GestureEncoder.forward input names before training."
                )
            print(
                f"[ABLATION] Enabled={sorted(enabled)} "
                f"disabled={sorted(disabled)}; masking verified",
                flush=True,
            )

        return original_forward(*bound.args, **bound.kwargs)

    GestureEncoder.forward = patched_forward


def invoke(stage, config_name, forwarded_args):
    enabled = CONFIGS[config_name]
    neutralize_model_inputs(enabled)

    scripts = {
        "train": "train_architecture_experiment.py",
        "extract": "extract_architecture_embeddings.py",
    }
    script = ROOT / scripts[stage]

    if not script.is_file():
        raise FileNotFoundError(script)

    sys.argv = [str(script), *forwarded_args]
    runpy.run_path(str(script), run_name="__main__")


def execute(command):
    print("\n[COMMAND]", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_experiment(name):
    if name == "full":
        if not scenario_file(BASELINE).exists():
            raise FileNotFoundError(
                "Existing full baseline evaluation not found: "
                + str(scenario_file(BASELINE))
            )
        print("[REUSE] Existing full CNN baseline")
        return

    run_root = RESULTS / f"{name}_supcon_seed{SEED}"

    if run_root.exists():
        if scenario_file(run_root).is_file():
            print(f"[SKIP] Completed experiment: {name}")
            return
        raise FileExistsError(
            f"Incomplete run exists: {run_root}\n"
            "Preserved to prevent overwriting. Inspect it before retrying."
        )

    prefix = [
        sys.executable, "-u", str(ROOT / "modality_ablation.py"),
        "invoke",
    ]

    print(f"\n{'=' * 60}\n[START] {name}\n{'=' * 60}", flush=True)

    execute([
        *prefix, "train", name,
        "--scenario", "global",
        "--architecture", "baseline",
        "--objective", "supcon",
        "--temperature", str(TEMPERATURE),
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--output-root", str(run_root),
    ])

    execute([
        *prefix, "extract", name,
        "--model-id", "global",
        "--role", "validation",
        "--cache", str(
            ROOT / "output_data"
            / "gesture_cache_validation_first_four"
        ),
        "--results", str(run_root),
    ])

    execute([
        sys.executable, "-u",
        str(ROOT / "evaluate_architecture_run.py"),
        "--model-id", "global",
        "--results", str(run_root),
        "--methods", "centroid",
        "--workers", "4",
    ])

    if not scenario_file(run_root).is_file():
        raise RuntimeError("Evaluation CSV was not generated.")

    manifest = {
        "configuration": name,
        "enabled_modalities": list(CONFIGS[name]),
        "disabled_modalities": sorted(
            {"touch", "accel", "gyro"} - set(CONFIGS[name])
        ),
        "architecture": "baseline",
        "objective": "supcon",
        "seed": SEED,
        "temperature": TEMPERATURE,
        "max_epochs": EPOCHS,
        "selection_metric": "validation macro AUC k=1",
        "enrollment_method": "centroid",
        "role": "validation",
        "test_used": False,
        "modality_removal": "fixed zero inputs and dummy masks",
    }

    (run_root / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] {name}", flush=True)


def read_k1(root):
    path = scenario_file(root)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))

    selected = []

    for row in all_rows:
        key = next(
            (row.get(col) for col in ("k", "window_size", "window_k")
             if row.get(col) not in (None, "")),
            None,
        )
        if key is None:
            raise ValueError(f"Missing window-size column in {path}")

        if int(float(key)) == 1:
            selected.append(row)

    names = {r["scenario"] for r in selected}
    if names != set(SCENARIOS) or len(selected) != 8:
        raise ValueError(
            f"Expected exactly 8 scenarios in {path}; got {names}"
        )

    for row in selected:
        row["auc"] = float(row["macro_auc"])
        row["eer"] = float(row["macro_eer"])

    return selected


def save_csv(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([
            {key: row.get(key, "") for key in columns}
            for row in rows
        ])


def summarize():
    import math

    by_name = {}

    for name in CONFIGS:
        root = (
            BASELINE if name == "full"
            else RESULTS / f"{name}_supcon_seed{SEED}"
        )
        by_name[name] = {
            row["scenario"]: row for row in read_k1(root)
        }

    summary_rows = []
    scenario_rows = []
    delta_rows = []

    for name, modalities in CONFIGS.items():
        rows = by_name[name]
        avg_auc = sum(r["auc"] for r in rows.values()) / 8
        avg_eer = sum(r["eer"] for r in rows.values()) / 8

        if not (math.isfinite(avg_auc) and math.isfinite(avg_eer)):
            raise ValueError(f"Non-finite metrics for {name}")

        baseline_auc = (
            sum(r["auc"] for r in by_name["full"].values()) / 8
        )

        summary_rows.append({
            "configuration": name,
            "modalities": "+".join(modalities),
            "macro_auc": avg_auc,
            "macro_eer": avg_eer,
            "delta_auc_vs_full": avg_auc - baseline_auc,
            "scenario_count": 8,
            "seed": SEED,
        })

        for scenario in SCENARIOS:
            row = rows[scenario]
            full_row = by_name["full"][scenario]

            scenario_rows.append({
                "configuration": name,
                "scenario": scenario,
                "auc": row["auc"],
                "eer": row["eer"],
            })

            if name != "full":
                delta_rows.append({
                    "configuration": name,
                    "scenario": scenario,
                    "delta_auc_vs_full": (
                        row["auc"] - full_row["auc"]
                    ),
                    "delta_eer_vs_full": (
                        row["eer"] - full_row["eer"]
                    ),
                })

    output = RESULTS / "_publication_summary_v1"
    output.mkdir(parents=True, exist_ok=False)

    save_csv(
        output / "summary.csv",
        sorted(summary_rows, key=lambda r: -r["macro_auc"]),
        [
            "configuration", "modalities", "macro_auc",
            "macro_eer", "delta_auc_vs_full",
            "scenario_count", "seed",
        ],
    )

    save_csv(
        output / "scenarios.csv",
        scenario_rows,
        ["configuration", "scenario", "auc", "eer"],
    )

    save_csv(
        output / "deltas_vs_full.csv",
        delta_rows,
        [
            "configuration", "scenario",
            "delta_auc_vs_full", "delta_eer_vs_full",
        ],
    )

    print("\n[RESULTS] Validation k=1\n", flush=True)
    for row in sorted(summary_rows, key=lambda r: -r["macro_auc"]):
        print(
            f"{row['configuration']:15} "
            f"AUC={row['macro_auc']:.5f} "
            f"EER={row['macro_eer']:.2%} "
            f"delta_AUC={row['delta_auc_vs_full']:+.5f}"
        )

    print(f"\nSaved to: {output}")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "invoke":
        invoke(sys.argv[2], sys.argv[3], sys.argv[4:])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "summarize"])
    parser.add_argument("--only", choices=list(CONFIGS))
    args = parser.parse_args()

    if args.action == "run":
        names = [args.only] if args.only else list(CONFIGS)
        for name in names:
            run_experiment(name)
    else:
        summarize()


if __name__ == "__main__":
    main()
