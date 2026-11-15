
#!/usr/bin/env python3
"""Publication-ready black-and-white SVG figures for the HMOG/FETA study.

Figures:
  1. Test EER vs aggregation size for the three best modality combinations.
  2. Test EER vs aggregation size across four scenarios (full model).
  3. Macro ROC curves at k=20 for three contrasting scenarios.
  4. Test EER vs aggregation size for two contrasting participants.

No training or evaluation is performed.
Existing experiment results are never modified.
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

PUBLICATION = (
    ROOT
    / "experiments"
    / "modality_ablation"
    / "_publication_test_v1"
)

SUMMARY_CSV = PUBLICATION / "test_summary_all_windows.csv"
SCENARIOS_CSV = PUBLICATION / "test_all_scenarios_windows.csv"

BASELINE = (
    ROOT
    / "experiments"
    / "architecture_study"
    / "baseline_supcon_seed20261112"
    / "global"
)

EVALUATION = (
    BASELINE
    / "evaluation"
    / "method_study"
    / "test"
    / "centroid"
)

PER_USER_CSV = EVALUATION / "per_user_metrics.csv"

OUTPUT = PUBLICATION / "figures_svg_v3"


# ------------------------------------------------------------
# Labels and styles
# ------------------------------------------------------------

SCENARIO_NAMES = {
    "hmog_walking": "HMOG: reading walking",
    "hmog_sitting": "HMOG: reading sitting",
    "feta_6s_scroll": "FETA 6s: scroll",
    "feta_6s_swipe": "FETA 6s: swipe",
    "feta_7_scroll": "FETA 7: scroll",
    "feta_7_swipe": "FETA 7: swipe",
    "feta_x_scroll": "FETA X: scroll",
    "feta_x_swipe": "FETA X: swipe",
}

CONFIG_NAMES = {
    "full": "T + A + G",
    "touch_accel": "T + A",
    "touch_gyro": "T + G",
    "touch": "T",
    "accel_gyro": "A + G",
    "accel": "A",
    "gyro": "G",
}

# Black-and-white printing:
# linestyle + marker distinguish EER series.
# ROC curves use only linestyle, without markers.
STYLES = [
    ("-", "o"),
    ("--", "s"),
    ("-.", "^"),
    (":", "D"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
})


# ------------------------------------------------------------
# Common plotting functions
# ------------------------------------------------------------

def save_figure(fig, filename):
    """Save SVG without overwriting an existing figure."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / filename

    if path.exists():
        print(f"[SKIP] Already exists: {path}")
    else:
        fig.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
        print(f"[SAVED] {path}")

    plt.close(fig)


def style_axes(ax):
    """Consistent minimal black-and-white presentation."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        color="0.80",
        linestyle=":",
        linewidth=0.65,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=0.8,
    )


def style_eer_axes(ax, max_eer):
    """Axes begin visually near zero, with visible origin and margins."""

    upper = max(
        10,
        math.ceil(max_eer / 5) * 5,
    )

    # Extra room outside the data range prevents marker clipping.
    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(-1.0, upper + 2.0)

    # Explicitly preserve the origin tick marks.
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks(np.arange(0, upper + 1, 5))

    ax.set_xlabel("Кількість агрегованих жестів, k")
    ax.set_ylabel("EER, %")

    style_axes(ax)


def plot_series(
    ax,
    x,
    y,
    label,
    index,
    roc=False,
):
    """Plot a black line, optionally with markers."""

    linestyle, marker = STYLES[index % len(STYLES)]

    ax.plot(
        x,
        y,
        color="black",
        linestyle=linestyle,
        linewidth=1.55 if roc else 1.4,
        marker=None if roc else marker,
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1,
        label=label,
        zorder=3,
    )


def validate_windows(group, description):
    actual = sorted(group["k"].astype(int).tolist())
    expected = list(range(1, 21))

    if actual != expected:
        raise ValueError(
            f"Incomplete k=1..20 for {description}. "
            f"Received: {actual}"
        )


# ------------------------------------------------------------
# Figure 1: Three best modality configurations
# ------------------------------------------------------------

def figure_1(summary):
    """Test EER vs k for three best k=1 modality configurations."""

    ranking = (
        summary.loc[summary["k"].eq(1)]
        .sort_values("test_macro_eer")
        .head(3)
    )

    if len(ranking) != 3:
        raise ValueError("Expected three modality configurations")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    max_eer = 0.0

    for index, configuration in enumerate(
        ranking["configuration"]
    ):
        group = (
            summary.loc[
                summary["configuration"].eq(configuration)
            ]
            .sort_values("k")
        )

        validate_windows(group, configuration)

        x = group["k"].to_numpy()
        y = 100 * group["test_macro_eer"].to_numpy()

        max_eer = max(max_eer, float(np.nanmax(y)))

        plot_series(
            ax,
            x,
            y,
            CONFIG_NAMES[configuration],
            index,
        )

    style_eer_axes(ax, max_eer)

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig1_top3_eer_origin.svg",
    )


# ------------------------------------------------------------
# Figure 2: Different scenarios, full T+A+G model
# ------------------------------------------------------------

def figure_2(scenarios):
    """Test EER vs k across four contrasting scenarios."""

    chosen = [
        "hmog_walking",
        "hmog_sitting",
        "feta_x_scroll",
        "feta_x_swipe",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.7))

    max_eer = 0.0

    for index, scenario in enumerate(chosen):
        group = scenarios.loc[
            scenarios["configuration"].eq("full")
            & scenarios["scenario"].eq(scenario)
        ].sort_values("k")

        validate_windows(group, scenario)

        x = group["k"].to_numpy()
        y = 100 * group["test_eer"].to_numpy()

        max_eer = max(max_eer, float(np.nanmax(y)))

        plot_series(
            ax,
            x,
            y,
            SCENARIO_NAMES[scenario],
            index,
        )

    style_eer_axes(ax, max_eer)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig2_scenarios_eer_origin.svg",
    )


# ------------------------------------------------------------
# Figure 3: ROC curves at k=20
# ------------------------------------------------------------

def load_macro_roc(scenario, k=20):
    """Read original saved macro ROC data, not approximate from AUC."""

    prefix = f"{scenario}_k{k:02d}"

    scenario_file = (
        EVALUATION / f"{prefix}_scenario_roc.npz"
    )

    if scenario_file.is_file():
        with np.load(
            scenario_file,
            allow_pickle=False,
        ) as data:
            fpr = data["fpr_grid"].copy()
            tpr = data["macro_tpr"].copy()

        return fpr, tpr

    # Fallback: reproduce the original evaluator's
    # macro interpolation from saved per-user ROC curves.
    user_file = (
        EVALUATION / f"{prefix}_per_user_rocs.npz"
    )

    if user_file.is_file():
        with np.load(
            user_file,
            allow_pickle=False,
        ) as data:
            offsets = data["offsets"].copy()
            all_fpr = data["fpr"].copy()
            all_tpr = data["tpr"].copy()

        grid = np.linspace(0, 1, 201)
        interpolated = []

        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])

            interpolated.append(
                np.interp(
                    grid,
                    all_fpr[start:end],
                    all_tpr[start:end],
                )
            )

        if not interpolated:
            raise ValueError(
                f"No ROC data for {scenario}"
            )

        return (
            grid,
            np.mean(interpolated, axis=0),
        )

    raise FileNotFoundError(
        f"Missing ROC files for {scenario}, k={k}:\n"
        f"{scenario_file}\n"
        f"{user_file}"
    )


def figure_3(scenarios):
    """Macro ROC: HMOG walking, HMOG sitting, median FETA."""

    chosen = [
        "hmog_walking",
        "feta_7_swipe",
        "hmog_sitting",
    ]

    full_k20 = scenarios.loc[
        scenarios["configuration"].eq("full")
        & scenarios["k"].eq(20)
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.9))

    for index, scenario in enumerate(chosen):
        fpr, tpr = load_macro_roc(
            scenario,
            k=20,
        )

        row = full_k20.loc[
            full_k20["scenario"].eq(scenario)
        ]

        if len(row) != 1:
            raise ValueError(
                f"Missing k=20 metrics for {scenario}"
            )

        auc = float(row.iloc[0]["test_auc"])

        label = (
            f"{SCENARIO_NAMES[scenario]} "
            f"(AUC={auc:.3f})"
        )

        # No point markers on ROC curves.
        plot_series(
            ax,
            fpr,
            tpr,
            label,
            index,
            roc=True,
        )

    # Chance-level reference.
    ax.plot(
        [0, 1],
        [0, 1],
        color="0.60",
        linestyle=(0, (1.5, 3)),
        linewidth=0.9,
        marker=None,
        label="Випадкове рішення",
        zorder=1,
    )

    # 2.5% margin around the unit square.
    # ROC endpoints (0,0) and (1,1) remain fully visible.
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")

    style_axes(ax)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig3_macro_roc_k20_nomarkers.svg",
    )


# ------------------------------------------------------------
# Figure 4: Best and worst performing participants
# ------------------------------------------------------------

def load_per_user():
    """Load test-only per-user centroid metrics."""

    path = PER_USER_CSV

    if not path.is_file():
        for fallback in (
            ROOT / "per_user_metrics.csv",
            ROOT / "per_user_metrics(2).csv",
        ):
            if fallback.is_file():
                path = fallback
                break

    if not path.is_file():
        raise FileNotFoundError(
            f"Per-user results not found: {PER_USER_CSV}"
        )

    data = pd.read_csv(
        path,
        dtype={"user_id": str},
    )

    if "enrollment_method" in data.columns:
        data = data.loc[
            data["enrollment_method"].eq("centroid")
        ]

    data = data.loc[
        data["role"].eq("test")
        & data["model_id"].eq("global")
    ].copy()

    data["k"] = data["k"].astype(int)
    data["eer"] = pd.to_numeric(
        data["eer"],
        errors="coerce",
    )
    data["auc"] = pd.to_numeric(
        data["auc"],
        errors="coerce",
    )

    return data


def device_name(scenario):
    if scenario.startswith("feta_6s"):
        return "FETA iPhone 6s"
    if scenario.startswith("feta_7"):
        return "FETA iPhone 7"
    if scenario.startswith("feta_x"):
        return "FETA iPhone X"
    return "HMOG Samsung Galaxy S4"


def figure_4(data):
    """Compare lowest and highest mean k=1 EER users."""

    base = data.loc[
        data["k"].eq(1)
        & data["status"].eq("ok")
        & data["eer"].notna()
    ].copy()

    grouped = (
        base.groupby("user_id")
        .agg(
            mean_eer=("eer", "mean"),
            scenario_count=("scenario", "nunique"),
        )
        .reset_index()
    )

    eligible = (
        grouped.loc[
            grouped["scenario_count"].ge(2)
        ]
        .sort_values(["mean_eer", "user_id"])
    )

    if len(eligible) < 2:
        raise ValueError(
            "Not enough participants with >=2 scenarios"
        )

    chosen = [
        ("P1 — низька похибка", eligible.iloc[0]),
        ("P2 — висока похибка", eligible.iloc[-1]),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.3),
        sharex=True,
        sharey=True,
    )

    max_eer = 0.0
    records = []

    for ax, (title, participant) in zip(axes, chosen):
        user_id = participant["user_id"]

        user_data = data.loc[
            data["user_id"].eq(user_id)
        ].copy()

        user_scenarios = sorted(
            base.loc[
                base["user_id"].eq(user_id),
                "scenario",
            ].unique()
        )

        mean_eer_pct = 100 * float(
            participant["mean_eer"]
        )

        ax.set_title(
            f"{title}\n"
            f"{device_name(user_scenarios[0])}\n"
            f"mean EER, k=1: {mean_eer_pct:.2f}%",
            fontsize=10,
        )

        for index, scenario in enumerate(user_scenarios):
            subset = (
                user_data.loc[
                    user_data["scenario"].eq(scenario)
                    & user_data["status"].eq("ok"),
                    ["k", "eer", "auc"],
                ]
                .set_index("k")
                .reindex(range(1, 21))
            )

            x = np.arange(1, 21)
            y = 100 * subset["eer"].to_numpy()

            if np.isfinite(y).any():
                max_eer = max(
                    max_eer,
                    float(np.nanmax(y)),
                )

            if scenario.endswith("scroll"):
                task = "scroll"
            elif scenario.endswith("swipe"):
                task = "swipe"
            else:
                task = SCENARIO_NAMES[scenario]

            plot_series(
                ax,
                x,
                y,
                task,
                index,
            )

            for k in (1, 20):
                original = user_data.loc[
                    user_data["scenario"].eq(scenario)
                    & user_data["k"].eq(k)
                ]

                if original.empty:
                    continue

                row = original.iloc[0]

                records.append({
                    "participant": title.split()[0],
                    "user_id": user_id,
                    "scenario": scenario,
                    "k": k,
                    "status": row["status"],
                    "eer": row["eer"],
                    "auc": row["auc"],
                })

        ax.legend(
            frameon=False,
            loc="upper left",
        )

        style_axes(ax)

    upper = max(
        10,
        math.ceil(max_eer / 10) * 10,
    )

    for ax in axes:
        # Margins ensure k=1, k=20 and EER=0
        # are not clipped by the plot boundary.
        ax.set_xlim(-0.6, 20.6)
        ax.set_ylim(-2.0, upper + 2.0)

        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks(np.arange(0, upper + 1, 10))

        ax.set_xlabel("Кількість агрегованих жестів, k")

    axes[0].set_ylabel("EER користувача, %")

    fig.tight_layout()

    save_figure(
        fig,
        "fig4_participant_extremes.svg",
    )

    selection_path = (
        OUTPUT / "participant_examples_internal.csv"
    )

    if not selection_path.exists():
        pd.DataFrame(records).to_csv(
            selection_path,
            index=False,
            encoding="utf-8",
        )

    print(
        "[PARTICIPANTS]",
        f"Eligible: {len(eligible)}",
        f"Best: {chosen[0][1]['user_id']}",
        f"Worst: {chosen[1][1]['user_id']}",
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(SUMMARY_CSV)

    if not SCENARIOS_CSV.is_file():
        raise FileNotFoundError(SCENARIOS_CSV)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    scenarios = pd.read_csv(SCENARIOS_CSV)

    summary["k"] = summary["k"].astype(int)
    scenarios["k"] = scenarios["k"].astype(int)

    if len(summary) != 7 * 20:
        raise ValueError(
            "Expected 140 summary rows"
        )

    if len(scenarios) != 7 * 8 * 20:
        raise ValueError(
            "Expected 1120 scenario rows"
        )

    print("[FIGURE 1] Top 3 configurations")
    figure_1(summary)

    print("[FIGURE 2] Different scenarios")
    figure_2(scenarios)

    print("[FIGURE 3] Macro ROC, k=20")
    try:
        figure_3(scenarios)
    except FileNotFoundError as exc:
        print("[ROC NOT AVAILABLE]")
        print(exc)

    print("[FIGURE 4] Participant comparison")
    try:
        per_user = load_per_user()
        figure_4(per_user)
    except FileNotFoundError as exc:
        print("[PER-USER NOT AVAILABLE]")
        print(exc)

    print("\nCompleted.")
    print("SVG directory:", OUTPUT)


if __name__ == "__main__":
    main()
