from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.common import FOUR_CONDITIONS, ensure_dirs, load_config


GENERAL_METRICS = [
    "answer_relevance",
    "biomedical_factual_correctness",
    "completeness",
    "uncertainty_abstention",
]
IGKF_METRICS = ["context_groundedness", "context_utilization"]
CONDITION_LABELS = {
    "qwen3_8b_baseline": "Qwen3 baseline",
    "qwen3_8b_igkf": "Qwen3 + IGKF",
    "gpt_5_4_mini_baseline": "GPT-5.4 mini baseline",
    "gpt_5_4_mini_igkf": "GPT-5.4 mini + IGKF",
}
METRIC_LABELS = {
    "answer_relevance": "Answer relevance",
    "biomedical_factual_correctness": "Biomedical factual correctness",
    "completeness": "Completeness",
    "uncertainty_abstention": "Uncertainty / abstention",
    "context_groundedness": "Context groundedness",
    "context_utilization": "Context utilization",
}
PALETTE = {
    "qwen3_8b_baseline": "#4B5563",
    "qwen3_8b_igkf": "#0F766E",
    "gpt_5_4_mini_baseline": "#6B7280",
    "gpt_5_4_mini_igkf": "#2563EB",
}


def bootstrap_ci(values: Iterable[float], *, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, float(lo), float(hi)


def save_figure(fig, figures: Path, stem: str) -> None:
    for ext in ["png", "svg", "pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 400
        fig.savefig(figures / f"{stem}.{ext}", **kwargs)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_overall_metric_cis(df: pd.DataFrame, figures: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(FOUR_CONDITIONS))
    for ax, metric in zip(axes, GENERAL_METRICS):
        means, lows, highs = [], [], []
        for condition in FOUR_CONDITIONS:
            mean, lo, hi = bootstrap_ci(df.loc[df["condition"] == condition, metric], rng=rng)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        ax.bar(
            x,
            means,
            yerr=yerr,
            capsize=4,
            color=[PALETTE[c] for c in FOUR_CONDITIONS],
            edgecolor="#111827",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "ecolor": "#111827"},
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=11, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in FOUR_CONDITIONS], rotation=25, ha="right")
        ax.set_ylim(1, 5.15)
        ax.set_ylabel("Mean judge score (95% bootstrap CI)")
        style_axes(ax)
    fig.suptitle("Four-condition answer quality", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, figures, "figure_1_four_condition_quality_bootstrap_ci")
    plt.close(fig)


def plot_paired_delta_cis(df: pd.DataFrame, figures: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed + 1)
    comparisons = [
        ("qwen3_8b_baseline", "qwen3_8b_igkf", "Qwen3"),
        ("gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf", "GPT-5.4 mini"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    offsets = [-0.18, 0.18]
    base_x = np.arange(len(GENERAL_METRICS))
    for offset, (baseline, igkf, label) in zip(offsets, comparisons):
        wide = df[df["condition"].isin([baseline, igkf])].pivot(index="question_id", columns="condition", values=GENERAL_METRICS)
        means, lows, highs = [], [], []
        for metric in GENERAL_METRICS:
            delta = wide[(metric, igkf)] - wide[(metric, baseline)]
            mean, lo, hi = bootstrap_ci(delta.dropna(), rng=rng)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        color = PALETTE[igkf]
        ax.errorbar(
            base_x + offset,
            means,
            yerr=yerr,
            fmt="o",
            capsize=4,
            markersize=7,
            linewidth=1.5,
            color=color,
            label=label,
        )
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_xticks(base_x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in GENERAL_METRICS], rotation=20, ha="right")
    ax.set_ylabel("Mean paired IGKF effect\n(IGKF - baseline, 95% bootstrap CI)")
    ax.set_title("Paired improvement from IGKF", fontsize=14, weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figures, "figure_2_paired_igkf_effect_bootstrap_ci")
    plt.close(fig)


def plot_igkf_context_cis(df: pd.DataFrame, figures: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed + 2)
    conditions = ["qwen3_8b_igkf", "gpt_5_4_mini_igkf"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.34
    x = np.arange(len(IGKF_METRICS))
    for idx, condition in enumerate(conditions):
        means, lows, highs = [], [], []
        for metric in IGKF_METRICS:
            mean, lo, hi = bootstrap_ci(df.loc[df["condition"] == condition, metric], rng=rng)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            means,
            width=width,
            yerr=yerr,
            capsize=4,
            color=PALETTE[condition],
            edgecolor="#111827",
            linewidth=0.6,
            label=CONDITION_LABELS[condition],
            error_kw={"elinewidth": 1.0, "ecolor": "#111827"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in IGKF_METRICS])
    ax.set_ylim(1, 5.15)
    ax.set_ylabel("Mean judge score (95% bootstrap CI)")
    ax.set_title("IGKF-specific context evaluation", fontsize=14, weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figures, "figure_3_igkf_context_metrics_bootstrap_ci")
    plt.close(fig)


def plot_category_factual_delta(df: pd.DataFrame, figures: Path, seed: int, min_n: int = 5) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed + 3)
    rows = []
    for category, category_df in df.groupby("category"):
        for baseline, igkf, model_label, color_key in [
            ("qwen3_8b_baseline", "qwen3_8b_igkf", "Qwen3", "qwen3_8b_igkf"),
            ("gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf", "GPT-5.4 mini", "gpt_5_4_mini_igkf"),
        ]:
            wide = category_df[category_df["condition"].isin([baseline, igkf])].pivot(
                index="question_id",
                columns="condition",
                values="biomedical_factual_correctness",
            ).dropna()
            if len(wide) < min_n:
                continue
            delta = wide[igkf] - wide[baseline]
            mean, lo, hi = bootstrap_ci(delta, rng=rng)
            rows.append({
                "category": category,
                "model": model_label,
                "color_key": color_key,
                "n": len(wide),
                "mean": mean,
                "lo": lo,
                "hi": hi,
            })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    category_order = (
        plot_df.groupby("category")["mean"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y_lookup = {cat: idx for idx, cat in enumerate(category_order)}
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.48 * len(category_order) + 1.5)))
    offsets = {"Qwen3": -0.13, "GPT-5.4 mini": 0.13}
    for _, row in plot_df.iterrows():
        y = y_lookup[row["category"]] + offsets[row["model"]]
        ax.errorbar(
            row["mean"],
            y,
            xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o",
            capsize=3,
            markersize=6,
            color=PALETTE[row["color_key"]],
            label=row["model"],
        )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="lower right")
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_yticks(range(len(category_order)))
    ax.set_yticklabels(category_order)
    ax.set_xlabel("Mean paired factual-correctness effect\n(IGKF - baseline, 95% bootstrap CI)")
    ax.set_title("IGKF factual-correctness gains by category", fontsize=14, weight="bold")
    style_axes(ax)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figures, "figure_4_category_factual_delta_bootstrap_ci")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-facing figures from saved scores.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Reserved for reproducibility notes; figures use 10,000 samples.")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    scores_path = output_root / "statistics" / "judge_scores_long.csv"
    if not scores_path.exists():
        print("No judge_scores_long.csv available; figures not generated.")
        return

    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    df = pd.read_csv(scores_path)
    figures = output_root / "figures"
    seed = int(config["experiment"]["random_seed"])
    plot_overall_metric_cis(df, figures, seed)
    plot_paired_delta_cis(df, figures, seed)
    plot_igkf_context_cis(df, figures, seed)
    plot_category_factual_delta(df, figures, seed)
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    main()
