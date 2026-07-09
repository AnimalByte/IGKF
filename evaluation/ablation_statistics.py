from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from evaluation.ablation_common import ABLATION_CONDITIONS, CANONICAL_REFERENCE_CONDITIONS, NEW_ABLATION_CONDITIONS, load_ablation_config
from evaluation.common import judge_path, load_questions, read_json, write_json_no_overwrite
from evaluation.statistical_analysis import GENERAL_METRICS, holm_adjust


CONTRASTS = [
    ("qwen3_8b_graph_only", "qwen3_8b_baseline", "graph-only vs baseline"),
    ("qwen3_8b_literature_only", "qwen3_8b_baseline", "literature-only vs baseline"),
    ("qwen3_8b_igkf", "qwen3_8b_baseline", "full IGKF vs baseline"),
    ("qwen3_8b_igkf", "qwen3_8b_graph_only", "full IGKF vs graph-only"),
    ("qwen3_8b_igkf", "qwen3_8b_literature_only", "full IGKF vs literature-only"),
]
SECONDARY_CONTRASTS = [("qwen3_8b_graph_only", "qwen3_8b_literature_only", "graph-only vs literature-only")]


def score_path(config: Dict, condition: str, question_id: str) -> Path:
    if condition in CANONICAL_REFERENCE_CONDITIONS:
        return judge_path(Path(config["experiment"]["canonical_output_root"]), condition, question_id)
    return judge_path(Path(config["experiment"]["output_root"]), condition, question_id)


def load_scores(config: Dict) -> pd.DataFrame:
    rows = []
    for question in load_questions(config["experiment"]["benchmark_path"]):
        for condition in ABLATION_CONDITIONS:
            path = score_path(config, condition, question.question_id)
            if not path.exists():
                continue
            obj = read_json(path)
            parsed = obj.get("parsed_scores")
            if not parsed:
                continue
            row = {"question_id": question.question_id, "condition": condition, **question.annotations}
            for metric in GENERAL_METRICS:
                score = parsed.get(metric, {}).get("score")
                row[metric] = None if score is None else float(score)
            rows.append(row)
    return pd.DataFrame(rows)


def describe_condition_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, sub in df.groupby("condition"):
        for metric in GENERAL_METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "condition": condition,
                "metric": metric,
                "N": int(vals.size),
                "median": float(vals.median()),
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, contrasts: List[Tuple[str, str, str]], zero_method: str, family: str) -> pd.DataFrame:
    rows = []
    pvals = []
    for right, left, label in contrasts:
        for metric in GENERAL_METRICS:
            wide = df[df["condition"].isin([left, right])].pivot(index="question_id", columns="condition", values=metric)
            if left not in wide.columns or right not in wide.columns:
                continue
            wide = wide[[left, right]].dropna()
            if wide.empty:
                continue
            diffs = wide[right] - wide[left]
            if (diffs != 0).sum() == 0:
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(wide[right], wide[left], zero_method=zero_method, alternative="two-sided")
            rows.append({
                "comparison": label,
                "left_condition": left,
                "right_condition": right,
                "metric": metric,
                "paired_N": int(len(wide)),
                "missing_pair_count": int(205 - len(wide)),
                "left_median": float(wide[left].median()),
                "right_median": float(wide[right].median()),
                "median_paired_difference": float(diffs.median()),
                "q1_paired_difference": float(diffs.quantile(0.25)),
                "q3_paired_difference": float(diffs.quantile(0.75)),
                "mean_paired_difference": float(diffs.mean()),
                "percent_improved": float((diffs > 0).mean() * 100),
                "percent_unchanged": float((diffs == 0).mean() * 100),
                "percent_worsened": float((diffs < 0).mean() * 100),
                "wilcoxon_statistic": float(stat),
                "raw_p_value": float(p),
                "zero_method": zero_method,
                "contrast_family": family,
            })
            pvals.append(float(p))
    adjusted = holm_adjust(pvals) if pvals else []
    for row, adj in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = adj
        row["multiple_comparison_correction"] = "holm"
    return pd.DataFrame(rows)


def friedman_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pvals = []
    for metric in GENERAL_METRICS:
        wide = df.pivot(index="question_id", columns="condition", values=metric)
        if any(condition not in wide.columns for condition in ABLATION_CONDITIONS):
            continue
        wide = wide[list(ABLATION_CONDITIONS)].dropna()
        if wide.empty:
            continue
        stat, p = friedmanchisquare(*(wide[condition] for condition in ABLATION_CONDITIONS))
        rows.append({
            "metric": metric,
            "method": "Friedman omnibus test across four paired Qwen ablation conditions",
            "paired_N": int(len(wide)),
            "statistic": float(stat),
            "raw_p_value": float(p),
        })
        pvals.append(float(p))
    adjusted = holm_adjust(pvals) if pvals else []
    for row, adj in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = adj
        row["multiple_comparison_correction"] = "holm_across_four_omnibus_metric_tests"
    return pd.DataFrame(rows)


def make_placeholder_figures(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty or not all(condition in set(df["condition"]) for condition in ABLATION_CONDITIONS):
        return
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metric = "biomedical_factual_correctness"
    plot_df = df.dropna(subset=[metric])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    data = [plot_df.loc[plot_df["condition"] == condition, metric].values for condition in ABLATION_CONDITIONS]
    ax.boxplot(data, labels=ABLATION_CONDITIONS, showmeans=True)
    ax.set_ylabel("Judge score")
    ax.set_title("Qwen ablation factual-correctness scores")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures / f"qwen_ablation_factual_correctness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen ablation statistics after judge scores exist.")
    parser.add_argument("--config", default="evaluation/configs/qwen_ablation.yaml")
    args = parser.parse_args()
    config = load_ablation_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    stats = output_root / "statistics"
    tables = output_root / "tables"
    stats.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = load_scores(config)
    if df.empty or not all(condition in set(df.get("condition", [])) for condition in ABLATION_CONDITIONS):
        status = {
            "status": "judge_scores_incomplete",
            "message": "Ablation statistics are prepared but require judge scores for qwen3_8b_graph_only and qwen3_8b_literature_only.",
            "openai_api_called": False,
        }
        write_json_no_overwrite(stats / "ablation_statistics_status.json", status, force=True)
        print(status["message"])
        return

    df.to_csv(stats / "qwen_ablation_scores_long.csv", index=False)
    describe = describe_condition_scores(df)
    describe.to_csv(stats / "qwen_ablation_condition_summary.csv", index=False)
    describe.to_csv(tables / "qwen_ablation_condition_summary.csv", index=False)
    friedman = friedman_table(df)
    friedman.to_csv(stats / "qwen_ablation_friedman_omnibus.csv", index=False)
    friedman.to_csv(tables / "qwen_ablation_friedman_omnibus.csv", index=False)
    contrasts = paired_table(df, CONTRASTS, config["statistics"]["wilcoxon_zero_method"], "declared_primary_ablation_contrasts")
    contrasts.to_csv(stats / "qwen_ablation_planned_contrasts.csv", index=False)
    contrasts.to_csv(tables / "qwen_ablation_planned_contrasts.csv", index=False)
    secondary = paired_table(df, SECONDARY_CONTRASTS, config["statistics"]["wilcoxon_zero_method"], "secondary_ablation_contrast")
    secondary.to_csv(stats / "qwen_ablation_secondary_contrast.csv", index=False)
    secondary.to_csv(tables / "qwen_ablation_secondary_contrast.csv", index=False)
    make_placeholder_figures(df, output_root)
    print(f"Wrote Qwen ablation statistics to {stats}")


if __name__ == "__main__":
    main()
