from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import wilcoxon

from evaluation.common import FOUR_CONDITIONS, ensure_dirs, judge_path, load_config, load_questions, read_json, write_json_no_overwrite


GENERAL_METRICS = ["answer_relevance", "biomedical_factual_correctness", "completeness", "uncertainty_abstention"]
IGKF_METRICS = ["context_groundedness", "context_utilization"]
METRICS = GENERAL_METRICS + IGKF_METRICS
PRIMARY_COMPARISONS = [
    ("qwen3_8b_baseline", "qwen3_8b_igkf"),
    ("gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf"),
]


def load_scores(config: Dict) -> pd.DataFrame:
    output_root = Path(config["experiment"]["output_root"])
    rows = []
    for question in load_questions(config["experiment"]["benchmark_path"]):
        for condition in FOUR_CONDITIONS:
            path = judge_path(output_root, condition, question.question_id)
            if not path.exists():
                continue
            obj = read_json(path)
            parsed = obj.get("parsed_scores")
            if not parsed:
                continue
            row = {"question_id": question.question_id, "condition": condition, **question.annotations}
            row["factual_correctness_evaluable"] = obj.get("factual_correctness_evaluable", True)
            for metric in METRICS:
                score = parsed.get(metric, {}).get("score")
                row[metric] = None if score is None else float(score)
            rows.append(row)
    return pd.DataFrame(rows)


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, sub in df.groupby("condition"):
        row = {"condition": condition, "N": int(len(sub))}
        for metric in METRICS:
            if metric not in sub:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            row[f"median_{metric}"] = float(vals.median())
            row[f"iqr_{metric}"] = iqr(vals)
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(pvals: List[float]) -> List[float]:
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [math.nan] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, (idx, p) in enumerate(indexed):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def paired_tests(df: pd.DataFrame, zero_method: str) -> pd.DataFrame:
    rows = []
    pvals = []
    for left, right in PRIMARY_COMPARISONS:
        for metric in GENERAL_METRICS:
            if metric not in df:
                continue
            wide = df[df["condition"].isin([left, right])].pivot(index="question_id", columns="condition", values=metric)
            if left not in wide.columns or right not in wide.columns:
                continue
            wide = wide[[left, right]].dropna()
            if len(wide) == 0:
                continue
            diffs = wide[right] - wide[left]
            nonzero = diffs[diffs != 0]
            if len(nonzero) == 0:
                stat = 0.0
                p = 1.0
            else:
                stat, p = wilcoxon(wide[right], wide[left], zero_method=zero_method, alternative="two-sided")
            effect = float(stat / (len(wide) * (len(wide) + 1) / 2)) if len(wide) else math.nan
            rows.append({
                "comparison": f"{left} vs {right}",
                "metric": metric,
                "n_paired": int(len(wide)),
                "n_zero_differences": int((diffs == 0).sum()),
                "baseline_median": float(wide[left].median()),
                "igkf_median": float(wide[right].median()),
                "median_paired_difference": float(diffs.median()),
                "wilcoxon_statistic": float(stat),
                "raw_p_value": float(p),
                "effect_size_rank_biserial_proxy": effect,
                "zero_method": zero_method,
                "missing_pair_count": int(205 - len(wide)),
            })
            pvals.append(float(p))
    adjusted = holm_adjust(pvals) if pvals else []
    for row, adj in zip(rows, adjusted):
        row["adjusted_p_value"] = adj
        row["multiple_comparison_correction"] = "holm"
    return pd.DataFrame(rows)


def stratified_describe(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (group_value, condition), sub in df.groupby([group_col, "condition"]):
        row = {group_col: group_value, "condition": condition, "N": int(len(sub))}
        for metric in GENERAL_METRICS:
            vals = pd.to_numeric(sub.get(metric), errors="coerce").dropna()
            if vals.empty:
                continue
            row[f"median_{metric}"] = float(vals.median())
            row[f"iqr_{metric}"] = iqr(vals)
            row[f"mean_{metric}"] = float(vals.mean())
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def interaction_tests(df: pd.DataFrame, zero_method: str) -> pd.DataFrame:
    rows = []
    for metric in GENERAL_METRICS:
        wide = df[df["condition"].isin(FOUR_CONDITIONS)].pivot(index="question_id", columns="condition", values=metric)
        needed = ["qwen3_8b_baseline", "qwen3_8b_igkf", "gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf"]
        if any(c not in wide.columns for c in needed):
            continue
        wide = wide[needed].dropna()
        if wide.empty:
            continue
        delta_qwen = wide["qwen3_8b_igkf"] - wide["qwen3_8b_baseline"]
        delta_gpt = wide["gpt_5_4_mini_igkf"] - wide["gpt_5_4_mini_baseline"]
        interaction = delta_qwen - delta_gpt
        if (interaction != 0).sum() == 0:
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(interaction, zero_method=zero_method, alternative="two-sided")
        rows.append({
            "metric": metric,
            "method": "Wilcoxon signed-rank test on per-question interaction_difference = delta_qwen - delta_gpt",
            "n_paired": int(len(wide)),
            "median_delta_qwen": float(delta_qwen.median()),
            "median_delta_gpt": float(delta_gpt.median()),
            "median_interaction_difference": float(interaction.median()),
            "wilcoxon_statistic": float(stat),
            "raw_p_value": float(p),
            "zero_method": zero_method,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wilcoxon-based paper statistics from saved judge outputs.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    df = load_scores(config)
    stats_dir = output_root / "statistics"
    if df.empty:
        write_json_no_overwrite(stats_dir / "statistics_status.json", {"status": "no_scores_available"}, force=True)
        print("No judge scores available.")
        return
    score_csv = stats_dir / "judge_scores_long.csv"
    df.to_csv(score_csv, index=False)
    desc = describe(df)
    desc.to_csv(stats_dir / "condition_summary.csv", index=False)
    tests = paired_tests(df, config["statistics"]["wilcoxon_zero_method"])
    tests.to_csv(stats_dir / "wilcoxon_primary_comparisons.csv", index=False)
    interaction_tests(df, config["statistics"]["wilcoxon_zero_method"]).to_csv(stats_dir / "wilcoxon_interaction_tests.csv", index=False)
    for group_col, name in [
        ("category", "category_level_summary.csv"),
        ("answerability", "answerability_stratified_summary.csv"),
        ("expected_behavior", "expected_behavior_stratified_summary.csv"),
        ("manual_review_required", "manual_review_summary.csv"),
    ]:
        stratified_describe(df, group_col).to_csv(stats_dir / name, index=False)
    macro = stratified_describe(df, "category")
    if not macro.empty:
        numeric_cols = [c for c in macro.columns if c.startswith(("median_", "iqr_", "mean_", "sd_"))]
        macro.groupby("condition")[numeric_cols].mean().reset_index().to_csv(stats_dir / "macro_average_category_summary.csv", index=False)
    print(f"Wrote {score_csv}, condition_summary.csv, wilcoxon_primary_comparisons.csv")


if __name__ == "__main__":
    main()
