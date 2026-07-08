from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.common import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-facing tables from saved statistics.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    stats = output_root / "statistics"
    tables = output_root / "tables"

    summary = stats / "condition_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        keep = [c for c in df.columns if c == "condition" or c == "N" or c.startswith(("median_", "iqr_", "mean_", "sd_"))]
        df[keep].to_csv(tables / "overall_four_condition_summary.csv", index=False)

    effects = stats / "wilcoxon_primary_comparisons.csv"
    if effects.exists():
        pd.read_csv(effects).to_csv(tables / "within_model_igkf_effects.csv", index=False)

    scores = stats / "judge_scores_long.csv"
    if scores.exists():
        df = pd.read_csv(scores)
        rag = df[df["condition"].isin(["qwen3_8b_igkf", "gpt_5_4_mini_igkf"])]
        if not rag.empty:
            rag.groupby("condition")[["context_groundedness", "context_utilization"]].agg(["count", "median", "mean", "std"]).to_csv(
                tables / "rag_groundedness_utilization.csv"
            )
    for name in [
        "category_level_summary.csv",
        "macro_average_category_summary.csv",
        "answerability_stratified_summary.csv",
        "expected_behavior_stratified_summary.csv",
        "manual_review_summary.csv",
        "wilcoxon_interaction_tests.csv",
    ]:
        path = stats / name
        if path.exists():
            pd.read_csv(path).to_csv(tables / name, index=False)

    pairwise = output_root / "pairwise_judge"
    for name in [
        "final_pairwise_graphrag_qwen_vs_mini_pairwise_summary.csv",
        "final_pairwise_graphrag_qwen_vs_mini_pairwise_results.csv",
    ]:
        path = pairwise / name
        if path.exists():
            pd.read_csv(path).to_csv(tables / name, index=False)
    stats_json = pairwise / "final_pairwise_graphrag_qwen_vs_mini_pairwise_statistics.json"
    if stats_json.exists():
        (tables / stats_json.name).write_text(stats_json.read_text())
    print(f"Tables written to {tables}")


if __name__ == "__main__":
    main()
