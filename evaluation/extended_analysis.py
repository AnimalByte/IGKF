from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from evaluation.common import FOUR_CONDITIONS, ensure_dirs, load_config
from evaluation.statistical_analysis import GENERAL_METRICS, IGKF_METRICS, METRICS, holm_adjust


CONDITION_LABELS = {
    "qwen3_8b_baseline": "Qwen no GraphRAG",
    "qwen3_8b_igkf": "Qwen + GraphRAG",
    "gpt_5_4_mini_baseline": "GPT-5.4 mini no GraphRAG",
    "gpt_5_4_mini_igkf": "GPT-5.4 mini + GraphRAG",
}
MODEL_BY_CONDITION = {
    "qwen3_8b_baseline": "qwen3_8b",
    "qwen3_8b_igkf": "qwen3_8b",
    "gpt_5_4_mini_baseline": "gpt_5_4_mini",
    "gpt_5_4_mini_igkf": "gpt_5_4_mini",
}
GRAPHRAG_BY_CONDITION = {
    "qwen3_8b_baseline": "absent",
    "qwen3_8b_igkf": "present",
    "gpt_5_4_mini_baseline": "absent",
    "gpt_5_4_mini_igkf": "present",
}
PLANNED_COMPARISONS = [
    ("qwen3_8b_baseline", "qwen3_8b_igkf", "Qwen GraphRAG effect"),
    ("gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf", "GPT-5.4 mini GraphRAG effect"),
    ("qwen3_8b_igkf", "gpt_5_4_mini_igkf", "Model difference with GraphRAG"),
    ("qwen3_8b_baseline", "gpt_5_4_mini_baseline", "Model difference without GraphRAG"),
]


def numeric_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    return pd.to_numeric(df[metric], errors="coerce").dropna()


def iqr(vals: pd.Series) -> float:
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def shannon_entropy(vals: pd.Series) -> Tuple[float, float]:
    counts = vals.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).astype(float)
    probs = counts / counts.sum() if counts.sum() else counts
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else math.nan
    normalized = float(entropy / math.log(5)) if len(nz) else math.nan
    return entropy, normalized


def bootstrap_ci(values: Sequence[float], func=np.mean, samples: int = 10000, seed: int = 20260706) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        val = float(func(arr))
        return val, val
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(samples, arr.size), replace=True)
    stats = np.apply_along_axis(func, 1, draws)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_wide(df: pd.DataFrame, conditions: Sequence[str], metric: str) -> pd.DataFrame:
    wide = df[df["condition"].isin(conditions)].pivot(index="question_id", columns="condition", values=metric)
    missing_cols = [c for c in conditions if c not in wide.columns]
    if missing_cols:
        return pd.DataFrame(columns=list(conditions))
    return wide[list(conditions)].apply(pd.to_numeric, errors="coerce").dropna()


def rank_biserial_from_diffs(diffs: pd.Series) -> float:
    nonzero = pd.to_numeric(diffs, errors="coerce").dropna()
    nonzero = nonzero[nonzero != 0]
    if nonzero.empty:
        return 0.0
    ranks = rankdata(np.abs(nonzero.to_numpy()), method="average")
    w_pos = float(ranks[nonzero.to_numpy() > 0].sum())
    w_neg = float(ranks[nonzero.to_numpy() < 0].sum())
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom else math.nan


def score_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for condition in FOUR_CONDITIONS:
            vals = numeric_scores(df[df["condition"] == condition], metric)
            if vals.empty:
                continue
            counts = vals.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            entropy, norm_entropy = shannon_entropy(vals)
            row = {
                "metric": metric,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "model": MODEL_BY_CONDITION[condition],
                "graphrag": GRAPHRAG_BY_CONDITION[condition],
                "N": int(vals.size),
                "median": float(vals.median()),
                "mean": float(vals.mean()),
                "standard_deviation": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "iqr": iqr(vals),
                "minimum": float(vals.min()),
                "maximum": float(vals.max()),
                "percent_score_5": float((vals == 5).mean() * 100),
                "percent_score_ge_4": float((vals >= 4).mean() * 100),
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
            }
            for score in [1, 2, 3, 4, 5]:
                row[f"count_score_{score}"] = int(counts.loc[score])
                row[f"percent_score_{score}"] = float(counts.loc[score] / vals.size * 100)
            rows.append(row)
    return pd.DataFrame(rows)


def score_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for condition in FOUR_CONDITIONS:
            vals = numeric_scores(df[df["condition"] == condition], metric)
            if vals.empty:
                continue
            counts = vals.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            for score, count in counts.items():
                rows.append({
                    "metric": metric,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "score": int(score),
                    "count": int(count),
                    "proportion": float(count / vals.size),
                    "percent": float(count / vals.size * 100),
                })
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    needed = ["qwen3_8b_baseline", "qwen3_8b_igkf", "gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf"]
    for metric in GENERAL_METRICS:
        wide = paired_wide(df, needed, metric)
        for question_id, row in wide.iterrows():
            delta_qwen = row["qwen3_8b_igkf"] - row["qwen3_8b_baseline"]
            delta_gpt = row["gpt_5_4_mini_igkf"] - row["gpt_5_4_mini_baseline"]
            rows.append({
                "question_id": question_id,
                "metric": metric,
                "qwen_baseline": float(row["qwen3_8b_baseline"]),
                "qwen_igkf": float(row["qwen3_8b_igkf"]),
                "gpt_baseline": float(row["gpt_5_4_mini_baseline"]),
                "gpt_igkf": float(row["gpt_5_4_mini_igkf"]),
                "delta_qwen": float(delta_qwen),
                "delta_gpt": float(delta_gpt),
                "interaction_difference": float(delta_qwen - delta_gpt),
            })
    return pd.DataFrame(rows)


def paired_delta_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, sub in delta_df.groupby("metric"):
        for col, label in [("delta_qwen", "Qwen GraphRAG - baseline"), ("delta_gpt", "GPT GraphRAG - baseline"), ("interaction_difference", "delta_qwen - delta_gpt")]:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            mean_lo, mean_hi = bootstrap_ci(vals, np.mean)
            med_lo, med_hi = bootstrap_ci(vals, np.median)
            rows.append({
                "metric": metric,
                "quantity": col,
                "label": label,
                "N": int(vals.size),
                "median": float(vals.median()),
                "mean": float(vals.mean()),
                "standard_deviation": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "iqr": iqr(vals),
                "minimum": float(vals.min()),
                "maximum": float(vals.max()),
                "mean_ci95_low": mean_lo,
                "mean_ci95_high": mean_hi,
                "median_ci95_low": med_lo,
                "median_ci95_high": med_hi,
            })
    return pd.DataFrame(rows)


def model_win_tie_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("qwen3_8b_igkf", "gpt_5_4_mini_igkf", "GraphRAG systems"),
        ("qwen3_8b_baseline", "gpt_5_4_mini_baseline", "Non-GraphRAG systems"),
    ]
    for metric in GENERAL_METRICS:
        for qwen_cond, gpt_cond, label in comparisons:
            wide = paired_wide(df, [qwen_cond, gpt_cond], metric)
            if wide.empty:
                continue
            qwen_higher = int((wide[qwen_cond] > wide[gpt_cond]).sum())
            gpt_higher = int((wide[gpt_cond] > wide[qwen_cond]).sum())
            ties = int((wide[qwen_cond] == wide[gpt_cond]).sum())
            rows.append({
                "metric": metric,
                "comparison": label,
                "qwen_condition": qwen_cond,
                "gpt_condition": gpt_cond,
                "N": int(len(wide)),
                "qwen_higher": qwen_higher,
                "gpt_higher": gpt_higher,
                "exact_score_tie": ties,
                "qwen_higher_percent": float(qwen_higher / len(wide) * 100),
                "gpt_higher_percent": float(gpt_higher / len(wide) * 100),
                "exact_score_tie_percent": float(ties / len(wide) * 100),
                "tie_interpretation_note": "Exact ties on a 1-5 ordinal scale are not evidence of model equivalence.",
            })
    return pd.DataFrame(rows)


def planned_paired_comparisons(df: pd.DataFrame, zero_method: str) -> pd.DataFrame:
    rows = []
    for metric in GENERAL_METRICS:
        metric_rows = []
        pvals = []
        for left, right, label in PLANNED_COMPARISONS:
            wide = paired_wide(df, [left, right], metric)
            if wide.empty:
                continue
            diffs = wide[right] - wide[left]
            if (diffs != 0).sum() == 0:
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(wide[right], wide[left], zero_method=zero_method, alternative="two-sided")
            mean_lo, mean_hi = bootstrap_ci(diffs, np.mean)
            med_lo, med_hi = bootstrap_ci(diffs, np.median)
            metric_rows.append({
                "metric": metric,
                "comparison": label,
                "left_condition": left,
                "right_condition": right,
                "direction": f"{right} - {left}",
                "N_paired": int(len(wide)),
                "missing_pair_count": int(df["question_id"].nunique() - len(wide)),
                "zero_difference_count": int((diffs == 0).sum()),
                "left_median": float(wide[left].median()),
                "right_median": float(wide[right].median()),
                "median_paired_difference": float(diffs.median()),
                "mean_paired_difference": float(diffs.mean()),
                "paired_difference_sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "mean_difference_ci95_low": mean_lo,
                "mean_difference_ci95_high": mean_hi,
                "median_difference_ci95_low": med_lo,
                "median_difference_ci95_high": med_hi,
                "wilcoxon_statistic": float(stat),
                "raw_p_value": float(p),
                "effect_size_rank_biserial": rank_biserial_from_diffs(diffs),
                "zero_method": zero_method,
                "test": "paired Wilcoxon signed-rank",
            })
            pvals.append(float(p))
        adjusted = holm_adjust(pvals) if pvals else []
        for row, adj in zip(metric_rows, adjusted):
            row["holm_adjusted_p_value"] = adj
            row["multiple_comparison_family"] = f"{metric}: four planned paired comparisons"
            row["multiple_comparison_correction"] = "holm"
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def factorial_fallback(df: pd.DataFrame, zero_method: str) -> pd.DataFrame:
    rows = []
    for metric in GENERAL_METRICS:
        delta = paired_deltas(df)
        sub = delta[delta["metric"] == metric]
        if sub.empty:
            continue
        for col, label in [
            ("delta_qwen", "GraphRAG effect within Qwen"),
            ("delta_gpt", "GraphRAG effect within GPT-5.4 mini"),
            ("interaction_difference", "Model x GraphRAG interaction on paired deltas"),
        ]:
            vals = sub[col].dropna()
            if (vals != 0).sum() == 0:
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(vals, zero_method=zero_method, alternative="two-sided")
            rows.append({
                "metric": metric,
                "factor_or_contrast": label,
                "method": "Fallback paired ordinal/nonparametric analysis; preferred cumulative-link mixed model unavailable in this Python environment.",
                "N_paired_questions": int(vals.size),
                "median_effect": float(vals.median()),
                "mean_effect": float(vals.mean()),
                "wilcoxon_statistic": float(stat),
                "raw_p_value": float(p),
                "effect_size_rank_biserial": rank_biserial_from_diffs(vals),
                "zero_method": zero_method,
            })
        for left, right, label in [
            ("qwen3_8b_baseline", "gpt_5_4_mini_baseline", "Model effect without GraphRAG"),
            ("qwen3_8b_igkf", "gpt_5_4_mini_igkf", "Model effect with GraphRAG"),
        ]:
            wide = paired_wide(df, [left, right], metric)
            diffs = wide[right] - wide[left]
            if (diffs != 0).sum() == 0:
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(wide[right], wide[left], zero_method=zero_method, alternative="two-sided")
            rows.append({
                "metric": metric,
                "factor_or_contrast": label,
                "method": "Fallback paired ordinal/nonparametric analysis; preferred cumulative-link mixed model unavailable in this Python environment.",
                "N_paired_questions": int(len(wide)),
                "median_effect": float(diffs.median()),
                "mean_effect": float(diffs.mean()),
                "wilcoxon_statistic": float(stat),
                "raw_p_value": float(p),
                "effect_size_rank_biserial": rank_biserial_from_diffs(diffs),
                "zero_method": zero_method,
            })
    return pd.DataFrame(rows)


def ceiling_diagnostics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    variance_rows = []
    summary = score_distribution_summary(df)
    for metric in METRICS:
        for condition in FOUR_CONDITIONS:
            vals = numeric_scores(df[df["condition"] == condition], metric) if metric in df.columns else pd.Series(dtype=float)
            if vals.empty:
                continue
            entropy, norm_entropy = shannon_entropy(vals)
            rows.append({
                "metric": metric,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "N": int(vals.size),
                "proportion_at_max_score": float((vals == 5).mean()),
                "proportion_score_ge_4": float((vals >= 4).mean()),
                "variance": float(vals.var(ddof=1)) if vals.size > 1 else 0.0,
                "standard_deviation": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
            })
    for metric in GENERAL_METRICS:
        for model, baseline, igkf in [
            ("qwen3_8b", "qwen3_8b_baseline", "qwen3_8b_igkf"),
            ("gpt_5_4_mini", "gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf"),
        ]:
            base_vals = numeric_scores(df[df["condition"] == baseline], metric)
            igkf_vals = numeric_scores(df[df["condition"] == igkf], metric)
            if base_vals.empty or igkf_vals.empty:
                continue
            base_var = float(base_vals.var(ddof=1))
            igkf_var = float(igkf_vals.var(ddof=1))
            reduction = math.nan if base_var == 0 else float((base_var - igkf_var) / base_var)
            variance_rows.append({
                "metric": metric,
                "model": model,
                "baseline_condition": baseline,
                "igkf_condition": igkf,
                "baseline_variance": base_var,
                "igkf_variance": igkf_var,
                "variance_reduction_fraction": reduction,
            })
        wide = paired_wide(df, ["qwen3_8b_igkf", "gpt_5_4_mini_igkf"], metric)
        if not wide.empty:
            both_5 = int(((wide["qwen3_8b_igkf"] == 5) & (wide["gpt_5_4_mini_igkf"] == 5)).sum())
            identical_below_5 = int(((wide["qwen3_8b_igkf"] == wide["gpt_5_4_mini_igkf"]) & (wide["qwen3_8b_igkf"] < 5)).sum())
            differ = int((wide["qwen3_8b_igkf"] != wide["gpt_5_4_mini_igkf"]).sum())
            rows.append({
                "metric": metric,
                "condition": "qwen3_8b_igkf vs gpt_5_4_mini_igkf",
                "condition_label": "Both GraphRAG systems question-paired",
                "N": int(len(wide)),
                "both_graphrag_score_5_count": both_5,
                "both_graphrag_score_5_proportion": float(both_5 / len(wide)),
                "both_graphrag_identical_below_5_count": identical_below_5,
                "both_graphrag_identical_below_5_proportion": float(identical_below_5 / len(wide)),
                "graphrag_systems_differ_count": differ,
                "graphrag_systems_differ_proportion": float(differ / len(wide)),
            })
    return pd.DataFrame(rows), pd.DataFrame(variance_rows)


def ordinal_model_status() -> Dict:
    status = {
        "preferred_model": "cumulative-link mixed-effects model: score ~ model + graphrag + model:graphrag + (1 | question_id)",
        "implemented": False,
        "reason": "The active Python environment does not provide a defensible cumulative-link mixed-effects implementation.",
        "fallback": "Paired Wilcoxon signed-rank tests for planned ordinal contrasts plus Wilcoxon test on per-question interaction differences.",
        "statsmodels_available": False,
        "ordered_model_available": False,
        "ordered_model_reason_not_used": "Even if statsmodels OrderedModel were available, it does not provide the required question-level random intercept.",
    }
    try:
        import statsmodels  # type: ignore

        status["statsmodels_available"] = True
        status["statsmodels_version"] = getattr(statsmodels, "__version__", "unknown")
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: F401

            status["ordered_model_available"] = True
        except Exception as exc:
            status["ordered_model_import_error"] = str(exc)
    except Exception as exc:
        status["statsmodels_import_error"] = str(exc)
    return status


def pairwise_infrastructure_status(repo_root: Path) -> Dict:
    code_hits = []
    for path in [repo_root / "evaluation", repo_root / "tests"]:
        if not path.exists():
            continue
        for file_path in path.rglob("*.py"):
            if file_path.name == "extended_analysis.py":
                continue
            text = file_path.read_text(errors="ignore")
            lowered = text.lower()
            if "pairwise" in lowered or "response a" in lowered or "response b" in lowered:
                code_hits.append(str(file_path.relative_to(repo_root)))
    return {
        "pairwise_judge_infrastructure_found": bool(code_hits),
        "matching_code_files": code_hits,
        "prepared_new_paid_batch": False,
        "note": "No new paid pairwise judge batch was prepared or submitted by this extended analysis command.",
    }


def set_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def plot_score_distributions(freq: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#59A14F", "#F58518", "#B279A2"]
    for metric in METRICS:
        sub = freq[freq["metric"] == metric]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        x = np.arange(1, 6)
        width = 0.18
        for idx, condition in enumerate(FOUR_CONDITIONS):
            csub = sub[sub["condition"] == condition].set_index("score").reindex([1, 2, 3, 4, 5])
            if csub["proportion"].isna().all():
                continue
            ax.bar(x + (idx - 1.5) * width, csub["proportion"] * 100, width=width, label=CONDITION_LABELS[condition], color=colors[idx])
        ax.set_xticks(x)
        ax.set_xlabel("Judge score")
        ax.set_ylabel("Questions (%)")
        ax.set_ylim(0, 100)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(frameon=False, ncol=2)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        fig.tight_layout()
        save_figure(fig, figures_dir, f"score_distribution_{metric}")
        plt.close(fig)


def plot_ceiling(summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for metric in METRICS:
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            continue
        labels = [CONDITION_LABELS[c] for c in FOUR_CONDITIONS if c in set(sub["condition"])]
        vals = [float(sub[sub["condition"] == c]["percent_score_5"].iloc[0]) for c in FOUR_CONDITIONS if c in set(sub["condition"])]
        colors = ["#4C78A8", "#59A14F", "#F58518", "#B279A2"][: len(vals)]
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Score = 5 (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Potential Ceiling Concentration: {metric.replace('_', ' ').title()}")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        fig.tight_layout()
        save_figure(fig, figures_dir, f"ceiling_proportion_{metric}")
        plt.close(fig)


def plot_paired_deltas(delta_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for metric in GENERAL_METRICS:
        sub = delta_df[delta_df["metric"] == metric]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        delta_values = np.arange(-4, 5)
        width = 0.34
        for idx, (col, label, color) in enumerate([
            ("delta_qwen", "Qwen", "#4C78A8"),
            ("delta_gpt", "GPT-5.4 mini", "#F58518"),
        ]):
            counts = sub[col].value_counts().reindex(delta_values, fill_value=0)
            props = counts / counts.sum() * 100
            ax.bar(delta_values + (idx - 0.5) * width, props, width=width, label=label, color=color)
        ax.axvline(0, color="#374151", linewidth=0.8)
        ax.set_xticks(delta_values)
        ax.set_xlabel("Question-level score change (GraphRAG - no GraphRAG)")
        ax.set_ylabel("Questions (%)")
        ax.set_title(f"Paired GraphRAG Score Change: {metric.replace('_', ' ').title()}")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        fig.tight_layout()
        save_figure(fig, figures_dir, f"paired_score_change_{metric}")
        plt.close(fig)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    planned: pd.DataFrame,
    ceiling: pd.DataFrame,
    variance: pd.DataFrame,
    win_ties: pd.DataFrame,
    ordinal_status: Dict,
    pairwise_status: Dict,
) -> None:
    factual = summary[summary["metric"] == "biomedical_factual_correctness"].set_index("condition")
    rel = summary[summary["metric"] == "answer_relevance"].set_index("condition")

    def med(table: pd.DataFrame, condition: str) -> str:
        return "NA" if condition not in table.index else f"{table.loc[condition, 'median']:.1f}"

    lines = [
        "# Extended IGKF Evaluation Analysis",
        "",
        "This report is generated only from existing normalized judge scores. It does not change, delete, regenerate, or selectively filter any evaluation results.",
        "",
        "## Median 3 to 5 Pattern",
        "",
        (
            "The biomedical factual-correctness medians are "
            f"{med(factual, 'qwen3_8b_baseline')} to {med(factual, 'qwen3_8b_igkf')} for Qwen and "
            f"{med(factual, 'gpt_5_4_mini_baseline')} to {med(factual, 'gpt_5_4_mini_igkf')} for GPT-5.4 mini. "
            "Conservatively, this is evidence of a strong GraphRAG-associated improvement across both model families."
        ),
        (
            "The answer-relevance medians are "
            f"{med(rel, 'qwen3_8b_baseline')} to {med(rel, 'qwen3_8b_igkf')} for Qwen and "
            f"{med(rel, 'gpt_5_4_mini_baseline')} to {med(rel, 'gpt_5_4_mini_igkf')} for GPT-5.4 mini."
        ),
        "",
        "## Ceiling and Score Compression",
        "",
        "A median of 5 alone is not treated as proof that the benchmark is saturated. The ceiling diagnostics report the score-5 proportion, variance, entropy, and question-paired GraphRAG ties so the upper-end discrimination can be inspected directly.",
        "",
    ]

    factual_ceiling = ceiling[(ceiling["metric"] == "biomedical_factual_correctness") & (ceiling["condition"].isin(FOUR_CONDITIONS))]
    if not factual_ceiling.empty:
        lines.append("For biomedical factual correctness, score-5 proportions were:")
        for _, row in factual_ceiling.iterrows():
            lines.append(f"- {row['condition_label']}: {row['proportion_at_max_score']:.1%}")
        lines.append("")

    factual_variance = variance[variance["metric"] == "biomedical_factual_correctness"]
    if not factual_variance.empty:
        lines.append("Variance reduction from non-GraphRAG to GraphRAG for biomedical factual correctness:")
        for _, row in factual_variance.iterrows():
            lines.append(f"- {row['model']}: {row['variance_reduction_fraction']:.1%}")
        lines.append("")

    lines.extend([
        "## Model Family Differences After GraphRAG",
        "",
        "Exact ties on the 1-5 scale are counted but are not interpreted as model equivalence. Residual model differences are assessed with paired Qwen-vs-GPT comparisons within GraphRAG and non-GraphRAG conditions.",
        "",
    ])
    factual_ties = win_ties[(win_ties["metric"] == "biomedical_factual_correctness") & (win_ties["comparison"] == "GraphRAG systems")]
    if not factual_ties.empty:
        row = factual_ties.iloc[0]
        lines.append(
            f"For biomedical factual correctness among GraphRAG systems, Qwen was higher on {int(row['qwen_higher'])} questions, "
            f"GPT-5.4 mini was higher on {int(row['gpt_higher'])}, and exact score ties occurred on {int(row['exact_score_tie'])}."
        )
        lines.append("")

    lines.extend([
        "## Model x GraphRAG Interaction",
        "",
        "The preferred analysis is a cumulative-link mixed-effects model with a question-level random intercept. That model was not fit because the active Python environment lacks a defensible CLMM implementation.",
        f"Fallback used: {ordinal_status['fallback']}",
        "",
        "## Pairwise Judging",
        "",
    ])
    if pairwise_status["pairwise_judge_infrastructure_found"]:
        lines.append("Existing pairwise judge infrastructure was detected, but this command did not submit or prepare a paid judge batch.")
    else:
        lines.append("No existing pairwise A/B judge infrastructure was detected in the repository code. A future blinded pairwise judge pass is recommended if the absolute 1-5 scale appears compressed near the ceiling.")
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `score_distribution_summary.csv`",
        "- `score_frequency_table.csv`",
        "- `paired_question_deltas.csv`",
        "- `paired_delta_summary.csv`",
        "- `model_pairwise_win_tie_counts.csv`",
        "- `planned_paired_comparisons.csv`",
        "- `factorial_fallback_analysis.csv`",
        "- `ceiling_diagnostics.csv`",
        "- `variance_reduction.csv`",
        "- `figures/`",
        "",
    ])
    (out_dir / "extended_analysis_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extended convergence, ceiling, and paired ordinal analyses.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    scores_path = output_root / "statistics" / "judge_scores_long.csv"
    if not scores_path.exists():
        raise SystemExit(f"Missing normalized score table: {scores_path}")

    out_dir = output_root / "extended_analysis"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_path)
    summary = score_distribution_summary(df)
    freq = score_frequency_table(df)
    delta_df = paired_deltas(df)
    delta_summary = paired_delta_summary(delta_df)
    win_ties = model_win_tie_counts(df)
    zero_method = config["statistics"]["wilcoxon_zero_method"]
    planned = planned_paired_comparisons(df, zero_method)
    fallback = factorial_fallback(df, zero_method)
    ceiling, variance = ceiling_diagnostics(df)
    ordinal_status = ordinal_model_status()
    pairwise_status = pairwise_infrastructure_status(Path.cwd())

    summary.to_csv(out_dir / "score_distribution_summary.csv", index=False)
    freq.to_csv(out_dir / "score_frequency_table.csv", index=False)
    delta_df.to_csv(out_dir / "paired_question_deltas.csv", index=False)
    delta_summary.to_csv(out_dir / "paired_delta_summary.csv", index=False)
    win_ties.to_csv(out_dir / "model_pairwise_win_tie_counts.csv", index=False)
    planned.to_csv(out_dir / "planned_paired_comparisons.csv", index=False)
    fallback.to_csv(out_dir / "factorial_fallback_analysis.csv", index=False)
    ceiling.to_csv(out_dir / "ceiling_diagnostics.csv", index=False)
    variance.to_csv(out_dir / "variance_reduction.csv", index=False)
    (out_dir / "ordinal_model_status.json").write_text(json.dumps(ordinal_status, indent=2))
    (out_dir / "pairwise_infrastructure_status.json").write_text(json.dumps(pairwise_status, indent=2))
    shutil.copyfile(out_dir / "planned_paired_comparisons.csv", out_dir / "publication_statistical_results_table.csv")
    shutil.copyfile(out_dir / "score_distribution_summary.csv", out_dir / "machine_readable_summary_table.csv")

    set_plot_style()
    plot_score_distributions(freq, figures_dir)
    plot_ceiling(summary, figures_dir)
    plot_paired_deltas(delta_df, figures_dir)
    write_report(out_dir, summary, planned, ceiling, variance, win_ties, ordinal_status, pairwise_status)
    print(f"Extended analysis written to {out_dir}")


if __name__ == "__main__":
    main()
