from __future__ import annotations

import unittest

import pandas as pd

from evaluation.extended_analysis import (
    model_win_tie_counts,
    paired_deltas,
    planned_paired_comparisons,
    score_distribution_summary,
)
from evaluation.common import load_config
from evaluation.pairwise_judge import (
    MINI_IGKF,
    QWEN_IGKF,
    assert_no_model_leakage,
    build_pairwise_prompt,
    decode_winner,
    paired_question_records,
    randomized_mapping,
    reconcile_pairwise,
    reverse_mapping,
)


class ExtendedAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame([
            {"question_id": "Q001", "condition": "qwen3_8b_baseline", "answer_relevance": 3, "biomedical_factual_correctness": 3, "completeness": 2, "uncertainty_abstention": 3},
            {"question_id": "Q001", "condition": "qwen3_8b_igkf", "answer_relevance": 5, "biomedical_factual_correctness": 5, "completeness": 4, "uncertainty_abstention": 5},
            {"question_id": "Q001", "condition": "gpt_5_4_mini_baseline", "answer_relevance": 4, "biomedical_factual_correctness": 3, "completeness": 3, "uncertainty_abstention": 4},
            {"question_id": "Q001", "condition": "gpt_5_4_mini_igkf", "answer_relevance": 5, "biomedical_factual_correctness": 5, "completeness": 5, "uncertainty_abstention": 5},
            {"question_id": "Q002", "condition": "qwen3_8b_baseline", "answer_relevance": 4, "biomedical_factual_correctness": 2, "completeness": 2, "uncertainty_abstention": 3},
            {"question_id": "Q002", "condition": "qwen3_8b_igkf", "answer_relevance": 4, "biomedical_factual_correctness": 4, "completeness": 4, "uncertainty_abstention": 4},
            {"question_id": "Q002", "condition": "gpt_5_4_mini_baseline", "answer_relevance": 4, "biomedical_factual_correctness": 4, "completeness": 3, "uncertainty_abstention": 4},
            {"question_id": "Q002", "condition": "gpt_5_4_mini_igkf", "answer_relevance": 5, "biomedical_factual_correctness": 5, "completeness": 5, "uncertainty_abstention": 5},
        ])

    def test_score_distribution_includes_ceiling_percent(self):
        out = score_distribution_summary(self.df)
        row = out[(out["metric"] == "answer_relevance") & (out["condition"] == "qwen3_8b_igkf")].iloc[0]
        self.assertEqual(row["N"], 2)
        self.assertEqual(row["percent_score_5"], 50.0)
        self.assertEqual(row["count_score_4"], 1)
        self.assertEqual(row["count_score_5"], 1)

    def test_paired_deltas_preserve_question_pairing(self):
        out = paired_deltas(self.df)
        row = out[(out["metric"] == "biomedical_factual_correctness") & (out["question_id"] == "Q001")].iloc[0]
        self.assertEqual(row["delta_qwen"], 2)
        self.assertEqual(row["delta_gpt"], 2)
        self.assertEqual(row["interaction_difference"], 0)

    def test_model_win_tie_counts(self):
        out = model_win_tie_counts(self.df)
        row = out[(out["metric"] == "answer_relevance") & (out["comparison"] == "GraphRAG systems")].iloc[0]
        self.assertEqual(row["N"], 2)
        self.assertEqual(row["gpt_higher"], 1)
        self.assertEqual(row["exact_score_tie"], 1)

    def test_planned_comparisons_apply_holm_by_metric_family(self):
        out = planned_paired_comparisons(self.df, "wilcox")
        metric_rows = out[out["metric"] == "answer_relevance"]
        self.assertEqual(len(metric_rows), 4)
        self.assertTrue((metric_rows["multiple_comparison_correction"] == "holm").all())


class PairwiseJudgeTests(unittest.TestCase):
    def test_ab_randomization_is_reproducible(self):
        first = randomized_mapping(20260706, "Q001")
        second = randomized_mapping(20260706, "Q001")
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"A", "B"})
        self.assertEqual(set(first.values()), {QWEN_IGKF, MINI_IGKF})

    def test_reverse_orientation_decoding(self):
        mapping = {"A": QWEN_IGKF, "B": MINI_IGKF}
        reversed_mapping = reverse_mapping(mapping)
        self.assertEqual(decode_winner("A", mapping), "QWEN")
        self.assertEqual(decode_winner("A", reversed_mapping), "MINI")
        self.assertEqual(decode_winner("TIE", mapping), "TIE")

    def test_reconciliation_logic(self):
        self.assertEqual(reconcile_pairwise(["QWEN", "QWEN"]), ("QWEN_WIN", False))
        self.assertEqual(reconcile_pairwise(["MINI", "MINI"]), ("MINI_WIN", False))
        self.assertEqual(reconcile_pairwise(["TIE", "TIE"]), ("TIE", False))
        self.assertEqual(reconcile_pairwise(["QWEN", "TIE"]), ("TIE", False))
        self.assertEqual(reconcile_pairwise(["QWEN", "MINI"]), ("POSITION_UNSTABLE", True))

    def test_pairwise_prompt_does_not_leak_model_names(self):
        record = {
            "question_id": "QTEST",
            "question_text": "Which answer is better supported?",
            "qwen_answer": "Response about AR supported by the context.",
            "mini_answer": "Another response about AR supported by the context.",
            "reference_context": "Graph Context: Entity: AR. Literature Context: androgen receptor evidence.",
            "context_hash": "abc",
            "retrieval_package_path": "none",
        }
        prompt = build_pairwise_prompt(record, {"A": QWEN_IGKF, "B": MINI_IGKF})
        assert_no_model_leakage(prompt)
        lowered = prompt.lower()
        self.assertNotIn("qwen", lowered)
        self.assertNotIn("gpt", lowered)
        self.assertNotIn("openai", lowered)

    def test_response_pair_alignment_by_question_id(self):
        config = load_config("evaluation/configs/paper_experiment.yaml")
        records = paired_question_records(config, limit=2)
        self.assertEqual([r["question_id"] for r in records], ["Q001", "Q002"])
        self.assertTrue(all(r["qwen_answer"] for r in records))
        self.assertTrue(all(r["mini_answer"] for r in records))
        self.assertTrue(all(r["context_hash"] for r in records))


if __name__ == "__main__":
    unittest.main()
