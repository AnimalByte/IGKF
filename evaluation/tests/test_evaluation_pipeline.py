from __future__ import annotations

import tempfile
import unittest
import re
from pathlib import Path

import pandas as pd

from evaluation.common import (
    FOUR_CONDITIONS,
    generation_path,
    load_config,
    load_questions,
    validate_conditions,
    write_json_no_overwrite,
)
from evaluation.judge_answers import parse_judge_json
from evaluation.qwen_generation import Qwen3LocalGenerator
from evaluation.statistical_analysis import paired_tests


CONFIG = "evaluation/configs/paper_experiment.yaml"


class EvaluationPipelineTests(unittest.TestCase):
    def test_config_conditions_and_qwen_non_thinking(self):
        config = load_config(CONFIG)
        self.assertEqual(tuple(config["conditions"]), FOUR_CONDITIONS)
        self.assertEqual(config["local_model"]["model_name"], "Qwen3-8B")
        self.assertIs(config["local_model"]["thinking_mode"], False)

    def test_qwen_template_non_thinking_validation(self):
        config = load_config(CONFIG)
        qwen = Qwen3LocalGenerator(config["local_model"], load_model=False)
        prompt = qwen.build_prompt("You are a test assistant.", "Say hello.")
        self.assertRegex(prompt, re.compile(r"<think>\s*</think>\s*$"))
        self.assertIn("hello", prompt.lower())

    def test_exact_benchmark_count_and_stable_ids(self):
        config = load_config(CONFIG)
        questions = load_questions(config["experiment"]["benchmark_path"])
        self.assertEqual(len(questions), 205)
        self.assertEqual(questions[0].question_id, "Q001")
        self.assertEqual(questions[-1].question_id, "Q205")
        self.assertEqual(len({q.question_text for q in questions}), 205)
        self.assertTrue(all(q.category for q in questions))
        self.assertTrue(all(isinstance(q.manual_review_required, bool) for q in questions))

    def test_condition_validation(self):
        self.assertEqual(validate_conditions(["qwen3_8b_baseline"]), ["qwen3_8b_baseline"])
        with self.assertRaises(ValueError):
            validate_conditions(["bad_condition"])

    def test_skip_existing_and_force_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = generation_path(root, "qwen3_8b_baseline", "q001")
            write_json_no_overwrite(path, {"a": 1}, force=False)
            with self.assertRaises(FileExistsError):
                write_json_no_overwrite(path, {"a": 2}, force=False)
            write_json_no_overwrite(path, {"a": 3}, force=True)

    def test_judge_parser_accepts_valid_json(self):
        parsed = parse_judge_json(
            '{"answer_relevance":{"score":4,"rationale":"ok"},'
            '"biomedical_factual_correctness":{"score":5,"rationale":"ok"},'
            '"context_groundedness":{"score":null,"rationale":"na"},'
            '"context_utilization":{"score":null,"rationale":"na"}}'
        )
        self.assertEqual(parsed["answer_relevance"]["score"], 4)

    def test_judge_parser_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            parse_judge_json('{"answer_relevance":{"score":9,"rationale":"bad"}}')

    def test_statistics_pairs_by_question_id(self):
        df = pd.DataFrame([
            {"question_id": "q001", "condition": "qwen3_8b_baseline", "answer_relevance": 2, "biomedical_factual_correctness": 2},
            {"question_id": "q001", "condition": "qwen3_8b_igkf", "answer_relevance": 4, "biomedical_factual_correctness": 3},
            {"question_id": "q002", "condition": "qwen3_8b_baseline", "answer_relevance": 3, "biomedical_factual_correctness": 3},
            {"question_id": "q002", "condition": "qwen3_8b_igkf", "answer_relevance": 3, "biomedical_factual_correctness": 5},
        ])
        out = paired_tests(df, "wilcox")
        row = out[(out["comparison"] == "qwen3_8b_baseline vs qwen3_8b_igkf") & (out["metric"] == "answer_relevance")].iloc[0]
        self.assertEqual(row["n_paired"], 2)
        self.assertEqual(row["n_zero_differences"], 1)


if __name__ == "__main__":
    unittest.main()
