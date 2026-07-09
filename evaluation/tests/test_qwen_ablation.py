from __future__ import annotations

import unittest

from evaluation.ablation_common import load_ablation_config
from evaluation.run_qwen_ablation import assert_package_isolated, build_package


class QwenAblationTests(unittest.TestCase):
    def test_ablation_config_preserves_paper_qwen_settings(self):
        config = load_ablation_config("evaluation/configs/qwen_ablation.yaml")
        self.assertEqual(config["local_model"]["model_name"], "Qwen3-8B")
        self.assertEqual(config["local_model"]["quantization"], "Q4_K_M")
        self.assertEqual(config["local_model"]["context_length"], 32768)
        self.assertEqual(config["local_model"]["seed"], 20260706)
        self.assertIs(config["local_model"]["thinking_mode"], False)
        self.assertEqual(config["local_model"]["generation"]["temperature"], 0.3)
        self.assertEqual(config["retrieval"]["initial_k"], 15)
        self.assertEqual(config["retrieval"]["rerank_top_n"], 3)

    def test_graph_only_rejects_vector_context_and_abstract_markers(self):
        good = {
            "user_prompt": "**Graph Context (Structured Data):**\nGene evidence",
            "vector_context": "",
            "retrieval_metadata": {
                "candidate_pmids": [],
                "selected_pmids": [],
                "selected_abstracts": [],
            },
        }
        assert_package_isolated("qwen3_8b_graph_only", good, "question")
        bad = {
            "user_prompt": "--- Abstract 1 (PMID: 1) ---",
            "vector_context": "abstract",
            "retrieval_metadata": {"selected_pmids": ["1"]},
        }
        with self.assertRaises(AssertionError):
            assert_package_isolated("qwen3_8b_graph_only", bad, "question")

    def test_literature_only_requires_original_query_and_no_graph_context(self):
        question = "What is the role of rs6152?"
        good = {
            "user_prompt": "**Literature Context (Retrieved Abstracts):**\n--- Abstract 1 (PMID: 1) ---",
            "graph_context": "",
            "retrieval_metadata": {
                "graph_context": "",
                "expanded_retrieval_query": question,
                "candidate_count": 15,
                "reranked_count": 3,
                "selected_pmids": ["1"],
                "scores": [1.2],
            },
        }
        assert_package_isolated("qwen3_8b_literature_only", good, question)
        bad = {
            "user_prompt": "**Graph Context (Structured Data):**\nGene evidence",
            "graph_context": "Gene evidence",
            "retrieval_metadata": {
                "graph_context": "Gene evidence",
                "expanded_retrieval_query": question + " Gene evidence",
                "candidate_count": 16,
                "reranked_count": 4,
            },
        }
        with self.assertRaises(AssertionError):
            assert_package_isolated("qwen3_8b_literature_only", bad, question)

    def test_graph_only_build_path_does_not_call_literature_retrieval(self):
        class FakeEngine:
            def prepare_graph_only_prompt(self, question):
                return "graph prompt", "graph evidence", "", {"retrieval_mode": "graph_only"}

            def prepare_literature_only_prompt(self, question):
                raise AssertionError("literature path should not be called")

        package = build_package("qwen3_8b_graph_only", "QX", "question", FakeEngine())
        self.assertEqual(package["graph_context"], "graph evidence")
        self.assertEqual(package["vector_context"], "")

    def test_literature_only_build_path_does_not_call_graph_retrieval(self):
        class FakeEngine:
            def prepare_graph_only_prompt(self, question):
                raise AssertionError("graph path should not be called")

            def prepare_literature_only_prompt(self, question):
                return "literature prompt", "", "abstract evidence", {
                    "retrieval_mode": "literature_only",
                    "expanded_retrieval_query": question,
                    "candidate_count": 1,
                    "reranked_count": 1,
                }

        package = build_package("qwen3_8b_literature_only", "QX", "question", FakeEngine())
        self.assertEqual(package["graph_context"], "")
        self.assertEqual(package["vector_context"], "abstract evidence")


if __name__ == "__main__":
    unittest.main()
