#!/usr/bin/env python
"""
Evaluate retrieval quality (Recall@k, MRR@k, nDCG@k) for IGKF RAG retrieval.

Benchmark format (JSONL):
{"question": "...", "relevant_pmids": ["12345", "67890"]}

Usage:
python 15_retrieval_evaluation.py --benchmark benchmark_retrieval.jsonl
python 15_retrieval_evaluation.py --benchmark benchmark_retrieval.jsonl --rerankers "cross-encoder/ms-marco-MiniLM-L-6-v2" "ncbi/MedCPT-Cross-Encoder"
"""

import json
import math
import argparse
from typing import Dict, List, Tuple

from query_engine import GraphRAGQueryEngine
from config import RERANKER_CANDIDATES, RERANK_TOP_N


def read_benchmark(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "relevant_pmids" not in row:
                raise ValueError("Each JSONL row must contain `question` and `relevant_pmids`.")
            rows.append({
                "question": row["question"],
                "relevant_pmids": [str(p) for p in row["relevant_pmids"]],
            })
    return rows


def recall_at_k(pred: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(pred[:k])
    return len(top.intersection(set(gold))) / len(set(gold))


def mrr_at_k(pred: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    for i, pid in enumerate(pred[:k], start=1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    dcg = 0.0
    for i, pid in enumerate(pred[:k], start=1):
        rel = 1.0 if pid in gold_set else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(k, len(gold_set))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_predictions(predictions: List[List[str]], benchmark: List[Dict], k: int) -> Dict[str, float]:
    recalls, mrrs, ndcgs = [], [], []
    for pred, row in zip(predictions, benchmark):
        gold = row["relevant_pmids"]
        recalls.append(recall_at_k(pred, gold, k))
        mrrs.append(mrr_at_k(pred, gold, k))
        ndcgs.append(ndcg_at_k(pred, gold, k))
    n = len(benchmark) if benchmark else 1
    return {
        f"Recall@{k}": sum(recalls) / n,
        f"MRR@{k}": sum(mrrs) / n,
        f"nDCG@{k}": sum(ndcgs) / n,
    }


def run_eval(benchmark: List[Dict], reranker_model: str, k: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    engine = GraphRAGQueryEngine(llm=None, reranker_model_name=reranker_model)
    reranked_predictions = []
    baseline_predictions = []

    for row in benchmark:
        engine.retrieve_context(row["question"])
        meta = engine.get_last_retrieval_metadata()
        reranked = [str(x) for x in meta.get("selected_pmids", [])][:k]
        baseline = [str(x) for x in meta.get("candidate_pmids", [])][:k]
        reranked_predictions.append(reranked)
        baseline_predictions.append(baseline)

    engine.close()
    return (
        evaluate_predictions(reranked_predictions, benchmark, k),
        evaluate_predictions(baseline_predictions, benchmark, k),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate IGKF retrieval + reranking quality.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL file.")
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=RERANKER_CANDIDATES,
        help="Reranker model names to evaluate.",
    )
    parser.add_argument("--k", type=int, default=RERANK_TOP_N, help="Cutoff k for Recall/MRR/nDCG.")
    args = parser.parse_args()

    benchmark = read_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark)} benchmark questions.")
    print(f"Evaluating @k={args.k}\n")

    for model_name in args.rerankers:
        print(f"=== Reranker: {model_name} ===")
        reranked, baseline = run_eval(benchmark, model_name, args.k)
        print("Reranked metrics:")
        for k, v in reranked.items():
            print(f"  {k}: {v:.4f}")
        print("Baseline (bi-encoder top-k before rerank) metrics:")
        for k, v in baseline.items():
            print(f"  {k}: {v:.4f}")
        print()


if __name__ == "__main__":
    main()
