from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

from evaluation.common import ensure_dirs, load_config, write_json_no_overwrite


REQUIRED_COLUMNS = {"question_id", "PMID", "relevance_grade", "reviewer", "notes", "adjudication_status"}


def recall_at_k(pred: List[str], gold: List[str], k: int) -> float:
    return len(set(pred[:k]).intersection(gold)) / len(set(gold)) if gold else 0.0


def mrr_at_k(pred: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    for i, pmid in enumerate(pred[:k], 1):
        if pmid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: List[str], gold_grades: Dict[str, float], k: int) -> float:
    dcg = 0.0
    for i, pmid in enumerate(pred[:k], 1):
        rel = gold_grades.get(pmid, 0.0)
        dcg += rel / math.log2(i + 1)
    ideal = sorted(gold_grades.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval only when manual relevance judgments exist.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--judgments", default="evaluation/mappings/relevance_judgments_template.csv")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    judgments = Path(args.judgments)
    if not judgments.exists():
        write_json_no_overwrite(
            output_root / "retrieval_metrics" / "retrieval_status.json",
            {
                "status": "not_evaluated",
                "reason": "No manual relevance-judgment file exists. Retrieval metrics would be circular without gold labels.",
                "required_schema": sorted(REQUIRED_COLUMNS),
            },
            force=True,
        )
        print("Retrieval metrics not evaluated: no relevance judgments.")
        return

    with judgments.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != REQUIRED_COLUMNS:
            raise ValueError(f"Judgment schema must be exactly {sorted(REQUIRED_COLUMNS)}")
        rows = [r for r in reader if r.get("adjudication_status") == "adjudicated"]
    if not rows:
        write_json_no_overwrite(
            output_root / "retrieval_metrics" / "retrieval_status.json",
            {"status": "not_evaluated", "reason": "No adjudicated relevance judgments available."},
            force=True,
        )
        print("Retrieval metrics not evaluated: no adjudicated judgments.")
        return
    raise NotImplementedError(
        "Manual judgments are present, but linking them to saved retrieval outputs should be reviewed before reporting metrics."
    )


if __name__ == "__main__":
    main()
