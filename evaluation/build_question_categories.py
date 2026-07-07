from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.common import load_config, load_questions


def categorize(question: str) -> tuple[str, str, bool, str]:
    q = question.lower()
    hits = []
    rules = [
        ("variant interpretation", ["variant", "snp", "rs", "mutation"]),
        ("gene interpretation", ["gene", "locus", "lgr4", "ar ", "edar", "wnt"]),
        ("pathway reasoning", ["pathway", "go term", "biological process", "mechanism"]),
        ("drug or therapeutic reasoning", ["drug", "treatment", "therapeutic", "dutasteride", "finasteride"]),
        ("regulatory reasoning", ["regulatory", "enhancer", "promoter", "encode", "chromatin", "active in"]),
        ("multi-hop or cross-entity reasoning", ["connect", "link", "relationship", "between", "interact", "cross"]),
        ("literature synthesis", ["pubmed", "literature", "abstract", "evidence", "study"]),
    ]
    for category, terms in rules:
        if any(term in q for term in terms):
            hits.append(category)
    if not hits:
        return "general biomedical or control questions", "medium", True, "No specific category keyword matched."
    if len(hits) == 1:
        return hits[0], "medium", False, "Assigned by transparent keyword rule."
    preferred_order = [
        "drug or therapeutic reasoning",
        "regulatory reasoning",
        "variant interpretation",
        "pathway reasoning",
        "gene interpretation",
        "literature synthesis",
        "multi-hop or cross-entity reasoning",
    ]
    for category in preferred_order:
        if category in hits:
            return category, "low", True, f"Multiple possible categories matched: {', '.join(hits)}."
    return hits[0], "low", True, f"Multiple possible categories matched: {', '.join(hits)}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative question category mapping.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--output", default="evaluation/mappings/question_categories.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    questions = load_questions(config["experiment"]["benchmark_path"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question_text",
                "proposed_category",
                "category_confidence",
                "manual_review_required",
                "notes",
            ],
        )
        writer.writeheader()
        for question in questions:
            category, confidence, review, notes = categorize(question.question_text)
            writer.writerow({
                "question_id": question.question_id,
                "question_text": question.question_text,
                "proposed_category": category,
                "category_confidence": confidence,
                "manual_review_required": str(review).lower(),
                "notes": notes,
            })
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
