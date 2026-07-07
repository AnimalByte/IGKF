from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.common import (
    FOUR_CONDITIONS,
    condition_uses_igkf,
    ensure_dirs,
    generation_path,
    judge_path,
    load_config,
    load_questions,
    parse_csv_arg,
    read_json,
    utc_now_iso,
    validate_conditions,
    write_json_no_overwrite,
)


def build_judge_prompt(generation: Dict[str, Any], config: Dict[str, Any]) -> str:
    base = {
        "question": generation["question_text"],
        "answer": generation.get("parsed_answer") or "",
        "metrics": {
            "answer_relevance": "How directly, completely, and appropriately the answer addresses the actual question.",
            "biomedical_factual_correctness": "Whether biomedical claims are scientifically accurate and avoid unsupported genetic, clinical, regulatory, therapeutic, or pathway claims.",
        },
    }
    if condition_uses_igkf(generation["condition"]):
        base["supplied_context"] = (generation.get("retrieval_metadata") or {}).get("final_context", "")
        base["metrics"]["context_groundedness"] = "For IGKF answers only: whether evidence-based claims are supported by the exact supplied graph and literature context."
        base["metrics"]["context_utilization"] = "For IGKF answers only: whether the answer appropriately uses relevant supplied evidence without ignoring important context or adding unsupported claims."
    return (
        "You are a strict biomedical answer evaluator. Do not reward length by itself. "
        "Return only valid JSON. Do not mention model names or experimental conditions.\n\n"
        f"{json.dumps(base, indent=2)}\n\n"
        "Return schema:\n"
        "{\n"
        '  "answer_relevance": {"score": 1-5, "rationale": "..."},\n'
        '  "biomedical_factual_correctness": {"score": 1-5, "rationale": "..."},\n'
        '  "context_groundedness": {"score": 1-5 or null, "rationale": "..."},\n'
        '  "context_utilization": {"score": 1-5 or null, "rationale": "..."}\n'
        "}\n"
    )


def parse_judge_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response.")
    parsed = json.loads(match.group(0))
    for metric in ["answer_relevance", "biomedical_factual_correctness", "context_groundedness", "context_utilization"]:
        if metric not in parsed:
            raise ValueError(f"Missing judge metric: {metric}")
        value = parsed[metric]
        if not isinstance(value, dict) or "score" not in value or "rationale" not in value:
            raise ValueError(f"Malformed judge metric: {metric}")
        score = value["score"]
        if score is not None and not (1 <= float(score) <= 5):
            raise ValueError(f"Judge score out of range for {metric}: {score}")
    return parsed


class JudgeClient:
    def __init__(self, model: str):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for non-dry-run judging.")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
        )
        return getattr(response, "output_text", "") or json.dumps(response.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge saved IGKF paper generations.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    questions = load_questions(config["experiment"]["benchmark_path"])
    if args.limit is not None:
        questions = questions[: args.limit]
    conditions = validate_conditions(parse_csv_arg(args.conditions))

    if args.dry_run:
        print(f"Judge dry run OK: {len(questions)} questions x {len(conditions)} conditions")
        return

    client = JudgeClient(config["openai"]["judge_model"])
    max_retries = int(config["judging"]["max_retries"])
    for condition in conditions:
        for question in questions:
            gen_path = generation_path(output_root, condition, question.question_id)
            if not gen_path.exists():
                print(f"missing generation {condition}/{question.question_id}")
                continue
            out_path = judge_path(output_root, condition, question.question_id)
            if out_path.exists() and not args.force:
                print(f"skip existing judge {condition}/{question.question_id}")
                continue
            generation = read_json(gen_path)
            prompt = build_judge_prompt(generation, config)
            raw_attempts = []
            parsed = None
            error = None
            for attempt in range(max_retries + 1):
                try:
                    raw = client.invoke(prompt)
                    raw_attempts.append(raw)
                    parsed = parse_judge_json(raw)
                    error = None
                    break
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc), "attempt": attempt}
            payload = {
                "schema_version": "paper_judge_v1",
                "timestamp": utc_now_iso(),
                "question_id": question.question_id,
                "condition": condition,
                "judge_model": config["openai"]["judge_model"],
                "judge_prompt": prompt,
                "raw_judge_responses": raw_attempts,
                "parsed_scores": parsed,
                "error": error,
            }
            write_json_no_overwrite(out_path, payload, force=args.force)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
