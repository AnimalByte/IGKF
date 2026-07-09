from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from evaluation.ablation_common import NEW_ABLATION_CONDITIONS, load_ablation_config, selected_questions
from evaluation.common import generation_path, parse_csv_arg, read_json, utc_now_iso, write_json_no_overwrite
from evaluation.judge_answers import build_judge_prompt


def make_judge_generation_view(generation: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ablation generations to the paper judge prompt schema without changing scores."""
    view = dict(generation)
    view["condition"] = "qwen3_8b_igkf"
    if generation["condition"] == "qwen3_8b_graph_only":
        final_context = f"Graph Context:\n{generation.get('graph_context', '')}"
    else:
        final_context = f"Literature Context:\n{generation.get('vector_context', '')}"
    metadata = dict(generation.get("retrieval_metadata") or {})
    metadata["final_context"] = final_context
    view["retrieval_metadata"] = metadata
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare no-submit judge inputs for Qwen ablation arms.")
    parser.add_argument("--config", default="evaluation/configs/qwen_ablation.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-ids")
    parser.add_argument("--conditions", default=",".join(NEW_ABLATION_CONDITIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_ablation_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    questions = selected_questions(config, args.limit, parse_csv_arg(args.question_ids))
    conditions = parse_csv_arg(args.conditions) or list(NEW_ABLATION_CONDITIONS)
    invalid = set(conditions) - set(NEW_ABLATION_CONDITIONS)
    if invalid:
        raise ValueError(f"Only new ablation conditions can be judged here: {sorted(invalid)}")

    judge_cfg = {
        "openai": {"judge_model": config["judging"]["judge_model"]},
        "judging": config["judging"],
    }
    out_path = output_root / "judge_inputs" / f"qwen_ablation_judge_preview_{int(time.time())}.jsonl"
    if out_path.exists() and not args.force:
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for condition in conditions:
            for question in questions:
                path = generation_path(output_root, condition, question.question_id)
                if not path.exists():
                    continue
                generation = read_json(path)
                if generation.get("error"):
                    continue
                prompt = build_judge_prompt(make_judge_generation_view(generation), judge_cfg)
                record = {
                    "schema_version": "qwen_ablation_judge_input_preview_v1",
                    "created_at": utc_now_iso(),
                    "question_id": question.question_id,
                    "condition": condition,
                    "judge_model": config["judging"]["judge_model"],
                    "judge_reasoning_effort": config["judging"]["judge_reasoning_effort"],
                    "judge_prompt": prompt,
                    "note": "Prepared locally only; no OpenAI API call or Batch submission performed.",
                }
                f.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
    manifest = {
        "schema_version": "qwen_ablation_judge_preview_manifest_v1",
        "created_at": utc_now_iso(),
        "preview_path": str(out_path),
        "request_count": count,
        "conditions": conditions,
        "api_submitted": False,
    }
    manifest_path = output_root / "metadata" / f"qwen_ablation_judge_preview_{int(time.time())}.json"
    write_json_no_overwrite(manifest_path, manifest, force=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
