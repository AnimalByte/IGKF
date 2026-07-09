from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.ablation_common import NEW_ABLATION_CONDITIONS, load_ablation_config
from evaluation.common import (
    filter_questions,
    generation_path,
    judge_path,
    load_questions,
    parse_csv_arg,
    read_json,
    retrieval_package_path,
    utc_now_iso,
    write_json_no_overwrite,
)
from evaluation.openai_batch import build_judge_pass_a_prompt, estimate_tokens, extract_response_text, response_body, score_to_float
from evaluation.statistical_analysis import GENERAL_METRICS


PASS_NAME = "qwen_ablation_answer_quality"


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for ablation Batch judging.")
    from openai import OpenAI

    return OpenAI()


def evidence_policy(question: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    canonical_root = Path(config["experiment"]["canonical_output_root"])
    package_path = retrieval_package_path(canonical_root, question.question_id)
    study_specific = question.answerability not in {"general_biomedical", "out_of_scope_or_insufficient"}
    if study_specific and package_path.exists():
        package = read_json(package_path)
        return {
            "factual_correctness_evaluable": True,
            "reference_evidence_packet_id": str(package_path),
            "reference_evidence_packet": package.get("final_context", ""),
        }
    if study_specific:
        return {
            "factual_correctness_evaluable": False,
            "reference_evidence_packet_id": None,
            "reference_evidence_packet": None,
        }
    return {
        "factual_correctness_evaluable": True,
        "reference_evidence_packet_id": None,
        "reference_evidence_packet": None,
    }


def load_generation_answer(output_root: Path, condition: str, question_id: str) -> Dict[str, Any]:
    path = generation_path(output_root, condition, question_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing ablation generation: {path}")
    obj = read_json(path)
    if obj.get("error"):
        raise RuntimeError(f"Ablation generation has error for {condition}/{question_id}: {obj['error']}")
    return obj


def prepare_batch(config: Dict[str, Any], *, limit: int | None, question_ids: Iterable[str] | None, force: bool) -> Dict[str, Any]:
    output_root = Path(config["experiment"]["output_root"])
    for rel in ["batch_inputs", "batch_metadata", "batch_outputs", "judge_scores"]:
        (output_root / rel).mkdir(parents=True, exist_ok=True)
    questions = filter_questions(load_questions(config["experiment"]["benchmark_path"]), limit, question_ids)
    judge_model = config["judging"]["judge_model"]
    reasoning_effort = config["judging"]["judge_reasoning_effort"]
    max_output_tokens = int(config["judging"]["judge_max_output_tokens"])
    seed = int(config["experiment"]["random_seed"])

    requests: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    input_token_estimate = 0
    for question in questions:
        rng = random.Random(f"{seed}:{question.question_id}:{PASS_NAME}")
        answer_items = []
        for idx, condition in enumerate(NEW_ABLATION_CONDITIONS, start=1):
            generation = load_generation_answer(output_root, condition, question.question_id)
            answer_id = f"AB{idx}"
            answer_items.append({"answer_id": answer_id, "answer": generation.get("parsed_answer") or ""})
            mappings.append({
                "pass": PASS_NAME,
                "question_id": question.question_id,
                "answer_id": answer_id,
                "condition": condition,
                "randomization_seed": f"{seed}:{question.question_id}:{PASS_NAME}",
            })
        rng.shuffle(answer_items)
        prompt = build_judge_pass_a_prompt(question, answer_items, evidence_policy(question, config))
        metadata = {"pass": PASS_NAME, "question_id": question.question_id}
        requests.append({
            "custom_id": f"{PASS_NAME}::{question.question_id}",
            "method": "POST",
            "url": "/v1/responses",
            "body": response_body(judge_model, reasoning_effort, max_output_tokens, prompt, metadata),
        })
        input_token_estimate += estimate_tokens(prompt)

    stamp = int(time.time())
    source_jsonl = output_root / "batch_inputs" / f"{PASS_NAME}_{stamp}.jsonl"
    if source_jsonl.exists() and not force:
        raise FileExistsError(source_jsonl)
    with source_jsonl.open("w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, sort_keys=True) + "\n")

    mapping_path = output_root / "batch_metadata" / f"{source_jsonl.stem}_answer_mapping.json"
    write_json_no_overwrite(mapping_path, {"created_at": utc_now_iso(), "mappings": mappings}, force=force)
    manifest = {
        "schema_version": "qwen_ablation_batch_judge_manifest_v1",
        "created_at": utc_now_iso(),
        "source_jsonl": str(source_jsonl),
        "answer_mapping": str(mapping_path),
        "judge_pass": PASS_NAME,
        "request_count": len(requests),
        "estimated_input_tokens": input_token_estimate,
        "configured_output_token_budget": len(requests) * max_output_tokens,
        "model": judge_model,
        "reasoning_effort": reasoning_effort,
        "completion_window": "24h",
        "submitted": False,
    }
    manifest_path = output_root / "batch_metadata" / f"{source_jsonl.stem}_manifest.json"
    write_json_no_overwrite(manifest_path, manifest, force=force)
    return manifest | {"manifest_path": str(manifest_path)}


def submit_batch(manifest_path: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    client = get_client()
    with Path(manifest["source_jsonl"]).open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=manifest["completion_window"],
    )
    manifest.update({
        "submitted": True,
        "submitted_at": utc_now_iso(),
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "batch_status": batch.model_dump(mode="json"),
    })
    write_json_no_overwrite(manifest_path, manifest, force=True)
    return manifest


def refresh_status(manifest_path: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    client = get_client()
    batch = client.batches.retrieve(manifest["batch_id"])
    manifest["batch_status"] = batch.model_dump(mode="json")
    manifest["status_refreshed_at"] = utc_now_iso()
    write_json_no_overwrite(manifest_path, manifest, force=True)
    return manifest


def download_batch(manifest_path: Path) -> Dict[str, Any]:
    manifest = refresh_status(manifest_path)
    output_file_id = (manifest.get("batch_status") or {}).get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch has no output_file_id yet.")
    client = get_client()
    content = client.files.content(output_file_id).read()
    output_root = Path(load_ablation_config("evaluation/configs/qwen_ablation.yaml")["experiment"]["output_root"])
    out = output_root / "batch_outputs" / f"{manifest['batch_id']}_raw.jsonl"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_bytes(content)
    manifest["output_file_id"] = output_file_id
    manifest["raw_output_jsonl"] = str(out)
    manifest["downloaded_at"] = utc_now_iso()
    write_json_no_overwrite(manifest_path, manifest, force=True)
    return manifest


def normalize_batch(config: Dict[str, Any], manifest_path: Path, *, force: bool) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    raw_path = manifest.get("raw_output_jsonl")
    if not raw_path:
        raise RuntimeError("Manifest has no raw_output_jsonl. Download the completed batch first.")
    mappings = read_json(Path(manifest["answer_mapping"]))["mappings"]
    map_by_answer = {(m["question_id"], m["answer_id"]): m for m in mappings}
    questions = {q.question_id: q for q in load_questions(config["experiment"]["benchmark_path"])}
    output_root = Path(config["experiment"]["output_root"])
    normalized: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with Path(raw_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            parts = custom_id.split("::")
            response = obj.get("response") or {}
            body = response.get("body") or {}
            raw_text = extract_response_text(body)
            try:
                parsed_response = json.loads(raw_text)
            except Exception as exc:
                errors.append({
                    "line": line_no,
                    "custom_id": custom_id,
                    "error": f"unparseable judge response: {type(exc).__name__}: {exc}",
                    "status": body.get("status"),
                    "incomplete_details": body.get("incomplete_details"),
                    "usage": body.get("usage"),
                })
                continue
            if len(parts) != 2 or parts[0] != PASS_NAME:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected custom_id"})
                continue
            question_id = parts[1]
            question = questions.get(question_id)
            if question is None:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unknown question_id"})
                continue
            evidence = evidence_policy(question, config)
            answers = parsed_response.get("answers")
            if not isinstance(answers, list):
                errors.append({"line": line_no, "custom_id": custom_id, "error": "response missing answers list"})
                continue
            for answer in answers:
                answer_id = answer.get("answer_id")
                mapped = map_by_answer.get((question_id, answer_id))
                if not mapped:
                    errors.append({"line": line_no, "custom_id": custom_id, "answer_id": answer_id, "error": "missing answer mapping"})
                    continue
                condition = mapped["condition"]
                parsed_scores = {}
                for metric in GENERAL_METRICS:
                    metric_obj = answer.get(metric)
                    if not isinstance(metric_obj, dict) or "score" not in metric_obj:
                        errors.append({"line": line_no, "custom_id": custom_id, "answer_id": answer_id, "error": f"missing metric {metric}"})
                        parsed_scores = {}
                        break
                    parsed_scores[metric] = {
                        "score": score_to_float(metric_obj.get("score")),
                        "rationale": metric_obj.get("rationale"),
                    }
                if not parsed_scores:
                    continue
                out = judge_path(output_root, condition, question_id)
                payload = {
                    "schema_version": "qwen_ablation_judge_batch_v1",
                    "timestamp": utc_now_iso(),
                    "question_id": question_id,
                    "condition": condition,
                    "benchmark_annotations": question.annotations,
                    "judge_model": body.get("model") or manifest["model"],
                    "judge_pass": PASS_NAME,
                    "judge_batch_metadata": {
                        "manifest": str(manifest_path),
                        "custom_id": custom_id,
                        "answer_id": answer_id,
                        "anonymous_answer_mapping": mapped,
                        "raw_judge_response": obj,
                    },
                    "parsed_scores": parsed_scores,
                    "factual_correctness_evaluable": evidence["factual_correctness_evaluable"],
                    "reference_evidence_packet_id": evidence["reference_evidence_packet_id"],
                    "error": None,
                }
                write_json_no_overwrite(out, payload, force=force)
                normalized.append({"question_id": question_id, "condition": condition, "path": str(out)})

    summary = {
        "schema_version": "qwen_ablation_judge_normalization_v1",
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path),
        "raw_output_jsonl": raw_path,
        "normalized_count": len(normalized),
        "error_count": len(errors),
        "errors": errors,
        "conditions": sorted({item["condition"] for item in normalized}),
    }
    summary_path = output_root / "batch_metadata" / f"{Path(raw_path).stem}_ablation_judge_normalization_summary.json"
    write_json_no_overwrite(summary_path, summary, force=force)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, submit, download, and normalize Qwen ablation judge Batch jobs.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", default="evaluation/configs/qwen_ablation.yaml")
    prep.add_argument("--limit", type=int)
    prep.add_argument("--question-ids")
    prep.add_argument("--force", action="store_true")

    submit = sub.add_parser("submit")
    submit.add_argument("manifest")

    status = sub.add_parser("status")
    status.add_argument("manifest")

    download = sub.add_parser("download")
    download.add_argument("manifest")

    normalize = sub.add_parser("normalize")
    normalize.add_argument("manifest")
    normalize.add_argument("--config", default="evaluation/configs/qwen_ablation.yaml")
    normalize.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        config = load_ablation_config(args.config)
        print(json.dumps(prepare_batch(config, limit=args.limit, question_ids=parse_csv_arg(args.question_ids), force=args.force), indent=2, sort_keys=True))
    elif args.command == "submit":
        print(json.dumps(submit_batch(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "status":
        print(json.dumps(refresh_status(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "download":
        print(json.dumps(download_batch(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "normalize":
        config = load_ablation_config(args.config)
        print(json.dumps(normalize_batch(config, Path(args.manifest), force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
