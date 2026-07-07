from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.common import (
    FOUR_CONDITIONS,
    condition_uses_igkf,
    ensure_dirs,
    filter_questions,
    generation_path,
    judge_path,
    load_config,
    load_questions,
    parse_csv_arg,
    read_json,
    retrieval_package_path,
    utc_now_iso,
    validate_conditions,
    write_json_no_overwrite,
)
from evaluation.run_generations import build_baseline_user_prompt, get_igkf_engine, get_or_create_retrieval_package
from evaluation.run_generations import result_skeleton


BATCH_GENERATION_CONDITIONS = ("gpt_5_4_mini_baseline", "gpt_5_4_mini_igkf")
JUDGE_PASS_A = "judge_pass_a_general_quality"
JUDGE_PASS_B = "judge_pass_b_igkf_specific"


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def build_generation_request(
    *,
    condition: str,
    question_id: str,
    question_text: str,
    system_prompt: str,
    user_prompt: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    openai_cfg = config["openai"]
    return {
        "custom_id": f"generation::{condition}::{question_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": openai_cfg["generator_model"],
            "reasoning": {"effort": openai_cfg["generator_reasoning_effort"]},
            "max_output_tokens": int(openai_cfg["max_output_tokens"]),
            "metadata": {"question_id": question_id, "condition": condition},
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def response_body(model: str, reasoning_effort: str, max_output_tokens: int, prompt: str, metadata: Dict[str, str]) -> Dict[str, Any]:
    return {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "metadata": metadata,
        "input": [{"role": "user", "content": prompt}],
    }


def prepare_generation_batch(
    *,
    config: Dict[str, Any],
    conditions: Iterable[str],
    limit: int | None,
    question_ids: Iterable[str] | None,
    force: bool,
) -> Dict[str, Any]:
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    questions = filter_questions(load_questions(config["experiment"]["benchmark_path"]), limit, question_ids)
    conditions = list(conditions)
    invalid = set(conditions) - set(BATCH_GENERATION_CONDITIONS)
    if invalid:
        raise ValueError(f"Batch generation only supports GPT conditions: {sorted(invalid)}")

    igkf_engine = None
    if any(condition_uses_igkf(c) for c in conditions):
        igkf_engine = get_igkf_engine()

    requests: List[Dict[str, Any]] = []
    input_token_estimate = 0
    try:
        for condition in conditions:
            uses_igkf = condition_uses_igkf(condition)
            system_prompt = config["prompts"]["igkf_system_prompt"] if uses_igkf else config["prompts"]["baseline_system_prompt"]
            for q in questions:
                if uses_igkf:
                    package = get_or_create_retrieval_package(
                        output_root=output_root,
                        question_id=q.question_id,
                        question_text=q.question_text,
                        igkf_engine=igkf_engine,
                        force=False,
                    )
                    user_prompt = package["user_prompt"]
                else:
                    user_prompt = build_baseline_user_prompt(q.question_text)
                requests.append(build_generation_request(
                    condition=condition,
                    question_id=q.question_id,
                    question_text=q.question_text,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    config=config,
                ))
                input_token_estimate += estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    finally:
        if igkf_engine is not None:
            igkf_engine.close()

    stamp = int(time.time())
    name = "gpt_generation"
    if limit:
        name += f"_limit{limit}"
    out_jsonl = output_root / "batch_inputs" / f"{name}_{stamp}.jsonl"
    if out_jsonl.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "openai_batch_generation_manifest_v1",
        "created_at": utc_now_iso(),
        "source_jsonl": str(out_jsonl),
        "conditions": conditions,
        "request_count": len(requests),
        "estimated_input_tokens": input_token_estimate,
        "configured_output_token_budget": len(requests) * int(config["openai"]["max_output_tokens"]),
        "model": config["openai"]["generator_model"],
        "reasoning_effort": config["openai"]["generator_reasoning_effort"],
        "completion_window": config["openai"]["batch_completion_window"],
        "submitted": False,
    }
    manifest_path = output_root / "batch_metadata" / f"{out_jsonl.stem}_manifest.json"
    write_json_no_overwrite(manifest_path, manifest, force=force)
    return manifest


def load_generation_answer(output_root: Path, condition: str, question_id: str) -> Dict[str, Any]:
    path = generation_path(output_root, condition, question_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing generation: {path}")
    obj = read_json(path)
    if obj.get("error"):
        raise RuntimeError(f"Generation has error for {condition}/{question_id}: {obj['error']}")
    return obj


def factual_evidence_policy(question: Any, output_root: Path) -> Dict[str, Any]:
    package_path = retrieval_package_path(output_root, question.question_id)
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


def build_judge_pass_a_prompt(question: Any, answers: List[Dict[str, str]], evidence: Dict[str, Any]) -> str:
    payload = {
        "question": question.question_text,
        "reference_evidence_packet": evidence["reference_evidence_packet"],
        "factual_correctness_evaluable": evidence["factual_correctness_evaluable"],
        "answers": answers,
        "instructions": {
            "blindness": "Do not infer or discuss model identity, condition identity, or whether an answer used retrieval.",
            "metadata_policy": "Benchmark category and expected behavior are intentionally omitted and must not be inferred as scoring instructions.",
            "scales": "Use anchored ordinal 1-5 scores. If factual_correctness_evaluable is false, return null for biomedical_factual_correctness.",
            "answer_relevance": "1=does not address the question; 3=partially addresses it; 5=directly and appropriately addresses it.",
            "biomedical_factual_correctness": "1=material biomedical errors; 3=mixed/uncertain accuracy; 5=scientifically accurate against reference evidence where supplied.",
            "completeness": "1=omits most necessary components; 3=partially complete; 5=covers the important components without unnecessary padding.",
            "uncertainty_abstention": "1=invents unsupported conclusions; 3=some qualification; 5=appropriately qualifies uncertainty or abstains when evidence is insufficient.",
        },
        "return_schema": {
            "answers": [
                {
                    "answer_id": "anonymous id from input",
                    "answer_relevance": {"score": "1-5", "rationale": "short"},
                    "biomedical_factual_correctness": {"score": "1-5 or null", "rationale": "short"},
                    "completeness": {"score": "1-5", "rationale": "short"},
                    "uncertainty_abstention": {"score": "1-5", "rationale": "short"},
                }
            ]
        },
    }
    return "Return only valid JSON for this biomedical answer-quality evaluation.\n\n" + json.dumps(payload, indent=2)


def build_judge_pass_b_prompt(question: Any, answer_id: str, answer: str, context: str) -> str:
    payload = {
        "question": question.question_text,
        "answer_id": answer_id,
        "answer": answer,
        "exact_igkf_context_supplied_to_generator": context,
        "instructions": {
            "blindness": "Do not infer or discuss model identity.",
            "context_groundedness": "1=claims contradict or are unsupported by supplied context; 3=partially grounded; 5=evidence-based claims are supported by the supplied context.",
            "context_utilization": "1=ignores important supplied evidence or relies on unsupported external claims; 3=uses some relevant evidence; 5=appropriately uses the important supplied evidence.",
        },
        "return_schema": {
            "answer_id": answer_id,
            "context_groundedness": {"score": "1-5", "rationale": "short"},
            "context_utilization": {"score": "1-5", "rationale": "short"},
        },
    }
    return "Return only valid JSON for this IGKF-specific groundedness evaluation.\n\n" + json.dumps(payload, indent=2)


def prepare_judge_batch(
    *,
    config: Dict[str, Any],
    pass_name: str,
    limit: int | None,
    question_ids: Iterable[str] | None,
    force: bool,
) -> Dict[str, Any]:
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    questions = filter_questions(load_questions(config["experiment"]["benchmark_path"]), limit, question_ids)
    seed = int(config["experiment"]["random_seed"])
    requests: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    input_token_estimate = 0
    judge_cfg = config["openai"]

    for question in questions:
        if pass_name == JUDGE_PASS_A:
            rng = random.Random(f"{seed}:{question.question_id}:pass_a")
            answer_items = []
            for idx, condition in enumerate(FOUR_CONDITIONS, start=1):
                generation = load_generation_answer(output_root, condition, question.question_id)
                answer_id = f"A{idx}"
                answer_items.append({"answer_id": answer_id, "answer": generation.get("parsed_answer") or ""})
                mappings.append({
                    "pass": pass_name,
                    "question_id": question.question_id,
                    "answer_id": answer_id,
                    "condition": condition,
                    "randomization_seed": f"{seed}:{question.question_id}:pass_a",
                })
            rng.shuffle(answer_items)
            evidence = factual_evidence_policy(question, output_root)
            prompt = build_judge_pass_a_prompt(question, answer_items, evidence)
            metadata = {"pass": pass_name, "question_id": question.question_id}
            judge_max_output_tokens = int(judge_cfg.get("judge_max_output_tokens", judge_cfg["max_output_tokens"]))
            requests.append({
                "custom_id": f"{pass_name}::{question.question_id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": response_body(judge_cfg["judge_model"], judge_cfg["judge_reasoning_effort"], judge_max_output_tokens, prompt, metadata),
            })
            input_token_estimate += estimate_tokens(prompt)
        elif pass_name == JUDGE_PASS_B:
            judge_max_output_tokens = int(judge_cfg.get("judge_max_output_tokens", judge_cfg["max_output_tokens"]))
            for idx, condition in enumerate(["qwen3_8b_igkf", "gpt_5_4_mini_igkf"], start=1):
                generation = load_generation_answer(output_root, condition, question.question_id)
                package_path = retrieval_package_path(output_root, question.question_id)
                if not package_path.exists():
                    raise FileNotFoundError(f"Missing IGKF retrieval package for Pass B: {package_path}")
                package = read_json(package_path)
                context = package.get("final_context", "")
                context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
                answer_id = f"R{idx}"
                prompt = build_judge_pass_b_prompt(question, answer_id, generation.get("parsed_answer") or "", context)
                requests.append({
                    "custom_id": f"{pass_name}::{question.question_id}::{answer_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": response_body(judge_cfg["judge_model"], judge_cfg["judge_reasoning_effort"], judge_max_output_tokens, prompt, {"pass": pass_name, "question_id": question.question_id, "answer_id": answer_id}),
                })
                mappings.append({
                    "pass": pass_name,
                    "question_id": question.question_id,
                    "answer_id": answer_id,
                    "condition": condition,
                    "context_hash": context_hash,
                    "retrieval_package_path": str(package_path),
                })
                input_token_estimate += estimate_tokens(prompt)
        else:
            raise ValueError(f"Unknown judge pass: {pass_name}")

    stamp = int(time.time())
    out_jsonl = output_root / "batch_inputs" / f"{pass_name}_{stamp}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, sort_keys=True) + "\n")
    mapping_path = output_root / "batch_metadata" / f"{out_jsonl.stem}_answer_mapping.json"
    write_json_no_overwrite(mapping_path, {"created_at": utc_now_iso(), "mappings": mappings}, force=force)
    manifest = {
        "schema_version": "openai_batch_judge_manifest_v1",
        "created_at": utc_now_iso(),
        "source_jsonl": str(out_jsonl),
        "answer_mapping": str(mapping_path),
        "judge_pass": pass_name,
        "request_count": len(requests),
        "estimated_input_tokens": input_token_estimate,
        "configured_output_token_budget": len(requests) * int(judge_cfg.get("judge_max_output_tokens", judge_cfg["max_output_tokens"])),
        "model": judge_cfg["judge_model"],
        "reasoning_effort": judge_cfg["judge_reasoning_effort"],
        "completion_window": judge_cfg["batch_completion_window"],
        "submitted": False,
    }
    manifest_path = output_root / "batch_metadata" / f"{out_jsonl.stem}_manifest.json"
    write_json_no_overwrite(manifest_path, manifest, force=force)
    return manifest


def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI Batch submission/download.")
    from openai import OpenAI

    return OpenAI()


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
    status = manifest["batch_status"]
    output_file_id = status.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch has no output_file_id yet.")
    client = get_client()
    content = client.files.content(output_file_id).read()
    output_root = Path(load_config("evaluation/configs/paper_experiment.yaml")["experiment"]["output_root"])
    out = output_root / "batch_outputs" / f"{manifest['batch_id']}_raw.jsonl"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_bytes(content)
    manifest["output_file_id"] = output_file_id
    manifest["raw_output_jsonl"] = str(out)
    manifest["downloaded_at"] = utc_now_iso()
    write_json_no_overwrite(manifest_path, manifest, force=True)
    return manifest


def extract_response_text(body: Dict[str, Any]) -> str:
    texts: List[str] = []
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts).strip()


def normalize_generation_batch(manifest_path: Path, *, force: bool) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    raw_path = manifest.get("raw_output_jsonl")
    if not raw_path:
        raise RuntimeError("Manifest has no raw_output_jsonl. Download the completed batch first.")

    config = load_config("evaluation/configs/paper_experiment.yaml")
    output_root = Path(config["experiment"]["output_root"])
    questions = {q.question_id: q for q in load_questions(config["experiment"]["benchmark_path"])}
    normalized: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with Path(raw_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            parts = custom_id.split("::")
            if len(parts) != 3 or parts[0] != "generation":
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected custom_id"})
                continue
            _, condition, question_id = parts
            if condition not in BATCH_GENERATION_CONDITIONS:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected generation condition"})
                continue
            question = questions.get(question_id)
            if question is None:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unknown question_id"})
                continue

            response = obj.get("response") or {}
            body = response.get("body") or {}
            status_code = response.get("status_code")
            error_state = obj.get("error") or body.get("error")
            uses_igkf = condition_uses_igkf(condition)
            system_prompt = config["prompts"]["igkf_system_prompt"] if uses_igkf else config["prompts"]["baseline_system_prompt"]
            if uses_igkf:
                package_path = retrieval_package_path(output_root, question_id)
                package = read_json(package_path)
                user_prompt = package["user_prompt"]
                retrieval_metadata = package.get("retrieval_metadata", {})
                retrieval_metadata["retrieval_package_path"] = str(package_path)
            else:
                user_prompt = build_baseline_user_prompt(question.question_text)
                retrieval_metadata = None

            payload = result_skeleton(
                question_id,
                question.question_text,
                condition,
                config["openai"]["generator_model"],
                config["openai"],
            )
            payload.update({
                "benchmark_annotations": question.annotations,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "retrieval_metadata": retrieval_metadata,
                "raw_model_output": obj,
                "parsed_answer": extract_response_text(body),
                "error": error_state,
                "timestamp": utc_now_iso(),
                "generation_seconds": None,
                "openai_batch_metadata": {
                    "batch_id": manifest.get("batch_id"),
                    "input_file_id": manifest.get("input_file_id"),
                    "output_file_id": manifest.get("output_file_id"),
                    "custom_id": custom_id,
                    "status_code": status_code,
                    "request_id": response.get("request_id"),
                    "response_id": body.get("id"),
                    "created_at": body.get("created_at"),
                    "completed_at": body.get("completed_at"),
                    "usage": body.get("usage"),
                    "reasoning": body.get("reasoning"),
                },
            })
            out = generation_path(output_root, condition, question_id)
            write_json_no_overwrite(out, payload, force=force)
            normalized.append({"condition": condition, "question_id": question_id, "path": str(out)})

    summary = {
        "schema_version": "openai_generation_normalization_v1",
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path),
        "raw_output_jsonl": raw_path,
        "normalized_count": len(normalized),
        "error_count": len(errors),
        "errors": errors,
        "conditions": sorted({item["condition"] for item in normalized}),
    }
    summary_path = output_root / "batch_metadata" / f"{Path(raw_path).stem}_normalization_summary.json"
    write_json_no_overwrite(summary_path, summary, force=force)
    return summary


def score_to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def normalize_judge_batch(manifest_path: Path, *, force: bool) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    raw_path = manifest.get("raw_output_jsonl")
    if not raw_path:
        raise RuntimeError("Manifest has no raw_output_jsonl. Download the completed batch first.")
    mapping_path = manifest.get("answer_mapping")
    if not mapping_path:
        raise RuntimeError("Manifest has no answer_mapping.")

    config = load_config("evaluation/configs/paper_experiment.yaml")
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    questions = {q.question_id: q for q in load_questions(config["experiment"]["benchmark_path"])}
    mappings = read_json(Path(mapping_path))["mappings"]
    map_by_answer = {
        (item["question_id"], item["answer_id"]): item
        for item in mappings
    }

    normalized: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    pass_name = manifest["judge_pass"]

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

            if pass_name == JUDGE_PASS_A:
                if len(parts) != 2:
                    errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected Pass A custom_id"})
                    continue
                question_id = parts[1]
                question = questions.get(question_id)
                if question is None:
                    errors.append({"line": line_no, "custom_id": custom_id, "error": "unknown question_id"})
                    continue
                evidence = factual_evidence_policy(question, output_root)
                answers = parsed_response.get("answers")
                if not isinstance(answers, list):
                    errors.append({"line": line_no, "custom_id": custom_id, "error": "Pass A response missing answers list"})
                    continue
                for answer in answers:
                    answer_id = answer.get("answer_id")
                    mapped = map_by_answer.get((question_id, answer_id))
                    if not mapped:
                        errors.append({"line": line_no, "custom_id": custom_id, "answer_id": answer_id, "error": "missing answer mapping"})
                        continue
                    condition = mapped["condition"]
                    parsed_scores = {}
                    for metric in GENERAL_METRICS_FOR_BATCH():
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
                    existing = read_json(out) if out.exists() else {}
                    merged_scores = existing.get("parsed_scores") or {}
                    merged_scores.update(parsed_scores)
                    payload = {
                        **existing,
                        "schema_version": "paper_judge_batch_v1",
                        "timestamp": utc_now_iso(),
                        "question_id": question_id,
                        "condition": condition,
                        "benchmark_annotations": question.annotations,
                        "judge_model": body.get("model") or manifest["model"],
                        "judge_pass_a": {
                            "manifest": str(manifest_path),
                            "custom_id": custom_id,
                            "answer_id": answer_id,
                            "anonymous_answer_mapping": mapped,
                            "raw_judge_response": obj,
                        },
                        "parsed_scores": merged_scores,
                        "factual_correctness_evaluable": evidence["factual_correctness_evaluable"],
                        "reference_evidence_packet_id": evidence["reference_evidence_packet_id"],
                        "error": None,
                    }
                    write_json_no_overwrite(out, payload, force=force or out.exists())
                    normalized.append({"question_id": question_id, "condition": condition, "path": str(out)})

            elif pass_name == JUDGE_PASS_B:
                if len(parts) != 3:
                    errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected Pass B custom_id"})
                    continue
                question_id, answer_id = parts[1], parts[2]
                question = questions.get(question_id)
                mapped = map_by_answer.get((question_id, answer_id))
                if question is None or not mapped:
                    errors.append({"line": line_no, "custom_id": custom_id, "error": "unknown question or answer mapping"})
                    continue
                condition = mapped["condition"]
                parsed_scores = {}
                for metric in ["context_groundedness", "context_utilization"]:
                    metric_obj = parsed_response.get(metric)
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
                existing = read_json(out) if out.exists() else {}
                merged_scores = existing.get("parsed_scores") or {}
                merged_scores.update(parsed_scores)
                payload = {
                    **existing,
                    "schema_version": "paper_judge_batch_v1",
                    "timestamp": utc_now_iso(),
                    "question_id": question_id,
                    "condition": condition,
                    "benchmark_annotations": question.annotations,
                    "judge_model": body.get("model") or manifest["model"],
                    "judge_pass_b": {
                        "manifest": str(manifest_path),
                        "custom_id": custom_id,
                        "answer_id": answer_id,
                        "anonymous_answer_mapping": mapped,
                        "raw_judge_response": obj,
                    },
                    "parsed_scores": merged_scores,
                    "error": None,
                }
                write_json_no_overwrite(out, payload, force=force or out.exists())
                normalized.append({"question_id": question_id, "condition": condition, "path": str(out)})
            else:
                raise ValueError(f"Unsupported judge pass: {pass_name}")

    summary = {
        "schema_version": "openai_judge_normalization_v1",
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path),
        "raw_output_jsonl": raw_path,
        "judge_pass": pass_name,
        "normalized_count": len(normalized),
        "error_count": len(errors),
        "errors": errors,
    }
    summary_path = output_root / "batch_metadata" / f"{Path(raw_path).stem}_judge_normalization_summary.json"
    write_json_no_overwrite(summary_path, summary, force=force)
    return summary


def GENERAL_METRICS_FOR_BATCH() -> List[str]:
    return [
        "answer_relevance",
        "biomedical_factual_correctness",
        "completeness",
        "uncertainty_abstention",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or manage OpenAI Batch jobs for IGKF.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-generations")
    prep.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    prep.add_argument("--conditions", default=",".join(BATCH_GENERATION_CONDITIONS))
    prep.add_argument("--limit", type=int)
    prep.add_argument("--question-ids")
    prep.add_argument("--force", action="store_true")

    judge = sub.add_parser("prepare-judges")
    judge.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    judge.add_argument("--pass-name", choices=[JUDGE_PASS_A, JUDGE_PASS_B], required=True)
    judge.add_argument("--limit", type=int)
    judge.add_argument("--question-ids")
    judge.add_argument("--force", action="store_true")

    for name in ["submit", "status", "download"]:
        p = sub.add_parser(name)
        p.add_argument("manifest")

    norm = sub.add_parser("normalize-generations")
    norm.add_argument("manifest")
    norm.add_argument("--force", action="store_true")

    norm_judge = sub.add_parser("normalize-judges")
    norm_judge.add_argument("manifest")
    norm_judge.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-generations":
        config = load_config(args.config)
        conditions = validate_conditions(parse_csv_arg(args.conditions))
        manifest = prepare_generation_batch(
            config=config,
            conditions=conditions,
            limit=args.limit,
            question_ids=parse_csv_arg(args.question_ids),
            force=args.force,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "prepare-judges":
        config = load_config(args.config)
        manifest = prepare_judge_batch(
            config=config,
            pass_name=args.pass_name,
            limit=args.limit,
            question_ids=parse_csv_arg(args.question_ids),
            force=args.force,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "submit":
        print(json.dumps(submit_batch(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "status":
        print(json.dumps(refresh_status(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "download":
        print(json.dumps(download_batch(Path(args.manifest)), indent=2, sort_keys=True))
    elif args.command == "normalize-generations":
        print(json.dumps(normalize_generation_batch(Path(args.manifest), force=args.force), indent=2, sort_keys=True))
    elif args.command == "normalize-judges":
        print(json.dumps(normalize_judge_batch(Path(args.manifest), force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
