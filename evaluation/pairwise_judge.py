from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import numpy as np
from scipy.stats import binomtest

from evaluation.common import (
    ensure_dirs,
    filter_questions,
    generation_path,
    load_config,
    load_questions,
    parse_csv_arg,
    read_json,
    retrieval_package_path,
    utc_now_iso,
    write_json_no_overwrite,
)
from evaluation.openai_batch import estimate_tokens, extract_response_text, response_body


PAIRWISE_PASS = "pairwise_graphrag_qwen_vs_mini"
QWEN_IGKF = "qwen3_8b_igkf"
MINI_IGKF = "gpt_5_4_mini_igkf"
PAIRWISE_RUBRIC_VERSION = "pairwise_biomedical_rubric_v1"
PAIRWISE_PROMPT_VERSION = "pairwise_prompt_v1"
DEFAULT_PAIRWISE_MAX_OUTPUT_TOKENS = 512


def pairwise_output_dir(output_root: Path) -> Path:
    out = output_root / "pairwise_judge"
    out.mkdir(parents=True, exist_ok=True)
    return out


def answer_text(output_root: Path, condition: str, question_id: str) -> str:
    path = generation_path(output_root, condition, question_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing generation: {path}")
    obj = read_json(path)
    if obj.get("error"):
        raise RuntimeError(f"Generation has error for {condition}/{question_id}: {obj['error']}")
    answer = obj.get("parsed_answer") or ""
    if not answer.strip():
        raise ValueError(f"Generation has blank parsed_answer for {condition}/{question_id}")
    return answer


def paired_question_records(config: Dict[str, Any], limit: int | None = None, question_ids: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    output_root = Path(config["experiment"]["output_root"])
    questions = filter_questions(load_questions(config["experiment"]["benchmark_path"]), limit, question_ids)
    records: List[Dict[str, Any]] = []
    for question in questions:
        package_path = retrieval_package_path(output_root, question.question_id)
        if not package_path.exists():
            continue
        try:
            qwen_answer = answer_text(output_root, QWEN_IGKF, question.question_id)
            mini_answer = answer_text(output_root, MINI_IGKF, question.question_id)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        package = read_json(package_path)
        context = package.get("final_context", "")
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        records.append({
            "question_id": question.question_id,
            "question_text": question.question_text,
            "qwen_answer": qwen_answer,
            "mini_answer": mini_answer,
            "reference_context": context,
            "context_hash": context_hash,
            "retrieval_package_path": str(package_path),
        })
    return records


def randomized_mapping(seed: int, question_id: str) -> Dict[str, str]:
    rng = random.Random(f"{seed}:{question_id}:{PAIRWISE_PASS}")
    if rng.random() < 0.5:
        return {"A": QWEN_IGKF, "B": MINI_IGKF}
    return {"A": MINI_IGKF, "B": QWEN_IGKF}


def reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    return {"A": mapping["B"], "B": mapping["A"]}


def response_for_system(record: Dict[str, Any], condition: str) -> str:
    if condition == QWEN_IGKF:
        return record["qwen_answer"]
    if condition == MINI_IGKF:
        return record["mini_answer"]
    raise ValueError(f"Unexpected pairwise condition: {condition}")


def prompt_payload(record: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    return {
        "question": record["question_text"],
        "reference_evidence_context": record["reference_context"],
        "responses": {
            "A": response_for_system(record, mapping["A"]),
            "B": response_for_system(record, mapping["B"]),
        },
        "task": "Choose which response is better overall for biomedical research quality.",
        "blindness": {
            "do_not_infer_or_discuss": [
                "model identity",
                "provider identity",
                "training source",
                "which system produced either response",
            ],
            "responses_are_anonymous": True,
        },
        "rubric": {
            "biomedical_factual_correctness": "Prefer the response with more accurate biomedical claims and fewer materially incorrect genetic, mechanistic, pathway, clinical, regulatory, or therapeutic statements.",
            "evidence_support": "Prefer claims supported by the supplied reference context when the question depends on study-specific or evidence-specific facts.",
            "mechanistic_reasoning": "Prefer accurate mechanistic reasoning without speculative leaps.",
            "evidence_integration": "Prefer appropriate synthesis of graph and literature evidence without merely listing unrelated facts.",
            "completeness": "Prefer the response that addresses the important parts of the question without unnecessary padding.",
            "relevance": "Prefer the response that more directly answers the question.",
            "specificity": "Prefer useful, specific answers over vague generalities when specificity is supported.",
            "unsupported_claims_or_overstatement": "Penalize invented, overstated, or unsupported conclusions.",
            "appropriate_uncertainty": "Prefer calibrated uncertainty or abstention when evidence is insufficient.",
        },
        "tie_policy": "Return TIE only when neither response is meaningfully better overall.",
        "return_format": "Return exactly one JSON object with a single key named winner and value A, B, or TIE. No rationale. No extra keys.",
        "return_schema": {"winner": "A | B | TIE"},
        "prompt_version": PAIRWISE_PROMPT_VERSION,
        "rubric_version": PAIRWISE_RUBRIC_VERSION,
    }


def build_pairwise_prompt(record: Dict[str, Any], mapping: Dict[str, str]) -> str:
    payload = prompt_payload(record, mapping)
    return "Return only valid JSON for this blinded pairwise biomedical evaluation.\n\n" + json.dumps(payload, indent=2)


def assert_no_model_leakage(prompt: str) -> None:
    forbidden_patterns = {
        "qwen": r"\bqwen\b",
        "gpt": r"\bgpt\b|gpt-",
        "openai": r"\bopenai\b",
        "llama": r"\bllama\b",
        "graphrag": r"\bgraph[- ]?rag\b",
        "igkf_condition": r"\bigkf condition\b",
    }
    leaked = [name for name, pattern in forbidden_patterns.items() if re.search(pattern, prompt, flags=re.IGNORECASE)]
    if leaked:
        raise ValueError(f"Pairwise judge prompt leaks forbidden model/condition terms: {leaked}")


def build_pairwise_requests(
    *,
    config: Dict[str, Any],
    limit: int | None,
    question_ids: Iterable[str] | None,
    judge_model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    seed = int(config["experiment"]["random_seed"])
    judge_cfg = config["openai"]
    model = judge_model or judge_cfg["judge_model"]
    effort = reasoning_effort or judge_cfg["judge_reasoning_effort"]
    output_budget = int(max_output_tokens or judge_cfg.get("pairwise_judge_max_output_tokens", DEFAULT_PAIRWISE_MAX_OUTPUT_TOKENS))
    records = paired_question_records(config, limit=limit, question_ids=question_ids)
    requests: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    input_tokens = 0
    for record in records:
        original = randomized_mapping(seed, record["question_id"])
        for orientation, mapping in [("original", original), ("reversed", reverse_mapping(original))]:
            prompt = build_pairwise_prompt(record, mapping)
            assert_no_model_leakage(prompt)
            custom_id = f"{PAIRWISE_PASS}::{record['question_id']}::{orientation}"
            metadata = {
                "pass": PAIRWISE_PASS,
                "question_id": record["question_id"],
                "orientation": orientation,
                "prompt_version": PAIRWISE_PROMPT_VERSION,
                "rubric_version": PAIRWISE_RUBRIC_VERSION,
            }
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": response_body(model, effort, output_budget, prompt, metadata),
            })
            mappings.append({
                "pass": PAIRWISE_PASS,
                "question_id": record["question_id"],
                "orientation": orientation,
                "randomization_seed": f"{seed}:{record['question_id']}:{PAIRWISE_PASS}",
                "A_condition": mapping["A"],
                "B_condition": mapping["B"],
                "context_hash": record["context_hash"],
                "retrieval_package_path": record["retrieval_package_path"],
                "prompt_version": PAIRWISE_PROMPT_VERSION,
                "rubric_version": PAIRWISE_RUBRIC_VERSION,
            })
            input_tokens += estimate_tokens(prompt)
    return requests, mappings, input_tokens


def prepare_pairwise_batch(
    *,
    config: Dict[str, Any],
    limit: int | None,
    question_ids: Iterable[str] | None,
    force: bool,
    dry_run: bool,
    judge_model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> Dict[str, Any]:
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    pairwise_output_dir(output_root)
    requests, mappings, input_tokens = build_pairwise_requests(
        config=config,
        limit=limit,
        question_ids=question_ids,
        judge_model=judge_model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    judge_cfg = config["openai"]
    output_budget = int(max_output_tokens or judge_cfg.get("pairwise_judge_max_output_tokens", DEFAULT_PAIRWISE_MAX_OUTPUT_TOKENS))
    model = judge_model or judge_cfg["judge_model"]
    effort = reasoning_effort or judge_cfg["judge_reasoning_effort"]
    paired_questions = len({item["question_id"] for item in mappings})
    stamp = int(time.time())
    stem = f"{PAIRWISE_PASS}_{stamp}"
    if dry_run:
        preview_path = output_root / "pairwise_judge" / f"{stem}_dry_run_preview.json"
        preview = {
            "schema_version": "pairwise_judge_dry_run_v1",
            "created_at": utc_now_iso(),
            "paired_question_count": paired_questions,
            "request_count": len(requests),
            "estimated_input_tokens": input_tokens,
            "configured_output_token_budget": len(requests) * output_budget,
            "model": model,
            "reasoning_effort": effort,
            "first_request": requests[0] if requests else None,
            "first_mapping": mappings[0] if mappings else None,
        }
        write_json_no_overwrite(preview_path, preview, force=force)
        preview["preview_path"] = str(preview_path)
        return preview

    out_jsonl = output_root / "batch_inputs" / f"{stem}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, sort_keys=True) + "\n")
    mapping_path = output_root / "batch_metadata" / f"{stem}_hidden_mapping.json"
    write_json_no_overwrite(mapping_path, {"created_at": utc_now_iso(), "mappings": mappings}, force=force)
    manifest = {
        "schema_version": "openai_batch_pairwise_judge_manifest_v1",
        "created_at": utc_now_iso(),
        "source_jsonl": str(out_jsonl),
        "hidden_mapping": str(mapping_path),
        "judge_pass": PAIRWISE_PASS,
        "paired_question_count": paired_questions,
        "request_count": len(requests),
        "estimated_input_tokens": input_tokens,
        "configured_output_token_budget": len(requests) * output_budget,
        "model": model,
        "reasoning_effort": effort,
        "completion_window": judge_cfg["batch_completion_window"],
        "prompt_version": PAIRWISE_PROMPT_VERSION,
        "rubric_version": PAIRWISE_RUBRIC_VERSION,
        "submitted": False,
    }
    manifest_path = output_root / "batch_metadata" / f"{stem}_manifest.json"
    write_json_no_overwrite(manifest_path, manifest, force=force)
    return manifest


def parse_pairwise_winner(raw_text: str) -> str:
    text = raw_text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            value = str(obj.get("winner", "")).strip().upper()
            if value in {"A", "B", "TIE"}:
                return value
    except Exception:
        pass
    match = re.fullmatch(r"\s*(A|B|TIE)\s*\.?\s*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    raise ValueError(f"Cannot parse pairwise winner from response: {raw_text[:200]}")


def decode_winner(winner: str, mapping: Dict[str, Any]) -> str:
    if winner == "TIE":
        return "TIE"
    condition = mapping.get(f"{winner}_condition", mapping.get(winner))
    if condition == QWEN_IGKF:
        return "QWEN"
    if condition == MINI_IGKF:
        return "MINI"
    raise ValueError(f"Unknown mapped condition: {condition}")


def reconcile_pairwise(decoded_winners: Sequence[str]) -> Tuple[str, bool]:
    winners = [w for w in decoded_winners if w in {"QWEN", "MINI", "TIE"}]
    if len(winners) < 2:
        return "INCOMPLETE_OR_PARSE_ERROR", False
    non_ties = [w for w in winners if w != "TIE"]
    if not non_ties:
        return "TIE", False
    if len(set(non_ties)) == 1:
        if len(non_ties) == len(winners):
            return ("QWEN_WIN" if non_ties[0] == "QWEN" else "MINI_WIN"), False
        return "TIE", False
    return "POSITION_UNSTABLE", True


def normalize_pairwise_batch(manifest_path: Path, *, force: bool) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    raw_path = manifest.get("raw_output_jsonl")
    if not raw_path:
        raise RuntimeError("Manifest has no raw_output_jsonl. Download the completed batch first.")
    mapping_path = manifest.get("hidden_mapping")
    if not mapping_path:
        raise RuntimeError("Manifest has no hidden_mapping.")
    output_root = Path(load_config("evaluation/configs/paper_experiment.yaml")["experiment"]["output_root"])
    out_dir = pairwise_output_dir(output_root)
    mappings = read_json(Path(mapping_path))["mappings"]
    map_by_key = {(item["question_id"], item["orientation"]): item for item in mappings}
    raw_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with Path(raw_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            parts = custom_id.split("::")
            if len(parts) != 3 or parts[0] != PAIRWISE_PASS:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "unexpected custom_id"})
                continue
            _, question_id, orientation = parts
            mapping = map_by_key.get((question_id, orientation))
            if not mapping:
                errors.append({"line": line_no, "custom_id": custom_id, "error": "missing hidden mapping"})
                continue
            response = obj.get("response") or {}
            body = response.get("body") or {}
            raw_text = extract_response_text(body)
            try:
                winner = parse_pairwise_winner(raw_text)
                decoded = decode_winner(winner, mapping)
                parse_error = None
            except Exception as exc:
                winner = None
                decoded = "PARSE_ERROR"
                parse_error = f"{type(exc).__name__}: {exc}"
            raw_rows.append({
                "question_id": question_id,
                "orientation": orientation,
                "custom_id": custom_id,
                "randomization_seed": mapping["randomization_seed"],
                "A_condition": mapping["A_condition"],
                "B_condition": mapping["B_condition"],
                "context_hash": mapping["context_hash"],
                "retrieval_package_path": mapping["retrieval_package_path"],
                "judge_model": body.get("model") or manifest["model"],
                "prompt_version": mapping["prompt_version"],
                "rubric_version": mapping["rubric_version"],
                "raw_winner": winner,
                "decoded_winner": decoded,
                "raw_judge_response_text": raw_text,
                "parse_error": parse_error,
                "status_code": response.get("status_code"),
                "request_id": response.get("request_id"),
                "response_id": body.get("id"),
                "usage_json": json.dumps(body.get("usage"), sort_keys=True),
                "normalized_at": utc_now_iso(),
            })

    raw_df = pd.DataFrame(raw_rows)
    result_rows: List[Dict[str, Any]] = []
    if not raw_df.empty:
        for question_id, sub in raw_df.groupby("question_id"):
            by_orientation = {row["orientation"]: row for _, row in sub.iterrows()}
            original = by_orientation.get("original")
            reversed_row = by_orientation.get("reversed")
            decoded = []
            if original is not None:
                decoded.append(original["decoded_winner"])
            if reversed_row is not None:
                decoded.append(reversed_row["decoded_winner"])
            final, unstable = reconcile_pairwise(decoded)
            original_mapping = map_by_key.get((question_id, "original"), {})
            result_rows.append({
                "question_id": question_id,
                "randomized_orientation": json.dumps({
                    "original_A": original_mapping.get("A_condition"),
                    "original_B": original_mapping.get("B_condition"),
                }, sort_keys=True),
                "hidden_system_mapping": json.dumps({
                    "original": {
                        "A": original_mapping.get("A_condition"),
                        "B": original_mapping.get("B_condition"),
                    },
                    "reversed": {
                        "A": map_by_key.get((question_id, "reversed"), {}).get("A_condition"),
                        "B": map_by_key.get((question_id, "reversed"), {}).get("B_condition"),
                    },
                }, sort_keys=True),
                "judge_model": manifest["model"],
                "raw_first_orientation_judgment": None if original is None else original["raw_winner"],
                "raw_reversed_orientation_judgment": None if reversed_row is None else reversed_row["raw_winner"],
                "decoded_first_orientation_winner": None if original is None else original["decoded_winner"],
                "decoded_reversed_orientation_winner": None if reversed_row is None else reversed_row["decoded_winner"],
                "final_reconciled_outcome": final,
                "position_instability_flag": bool(unstable),
                "evaluation_run_id": manifest.get("batch_id") or Path(raw_path).stem,
                "evaluation_timestamp": utc_now_iso(),
                "prompt_version": manifest.get("prompt_version", PAIRWISE_PROMPT_VERSION),
                "rubric_version": manifest.get("rubric_version", PAIRWISE_RUBRIC_VERSION),
                "context_hash": original_mapping.get("context_hash"),
            })

    run_id = manifest.get("batch_id") or Path(raw_path).stem
    raw_judgments_path = out_dir / f"{run_id}_raw_pairwise_judgments.csv"
    results_path = out_dir / f"{run_id}_pairwise_results.csv"
    summary_path = out_dir / f"{run_id}_pairwise_summary.csv"
    stats_path = out_dir / f"{run_id}_pairwise_statistics.json"
    report_path = out_dir / f"{run_id}_pairwise_report.md"
    if raw_judgments_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {raw_judgments_path}")
    pd.DataFrame(raw_rows).to_csv(raw_judgments_path, index=False)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(results_path, index=False)
    summary = summarize_pairwise_results(results_df, manifest["model"])
    summary.to_csv(summary_path, index=False)
    stats = pairwise_exact_statistics(results_df, manifest["model"])
    stats["actual_usage"] = aggregate_usage(raw_df)
    stats["estimated_batch_cost_usd"] = estimate_pairwise_cost(stats["actual_usage"])
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    report_path.write_text(pairwise_report(summary, stats, manifest["model"]))
    figures = make_pairwise_plots(results_df, summary, out_dir, run_id, force=force)
    norm_summary = {
        "schema_version": "pairwise_judge_normalization_v1",
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path),
        "raw_output_jsonl": raw_path,
        "hidden_mapping": mapping_path,
        "raw_judgments_path": str(raw_judgments_path),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "statistics_path": str(stats_path),
        "report_path": str(report_path),
        "figure_paths": figures,
        "normalized_question_count": int(len(results_df)),
        "raw_judgment_count": int(len(raw_rows)),
        "error_count": len(errors),
        "errors": errors,
    }
    norm_path = out_dir / f"{run_id}_pairwise_normalization_summary.json"
    write_json_no_overwrite(norm_path, norm_summary, force=force)
    return norm_summary


def results_from_raw_judgments(raw_df: pd.DataFrame, run_id: str, judge_model: str) -> pd.DataFrame:
    result_rows: List[Dict[str, Any]] = []
    if raw_df.empty:
        return pd.DataFrame(result_rows)
    for question_id, sub in raw_df.groupby("question_id"):
        by_orientation = {row["orientation"]: row for _, row in sub.iterrows()}
        original = by_orientation.get("original")
        reversed_row = by_orientation.get("reversed")
        decoded = []
        if original is not None:
            decoded.append(original["decoded_winner"])
        if reversed_row is not None:
            decoded.append(reversed_row["decoded_winner"])
        final, unstable = reconcile_pairwise(decoded)
        result_rows.append({
            "question_id": question_id,
            "randomized_orientation": json.dumps({
                "original_A": None if original is None else original["A_condition"],
                "original_B": None if original is None else original["B_condition"],
            }, sort_keys=True),
            "hidden_system_mapping": json.dumps({
                "original": {
                    "A": None if original is None else original["A_condition"],
                    "B": None if original is None else original["B_condition"],
                },
                "reversed": {
                    "A": None if reversed_row is None else reversed_row["A_condition"],
                    "B": None if reversed_row is None else reversed_row["B_condition"],
                },
            }, sort_keys=True),
            "judge_model": judge_model,
            "raw_first_orientation_judgment": None if original is None else original["raw_winner"],
            "raw_reversed_orientation_judgment": None if reversed_row is None else reversed_row["raw_winner"],
            "decoded_first_orientation_winner": None if original is None else original["decoded_winner"],
            "decoded_reversed_orientation_winner": None if reversed_row is None else reversed_row["decoded_winner"],
            "final_reconciled_outcome": final,
            "position_instability_flag": bool(unstable),
            "evaluation_run_id": run_id,
            "evaluation_timestamp": utc_now_iso(),
            "prompt_version": None if original is None else original["prompt_version"],
            "rubric_version": None if original is None else original["rubric_version"],
            "context_hash": None if original is None else original["context_hash"],
        })
    return pd.DataFrame(result_rows)


def combine_pairwise_raw_judgments(raw_paths: Sequence[Path], output_prefix: str, force: bool) -> Dict[str, Any]:
    output_root = Path(load_config("evaluation/configs/paper_experiment.yaml")["experiment"]["output_root"])
    out_dir = pairwise_output_dir(output_root)
    frames = []
    for run_index, path in enumerate(raw_paths):
        frame = pd.read_csv(path)
        frame["source_raw_judgments_path"] = str(path)
        frame["source_run_order"] = run_index
        frames.append(frame)
    all_raw = pd.concat(frames, ignore_index=True)
    all_raw["parse_ok"] = all_raw["decoded_winner"].isin(["QWEN", "MINI", "TIE"])
    chosen_rows = []
    for _, sub in all_raw.sort_values(["source_run_order"]).groupby(["question_id", "orientation"], sort=False):
        ok = sub[sub["parse_ok"]]
        chosen_rows.append((ok.iloc[-1] if not ok.empty else sub.iloc[-1]).to_dict())
    chosen = pd.DataFrame(chosen_rows).sort_values(["question_id", "orientation"]).reset_index(drop=True)
    run_id = output_prefix
    raw_all_path = out_dir / f"{output_prefix}_all_paid_raw_pairwise_judgments.csv"
    chosen_path = out_dir / f"{output_prefix}_chosen_pairwise_judgments.csv"
    results_path = out_dir / f"{output_prefix}_pairwise_results.csv"
    summary_path = out_dir / f"{output_prefix}_pairwise_summary.csv"
    stats_path = out_dir / f"{output_prefix}_pairwise_statistics.json"
    report_path = out_dir / f"{output_prefix}_pairwise_report.md"
    for path in [raw_all_path, chosen_path, results_path, summary_path, stats_path, report_path]:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}")
    all_raw.to_csv(raw_all_path, index=False)
    chosen.to_csv(chosen_path, index=False)
    judge_model = "gpt-5.5"
    results = results_from_raw_judgments(chosen, run_id, judge_model)
    results.to_csv(results_path, index=False)
    summary = summarize_pairwise_results(results, judge_model)
    summary.to_csv(summary_path, index=False)
    stats = pairwise_exact_statistics(results, judge_model)
    stats["actual_usage"] = aggregate_usage(all_raw)
    stats["estimated_batch_cost_usd"] = estimate_pairwise_cost(stats["actual_usage"])
    stats["source_raw_judgment_files"] = [str(path) for path in raw_paths]
    stats["paid_raw_judgment_count"] = int(len(all_raw))
    stats["chosen_raw_judgment_count"] = int(len(chosen))
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    report_path.write_text(pairwise_report(summary, stats, judge_model))
    figures = make_pairwise_plots(results, summary, out_dir, run_id, force=force)
    return {
        "schema_version": "combined_pairwise_judge_analysis_v1",
        "created_at": utc_now_iso(),
        "raw_all_path": str(raw_all_path),
        "chosen_judgments_path": str(chosen_path),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "statistics_path": str(stats_path),
        "report_path": str(report_path),
        "figure_paths": figures,
        "question_count": int(len(results)),
        "paid_raw_judgment_count": int(len(all_raw)),
        "chosen_raw_judgment_count": int(len(chosen)),
    }


def summarize_pairwise_results(results_df: pd.DataFrame, judge_model: str) -> pd.DataFrame:
    total = int(len(results_df))
    counts = Counter(results_df["final_reconciled_outcome"]) if total else Counter()
    stable = results_df[results_df["final_reconciled_outcome"] != "POSITION_UNSTABLE"] if total else results_df
    stable_total = int(len(stable))
    rows = []
    for outcome in ["QWEN_WIN", "MINI_WIN", "TIE", "POSITION_UNSTABLE", "INCOMPLETE_OR_PARSE_ERROR"]:
        count = int(counts.get(outcome, 0))
        rows.append({
            "judge_model": judge_model,
            "outcome": outcome,
            "count": count,
            "percent_of_all_paired_questions": float(count / total * 100) if total else math.nan,
            "stable_denominator": stable_total,
            "percent_excluding_position_unstable": (
                float(count / stable_total * 100)
                if stable_total and outcome != "POSITION_UNSTABLE"
                else math.nan
            ),
            "total_paired_questions": total,
        })
    return pd.DataFrame(rows)


def aggregate_usage(raw_df: pd.DataFrame) -> Dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "requests_with_usage": 0,
    }
    if raw_df.empty or "usage_json" not in raw_df.columns:
        return totals
    for raw in raw_df["usage_json"].dropna():
        try:
            usage = json.loads(raw)
        except Exception:
            continue
        if not isinstance(usage, dict):
            continue
        totals["requests_with_usage"] += 1
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        totals["cached_input_tokens"] += int(input_details.get("cached_tokens") or 0)
        totals["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)
    return totals


def estimate_pairwise_cost(usage: Dict[str, Any]) -> Dict[str, Any]:
    # GPT-5.5 standard pricing is $5/M input and $30/M output; Batch API is 50% off.
    input_cost = usage["input_tokens"] / 1_000_000 * 2.50
    cached_savings_note = "Cached-token discounts are not separately applied here; Batch input pricing is used on total input tokens."
    output_cost = usage["output_tokens"] / 1_000_000 * 15.00
    return {
        "pricing_basis": "GPT-5.5 Batch API estimate: $2.50/M input tokens and $15.00/M output tokens.",
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
        "note": cached_savings_note,
    }


def pairwise_exact_statistics(results_df: pd.DataFrame, judge_model: str) -> Dict[str, Any]:
    qwen = int((results_df["final_reconciled_outcome"] == "QWEN_WIN").sum()) if not results_df.empty else 0
    mini = int((results_df["final_reconciled_outcome"] == "MINI_WIN").sum()) if not results_df.empty else 0
    ties = int((results_df["final_reconciled_outcome"] == "TIE").sum()) if not results_df.empty else 0
    unstable = int((results_df["final_reconciled_outcome"] == "POSITION_UNSTABLE").sum()) if not results_df.empty else 0
    decisive = qwen + mini
    if decisive:
        test = binomtest(qwen, decisive, p=0.5, alternative="two-sided")
        ci = test.proportion_ci(confidence_level=0.95, method="exact")
        p_value = float(test.pvalue)
        qwen_rate = float(qwen / decisive)
        ci_low, ci_high = float(ci.low), float(ci.high)
    else:
        p_value = math.nan
        qwen_rate = math.nan
        ci_low = math.nan
        ci_high = math.nan
    return {
        "judge_model": judge_model,
        "test": "exact binomial test among decisive, non-position-unstable outcomes",
        "qwen_win_count": qwen,
        "gpt_5_4_mini_win_count": mini,
        "tie_count": ties,
        "position_unstable_count": unstable,
        "decisive_comparison_count": decisive,
        "qwen_decisive_win_rate": qwen_rate,
        "qwen_decisive_win_rate_ci95_low": ci_low,
        "qwen_decisive_win_rate_ci95_high": ci_high,
        "raw_p_value": p_value,
        "interpretation_guardrail": "A nonsignificant result must not be interpreted as model equivalence.",
    }


def save_figure(fig, figures_dir: Path, name: str, force: bool) -> List[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "svg", "pdf"]:
        path = figures_dir / f"{name}.{ext}"
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    return paths


def make_pairwise_plots(results_df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, run_id: str, force: bool) -> List[str]:
    if results_df.empty:
        return []
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    figures_dir = out_dir / "figures"
    written: List[str] = []
    order = ["QWEN_WIN", "MINI_WIN", "TIE", "POSITION_UNSTABLE"]
    labels = ["Qwen win", "GPT-5.4 mini win", "Tie", "Position unstable"]
    colors = ["#4C78A8", "#F58518", "#59A14F", "#B279A2"]
    counts = []
    percents = []
    total = float(len(results_df))
    for outcome in order:
        count = int((results_df["final_reconciled_outcome"] == outcome).sum())
        counts.append(count)
        percents.append(count / total * 100 if total else 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(labels, percents, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Questions (%)")
    ax.set_ylim(0, max(100, max(percents) + 10))
    ax.set_title("Blinded Pairwise GraphRAG Outcomes")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    written.extend(save_figure(fig, figures_dir, f"{run_id}_pairwise_outcome_proportions", force))
    plt.close(fig)

    decisive = results_df[results_df["final_reconciled_outcome"].isin(["QWEN_WIN", "MINI_WIN"])]
    if not decisive.empty:
        qwen_count = int((decisive["final_reconciled_outcome"] == "QWEN_WIN").sum())
        mini_count = int((decisive["final_reconciled_outcome"] == "MINI_WIN").sum())
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        vals = np.array([qwen_count, mini_count], dtype=float)
        ax.bar(["Qwen", "GPT-5.4 mini"], vals / vals.sum() * 100, color=["#4C78A8", "#F58518"])
        for idx, count in enumerate([qwen_count, mini_count]):
            ax.text(idx, count / vals.sum() * 100 + 1, str(count), ha="center", va="bottom", fontsize=9)
        ax.axhline(50, color="#374151", linewidth=0.9, linestyle="--")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Decisive comparisons (%)")
        ax.set_title("Decisive Pairwise Win Share")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        fig.tight_layout()
        written.extend(save_figure(fig, figures_dir, f"{run_id}_pairwise_decisive_win_share", force))
        plt.close(fig)
    return written


def pairwise_report(summary: pd.DataFrame, stats: Dict[str, Any], judge_model: str) -> str:
    lines = [
        "# Blinded Pairwise GraphRAG Judge Analysis",
        "",
        "Question: Does direct blinded comparison distinguish Qwen + GraphRAG from GPT-5.4-mini-no-reasoning + GraphRAG despite both having a median absolute score of 5?",
        "",
        "This secondary analysis preserves the existing absolute 1-5 evaluation and uses only saved model responses.",
        "",
        f"Judge model: `{judge_model}`",
        f"Prompt version: `{PAIRWISE_PROMPT_VERSION}`",
        f"Rubric version: `{PAIRWISE_RUBRIC_VERSION}`",
        "",
        "Outcomes:",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {row['outcome']}: {int(row['count'])} ({row['percent_of_all_paired_questions']:.1f}% of all paired questions)")
    lines.extend([
        "",
        "Exact binomial test among decisive non-unstable outcomes:",
        f"- Qwen wins: {stats['qwen_win_count']}",
        f"- GPT-5.4 mini wins: {stats['gpt_5_4_mini_win_count']}",
        f"- Ties: {stats['tie_count']}",
        f"- Position-unstable: {stats['position_unstable_count']}",
        f"- Qwen decisive win rate: {stats['qwen_decisive_win_rate']}",
        f"- 95% CI: [{stats['qwen_decisive_win_rate_ci95_low']}, {stats['qwen_decisive_win_rate_ci95_high']}]",
        f"- Raw p-value: {stats['raw_p_value']}",
        f"- Estimated actual API cost: ${stats.get('estimated_batch_cost_usd', {}).get('total_cost_usd', math.nan):.4f}",
        "",
        "Interpretation should remain conservative: equal medians do not establish equivalence, and this pairwise analysis only assesses residual differences under this benchmark and judge.",
    ])
    return "\n".join(lines)


def compare_second_judge(primary_results: Path, secondary_results: Path, output_path: Path, force: bool = False) -> Dict[str, Any]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    first = pd.read_csv(primary_results)
    second = pd.read_csv(secondary_results)
    merged = first[["question_id", "final_reconciled_outcome"]].merge(
        second[["question_id", "final_reconciled_outcome"]],
        on="question_id",
        suffixes=("_primary", "_secondary"),
    )
    if merged.empty:
        raise ValueError("No overlapping question_id values between pairwise judge result files.")
    labels = sorted(set(merged["final_reconciled_outcome_primary"]) | set(merged["final_reconciled_outcome_secondary"]))
    table = pd.crosstab(merged["final_reconciled_outcome_primary"], merged["final_reconciled_outcome_secondary"]).reindex(index=labels, columns=labels, fill_value=0)
    observed = float((merged["final_reconciled_outcome_primary"] == merged["final_reconciled_outcome_secondary"]).mean())
    row_probs = table.sum(axis=1) / table.values.sum()
    col_probs = table.sum(axis=0) / table.values.sum()
    expected = float(sum(row_probs[label] * col_probs[label] for label in labels))
    kappa = float((observed - expected) / (1 - expected)) if expected < 1 else math.nan
    decisive = merged[
        merged["final_reconciled_outcome_primary"].isin(["QWEN_WIN", "MINI_WIN"])
        & merged["final_reconciled_outcome_secondary"].isin(["QWEN_WIN", "MINI_WIN"])
    ]
    decisive_agreement = float((decisive["final_reconciled_outcome_primary"] == decisive["final_reconciled_outcome_secondary"]).mean()) if not decisive.empty else math.nan
    result = {
        "overlapping_question_count": int(len(merged)),
        "overall_agreement": observed,
        "decisive_winner_agreement": decisive_agreement,
        "cohens_kappa": kappa,
        "disagreement_table": table.to_dict(),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, normalize, and analyze blinded pairwise GraphRAG judge batches.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    prep.add_argument("--limit", type=int)
    prep.add_argument("--question-ids")
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--judge-model")
    prep.add_argument("--reasoning-effort")
    prep.add_argument("--max-output-tokens", type=int)

    norm = sub.add_parser("normalize")
    norm.add_argument("manifest")
    norm.add_argument("--force", action="store_true")

    combine = sub.add_parser("combine")
    combine.add_argument("--raw-judgments", required=True, help="Comma-separated normalized raw pairwise judgment CSV files in priority order.")
    combine.add_argument("--output-prefix", default="final_pairwise_graphrag_qwen_vs_mini")
    combine.add_argument("--force", action="store_true")

    compare = sub.add_parser("compare-second-judge")
    compare.add_argument("primary_results")
    compare.add_argument("secondary_results")
    compare.add_argument("--output", required=True)
    compare.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        config = load_config(args.config)
        manifest = prepare_pairwise_batch(
            config=config,
            limit=args.limit,
            question_ids=parse_csv_arg(args.question_ids),
            force=args.force,
            dry_run=args.dry_run,
            judge_model=args.judge_model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "normalize":
        print(json.dumps(normalize_pairwise_batch(Path(args.manifest), force=args.force), indent=2, sort_keys=True))
    elif args.command == "combine":
        paths = [Path(item.strip()) for item in args.raw_judgments.split(",") if item.strip()]
        print(json.dumps(combine_pairwise_raw_judgments(paths, args.output_prefix, force=args.force), indent=2, sort_keys=True))
    elif args.command == "compare-second-judge":
        print(json.dumps(compare_second_judge(Path(args.primary_results), Path(args.secondary_results), Path(args.output), force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
