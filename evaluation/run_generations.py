from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.common import (
    collect_repro_metadata,
    condition_uses_igkf,
    condition_uses_openai,
    condition_uses_qwen,
    ensure_dirs,
    filter_questions,
    generation_path,
    load_config,
    load_questions,
    parse_csv_arg,
    read_json,
    retrieval_package_path,
    utc_now_iso,
    validate_conditions,
    write_json_no_overwrite,
)
from evaluation.openai_client import OpenAIGenerator
from evaluation.qwen_generation import Qwen3LocalGenerator


def build_baseline_user_prompt(question_text: str) -> str:
    return question_text


def result_skeleton(question_id: str, question_text: str, condition: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "paper_generation_v1",
        "question_id": question_id,
        "question_text": question_text,
        "benchmark_annotations": None,
        "condition": condition,
        "model_name": model_name,
        "timestamp": utc_now_iso(),
        "generation_parameters": config,
        "system_prompt": None,
        "user_prompt": None,
        "raw_model_output": None,
        "parsed_answer": None,
        "error": None,
        "retrieval_metadata": None,
    }


def get_igkf_engine():
    from query_engine import GraphRAGQueryEngine

    engine = GraphRAGQueryEngine(llm=None)
    preflight = engine.validate_experiment_ready()
    print(
        "IGKF preflight OK: "
        f"{len(preflight['present_expected_labels'])} expected labels, "
        f"{preflight['chroma_collection_count']} Chroma records, "
        f"retrieval_device={preflight['retrieval_model_device']}"
    )
    return engine


def build_retrieval_package(question_id: str, question_text: str, igkf_engine: Any) -> Dict[str, Any]:
    user_prompt, graph_context, vector_context, retrieval_metadata = igkf_engine.prepare_igkf_prompt(question_text)
    return {
        "schema_version": "igkf_retrieval_package_v1",
        "question_id": question_id,
        "question_text": question_text,
        "retrieval_timestamp": utc_now_iso(),
        "graph_context": graph_context,
        "vector_context": vector_context,
        "user_prompt": user_prompt,
        "final_context": f"Graph Context:\n{graph_context}\n\nVector Context:\n{vector_context}",
        "retrieval_metadata": retrieval_metadata,
    }


def get_or_create_retrieval_package(
    *,
    output_root: Path,
    question_id: str,
    question_text: str,
    igkf_engine: Any,
    force: bool,
) -> Dict[str, Any]:
    path = retrieval_package_path(output_root, question_id)
    if path.exists() and not force:
        return read_json(path)
    if igkf_engine is None:
        raise RuntimeError(f"IGKF retrieval package missing for {question_id}, and no IGKF engine is available.")
    package = build_retrieval_package(question_id, question_text, igkf_engine)
    write_json_no_overwrite(path, package, force=force)
    return package


def run_generation(
    *,
    condition: str,
    question_id: str,
    question_text: str,
    config: Dict[str, Any],
    qwen: Optional[Qwen3LocalGenerator],
    openai_generator: Optional[OpenAIGenerator],
    igkf_engine: Any,
    output_root: Path,
    annotations: Dict[str, Any],
    force_retrieval: bool = False,
) -> Dict[str, Any]:
    uses_igkf = condition_uses_igkf(condition)
    uses_qwen = condition_uses_qwen(condition)
    local_cfg = config["local_model"]
    system_prompt = config["prompts"]["igkf_system_prompt"] if uses_igkf else config["prompts"]["baseline_system_prompt"]

    if uses_igkf:
        package = get_or_create_retrieval_package(
            output_root=output_root,
            question_id=question_id,
            question_text=question_text,
            igkf_engine=igkf_engine,
            force=force_retrieval,
        )
        user_prompt = package["user_prompt"]
        retrieval_metadata = package.get("retrieval_metadata", {})
        retrieval_metadata["retrieval_package_path"] = str(retrieval_package_path(output_root, question_id))
    else:
        user_prompt = build_baseline_user_prompt(question_text)
        retrieval_metadata = None

    model_name = local_cfg["model_name"] if uses_qwen else config["openai"]["generator_model"]
    generation_cfg = local_cfg if uses_qwen else config["openai"]
    payload = result_skeleton(question_id, question_text, condition, model_name, generation_cfg)
    payload["benchmark_annotations"] = annotations
    payload["system_prompt"] = system_prompt
    payload["user_prompt"] = user_prompt
    payload["retrieval_metadata"] = retrieval_metadata

    start = time.perf_counter()
    if uses_qwen:
        if qwen is None:
            raise RuntimeError("Qwen generator is required for Qwen conditions.")
        generated = qwen.generate(system_prompt, user_prompt)
        payload.update({
            "raw_model_output": generated["raw_model_output"],
            "parsed_answer": generated["parsed_answer"],
            "qwen_metadata": qwen.metadata() | {
                "prompt_token_count": generated["prompt_token_count"],
                "output_token_count": generated["output_token_count"],
                "context_length": generated["context_length"],
                "requested_output_tokens": generated["requested_output_tokens"],
            },
        })
    else:
        if openai_generator is None:
            raise RuntimeError("OpenAI generator is required for GPT-5.4 mini conditions.")
        generated = openai_generator.generate(system_prompt, user_prompt)
        payload.update(generated)
    payload["generation_seconds"] = round(time.perf_counter() - start, 4)
    return payload


def write_dry_run_metadata(config: Dict[str, Any], config_path: Path, output_root: Path, selected_count: int, conditions: list[str]) -> None:
    metadata = collect_repro_metadata(config, config_path)
    metadata.update({
        "dry_run": True,
        "selected_question_count": selected_count,
        "selected_conditions": conditions,
    })
    write_json_no_overwrite(output_root / "metadata" / f"dry_run_{int(time.time())}.json", metadata, force=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical IGKF paper answer generation.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-ids", help="Comma-separated stable IDs, e.g. q001,q002")
    parser.add_argument("--conditions", help="Comma-separated condition names")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-question outputs.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)

    all_questions = load_questions(config["experiment"]["benchmark_path"])
    questions = filter_questions(all_questions, args.limit, parse_csv_arg(args.question_ids))
    conditions = validate_conditions(parse_csv_arg(args.conditions))

    # Validate Qwen tokenizer/non-thinking configuration even in dry-run mode.
    qwen_for_validation = Qwen3LocalGenerator(config["local_model"], load_model=False)

    if args.dry_run:
        write_dry_run_metadata(config, config_path, output_root, len(questions), conditions)
        print(f"Dry run OK: {len(questions)} questions x {len(conditions)} conditions")
        print(f"Qwen non-thinking mechanism: {qwen_for_validation.thinking_disable_mechanism}")
        return

    qwen: Optional[Qwen3LocalGenerator] = None
    if any(condition_uses_qwen(c) for c in conditions):
        qwen = Qwen3LocalGenerator(config["local_model"], load_model=True)

    openai_generator: Optional[OpenAIGenerator] = None
    if any(condition_uses_openai(c) for c in conditions):
        openai_generator = OpenAIGenerator(
            config["openai"]["generator_model"],
            int(config["openai"]["max_output_tokens"]),
        )

    igkf_engine = None
    if any(condition_uses_igkf(c) for c in conditions):
        igkf_engine = get_igkf_engine()

    metadata = collect_repro_metadata(config, config_path)
    metadata.update({"dry_run": False, "selected_conditions": conditions})
    write_json_no_overwrite(output_root / "metadata" / f"generation_run_{int(time.time())}.json", metadata, force=False)

    try:
        for condition in conditions:
            for question in questions:
                out_path = generation_path(output_root, condition, question.question_id)
                if out_path.exists() and not args.force:
                    print(f"skip existing {condition}/{question.question_id}")
                    continue
                try:
                    payload = run_generation(
                        condition=condition,
                        question_id=question.question_id,
                        question_text=question.question_text,
                        config=config,
                        qwen=qwen,
                        openai_generator=openai_generator,
                        igkf_engine=igkf_engine,
                        output_root=output_root,
                        annotations=question.annotations,
                        force_retrieval=args.force,
                    )
                except Exception as exc:
                    model_name = config["local_model"]["model_name"] if condition_uses_qwen(condition) else config["openai"]["generator_model"]
                    payload = result_skeleton(question.question_id, question.question_text, condition, model_name, {})
                    payload["benchmark_annotations"] = question.annotations
                    payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
                write_json_no_overwrite(out_path, payload, force=args.force)
                print(f"wrote {out_path}")
    finally:
        if igkf_engine is not None:
            igkf_engine.close()


if __name__ == "__main__":
    main()
