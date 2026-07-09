from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.ablation_common import (
    NEW_ABLATION_CONDITIONS,
    ablation_retrieval_package_path,
    canonical_generation_reference,
    ensure_ablation_dirs,
    load_ablation_config,
    selected_questions,
    write_ablation_run_metadata,
)
from evaluation.common import file_sha256, generation_path, parse_csv_arg, run_cmd, utc_now_iso, write_json_no_overwrite
from evaluation.qwen_generation import Qwen3LocalGenerator


def get_engine():
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


def build_package(condition: str, question_id: str, question_text: str, engine: Any) -> Dict[str, Any]:
    start = time.perf_counter()
    if condition == "qwen3_8b_graph_only":
        user_prompt, graph_context, vector_context, metadata = engine.prepare_graph_only_prompt(question_text)
    elif condition == "qwen3_8b_literature_only":
        user_prompt, graph_context, vector_context, metadata = engine.prepare_literature_only_prompt(question_text)
    else:
        raise ValueError(f"Unsupported ablation generation condition: {condition}")
    return {
        "schema_version": "qwen_ablation_retrieval_package_v1",
        "question_id": question_id,
        "question_text": question_text,
        "condition": condition,
        "retrieval_timestamp": utc_now_iso(),
        "retrieval_mode": metadata.get("retrieval_mode"),
        "graph_context": graph_context,
        "vector_context": vector_context,
        "user_prompt": user_prompt,
        "retrieval_metadata": metadata,
        "retrieval_seconds": round(time.perf_counter() - start, 4),
    }


def assert_package_isolated(condition: str, package: Dict[str, Any], question_text: str) -> None:
    metadata = package.get("retrieval_metadata") or {}
    prompt = package.get("user_prompt") or ""
    if condition == "qwen3_8b_graph_only":
        if package.get("vector_context"):
            raise AssertionError("graph-only package has non-empty vector_context")
        if metadata.get("candidate_pmids") or metadata.get("selected_pmids") or metadata.get("selected_abstracts"):
            raise AssertionError("graph-only package contains literature PMID/abstract metadata")
        if "--- Abstract" in prompt or "PMID:" in prompt:
            raise AssertionError("graph-only prompt contains abstract markers")
    elif condition == "qwen3_8b_literature_only":
        if package.get("graph_context"):
            raise AssertionError("literature-only package has non-empty graph_context")
        if metadata.get("graph_context"):
            raise AssertionError("literature-only metadata contains graph_context")
        if metadata.get("expanded_retrieval_query") != question_text:
            raise AssertionError("literature-only expanded_retrieval_query must equal original question")
        if metadata.get("candidate_count", 0) > 15:
            raise AssertionError("literature-only candidate_count exceeds 15")
        if metadata.get("reranked_count", 0) > 3:
            raise AssertionError("literature-only reranked_count exceeds 3")
        if "**Graph Context" in prompt or "Graph Context:" in prompt:
            raise AssertionError("literature-only prompt contains graph context marker")


def system_prompt_for(condition: str, config: Dict[str, Any]) -> str:
    if condition == "qwen3_8b_graph_only":
        return config["prompts"]["graph_only_system_prompt"]
    if condition == "qwen3_8b_literature_only":
        return config["prompts"]["literature_only_system_prompt"]
    raise ValueError(condition)


def result_payload(
    *,
    config: Dict[str, Any],
    config_path: Path,
    condition: str,
    question_id: str,
    question_text: str,
    annotations: Dict[str, Any],
    package: Dict[str, Any],
    generated: Dict[str, Any],
    qwen: Qwen3LocalGenerator,
    generation_seconds: float,
) -> Dict[str, Any]:
    metadata = package.get("retrieval_metadata") or {}
    return {
        "schema_version": "qwen_ablation_generation_v1",
        "question_id": question_id,
        "question_text": question_text,
        "benchmark_annotations": annotations,
        "condition": condition,
        "model_name": config["local_model"]["model_name"],
        "timestamp": utc_now_iso(),
        "generation_parameters": config["local_model"],
        "system_prompt": system_prompt_for(condition, config),
        "user_prompt": package["user_prompt"],
        "raw_model_output": generated["raw_model_output"],
        "parsed_answer": generated["parsed_answer"],
        "generation_seconds": round(generation_seconds, 4),
        "retrieval_mode": metadata.get("retrieval_mode"),
        "graph_context": package.get("graph_context", ""),
        "vector_context": package.get("vector_context", ""),
        "retrieval_metadata": metadata,
        "candidate_pmids": metadata.get("candidate_pmids", []),
        "selected_pmids": metadata.get("selected_pmids", []),
        "reranker_scores": metadata.get("scores", []),
        "original_retrieval_query": metadata.get("original_retrieval_query", question_text),
        "expanded_retrieval_query": metadata.get("expanded_retrieval_query"),
        "retrieval_package_path": str(ablation_retrieval_package_path(Path(config["experiment"]["output_root"]), condition, question_id)),
        "canonical_reference_paths": [
            canonical_generation_reference(config, "qwen3_8b_baseline", question_id),
            canonical_generation_reference(config, "qwen3_8b_igkf", question_id),
        ],
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "config_hash": file_sha256(config_path),
        "benchmark_hash": file_sha256(Path(config["experiment"]["benchmark_path"])),
        "qwen_metadata": qwen.metadata() | {
            "prompt_token_count": generated["prompt_token_count"],
            "output_token_count": generated["output_token_count"],
            "context_length": generated["context_length"],
            "requested_output_tokens": generated["requested_output_tokens"],
        },
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-Qwen-only IGKF component ablations.")
    parser.add_argument("--config", default="evaluation/configs/qwen_ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-ids")
    parser.add_argument("--conditions", default=",".join(NEW_ABLATION_CONDITIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_ablation_config(config_path)
    output_root = Path(config["experiment"]["output_root"])
    ensure_ablation_dirs(output_root)
    questions = selected_questions(config, args.limit, parse_csv_arg(args.question_ids))
    conditions = parse_csv_arg(args.conditions) or list(NEW_ABLATION_CONDITIONS)
    invalid = set(conditions) - set(NEW_ABLATION_CONDITIONS)
    if invalid:
        raise ValueError(f"This runner only generates new ablation conditions; invalid: {sorted(invalid)}")

    qwen_validation = Qwen3LocalGenerator(config["local_model"], load_model=False)
    metadata_path = write_ablation_run_metadata(config, config_path, questions, conditions, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run OK: {len(questions)} questions x {len(conditions)} new ablation conditions")
        print(f"Qwen non-thinking mechanism: {qwen_validation.thinking_disable_mechanism}")
        print(f"Metadata: {metadata_path}")
        return

    qwen = Qwen3LocalGenerator(config["local_model"], load_model=True)
    engine = get_engine()
    try:
        for condition in conditions:
            for question in questions:
                out_path = generation_path(output_root, condition, question.question_id)
                if out_path.exists() and not args.force:
                    print(f"skip existing {condition}/{question.question_id}")
                    continue
                package_path = ablation_retrieval_package_path(output_root, condition, question.question_id)
                try:
                    if package_path.exists() and not args.force:
                        package = __import__("evaluation.common", fromlist=["read_json"]).read_json(package_path)
                    else:
                        package = build_package(condition, question.question_id, question.question_text, engine)
                        assert_package_isolated(condition, package, question.question_text)
                        write_json_no_overwrite(package_path, package, force=args.force)
                    start = time.perf_counter()
                    generated = qwen.generate(system_prompt_for(condition, config), package["user_prompt"])
                    payload = result_payload(
                        config=config,
                        config_path=config_path,
                        condition=condition,
                        question_id=question.question_id,
                        question_text=question.question_text,
                        annotations=question.annotations,
                        package=package,
                        generated=generated,
                        qwen=qwen,
                        generation_seconds=time.perf_counter() - start,
                    )
                except Exception as exc:
                    payload = {
                        "schema_version": "qwen_ablation_generation_v1",
                        "question_id": question.question_id,
                        "question_text": question.question_text,
                        "benchmark_annotations": question.annotations,
                        "condition": condition,
                        "model_name": config["local_model"]["model_name"],
                        "timestamp": utc_now_iso(),
                        "generation_parameters": config["local_model"],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                write_json_no_overwrite(out_path, payload, force=args.force)
                print(f"wrote {out_path}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
