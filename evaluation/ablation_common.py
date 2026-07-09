from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluation.common import (
    Question,
    file_sha256,
    filter_questions,
    generation_path,
    load_questions,
    parse_csv_arg,
    read_json,
    read_yaml,
    run_cmd,
    utc_now_iso,
    write_json_no_overwrite,
)


ABLATION_CONDITIONS = (
    "qwen3_8b_baseline",
    "qwen3_8b_graph_only",
    "qwen3_8b_literature_only",
    "qwen3_8b_igkf",
)
NEW_ABLATION_CONDITIONS = ("qwen3_8b_graph_only", "qwen3_8b_literature_only")
CANONICAL_REFERENCE_CONDITIONS = ("qwen3_8b_baseline", "qwen3_8b_igkf")


def load_ablation_config(path: str | Path) -> Dict[str, Any]:
    config = read_yaml(Path(path))
    if tuple(config.get("conditions", [])) != ABLATION_CONDITIONS:
        raise ValueError(f"Ablation config must define exactly {ABLATION_CONDITIONS}")
    if tuple(config.get("new_generation_conditions", [])) != NEW_ABLATION_CONDITIONS:
        raise ValueError("Ablation config new_generation_conditions changed unexpectedly.")
    if tuple(config.get("canonical_reference_conditions", [])) != CANONICAL_REFERENCE_CONDITIONS:
        raise ValueError("Ablation config canonical reference conditions changed unexpectedly.")
    local = config.get("local_model", {})
    required = {
        "model_name": "Qwen3-8B",
        "gguf_filename": "Qwen3-8B-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "context_length": 32768,
        "seed": 20260706,
        "thinking_mode": False,
    }
    for key, expected in required.items():
        if local.get(key) != expected:
            raise ValueError(f"Ablation local_model.{key} must be {expected!r}; found {local.get(key)!r}")
    generation = local.get("generation", {})
    expected_generation = {
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "max_output_tokens": 768,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"Ablation generation.{key} must be {expected!r}; found {generation.get(key)!r}")
    retrieval = config.get("retrieval", {})
    expected_retrieval = {
        "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "initial_k": 15,
        "rerank_top_n": 3,
    }
    for key, expected in expected_retrieval.items():
        if retrieval.get(key) != expected:
            raise ValueError(f"Ablation retrieval.{key} must be {expected!r}; found {retrieval.get(key)!r}")
    return config


def ensure_ablation_dirs(output_root: Path) -> None:
    for rel in ["raw_generations", "retrieval_packages", "judge_inputs", "judge_scores", "statistics", "tables", "figures", "metadata"]:
        (output_root / rel).mkdir(parents=True, exist_ok=True)


def ablation_retrieval_package_path(output_root: Path, condition: str, question_id: str) -> Path:
    return output_root / "retrieval_packages" / condition / f"{question_id}.json"


def canonical_generation_reference(config: Dict[str, Any], condition: str, question_id: str) -> Dict[str, Any]:
    path = generation_path(Path(config["experiment"]["canonical_output_root"]), condition, question_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical generation for {condition}/{question_id}: {path}")
    return {
        "condition": condition,
        "question_id": question_id,
        "path": str(path),
        "sha256": file_sha256(path),
    }


def selected_questions(config: Dict[str, Any], limit: Optional[int], question_ids: Optional[Iterable[str]]) -> List[Question]:
    return filter_questions(load_questions(config["experiment"]["benchmark_path"]), limit, question_ids)


def write_ablation_run_metadata(config: Dict[str, Any], config_path: Path, questions: List[Question], conditions: List[str], dry_run: bool) -> Path:
    output_root = Path(config["experiment"]["output_root"])
    metadata = {
        "schema_version": "qwen_ablation_run_metadata_v1",
        "created_at": utc_now_iso(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "dry_run": dry_run,
        "selected_question_count": len(questions),
        "selected_question_ids": [q.question_id for q in questions],
        "selected_conditions": conditions,
        "benchmark_path": config["experiment"]["benchmark_path"],
        "config_sha256": file_sha256(config_path),
        "benchmark_sha256": file_sha256(Path(config["experiment"]["benchmark_path"])),
        "local_model": config["local_model"],
        "retrieval": config["retrieval"],
        "statistics": config["statistics"],
        "canonical_references": [
            canonical_generation_reference(config, condition, question.question_id)
            for condition in CANONICAL_REFERENCE_CONDITIONS
            for question in questions
        ],
    }
    path = output_root / "metadata" / f"ablation_run_{int(time.time())}.json"
    write_json_no_overwrite(path, metadata, force=False)
    return path


def load_generation_for_condition(config: Dict[str, Any], condition: str, question_id: str) -> Dict[str, Any]:
    if condition in CANONICAL_REFERENCE_CONDITIONS:
        return read_json(Path(canonical_generation_reference(config, condition, question_id)["path"]))
    return read_json(generation_path(Path(config["experiment"]["output_root"]), condition, question_id))
