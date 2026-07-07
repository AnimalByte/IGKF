from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


FOUR_CONDITIONS = (
    "qwen3_8b_baseline",
    "qwen3_8b_igkf",
    "gpt_5_4_mini_baseline",
    "gpt_5_4_mini_igkf",
)

REQUIRED_BENCHMARK_FIELDS = {
    "question_id",
    "question",
    "category",
    "answerability",
    "expected_behavior",
    "manual_review_required",
}


@dataclass(frozen=True)
class Question:
    question_id: str
    question_text: str
    source_index: int
    category: str
    answerability: str
    expected_behavior: str
    manual_review_required: bool

    @property
    def annotations(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "answerability": self.answerability,
            "expected_behavior": self.expected_behavior,
            "manual_review_required": self.manual_review_required,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> Dict[str, Any]:
    config = read_yaml(Path(path))
    conditions = tuple(config.get("conditions", []))
    if conditions != FOUR_CONDITIONS:
        raise ValueError(f"Config must define exactly {FOUR_CONDITIONS}; found {conditions}")
    local = config.get("local_model", {})
    if local.get("model_name") != "Qwen3-8B":
        raise ValueError("Qwen3-8B must be the only configured local generation model.")
    if local.get("thinking_mode") is not False:
        raise ValueError("Qwen3 thinking_mode must be explicitly false.")
    openai_cfg = config.get("openai", {})
    if openai_cfg.get("generator_model") != "gpt-5.4-mini-2026-03-17":
        raise ValueError("OpenAI generator model must be gpt-5.4-mini-2026-03-17.")
    if openai_cfg.get("generator_reasoning_effort") != "none":
        raise ValueError("GPT-5.4 mini generation must use reasoning effort none.")
    if openai_cfg.get("judge_model") != "gpt-5.5":
        raise ValueError("Judge model must be gpt-5.5.")
    if openai_cfg.get("judge_reasoning_effort") != "high":
        raise ValueError("Judge reasoning effort must be high.")
    if "Phi" in json.dumps(config, sort_keys=True):
        raise ValueError("Phi-specific configuration is not allowed in the paper experiment config.")
    return config


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_questions(path: str | Path) -> List[Question]:
    rows: List[Question] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    seen_normalized_text: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Blank line in benchmark at line {idx}")
            obj = json.loads(line)
            missing = REQUIRED_BENCHMARK_FIELDS - set(obj.keys())
            if missing:
                raise ValueError(f"Malformed benchmark schema at line {idx}; missing {sorted(missing)}")
            question_id = obj["question_id"]
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"Blank question_id at line {idx}")
            question_id = question_id.strip()
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question_id at line {idx}: {question_id}")
            seen_ids.add(question_id)
            text = obj["question"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Blank question at line {idx}")
            text = text.strip()
            if text in seen_text:
                raise ValueError(f"Duplicate question text at line {idx}: {text}")
            normalized = normalize_question_text(text)
            if normalized in seen_normalized_text:
                raise ValueError(f"Duplicate normalized question text at line {idx}: {text}")
            seen_text.add(text)
            seen_normalized_text.add(normalized)
            for field in ["category", "answerability", "expected_behavior"]:
                if not isinstance(obj[field], str) or not obj[field].strip():
                    raise ValueError(f"Unreadable {field} at line {idx}")
            if not isinstance(obj["manual_review_required"], bool):
                raise ValueError(f"manual_review_required must be boolean at line {idx}")
            rows.append(Question(
                question_id=question_id,
                question_text=text,
                source_index=idx,
                category=obj["category"].strip(),
                answerability=obj["answerability"].strip(),
                expected_behavior=obj["expected_behavior"].strip(),
                manual_review_required=obj["manual_review_required"],
            ))
    if len(rows) != 205:
        raise ValueError(f"Expected exactly 205 benchmark questions, found {len(rows)}")
    return rows


def validate_benchmark(path: str | Path) -> Dict[str, Any]:
    from collections import Counter

    benchmark_path = Path(path)
    questions = load_questions(benchmark_path)
    return {
        "benchmark_path": str(benchmark_path),
        "sha256": file_sha256(benchmark_path),
        "total_records": len(questions),
        "unique_question_ids": len({q.question_id for q in questions}),
        "duplicate_question_id_count": len(questions) - len({q.question_id for q in questions}),
        "duplicate_exact_question_count": len(questions) - len({q.question_text for q in questions}),
        "duplicate_normalized_question_count": len(questions) - len({normalize_question_text(q.question_text) for q in questions}),
        "category_counts": dict(Counter(q.category for q in questions)),
        "answerability_counts": dict(Counter(q.answerability for q in questions)),
        "expected_behavior_counts": dict(Counter(q.expected_behavior for q in questions)),
        "manual_review_count": sum(1 for q in questions if q.manual_review_required),
        "first_question_id": questions[0].question_id,
        "last_question_id": questions[-1].question_id,
    }


def filter_questions(questions: List[Question], limit: Optional[int], question_ids: Optional[Iterable[str]]) -> List[Question]:
    selected = questions
    if question_ids:
        wanted = set(question_ids)
        missing = wanted - {q.question_id for q in questions}
        if missing:
            raise ValueError(f"Unknown question IDs: {sorted(missing)}")
        selected = [q for q in selected if q.question_id in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_conditions(conditions: Optional[Iterable[str]]) -> List[str]:
    if not conditions:
        return list(FOUR_CONDITIONS)
    invalid = set(conditions) - set(FOUR_CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown conditions: {sorted(invalid)}")
    return list(conditions)


def ensure_dirs(output_root: Path) -> None:
    for rel in [
        "raw_generations",
        "retrieval_packages",
        "batch_inputs",
        "batch_metadata",
        "batch_outputs",
        "judge_scores",
        "retrieval_metrics",
        "statistics",
        "tables",
        "figures",
        "metadata",
    ]:
        (output_root / rel).mkdir(parents=True, exist_ok=True)


def generation_path(output_root: Path, condition: str, question_id: str) -> Path:
    return output_root / "raw_generations" / condition / f"{question_id}.json"


def judge_path(output_root: Path, condition: str, question_id: str) -> Path:
    return output_root / "judge_scores" / condition / f"{question_id}.json"


def retrieval_package_path(output_root: Path, question_id: str) -> Path:
    return output_root / "retrieval_packages" / f"{question_id}.json"


def write_json_no_overwrite(path: Path, payload: Dict[str, Any], force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None


def collect_repro_metadata(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    benchmark_path = Path(config["experiment"]["benchmark_path"])
    package_versions = {}
    for name in ["llama_cpp_python", "transformers", "torch", "chromadb", "ragas", "openai", "spacy", "scispacy"]:
        out = run_cmd([sys.executable, "-m", "pip", "show", name])
        if out:
            for line in out.splitlines():
                if line.startswith("Version:"):
                    package_versions[name] = line.split(":", 1)[1].strip()
                    break
    cuda = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    ram = run_cmd(["bash", "-lc", "free -h | awk '/Mem:/ {print $2}'"])
    return {
        "created_at": utc_now_iso(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_gpu": cuda,
        "system_ram": ram,
        "package_versions": package_versions,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_sha256(benchmark_path) if benchmark_path.exists() else None,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "local_model": config["local_model"],
        "openai": config["openai"],
        "retrieval": config["retrieval"],
        "statistics": config["statistics"],
    }


def condition_uses_openai(condition: str) -> bool:
    return condition.startswith("gpt_5_4_mini")


def condition_uses_igkf(condition: str) -> bool:
    return condition.endswith("_igkf")


def condition_uses_qwen(condition: str) -> bool:
    return condition.startswith("qwen3_8b")
