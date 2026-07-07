from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.common import collect_repro_metadata, ensure_dirs, file_sha256, load_config, utc_now_iso, write_json_no_overwrite


INCLUDE_PATTERNS = [
    "metadata/benchmark_validation_report.json",
    "metadata/generation_run_*.json",
    "metadata/smoke_*.json",
    "retrieval_metrics/*.json",
    "statistics/*.csv",
    "statistics/*.json",
    "tables/*.csv",
    "figures/*.png",
    "figures/*.svg",
    "figures/*.pdf",
    "batch_metadata/*_manifest.json",
    "batch_metadata/*_answer_mapping.json",
    "batch_metadata/*_normalization_summary.json",
    "batch_metadata/*_judge_normalization_summary.json",
    "batch_outputs/*_raw.jsonl",
]

SOURCE_PATTERNS = [
    "README.md",
    "REPRODUCING_EXPERIMENT.md",
    "requirements.txt",
    "environment.yml",
    "setup.sh",
    "run_pipeline.sh",
    "clean.sh",
    "evaluation/*.py",
    "evaluation/configs/*.yaml",
    "evaluation/tests/*.py",
]


def collect_files(output_root: Path, patterns: Iterable[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(output_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            records.append({
                "path": str(path),
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return records


def collect_root_files(patterns: Iterable[str]) -> List[Dict[str, Any]]:
    root = Path(".")
    records: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            records.append({
                "path": str(path),
                "relative_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hash manifest for publication experiment artifacts.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--output", default=None, help="Default: evaluation/outputs/metadata/publication_archive_manifest.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)

    manifest = {
        "schema_version": "igkf_publication_manifest_v1",
        "created_at": utc_now_iso(),
        "description": "Hash manifest for benchmark, configuration, OpenAI Batch files, normalized scores, statistics, tables, and figures.",
        "reproducibility_metadata": collect_repro_metadata(config, config_path),
        "source_files": collect_root_files(SOURCE_PATTERNS),
        "artifact_files": collect_files(output_root, INCLUDE_PATTERNS),
        "notes": [
            "Raw generations and retrieval packages are intentionally not enumerated here because they contain one file per question/condition; preserve the full evaluation/outputs directory for complete reproduction.",
            "OpenAI Batch raw JSONL outputs are included by hash so returned model outputs can be audited.",
            "Retrieval metrics should be reported only when independent adjudicated relevance judgments exist.",
        ],
    }

    out = Path(args.output) if args.output else output_root / "metadata" / "publication_archive_manifest.json"
    write_json_no_overwrite(out, manifest, force=args.force)
    print(f"Wrote {out}")
    print(f"Recorded {len(manifest['artifact_files'])} artifact files")


if __name__ == "__main__":
    main()
