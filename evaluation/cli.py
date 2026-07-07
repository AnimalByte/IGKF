from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_CONFIG = "evaluation/configs/paper_experiment.yaml"


def run_module(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.cli",
        description="Publication-oriented CLI wrapper for the IGKF paper experiment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-benchmark", help="Validate data/testset.jsonl and write a report.")
    add_common(validate)
    validate.add_argument("--force", action="store_true")

    dry = sub.add_parser("dry-run", help="Run generation and judging dry-run checks.")
    add_common(dry)

    gen = sub.add_parser("generate-local", help="Run local Qwen generation.")
    add_common(gen)
    gen.add_argument("--conditions", default="qwen3_8b_baseline,qwen3_8b_igkf")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--question-ids")
    gen.add_argument("--force", action="store_true")

    prep_gen = sub.add_parser("prepare-gpt-batch", help="Prepare GPT-5.4 mini Batch generation JSONL.")
    add_common(prep_gen)
    prep_gen.add_argument("--conditions", default="gpt_5_4_mini_baseline,gpt_5_4_mini_igkf")
    prep_gen.add_argument("--limit", type=int)
    prep_gen.add_argument("--question-ids")
    prep_gen.add_argument("--force", action="store_true")

    prep_judge = sub.add_parser("prepare-judge-batch", help="Prepare GPT-5.5 Batch judge JSONL.")
    add_common(prep_judge)
    prep_judge.add_argument("--pass-name", choices=["judge_pass_a_general_quality", "judge_pass_b_igkf_specific"], required=True)
    prep_judge.add_argument("--limit", type=int)
    prep_judge.add_argument("--question-ids")
    prep_judge.add_argument("--force", action="store_true")

    batch = sub.add_parser("batch", help="Submit, refresh, download, or normalize an OpenAI Batch manifest.")
    batch.add_argument("action", choices=["submit", "status", "download", "normalize-generations", "normalize-judges"])
    batch.add_argument("manifest")
    batch.add_argument("--force", action="store_true", help="Only used by normalize actions.")

    analysis = sub.add_parser("analyze", help="Run retrieval status, statistics, tables, figures, and publication manifest.")
    add_common(analysis)
    analysis.add_argument("--skip-retrieval", action="store_true")
    analysis.add_argument("--skip-figures", action="store_true")
    analysis.add_argument("--force-manifest", action="store_true")

    manifest = sub.add_parser("manifest", help="Create publication artifact hash manifest.")
    add_common(manifest)
    manifest.add_argument("--output")
    manifest.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "validate-benchmark":
        forwarded = ["--config", args.config]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.validate_benchmark", forwarded))

    if args.command == "dry-run":
        code = run_module("evaluation.run_generations", ["--config", args.config, "--dry-run"])
        if code:
            raise SystemExit(code)
        raise SystemExit(run_module("evaluation.judge_answers", ["--config", args.config, "--dry-run"]))

    if args.command == "generate-local":
        forwarded = ["--config", args.config, "--conditions", args.conditions]
        if args.limit is not None:
            forwarded += ["--limit", str(args.limit)]
        if args.question_ids:
            forwarded += ["--question-ids", args.question_ids]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.run_generations", forwarded))

    if args.command == "prepare-gpt-batch":
        forwarded = ["prepare-generations", "--config", args.config, "--conditions", args.conditions]
        if args.limit is not None:
            forwarded += ["--limit", str(args.limit)]
        if args.question_ids:
            forwarded += ["--question-ids", args.question_ids]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.openai_batch", forwarded))

    if args.command == "prepare-judge-batch":
        forwarded = ["prepare-judges", "--config", args.config, "--pass-name", args.pass_name]
        if args.limit is not None:
            forwarded += ["--limit", str(args.limit)]
        if args.question_ids:
            forwarded += ["--question-ids", args.question_ids]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.openai_batch", forwarded))

    if args.command == "batch":
        forwarded = [args.action, args.manifest]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.openai_batch", forwarded))

    if args.command == "analyze":
        if not args.skip_retrieval:
            code = run_module("evaluation.evaluate_retrieval", ["--config", args.config])
            if code:
                raise SystemExit(code)
        for module in ["evaluation.statistical_analysis", "evaluation.make_tables"]:
            code = run_module(module, ["--config", args.config])
            if code:
                raise SystemExit(code)
        if not args.skip_figures:
            code = run_module("evaluation.make_figures", ["--config", args.config])
            if code:
                raise SystemExit(code)
        forwarded = ["--config", args.config]
        if args.force_manifest:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.create_publication_manifest", forwarded))

    if args.command == "manifest":
        forwarded = ["--config", args.config]
        if args.output:
            forwarded += ["--output", args.output]
        if args.force:
            forwarded.append("--force")
        raise SystemExit(run_module("evaluation.create_publication_manifest", forwarded))

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
