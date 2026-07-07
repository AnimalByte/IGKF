from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.common import ensure_dirs, load_config, validate_benchmark, write_json_no_overwrite


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the annotated IGKF benchmark.")
    parser.add_argument("--config", default="evaluation/configs/paper_experiment.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["experiment"]["output_root"])
    ensure_dirs(output_root)
    report = validate_benchmark(config["experiment"]["benchmark_path"])
    out = output_root / "metadata" / "benchmark_validation_report.json"
    write_json_no_overwrite(out, report, force=args.force)
    print(f"Benchmark validation OK: {report['total_records']} records, {report['unique_question_ids']} unique IDs")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
