#!/usr/bin/env python3
"""Legacy RAGAS entry point.

The paper experiment now uses the canonical scripts under evaluation/. This
legacy script is intentionally disabled because its historical "faithfulness"
metric mixed context-grounded RAG evaluation with baseline answer assessment.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "0_ragas_evaluation.py is deprecated for the paper experiment.\n"
        "Use these canonical entry points instead:\n"
        "  python -m evaluation.run_generations --dry-run\n"
        "  python -m evaluation.judge_answers --dry-run\n"
        "  python -m evaluation.statistical_analysis\n"
        "  python -m evaluation.evaluate_retrieval\n"
        "  python -m evaluation.make_tables\n"
        "  python -m evaluation.make_figures\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
