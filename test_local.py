#!/usr/bin/env python
"""Legacy entrypoint retained only to prevent accidental use.

The paper experiment is now split into separate generation, judging, retrieval,
statistics, table, and figure scripts under evaluation/.
"""

import sys


def main() -> int:
    print(
        "test_local.py is deprecated. Use:\n"
        "  python -m evaluation.run_generations --dry-run\n"
        "  python -m evaluation.run_generations --conditions qwen3_8b_baseline --limit 3\n"
        "  python -m evaluation.judge_answers --dry-run\n"
        "  python -m evaluation.statistical_analysis\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
