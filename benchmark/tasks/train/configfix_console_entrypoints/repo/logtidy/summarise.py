"""The `logtidy-summary` command: how many records at each level.

The counting used to live in a separate ``stats`` module. That module was folded into
this one so the two commands do not disagree about what a record is.
"""

from __future__ import annotations

import argparse
import sys

from .parser import parse_lines
from .report import count_levels, format_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logtidy-summary", description="Count log records by level.")
    parser.add_argument("path", help="log file to read")
    parser.add_argument("--total", action="store_true", help="also print the record total")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `logtidy-summary` console script. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        with open(args.path, encoding="utf-8") as handle:
            records = parse_lines(handle)
    except OSError as exc:
        print(f"logtidy-summary: {exc}", file=sys.stderr)
        return 2

    counts = count_levels(records)
    print(format_counts(counts))
    if args.total:
        print(f"total  {sum(counts.values())}")
    return 0
