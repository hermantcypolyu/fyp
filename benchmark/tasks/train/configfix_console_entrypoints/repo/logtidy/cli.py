"""The `logtidy` command: filter a log file by level and re-render it canonically.

This module used to be called ``console`` and its entry function used to be called
``main``; both were renamed when the ``logtidy-summary`` command was split out, so that
each command's entry function is named after the command it serves.
"""

from __future__ import annotations

import argparse
import sys

from .parser import LEVELS, at_least, parse_lines
from .report import tidy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logtidy", description="Filter and re-render a log file.")
    parser.add_argument("path", help="log file to read")
    parser.add_argument(
        "--min-level",
        default="INFO",
        choices=LEVELS,
        help="drop records below this level (default: INFO)",
    )
    parser.add_argument("--output", help="write here instead of stdout")
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Entry point for the `logtidy` console script. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        with open(args.path, encoding="utf-8") as handle:
            records = parse_lines(handle)
    except OSError as exc:
        print(f"logtidy: {exc}", file=sys.stderr)
        return 2

    lines = tidy(records, at_least(args.min_level))
    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n" if text else "")
    else:
        print(text)
    return 0
