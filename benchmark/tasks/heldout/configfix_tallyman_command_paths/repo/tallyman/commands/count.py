import argparse
from typing import List, Optional

from tallyman.export import format_totals
from tallyman.filters import apply_filters, by_category, drop_below
from tallyman.reader import read_ledger
from tallyman.stats import totals


def main(argv: Optional[List[str]] = None) -> int:
    """Print the per-category totals of a ledger file."""
    parser = argparse.ArgumentParser(prog="tally", description="Total a ledger by category.")
    parser.add_argument("ledger", help="path to a category,amount file")
    parser.add_argument("--min", type=float, default=None, help="hide categories below this total")
    parser.add_argument("--category", action="append", default=None, help="restrict to a category")
    args = parser.parse_args(argv)

    rows = read_ledger(args.ledger)
    if args.category:
        rows = apply_filters(rows, [by_category(*args.category)])
    summed = totals(rows)
    if args.min is not None:
        summed = drop_below(summed, args.min)
    rendered = format_totals(summed)
    if rendered:
        print(rendered)
    return 0
