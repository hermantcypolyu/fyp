import argparse
from typing import List, Optional

from tallyman.export import format_summary
from tallyman.reader import read_ledger
from tallyman.stats import grand_total, top_category, totals


def main(argv: Optional[List[str]] = None) -> int:
    """Print per-category totals, the grand total and the leading category."""
    parser = argparse.ArgumentParser(prog="tally-report", description="Summarise a ledger.")
    parser.add_argument("ledger", help="path to a category,amount file")
    args = parser.parse_args(argv)

    rows = read_ledger(args.ledger)
    leader = top_category(rows)
    print(format_summary(totals(rows), grand_total(rows), leader or "-"))
    return 0
