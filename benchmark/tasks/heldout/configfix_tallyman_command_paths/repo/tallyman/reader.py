from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class Row:
    """One ledger entry: a category and the amount booked against it."""

    category: str
    amount: float


def parse_line(line: str) -> Optional[Row]:
    """Parse one `category,amount` line.

    Blank lines, comments and malformed rows are skipped by returning None, so a
    ledger with a banner or a stray note still loads.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = text.split(",")
    if len(parts) != 2:
        return None
    category = parts[0].strip()
    if not category:
        return None
    try:
        amount = float(parts[1].strip())
    except ValueError:
        return None
    return Row(category=category, amount=amount)


def parse_lines(lines: Iterable[str]) -> Iterator[Row]:
    for line in lines:
        row = parse_line(line)
        if row is not None:
            yield row


def read_ledger(path: str) -> List[Row]:
    with open(path, encoding="utf-8") as handle:
        return list(parse_lines(handle))
