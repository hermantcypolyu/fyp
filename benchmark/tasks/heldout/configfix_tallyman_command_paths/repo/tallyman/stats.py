from typing import Dict, Iterable, Optional

from tallyman.reader import Row


def totals(rows: Iterable[Row]) -> Dict[str, float]:
    """Sum the amounts booked against each category, in first-seen order."""
    out: Dict[str, float] = {}
    for row in rows:
        out[row.category] = out.get(row.category, 0.0) + row.amount
    return out


def grand_total(rows: Iterable[Row]) -> float:
    return sum(row.amount for row in rows)


def top_category(rows: Iterable[Row]) -> Optional[str]:
    """The category with the largest total, or None for an empty ledger.

    Ties are broken by first appearance, so the result is stable.
    """
    summed = totals(rows)
    if not summed:
        return None
    return max(summed, key=lambda key: summed[key])
