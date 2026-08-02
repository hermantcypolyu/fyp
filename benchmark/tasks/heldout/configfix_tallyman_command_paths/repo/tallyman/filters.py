"""Predicates for narrowing a ledger before it is totalled."""

from typing import Callable, Dict, Iterable, List, Optional

from tallyman.reader import Row

Predicate = Callable[[Row], bool]


def by_category(*names: str) -> Predicate:
    """Keep only rows booked against one of `names`, compared case-insensitively."""
    wanted = {name.strip().lower() for name in names if name.strip()}

    def predicate(row: Row) -> bool:
        return row.category.lower() in wanted

    return predicate


def by_amount(minimum: Optional[float] = None, maximum: Optional[float] = None) -> Predicate:
    """Keep rows whose amount falls within the inclusive bounds given."""

    def predicate(row: Row) -> bool:
        if minimum is not None and row.amount < minimum:
            return False
        if maximum is not None and row.amount > maximum:
            return False
        return True

    return predicate


def apply_filters(rows: Iterable[Row], predicates: Iterable[Predicate]) -> List[Row]:
    """Keep the rows satisfying every predicate."""
    predicates = list(predicates)
    return [row for row in rows if all(predicate(row) for predicate in predicates)]


def drop_below(summed: Dict[str, float], threshold: float) -> Dict[str, float]:
    """Remove categories whose total falls under `threshold`."""
    return {name: value for name, value in summed.items() if value >= threshold}
