"""A small chainable query over a list of dicts, ending in a paginated read."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .pagination import DEFAULT_PER_PAGE, Page, paginate

Record = dict[str, Any]


class Query:
    """Filter, sort and paginate a list of records.

    >>> Query([{"n": 2}, {"n": 1}]).order_by("n").page(1, per_page=1).items
    [{'n': 1}]
    """

    def __init__(self, records: Iterable[Record]) -> None:
        self._records: list[Record] = list(records)

    def where(self, **equals: Any) -> "Query":
        matched = [r for r in self._records if all(r.get(k) == v for k, v in equals.items())]
        return Query(matched)

    def filter(self, predicate: Callable[[Record], bool]) -> "Query":
        return Query([r for r in self._records if predicate(r)])

    def order_by(self, key: str, *, descending: bool = False) -> "Query":
        return Query(sorted(self._records, key=lambda r: r.get(key), reverse=descending))

    def page(self, number: int = 1, per_page: int = DEFAULT_PER_PAGE) -> Page[Record]:
        return paginate(self._records, page=number, per_page=per_page)

    def all(self) -> list[Record]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
