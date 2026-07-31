"""Offset-based pagination.

Pages are 1-indexed: page 1 is the first page. A collection is always considered to
have at least one page, so that an empty result set still renders a (blank) first page
rather than a "page 0 of 0".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_PER_PAGE = 10


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of results, plus enough context to render pagination controls."""

    items: list[T] = field(default_factory=list)
    page: int = 1
    per_page: int = DEFAULT_PER_PAGE
    total_items: int = 0
    total_pages: int = 1

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def start_index(self) -> int:
        """1-based index of the first item on this page, or 0 when the page is empty."""
        return (self.page - 1) * self.per_page + 1 if self.items else 0

    @property
    def end_index(self) -> int:
        """1-based index of the last item on this page, or 0 when the page is empty."""
        return self.start_index + len(self.items) - 1 if self.items else 0

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def total_pages(total_items: int, per_page: int = DEFAULT_PER_PAGE) -> int:
    """How many pages `total_items` records occupy at `per_page` records per page."""
    if per_page < 1:
        raise ValueError("per_page must be >= 1")
    if total_items < 0:
        raise ValueError("total_items must be >= 0")
    return total_items // per_page


def paginate(
    items: Sequence[T],
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> Page[T]:
    """Slice `items` into the requested page.

    Requesting a page past the end yields an empty page rather than raising, so that a
    stale bookmark degrades gracefully instead of returning a 500.
    """
    if page < 1:
        raise ValueError("page must be >= 1")

    count = len(items)
    pages = total_pages(count, per_page)

    if page > pages:
        return Page(items=[], page=page, per_page=per_page, total_items=count, total_pages=pages)

    start = (page - 1) * per_page
    end = start + per_page
    return Page(
        items=list(items[start:end]),
        page=page,
        per_page=per_page,
        total_items=count,
        total_pages=pages,
    )
