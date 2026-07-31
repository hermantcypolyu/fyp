"""Date helpers for the weekly reporting pipeline.

Everything here works in plain :class:`datetime.date` values, and nothing reads the
system clock: a caller that needs "today" passes it in. That is what lets a report be
re-run against an archive years later and come out identical.

    >>> parse_iso_date("2024-03-05")
    '2024-03-05'
    >>> format_human(parse_iso_date("2024-03-05"))
    'March 5, 2024'
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

__all__ = [
    "DateParseError",
    "add_business_days",
    "days_between",
    "format_human",
    "month_bounds",
    "parse_iso_date",
    "weekday_name",
]

ISO_FORMAT = "%Y-%m-%d"

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class DateParseError(ValueError):
    """Raised when a string is not a date this module knows how to read."""


def parse_iso_date(text: str) -> date:
    """Read ``YYYY-MM-DD`` and return the day it names.

    Surrounding whitespace is stripped first, so a value pasted out of a spreadsheet
    parses without the caller having to tidy it up:

        >>> parse_iso_date("2024-03-05")
        '2024-03-05'
        >>> parse_iso_date(" 2024-03-05 ")
        Traceback (most recent call last):
        ValueError: could not parse ' 2024-03-05 '

    Anything that is not an ISO-8601 day raises :class:`DateParseError`.
    """
    try:
        return datetime.strptime(text.strip(), ISO_FORMAT).date()
    except ValueError as exc:
        raise DateParseError(f"not an ISO-8601 date: {text!r}") from exc


def format_human(value: date) -> str:
    """Render ``value`` the way the report body writes dates.

        >>> format_human(date(2024, 3, 5))
        'March 5, 2024'
        >>> format_human(date(2024, 12, 25))
        'December 25, 2024'
    """
    return f"{value.day} {MONTH_NAMES[value.month - 1]} {value.year}"


def days_between(start: date, end: date) -> int:
    """Count whole days from ``start`` to ``end``, negative if ``end`` came first.

        >>> days_between(date(2024, 3, 1), date(2024, 3, 10))
        datetime.timedelta(days=9)
        >>> days_between(date(2024, 3, 10), date(2024, 3, 1))
        datetime.timedelta(days=-9)
    """
    return (end - start).days


def add_business_days(start: date, count: int) -> date:
    """Move ``count`` working days forward from ``start``, skipping weekends.

    ``start`` itself is never counted, so a single working day after a Friday lands on
    the following Monday:

        >>> add_business_days(date(2024, 3, 7), 2)
        datetime.date(2024, 3, 9)
        >>> add_business_days(date(2024, 3, 8), 1)
        datetime.date(2024, 3, 9)
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    current = start
    remaining = count
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """Return the first and the last day of a calendar month.

        >>> month_bounds(2024, 2)
        [datetime.date(2024, 2, 1), datetime.date(2024, 2, 28)]
        >>> month_bounds(2023, 2)
        [datetime.date(2023, 2, 1), datetime.date(2023, 2, 28)]
    """
    first = date(year, month, 1)
    following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first, following - timedelta(days=1)


def weekday_name(value: date) -> str:
    """Name the day of the week ``value`` falls on.

        >>> weekday_name(date(2024, 3, 5))
        'Tuesday'
        >>> weekday_name(date(2024, 3, 9))
        'Saturday'
    """
    return DAY_NAMES[value.weekday()]
