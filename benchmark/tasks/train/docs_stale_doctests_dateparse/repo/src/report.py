"""The one-line summary the pipeline emails out at the end of a reporting window.

Nothing in here has examples in its docstrings: these functions need a whole window to
say anything, which is too much setup to read well inline. The primitives they lean on
are documented with examples in :mod:`src.dateparse`.
"""

from __future__ import annotations

from datetime import date

from .dateparse import add_business_days, days_between, format_human, month_bounds, weekday_name

#: Working days the finance team get to sign off a window once it closes.
SIGN_OFF_DAYS = 3


def reporting_window(year: int, month: int) -> tuple[date, date]:
    """The window a monthly report covers: the whole calendar month."""
    return month_bounds(year, month)


def sign_off_deadline(window_end: date) -> date:
    """The working day by which a window that ended on ``window_end`` must be signed off."""
    return add_business_days(window_end, SIGN_OFF_DAYS)


def describe_window(start: date, end: date) -> str:
    """A human sentence naming both ends of a window and how long it ran."""
    span = days_between(start, end) + 1
    return (
        f"{format_human(start)} ({weekday_name(start)}) "
        f"to {format_human(end)} ({weekday_name(end)}): {span} days"
    )
