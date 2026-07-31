"""Minimal translation shim.

Upstream humanize routes every user-visible string through gettext so the library can be
localised, and derives thousands/decimal separators from the active locale. That machinery
is orthogonal to the defect under test and would have added ~140 lines of locale plumbing
to this task repo, so it is reduced here to the behaviour it has when no locale has been
activated - which is also the path the upstream tests exercise. The import sites in
number.py are unchanged.
"""

from __future__ import annotations

__all__ = ["decimal_separator", "thousands_separator"]


def _gettext(message: str) -> str:
    return message


def _pgettext(msgctxt: str, message: str) -> str:
    return message


def _ngettext(message: str, plural: str, num: int) -> str:
    return message if num == 1 else plural


def _gettext_noop(message: str) -> str:
    return message


def _ngettext_noop(singular: str, plural: str) -> tuple[str, str]:
    return singular, plural


def thousands_separator() -> str:
    """Thousands separator for the default (unactivated) locale."""
    return ","


def decimal_separator() -> str:
    """Decimal separator for the default (unactivated) locale."""
    return "."
