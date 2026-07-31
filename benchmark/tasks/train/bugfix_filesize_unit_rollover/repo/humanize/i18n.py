"""Minimal translation shim.

Upstream humanize routes every user-visible string through gettext so the library can be
localised. That machinery is orthogonal to the defect under test and would have added
~140 lines of locale plumbing to this task repo, so it is reduced here to the identity
function it resolves to when no locale has been activated - which is also the path the
upstream tests exercise. The import site in filesize.py is unchanged.
"""

from __future__ import annotations


def _gettext(message: str) -> str:
    """Return `message` unchanged; no locale is ever activated in this reduced repo."""
    return message
