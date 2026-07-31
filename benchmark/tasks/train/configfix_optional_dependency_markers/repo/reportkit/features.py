"""The optional-feature gate.

Every optional capability reportkit has is keyed by the name of the extra that installs
it. The names in `SUPPORTED_EXTRAS` are the authority: they are what `available()` and
`require()` accept, and they are what the install hints printed to users spell out, so
they have to be the same strings that appear in `[project.optional-dependencies]`.
"""

from __future__ import annotations

import importlib.util

#: The distribution name users type into pip.
DISTRIBUTION = "reportkit"

#: extra name -> the top-level module that installing that extra provides.
SUPPORTED_EXTRAS: dict[str, str] = {
    "yaml": "yaml",
    "excel": "openpyxl",
    "fast": "orjson",
}


def available(extra: str) -> bool:
    """Whether the module behind `extra` can be imported in this environment."""
    if extra not in SUPPORTED_EXTRAS:
        raise ValueError(f"unknown extra {extra!r}; expected one of {', '.join(sorted(SUPPORTED_EXTRAS))}")
    return importlib.util.find_spec(SUPPORTED_EXTRAS[extra]) is not None


def install_hint(extra: str) -> str:
    """The exact command a user should run to get `extra`."""
    if extra not in SUPPORTED_EXTRAS:
        raise ValueError(f"unknown extra {extra!r}; expected one of {', '.join(sorted(SUPPORTED_EXTRAS))}")
    return f"pip install {DISTRIBUTION}[{extra}]"


def require(extra: str) -> None:
    """Raise unless `extra` is installed, telling the user how to install it."""
    if not available(extra):
        raise ImportError(f"this feature needs the {extra!r} extra - {install_hint(extra)}")
