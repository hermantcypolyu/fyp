"""Applying a spec to a listing of paths.

Kept apart from the matcher itself: the matcher answers one question about one path,
and everything here is bookkeeping over a sequence of them.
"""

from __future__ import annotations

from typing import Iterable

from .pathspec import PathSpec


def filter_paths(spec: PathSpec, paths: Iterable[str]) -> list[str]:
    """The paths ``spec`` does not ignore, in the order they were given."""
    return [path for path in paths if not spec.is_ignored(path)]


def partition(spec: PathSpec, paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ``paths`` into (kept, ignored), each in the order they were given."""
    kept: list[str] = []
    ignored: list[str] = []
    for path in paths:
        (ignored if spec.is_ignored(path) else kept).append(path)
    return kept, ignored
