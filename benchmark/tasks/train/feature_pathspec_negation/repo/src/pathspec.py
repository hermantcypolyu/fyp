"""A gitignore-style path matcher.

Patterns are matched against repo-relative POSIX path strings, never against the
filesystem, so a spec can be applied to a listing that came out of a tarball, a git tree
or a fixture without touching a disk.

:mod:`fnmatch` is not enough for this. It has no notion of path segments, so its ``*``
happily crosses a ``/`` and its ``**`` means nothing at all; every rule here is compiled
to a regular expression that knows where the separators are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

#: The segment that stands for "any number of directories, including none".
DOUBLE_STAR = "**"


def _segment_regex(segment: str) -> str:
    """Translate the glob syntax inside a single path segment.

    ``*`` and ``?`` are deliberately segment-local: neither may swallow a ``/``, which
    is the whole difference between this and :func:`fnmatch.translate`.
    """
    out: list[str] = []
    for char in segment:
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
    return "".join(out)


def _compile(body: str, anchored: bool) -> re.Pattern[str]:
    """Compile a pattern body - no leading ``!``, no trailing ``/`` - to a full match."""
    segments = body.split("/")
    parts: list[str] = [] if anchored else ["(?:.*/)?"]
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == DOUBLE_STAR:
            parts.append(".*" if last else "(?:[^/]+/)*")
        else:
            parts.append(_segment_regex(segment))
            if not last:
                parts.append("/")
    return re.compile("".join(parts))


@dataclass(frozen=True)
class Rule:
    """One compiled line of a spec, keeping the text it was written as."""

    pattern: str
    dir_only: bool
    regex: re.Pattern[str]

    @classmethod
    def parse(cls, line: str) -> Rule:
        """Compile one already-stripped, already-significant line."""
        body = line
        dir_only = body.endswith("/")
        if dir_only:
            body = body[:-1]
        anchored = body.startswith("/")
        if anchored:
            body = body[1:]
        if "/" in body:
            anchored = True
        return cls(pattern=line, dir_only=dir_only, regex=_compile(body, anchored))

    def matches(self, candidate: str, is_dir: bool) -> bool:
        """Does this rule apply to ``candidate``, given whether it is a directory?"""
        if self.dir_only and not is_dir:
            return False
        return self.regex.fullmatch(candidate) is not None


def _significant(lines: Iterable[str]) -> Iterator[str]:
    """Drop blanks and comments, and strip surrounding whitespace off the rest."""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


def _candidates(path: str) -> list[tuple[str, bool]]:
    """Every prefix of ``path``, outermost first, tagged with "is it a directory?".

    A trailing ``/`` marks the path itself as a directory; every shorter prefix is one
    by construction. Leading slashes are stripped - paths here are repo-relative.
    """
    is_dir = path.endswith("/")
    body = path.strip("/")
    if not body:
        raise ValueError("path must not be empty")
    parts = body.split("/")
    return [
        ("/".join(parts[: index + 1]), True if index < len(parts) - 1 else is_dir)
        for index in range(len(parts))
    ]


class PathSpec:
    """A list of ignore patterns, in file order, applied to repo-relative paths."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._rules = [Rule.parse(line) for line in _significant(patterns)]

    @property
    def rules(self) -> tuple[Rule, ...]:
        """The compiled rules, in the order they were written."""
        return tuple(self._rules)

    def is_ignored(self, path: str) -> bool:
        """Is ``path`` excluded by this spec?

        The path is walked from its outermost directory inwards, because a spec that
        excludes a directory excludes everything inside it without the caller having to
        list the contents.
        """
        for candidate, is_dir in _candidates(path):
            for rule in self._rules:
                if rule.matches(candidate, is_dir):
                    return True
        return False
