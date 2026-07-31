"""Glob matching for the write allow/deny lists in `task.yaml`.

`fnmatch` treats `/` as an ordinary character, so `repo/*.py` would happily match
`repo/src/deep/nested.py` - which would silently widen every task's write permissions.
Since these globs are a **security boundary** (5.7's `tests_modified` and
`sandbox_escape` gates depend on them), the semantics are spelled out and tested here
rather than inherited from a convenient stdlib function.

Semantics, matching .gitignore/pathspec conventions:

| Pattern     | Meaning                                                  |
|-------------|----------------------------------------------------------|
| `*`         | any run of characters, **not** crossing `/`               |
| `?`         | exactly one character, not `/`                            |
| `[abc]`     | one character from the set                                |
| `**`        | as a whole segment: zero or more path segments            |
| `a/**`      | everything under `a/` (but not `a` itself)                |
| `**/x.py`   | `x.py` at any depth, including the top level              |

Paths are always compared as `/`-separated and relative to the task directory.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import PurePath

__all__ = ["glob_to_regex", "path_matches", "path_matches_any", "normalise"]

_GLOBSTAR = "\x00globstar\x00"


def normalise(path: str | PurePath) -> str:
    """`repo\\src\\a.py` -> `repo/src/a.py`.

    Also collapses `.` segments, repeated separators and a trailing separator. That is not
    cosmetic: `tests//test_x.py` and `./tests/test_x.py` name the same file as
    `tests/test_x.py`, and if the matcher disagreed, an agent could step around the
    `tests/**` deny-list by respelling the path.
    """
    text = str(path).replace("\\", "/")
    segments = [s for s in text.split("/") if s not in ("", ".")]
    return "/".join(segments)


def _segment_to_regex(seg: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(seg):
        ch = seg[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            close = seg.find("]", i + 1)
            if close == -1:
                out.append(re.escape(ch))
            else:
                body = seg[i + 1 : close]
                body = ("^" + body[1:]) if body.startswith("!") else body
                out.append(f"[{body}]")
                i = close
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


@lru_cache(maxsize=512)
def glob_to_regex(pattern: str) -> re.Pattern[str]:
    segments = normalise(pattern).split("/")
    parts: list[str] = []
    for index, seg in enumerate(segments):
        is_last = index == len(segments) - 1
        if seg == "**":
            # trailing `**` swallows the remainder; otherwise it spans whole segments
            parts.append(".*" if is_last else "(?:[^/]+/)*")
        else:
            parts.append(_segment_to_regex(seg) + ("" if is_last else "/"))
    return re.compile("^" + "".join(parts) + "$")


def path_matches(path: str | PurePath, pattern: str) -> bool:
    return bool(glob_to_regex(pattern).match(normalise(path)))


def path_matches_any(path: str | PurePath, patterns: list[str]) -> bool:
    text = normalise(path)
    return any(glob_to_regex(p).match(text) for p in patterns)
