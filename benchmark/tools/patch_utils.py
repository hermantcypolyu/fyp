"""Unified-diff generation and application, without depending on `git` or `patch`.

Why this exists: `git apply` works, but it is strict about CRLF, and half of this
project's tasks would fail on a Windows checkout for reasons that have nothing to do
with the task. Gold patches must apply identically everywhere, so we normalise line
endings and apply the diff ourselves.

Scope is deliberately narrow - it handles exactly the diffs `make_patch()` produces:
text files, `a/` and `b/` prefixes, file add/modify/delete. It is not a general
`patch(1)`. If you hand-write a diff and it fails to apply, regenerate it with
`benchmark/tools/make_gold_patch.py` rather than making this smarter.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PatchError", "FilePatch", "parse_unified_diff", "apply_patch", "make_patch", "patched_files"]

DEV_NULL = "/dev/null"
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
#: How far either side of the stated line number we will look for the context block.
_MAX_FUZZ = 200


class PatchError(Exception):
    """A patch could not be parsed or applied. The message is meant to be read by a human."""


@dataclass
class Hunk:
    old_start: int  # 1-based
    lines: list[str]  # each begins with ' ', '-' or '+'
    old_count: int = -1  # from the @@ header; -1 means "unknown"
    new_count: int = -1

    @property
    def old_block(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln[:1] in (" ", "-")]

    @property
    def new_block(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln[:1] in (" ", "+")]

    @property
    def is_full(self) -> bool:
        """True once the hunk holds exactly the line counts its header promised.

        This is what stops the blank line that follows the final hunk - an artefact of
        `text.split("\\n")` on a newline-terminated file - from being swallowed as a
        phantom empty context line. Without it the last hunk carries one extra line and
        never matches.
        """
        if self.old_count < 0 or self.new_count < 0:
            return False
        return len(self.old_block) >= self.old_count and len(self.new_block) >= self.new_count


@dataclass
class FilePatch:
    old_path: str | None  # None => file is being created
    new_path: str | None  # None => file is being deleted
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        p = self.new_path or self.old_path
        assert p is not None
        return p

    @property
    def is_creation(self) -> bool:
        return self.old_path is None

    @property
    def is_deletion(self) -> bool:
        return self.new_path is None


def _strip_prefix(raw: str) -> str | None:
    """`a/src/foo.py` -> `src/foo.py`; `/dev/null` -> None."""
    raw = raw.split("\t", 1)[0].strip()
    if raw == DEV_NULL:
        return None
    if raw[:2] in ("a/", "b/"):
        raw = raw[2:]
    return raw


def parse_unified_diff(text: str) -> list[FilePatch]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise PatchError(f"line {i + 1}: '---' header is not followed by a '+++' header")
            current = FilePatch(old_path=_strip_prefix(line[4:]), new_path=_strip_prefix(lines[i + 1][4:]))
            patches.append(current)
            hunk = None
            i += 2
            continue

        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                raise PatchError(f"line {i + 1}: hunk header before any '---'/'+++' file header")
            hunk = Hunk(
                old_start=int(m.group(1)),
                lines=[],
                old_count=int(m.group(2)) if m.group(2) is not None else 1,
                new_count=int(m.group(4)) if m.group(4) is not None else 1,
            )
            current.hunks.append(hunk)
            i += 1
            continue

        if hunk is not None and not hunk.is_full:
            if line[:1] in (" ", "-", "+"):
                hunk.lines.append(line)
            elif line == "":
                # An empty context line whose trailing space was stripped - editors and
                # mail clients do this constantly. Only accepted while the hunk is still
                # short of its declared length.
                hunk.lines.append(" ")
        # everything else (index lines, 'diff --git', '\ No newline', prose) is ignored
        i += 1

    if not patches:
        raise PatchError("no file headers found - is this actually a unified diff?")
    for p in patches:
        if not p.hunks and not p.is_deletion:
            raise PatchError(f"{p.path}: file header with no hunks")
    return patches


def _find_offset(src: list[str], block: list[str], want: int) -> int:
    """Return the 0-based index where `block` occurs, preferring `want`."""
    if not block:
        return max(0, min(want, len(src)))
    if src[want : want + len(block)] == block:
        return want
    for delta in range(1, _MAX_FUZZ + 1):
        for cand in (want - delta, want + delta):
            if 0 <= cand <= len(src) - len(block) and src[cand : cand + len(block)] == block:
                return cand
    raise PatchError(
        f"context not found near line {want + 1}. The file has changed since the patch was made.\n"
        f"  expected to find:\n"
        + "\n".join(f"    | {ln}" for ln in block[:6])
        + ("\n    | ..." if len(block) > 6 else "")
    )


def _apply_hunks(src: list[str], patch: FilePatch) -> list[str]:
    out: list[str] = []
    pos = 0
    for hunk in patch.hunks:
        start = _find_offset(src, hunk.old_block, max(0, hunk.old_start - 1))
        if start < pos:
            raise PatchError(f"{patch.path}: hunks are out of order or overlap at line {hunk.old_start}")
        out.extend(src[pos:start])
        out.extend(hunk.new_block)
        pos = start + len(hunk.old_block)
    out.extend(src[pos:])
    return out


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def apply_patch(root: Path, patch_text: str, *, reverse: bool = False, dry_run: bool = False) -> list[str]:
    """Apply a unified diff under `root`. Returns the paths touched, relative to root.

    All-or-nothing: everything is computed in memory first, so a failure on the third
    file does not leave the first two half-patched.
    """
    root = Path(root)
    patches = parse_unified_diff(patch_text)
    if reverse:
        patches = [_reverse(p) for p in patches]

    planned: list[tuple[Path, list[str] | None]] = []  # (target, new content or None to delete)
    for patch in patches:
        target = (root / patch.path).resolve()
        if not str(target).startswith(str(root.resolve())):
            raise PatchError(f"{patch.path}: patch escapes the target directory")

        if patch.is_deletion:
            if not target.is_file():
                raise PatchError(f"{patch.path}: patch deletes this file, but it does not exist")
            planned.append((target, None))
            continue

        if patch.is_creation:
            if target.exists():
                raise PatchError(f"{patch.path}: patch creates this file, but it already exists")
            src: list[str] = [""]
        else:
            if not target.is_file():
                raise PatchError(f"{patch.path}: file not found under {root}")
            src = _read_lines(target)
        planned.append((target, _apply_hunks(src, patch)))

    if dry_run:
        return [p.path for p in patches]

    for target, content in planned:
        if content is None:
            target.unlink()
        else:
            _write_lines(target, content)
    return [p.path for p in patches]


def _reverse(patch: FilePatch) -> FilePatch:
    flip = {" ": " ", "-": "+", "+": "-"}
    return FilePatch(
        old_path=patch.new_path,
        new_path=patch.old_path,
        hunks=[
            Hunk(
                old_start=1,  # recovered by fuzz search; reversed line numbers are unreliable
                lines=[flip[ln[:1]] + ln[1:] for ln in h.lines],
                old_count=h.new_count,
                new_count=h.old_count,
            )
            for h in patch.hunks
        ],
    )


def patched_files(patch_text: str) -> list[str]:
    """Paths a patch would touch, without applying it. Used by the write-glob checks."""
    return [p.path for p in parse_unified_diff(patch_text)]


def make_patch(before: Path, after: Path, *, context: int = 3) -> str:
    """Produce a unified diff turning `before/` into `after/`. Used when authoring tasks."""
    before, after = Path(before), Path(after)
    names = sorted(
        {str(p.relative_to(before)).replace("\\", "/") for p in before.rglob("*") if p.is_file()}
        | {str(p.relative_to(after)).replace("\\", "/") for p in after.rglob("*") if p.is_file()}
    )

    chunks: list[str] = []
    for name in names:
        if "__pycache__" in name or name.endswith(".pyc"):
            continue
        old_file, new_file = before / name, after / name
        old = _read_lines(old_file) if old_file.is_file() else None
        new = _read_lines(new_file) if new_file.is_file() else None
        if old == new:
            continue
        diff = difflib.unified_diff(
            old if old is not None else [],
            new if new is not None else [],
            fromfile=DEV_NULL if old is None else f"a/{name}",
            tofile=DEV_NULL if new is None else f"b/{name}",
            n=context,
            lineterm="",
        )
        chunks.append("\n".join(diff))
    return "\n".join(chunks) + "\n" if chunks else ""
