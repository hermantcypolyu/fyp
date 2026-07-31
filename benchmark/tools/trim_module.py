#!/usr/bin/env python
"""Reduce an extracted module to the definitions a task actually needs.

Real modules are bigger than a task repo should be. `humanize/number.py` is 600 lines
covering nine unrelated formatters; a task about `intword` needs three of them. Deleting
the rest by hand invites two mistakes: dropping something the kept code calls, and
accidentally *editing* the code that stays - which would quietly break the claim that
`source: real_repo` means the defect is byte-for-byte the one upstream shipped.

So this does it mechanically. Top-level definitions you name are copied out verbatim by
source segment; module docstring, imports and `__future__` lines are always kept; anything
else is dropped. It then reports which names the kept code still references but no longer
defines, so a missing dependency is visible immediately rather than as an import error
three commands later.

    uv run python benchmark/tools/trim_module.py \\
        benchmark/tasks/train/bugfix_intword_power_rollover/repo/humanize/number.py \\
        --keep NumberOrString --keep powers --keep human_powers --keep intword

    # the same trim must be applied to the reference solution, or the gold patch will
    # contain the deletions as well as the fix:
    uv run python benchmark/tools/trim_module.py <the solution/fixed copy> --keep ... --keep ...

`--dry-run` prints what would go and what would stay without touching the file.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
from pathlib import Path

ALWAYS_KEEP = (ast.Import, ast.ImportFrom)


def trim(source: str, keep: set[str]) -> tuple[str, list[str], list[str]]:
    """Return (new source, names kept, names dropped)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    chunks: list[tuple[ast.stmt, str]] = []
    kept: list[str] = []
    dropped: list[str] = []

    for node in tree.body:
        names = _names_bound(node)
        is_docstring = (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        )
        # `if TYPE_CHECKING:` blocks are import blocks in disguise; dropping one takes
        # TypeAlias, Iterable and friends with it.
        is_type_checking = isinstance(node, ast.If) and any(
            isinstance(n, ast.Name) and n.id == "TYPE_CHECKING" for n in ast.walk(node.test)
        )
        wanted = isinstance(node, ALWAYS_KEEP) or is_docstring or is_type_checking or bool(names & keep)

        if wanted:
            chunks.append((node, _segment(lines, node)))
            kept.extend(sorted(names))
        else:
            dropped.extend(sorted(names) or [type(node).__name__])

    text = ""
    prev: ast.stmt | None = None
    for node, chunk in chunks:
        body = chunk.rstrip("\n")
        if not text:
            text = body
            prev = node
            continue
        if isinstance(node, ALWAYS_KEEP) and isinstance(prev, ALWAYS_KEEP):
            gap = "\n"  # consecutive imports stay in one block
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            gap = "\n\n\n"  # PEP 8: two blank lines before a top-level definition
        else:
            gap = "\n\n"
        text += gap + body
        prev = node
    return text + "\n", kept, dropped


def _names_bound(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _segment(lines: list[str], node: ast.stmt) -> str:
    start = node.lineno - 1
    for deco in getattr(node, "decorator_list", []):
        start = min(start, deco.lineno - 1)
    return "".join(lines[start : node.end_lineno])


def _target_names(target: ast.expr) -> set[str]:
    """Every name bound by an assignment/loop target, including tuple unpacking."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def dangling_references(source: str) -> list[str]:
    """Names the trimmed module reads but no longer defines or imports."""
    tree = ast.parse(source)
    defined: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        defined |= _names_bound(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            defined |= {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            defined |= {a.arg for a in (args.vararg, args.kwarg) if a}
        elif isinstance(node, ast.comprehension):
            defined |= _target_names(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            defined |= _target_names(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            defined |= _target_names(node.optional_vars)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                defined |= _target_names(target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - defined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--keep", action="append", default=[], metavar="NAME", required=True, help="top-level name to keep; repeatable"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.path.read_text(encoding="utf-8")
    new_source, kept, dropped = trim(source, set(args.keep))

    missing = set(args.keep) - set(kept)
    if missing:
        print(f"not found in {args.path.name}: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    before = len([ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")])
    after = len([ln for ln in new_source.splitlines() if ln.strip() and not ln.strip().startswith("#")])

    print(f"{args.path}")
    print(f"  kept:    {', '.join(kept)}")
    print(f"  dropped: {', '.join(dropped) or '(nothing)'}")
    print(f"  {before} -> {after} lines of code")

    dangling = dangling_references(new_source)
    if dangling:
        print(f"  DANGLING: {', '.join(dangling)}")
        print("  ^^ the kept code references these but nothing defines them any more.")
        print("     Either --keep them too, or accept that they come from an import you have removed.")

    if args.dry_run:
        return 0
    args.path.write_text(new_source, encoding="utf-8", newline="\n")
    print("  written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
