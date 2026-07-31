#!/usr/bin/env python
"""Turn a real bug-fix commit into a task skeleton.

Given a repository and the SHA of a commit that fixed a bug, this extracts three
snapshots and lays them out the way `validate_tasks.py` expects:

| what | taken from | lands in |
| --- | --- | --- |
| the buggy source | the commit's **parent** | `repo/` |
| the fixed source | the **commit** | `solution/fixed/` |
| the test that proves it | the **commit** | `tests/` |

That is the whole trick behind `source: real_repo`: the fix is inverted by simply taking
the file from one commit earlier, so the defect is exactly the one a real maintainer
really shipped, and the test is exactly the one they wrote to catch it.

First-party imports are followed, so a module that depends on two siblings brings them
along. Everything is reported with line counts, because the result still has to fit in a
sub-500-line task repo (5.4 rule 6) - if it does not, extract fewer files or pick a
smaller commit.

    uv run python benchmark/tools/extract_task.py \\
        --clone ../_vendor/mining/humanize \\
        --sha 823ad6096e1e \\
        --source src/humanize/filesize.py \\
        --test tests/test_filesize.py \\
        --id bugfix_filesize_unit_rollover \\
        --category bug_fix --difficulty medium

What it does NOT do, and what you must still do by hand:

  * write the `instruction` (the symptom, in your own words, naming no test)
  * trim the extracted test file down to the assertions that matter, and remove any
    that depend on the rest of the upstream package
  * strip upstream imports the reduced repo no longer satisfies
  * set `difficulty` honestly and fill in `provenance.notes`
  * run `make gold-patches && make validate && make verify-gold`

See docs/task_authoring_guide.md.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.tools.new_task import CATEGORIES, DIFFICULTIES, ID_RE  # noqa: E402
from benchmark.tools.task_loader import SPLITS, TASKS_ROOT  # noqa: E402


def git(clone: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def show(clone: Path, rev: str, path: str) -> str | None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(clone), "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout if proc.returncode == 0 else None


def first_party_imports(source: str, package: str) -> set[str]:
    """Module names inside `package` that `source` imports, as dotted paths."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: `from .sibling import x`
                for alias in node.names:
                    found.add(f"{package}.{node.module or alias.name}" if node.module else f"{package}.{alias.name}")
            elif node.module and (node.module == package or node.module.startswith(package + ".")):
                found.add(node.module)
    return found


def module_to_path(dotted: str, package_root: str) -> list[str]:
    """`humanize.filesize` -> candidate repo-relative paths under `package_root`."""
    tail = dotted.split(".", 1)[1] if "." in dotted else ""
    base = f"{package_root}/{tail.replace('.', '/')}" if tail else package_root
    return [f"{base}.py", f"{base}/__init__.py"]


def loc(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))


def gather(clone: Path, rev: str, seeds: list[str], package_root: str, package: str, depth: int) -> dict[str, str]:
    """Fetch `seeds` at `rev`, following first-party imports up to `depth` levels."""
    out: dict[str, str] = {}
    frontier = list(seeds)
    for _ in range(depth + 1):
        nxt: list[str] = []
        for path in frontier:
            if path in out:
                continue
            text = show(clone, rev, path)
            if text is None:
                print(f"    (not present at {rev[:8]}: {path})", file=sys.stderr)
                continue
            out[path] = text
            for dotted in first_party_imports(text, package):
                for cand in module_to_path(dotted, package_root):
                    if cand not in out:
                        nxt.append(cand)
        frontier = nxt
        if not frontier:
            break
    return out


TASK_YAML = """\
id: {task_id}
category: {category}
difficulty: {difficulty}
split: {split}

# TODO: describe the SYMPTOM in your own words. Never name the hidden tests, and never
# paraphrase the upstream commit message closely enough to hand over the fix.
instruction: >
  TODO - upstream subject was: {subject}

repo:
  root: repo/
  loc: 0
  entrypoint_hint: {hint}

verification:
  runner: pytest
  test_files: [{test_files}]
  timeout_s: 60
  n_tests: 0

constraints:
  allowed_write_globs:
{allowed}
  forbidden_write_globs:
    - "tests/**"
    - "solution/**"
    - "**/conftest.py"
    - "task.yaml"
    - "repo/smoke/**"
  max_turns: 20
  max_tool_calls: 40

provenance:
  source: real_repo
  origin: {origin} @ {sha}
  license: {licence}
  notes: >
    TODO - upstream commit {short} ({cdate}) "{subject}".
    The source file was taken from the commit's PARENT, so the defect is the one that was
    actually shipped; the test is the one the maintainer wrote to catch it. Say here what
    was trimmed from the upstream module and why the remainder still stands alone.

metadata:
  expected_files_changed: {n_changed}
  created: {today}
  tags: []
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clone", required=True, type=Path, help="path to a local clone")
    parser.add_argument("--sha", required=True, help="the commit that FIXED the bug")
    parser.add_argument(
        "--source",
        required=True,
        action="append",
        metavar="PATH",
        help="repo-relative source file the fix touched; repeatable",
    )
    parser.add_argument(
        "--test",
        required=True,
        action="append",
        metavar="PATH",
        help="repo-relative test file the fix touched; repeatable",
    )
    parser.add_argument("--id", required=True)
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--difficulty", default="medium", choices=DIFFICULTIES)
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--package-root", help="e.g. src/humanize (inferred from --source if omitted)")
    parser.add_argument("--depth", type=int, default=1, help="how far to follow first-party imports")
    parser.add_argument("--origin", help="upstream URL (inferred from the clone's remote)")
    parser.add_argument("--licence", default="MIT")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not ID_RE.match(args.id):
        raise SystemExit(f"'{args.id}' is not a valid snake_case task id")
    clone = args.clone.resolve()
    if not (clone / ".git").exists():
        raise SystemExit(f"{clone} is not a git clone")

    sha = git(clone, "rev-parse", args.sha).strip()
    parent = git(clone, "rev-parse", f"{sha}^").strip()
    subject = git(clone, "log", "-1", "--pretty=%s", sha).strip().replace('"', "'")
    cdate = git(clone, "log", "-1", "--pretty=%ad", "--date=short", sha).strip()
    origin = args.origin or git(clone, "remote", "get-url", "origin").strip().removesuffix(".git")

    package_root = args.package_root or str(Path(args.source[0]).parent).replace("\\", "/")
    package = Path(package_root).name

    dest = TASKS_ROOT / args.split / args.id
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} already exists (--force to overwrite)")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"commit  {sha}\nparent  {parent}\nsubject {subject}\n")

    print("buggy source (from the parent) -> repo/")
    buggy = gather(clone, parent, args.source, package_root, package, args.depth)
    fixed = gather(clone, sha, args.source, package_root, package, args.depth)

    repo_loc = 0
    for path, text in sorted(buggy.items()):
        rel = Path(path).relative_to(Path(package_root).parent) if package_root in path else Path(path).name
        out = dest / "repo" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        n = loc(text)
        repo_loc += n
        print(f"  {n:>4}L  repo/{Path(rel).as_posix()}")

    print("\nfixed source (from the commit) -> solution/fixed/  [only where it differs]")
    n_changed = 0
    for path, text in sorted(fixed.items()):
        if buggy.get(path) == text:
            continue
        rel = Path(path).relative_to(Path(package_root).parent) if package_root in path else Path(path).name
        out = dest / "solution" / "fixed" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        n_changed += 1
        print(f"  {loc(text):>4}L  solution/fixed/{Path(rel).as_posix()}")

    print("\ntest (from the commit) -> tests/")
    test_names = []
    for path in args.test:
        text = show(clone, sha, path)
        if text is None:
            print(f"  MISSING at {sha[:8]}: {path}", file=sys.stderr)
            continue
        name = Path(path).name
        out = dest / "tests" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        test_names.append(f"tests/{name}")
        print(f"  {loc(text):>4}L  tests/{name}")

    (dest / "smoke").mkdir(exist_ok=True)

    yaml_path = dest / "task.yaml"
    if not yaml_path.exists() or args.force:
        yaml_path.write_text(
            TASK_YAML.format(
                task_id=args.id,
                category=args.category,
                difficulty=args.difficulty,
                split=args.split,
                subject=subject,
                hint=f"{package}/{Path(args.source[0]).name}",
                test_files=", ".join(test_names) or "tests/TODO.py",
                allowed="\n".join(f'    - "repo/{package}/**"' for _ in [0]),
                origin=origin,
                sha=sha,
                short=sha[:12],
                cdate=cdate,
                licence=args.licence,
                n_changed=max(1, n_changed),
                today=date.today().isoformat(),
            ),
            encoding="utf-8",
            newline="\n",
        )

    print(f"\nrepo/ is {repo_loc} lines of code")
    if repo_loc > 500:
        print("  ^^ over the 500-line ceiling. Extract fewer modules (--depth 0) or pick a smaller commit.")
    print(f"\nscaffolded {dest.relative_to(REPO_ROOT)}")
    print("\nnow, by hand:")
    print("  1. trim the test file to assertions the reduced repo can actually satisfy")
    print("  2. strip imports the reduced repo no longer provides")
    print("  3. write the instruction, fill provenance.notes, set difficulty honestly")
    print("  4. make gold-patches && make validate && make verify-gold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
