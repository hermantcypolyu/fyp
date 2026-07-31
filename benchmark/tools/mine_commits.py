#!/usr/bin/env python
"""Find real bug-fix commits worth inverting into benchmark tasks.

AGENT_BUILD_SPEC 5.3 wants at least half the corpus to carry `source: real_repo`, because
invented bugs drift toward the tidy, textbook errors a code model has already seen a
thousand times. The recipe is: find a commit that fixed a real bug *and* added or changed
a test proving it, keep the test, invert the source change.

This script does the searching. It clones (or reuses) a repository, walks its history, and
reports commits that look like a fix, touch both source and tests, and are small enough
that the inverted version fits in a sub-300-line task repo.

It deliberately does **not** create tasks. Judging whether a diff is a *good* task - a
defect with several observable consequences, comprehensible from a symptom description
alone - is the part that needs a person. This just narrows thousands of commits down to
a few dozen worth reading.

    uv run python benchmark/tools/mine_commits.py --repo https://github.com/python-humanize/humanize
    uv run python benchmark/tools/mine_commits.py --config benchmark/tools/mining_sources.yaml
    uv run python benchmark/tools/mine_commits.py --repo <url> --show <sha>

Clones land in `../_vendor/mining/`, outside the repository and never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT.parent / "_vendor" / "mining"

#: Commit subjects that plausibly describe a bug fix. Kept broad on purpose - the cost of a
#: false positive is one line of output, the cost of a false negative is a missed task.
FIX_SUBJECT = re.compile(
    r"^(fix|bugfix|hotfix|patch)\b|^fix[(:\[]|\bfixes?\s+#\d+|\bbug\b|\bincorrect\b|"
    r"\bwrong\b|\boff.by.one\b|\bregression\b|\bcrash\b|\braises?\b.*\berror\b",
    re.IGNORECASE,
)

#: The five task categories in AGENT_BUILD_SPEC 5.2 are not all fed by bug fixes, and a
#: corpus that is 50% real_repo in one category and 0% in the other four is not really 50%
#: real. Each preset below finds the kind of commit that inverts into one category:
#:
#:   bug_fix       a defect was fixed and a test proves it     -> invert the source
#:   api_change    a public name/behaviour changed             -> keep the NEW source, the OLD tests
#:   feature       a capability was added, with tests          -> take the source from the parent
#:   config        packaging or re-export metadata was fixed   -> invert the config
#:   docs          a docstring or documented example was wrong -> invert the docstring
#:
#: `api_change` is the one that pays for itself: it is exactly the test_update shape, and
#: reversing the usual extraction (new source, stale tests) needs no invention at all.
KINDS: dict[str, re.Pattern[str]] = {
    "bug_fix": FIX_SUBJECT,
    "api_change": re.compile(
        r"\brename[ds]?\b|\bdeprecat|\bbreaking\b|\bremove[ds]?\b|\breplace[ds]?\b|"
        r"\bmigrat|\bdrop\b|^refactor[(:\[ ]|\bapi\b",
        re.IGNORECASE,
    ),
    "feature": re.compile(
        r"^(feat|feature)\b|^feat[(:\[]|\badd(s|ed|ing)?\b|\bsupport for\b|\bnew option\b", re.IGNORECASE
    ),
    "config": re.compile(
        r"packag|pyproject|setup\.(py|cfg)|\bimport error\b|__init__|\bexport|\brequires.python\b|\bwheel\b|\bversion\b",
        re.IGNORECASE,
    ),
    "docs": re.compile(r"^docs?[(:\[ ]|\bdocstring\b|\bdoctest\b|\bexample\b|\bdocumented?\b", re.IGNORECASE),
}

#: Per-kind extra requirement on the *paths* a commit touched. Returning False rejects it.
KIND_PATH_RULES: dict[str, str] = {
    "config": r"pyproject\.toml$|setup\.(py|cfg)$|__init__\.py$|_version\.py$",
}

#: Subjects that match FIX_SUBJECT but are not code fixes.
NOISE_SUBJECT = re.compile(
    r"\b(typo|changelog|readme|docs?|lint|format|black|isort|ruff|flake8|mypy|"
    r"pre-commit|bump|release|version|ci|workflow|dependabot|merge branch|revert)\b",
    re.IGNORECASE,
)

TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]+\.py$|_test\.py$", re.IGNORECASE)


@dataclass
class Candidate:
    sha: str
    date: str
    subject: str
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    source_lines: int = 0
    test_lines: int = 0

    @property
    def score(self) -> int:
        """Rough desirability. Small source change, one file, real test movement."""
        s = 0
        s += 30 if len(self.source_files) == 1 else max(0, 20 - 5 * len(self.source_files))
        s += 25 if 2 <= self.source_lines <= 20 else (10 if self.source_lines <= 40 else 0)
        s += 20 if self.test_lines >= 5 else 5
        s += 15 if len(self.test_files) == 1 else 5
        return s

    def as_dict(self) -> dict:
        return {
            "sha": self.sha,
            "date": self.date,
            "subject": self.subject,
            "source_files": self.source_files,
            "test_files": self.test_files,
            "source_lines": self.source_lines,
            "test_lines": self.test_lines,
            "score": self.score,
        }


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def ensure_clone(url: str) -> Path:
    VENDOR.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    dest = VENDOR / name
    if dest.is_dir():
        return dest
    print(f"cloning {url} -> {dest}", file=sys.stderr)
    # A full clone, deliberately. `--filter=blob:none` looks attractive for history-only
    # work, but `git log --numstat` needs every blob to compute a diffstat, so a partial
    # clone turns one local walk into thousands of individual network round-trips.
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "clone", "--quiet", url, str(dest)], check=True
    )
    return dest


def read_licence(repo: Path) -> str:
    for name in ("LICENSE", "LICENCE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        path = repo / name
        if path.is_file():
            head = path.read_text(encoding="utf-8", errors="replace")[:400].lower()
            for key, label in (
                ("mit license", "MIT"),
                ("apache license", "Apache-2.0"),
                ("bsd 3-clause", "BSD-3-Clause"),
                ("bsd 2-clause", "BSD-2-Clause"),
                ("redistribution and use in source", "BSD"),
                ("mozilla public license", "MPL-2.0"),
                ("gnu general public", "GPL - NOT USABLE"),
            ):
                if key in head:
                    return label
            return "unrecognised - read it yourself"
    return "no licence file found - DO NOT USE"


def mine(
    repo: Path,
    *,
    limit: int,
    max_source_lines: int,
    since: str | None,
    kind: str = "bug_fix",
) -> list[Candidate]:
    subject_re = KINDS[kind]
    path_rule = re.compile(KIND_PATH_RULES[kind]) if kind in KIND_PATH_RULES else None
    sep = "\x1e"
    args = ["log", f"--pretty=format:{sep}%H%x1f%ad%x1f%s", "--date=short", "--numstat", "--no-merges"]
    if since:
        args += [f"--since={since}"]
    raw = git(repo, *args)

    out: list[Candidate] = []
    for block in raw.split(sep):
        block = block.strip("\n")
        if not block:
            continue
        header, _, body = block.partition("\n")
        parts = header.split("\x1f")
        if len(parts) != 3:
            continue
        sha, date, subject = parts

        if not subject_re.search(subject) or NOISE_SUBJECT.search(subject):
            continue

        cand = Candidate(sha=sha, date=date, subject=subject)
        skip = False
        all_paths: list[str] = []
        for line in body.splitlines():
            cells = line.split("\t")
            if len(cells) != 3:
                continue
            added, removed, path = cells
            if added == "-" or removed == "-":  # binary
                skip = True
                break
            all_paths.append(path)
            churn = int(added) + int(removed)
            # config_fix tasks live in packaging metadata, which is not Python at all.
            if not path.endswith((".py", ".toml", ".cfg")):
                continue
            if TEST_PATH.search(path):
                cand.test_files.append(path)
                cand.test_lines += churn
            else:
                cand.source_files.append(path)
                cand.source_lines += churn

        if skip or not cand.source_files or not cand.test_files:
            continue
        if path_rule and not any(path_rule.search(p) for p in all_paths):
            continue
        if cand.source_lines == 0 or cand.source_lines > max_source_lines:
            continue
        if len(cand.source_files) > 3:
            continue
        out.append(cand)

    out.sort(key=lambda c: (-c.score, c.source_lines))
    return out[:limit]


def show(repo: Path, sha: str) -> None:
    print(git(repo, "show", "--stat", "--patch", sha))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", action="append", default=[], metavar="URL", help="repeatable")
    parser.add_argument("--show", metavar="SHA", help="print one commit's full diff and exit")
    parser.add_argument(
        "--kind",
        choices=sorted(KINDS),
        default="bug_fix",
        help="which task category to mine sources for (default bug_fix)",
    )
    parser.add_argument("--limit", type=int, default=25, help="candidates per repo (default 25)")
    parser.add_argument("--max-source-lines", type=int, default=60)
    parser.add_argument("--since", metavar="DATE", help="e.g. 2019-01-01")
    parser.add_argument("--json", metavar="PATH", help="write the full candidate list as JSON")
    args = parser.parse_args()

    if not args.repo:
        parser.error("give at least one --repo URL")

    report: dict[str, dict] = {}
    for url in args.repo:
        clone = ensure_clone(url)
        licence = read_licence(clone)

        if args.show:
            show(clone, args.show)
            return 0

        cands = mine(clone, limit=args.limit, max_source_lines=args.max_source_lines, since=args.since, kind=args.kind)
        head = git(clone, "rev-parse", "HEAD").strip()
        report[url] = {"licence": licence, "head": head, "candidates": [c.as_dict() for c in cands]}

        print(f"\n=== {url}")
        print(f"    licence: {licence}    head: {head}    kind: {args.kind}")
        if licence.startswith(("no licence", "GPL")):
            print("    ^^ unusable for redistribution; skipping the listing")
            continue
        if not cands:
            print("    no candidates")
            continue
        for c in cands:
            print(
                f"  {c.sha[:12]}  {c.date}  score {c.score:>3}  "
                f"src {c.source_lines:>3}L/{len(c.source_files)}f  test {c.test_lines:>3}L"
            )
            print(f"      {c.subject[:96]}")
            print(f"      src:  {', '.join(c.source_files)}")
            print(f"      test: {', '.join(c.test_files)}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
