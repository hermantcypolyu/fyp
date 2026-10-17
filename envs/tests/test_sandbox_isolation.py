"""Isolation proof: the hidden test suite is unreachable from inside the sandbox.

This is the security-critical file. Everything else in the project can be wrong and still
produce a study that means something; if this is wrong, every reward is suspect and the
results are worthless. So the vectors are enumerated explicitly rather than trusted to a
general argument, and each one is asserted *both* for reading and for writing:

- the plain path `tests/test_x.py`
- traversal, `../../../benchmark/tasks/train/<id>/tests/test_x.py`
- an absolute path to the real task directory
- a symlink planted inside the sandbox that points at the real task directory
- respellings that a naive matcher would miss: `tests//x.py`, `./tests/x.py`

Reading matters as much as writing. An agent that can *read* the hidden suite does not need
to modify it - it can special-case the exact assertions, which produces a perfect score and
a model that has learned nothing.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from benchmark.tools.task_loader import TASKS_ROOT, load_task
from envs.coding_env.sandbox import open_sandbox
from envs.coding_env.tools import GATE_SANDBOX_ESCAPE, GATE_TESTS_MODIFIED

TASK_ID = "bugfix_metric_zero_domain_error"
HIDDEN_TEST = "test_number.py"


@pytest.fixture(scope="module")
def task():
    return load_task(TASKS_ROOT / "train" / TASK_ID)


@pytest.fixture
def sb(task, tmp_path):
    with open_sandbox(task, parent=tmp_path) as sandbox:
        yield sandbox


def _digest(directory: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        h.update(path.relative_to(directory).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------- the guarantee is absence


def test_the_task_really_does_have_hidden_tests(task):
    """Control. Without this, every assertion below could pass on an empty corpus."""
    assert (task.tests_dir / HIDDEN_TEST).is_file()


def test_sandbox_contains_repo_and_smoke_but_no_tests(sb):
    assert sb.repo_dir.is_dir()
    assert (sb.root / "smoke").is_dir()
    assert not (sb.root / "tests").exists()
    assert not list(sb.root.rglob("tests"))


def test_no_file_in_the_sandbox_holds_a_hidden_assertion(sb, task):
    """Belt and braces: the hidden suite's text must not appear anywhere in the sandbox."""
    needle = (task.tests_dir / HIDDEN_TEST).read_text(encoding="utf-8").splitlines()
    marker = next(ln.strip() for ln in needle if ln.strip().startswith("assert"))
    for path in sb.root.rglob("*.py"):
        assert marker not in path.read_text(encoding="utf-8", errors="replace"), path


# ------------------------------------------------------- vector 1: the plain path


def test_plain_path_is_unreadable(sb):
    result = sb.toolbox.read_file(f"tests/{HIDDEN_TEST}")
    assert not result.ok
    assert "no such file" in result.output


def test_plain_path_is_unwritable(sb):
    result = sb.toolbox.write_file(f"tests/{HIDDEN_TEST}", "def test_x(): pass\n")
    assert not result.ok
    assert GATE_TESTS_MODIFIED in sb.toolbox.gates
    assert not (sb.root / "tests").exists()


# ------------------------------------------------------- vector 2: path traversal


def test_traversal_is_unreadable(sb, task):
    rel = Path(os.path.relpath(task.tests_dir / HIDDEN_TEST, sb.root)).as_posix()
    assert rel.startswith("..")  # the test would be vacuous otherwise
    result = sb.toolbox.read_file(rel)
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates


def test_traversal_is_unwritable(sb, task):
    target = task.tests_dir / HIDDEN_TEST
    before = target.read_bytes()
    rel = Path(os.path.relpath(target, sb.root)).as_posix()
    result = sb.toolbox.write_file(rel, "def test_x(): pass\n")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates
    assert target.read_bytes() == before


def test_traversal_out_of_the_sandbox_creates_nothing(sb):
    result = sb.toolbox.write_file("../../pwned.py", "x = 1")
    assert not result.ok
    assert not (sb.root.parent.parent / "pwned.py").exists()


# ------------------------------------------------------- vector 3: absolute paths


def test_absolute_path_is_unreadable(sb, task):
    result = sb.toolbox.read_file(str(task.tests_dir / HIDDEN_TEST))
    assert not result.ok
    assert "absolute" in result.output


def test_absolute_path_is_unwritable(sb, task):
    target = task.tests_dir / HIDDEN_TEST
    before = target.read_bytes()
    result = sb.toolbox.write_file(str(target), "def test_x(): pass\n")
    assert not result.ok
    assert target.read_bytes() == before


def test_absolute_path_inside_the_sandbox_is_also_refused(sb):
    """No absolute paths at all - allowing "harmless" ones re-opens the vector."""
    assert not sb.toolbox.read_file(str(sb.repo_dir / "humanize" / "number.py")).ok


# ------------------------------------------------------- vector 4: symlinks


@pytest.fixture
def symlink_to_hidden_tests(sb, task):
    """A link inside the sandbox pointing at the real task directory.

    Plain `os.symlink` needs Developer Mode or admin on Windows, so fall back to a
    directory junction, which does not. Without the fallback this vector would silently
    skip on the machine the project is actually developed on - an untested security claim.
    """
    link = sb.root / "peek"
    try:
        os.symlink(task.tests_dir, link, target_is_directory=True)
        return link
    except (OSError, NotImplementedError):
        pass
    if os.name != "nt":
        pytest.skip("cannot create symlinks here")
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["cmd", "/c", "mklink", "/J", str(link), str(task.tests_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not link.exists():
        pytest.skip(f"cannot create a link here: {proc.stderr.strip() or proc.stdout.strip()}")
    return link


def test_symlink_is_unreadable(sb, symlink_to_hidden_tests):
    result = sb.toolbox.read_file(f"peek/{HIDDEN_TEST}")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates


def test_symlink_is_unwritable(sb, task, symlink_to_hidden_tests):
    target = task.tests_dir / HIDDEN_TEST
    before = target.read_bytes()
    result = sb.toolbox.write_file(f"peek/{HIDDEN_TEST}", "def test_x(): pass\n")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates
    assert target.read_bytes() == before


def test_symlink_cannot_be_enumerated_through(sb, symlink_to_hidden_tests):
    result = sb.toolbox.list_files("peek")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates


# ------------------------------------------------------- vector 5: respellings


@pytest.mark.parametrize(
    "spelling",
    [f"tests//{HIDDEN_TEST}", f"./tests/{HIDDEN_TEST}", f"tests/./{HIDDEN_TEST}", f"tests\\{HIDDEN_TEST}"],
)
def test_respellings_still_hit_the_deny_list(sb, spelling):
    """`normalise` collapses these before matching; if it did not, the gate is bypassable."""
    result = sb.toolbox.write_file(spelling, "def test_x(): pass\n")
    assert not result.ok
    assert GATE_TESTS_MODIFIED in sb.toolbox.gates


# ------------------------------------------------------- enumeration


def test_list_files_cannot_see_out_of_the_sandbox(sb):
    result = sb.toolbox.list_files("..")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in sb.toolbox.gates


def test_list_files_never_mentions_a_tests_directory(sb):
    out = sb.toolbox.list_files().output
    assert "tests/" not in out
    assert HIDDEN_TEST not in out


# ------------------------------------------------------- the batch-1 guard


def test_run_tests_guard_fires_when_a_tests_directory_appears(sb):
    """Requirement 5. If the sandbox is ever misbuilt, grading must not proceed quietly."""
    (sb.root / "tests").mkdir()
    (sb.root / "tests" / HIDDEN_TEST).write_text("def test_x(): pass\n", encoding="utf-8")
    result = sb.toolbox.run_tests()
    assert not result.ok
    assert "misbuilt" in result.output


# ------------------------------------------------------- the corpus survives an attack


def test_a_hostile_episode_leaves_the_task_source_byte_identical(sb, task):
    """Every vector above, fired in sequence, against the real corpus on disk."""
    before = _digest(task.path)
    box = sb.toolbox
    attacks = [
        ("write_file", {"path": f"tests/{HIDDEN_TEST}", "content": "pass"}),
        ("write_file", {"path": str(task.tests_dir / HIDDEN_TEST), "content": "pass"}),
        ("write_file", {"path": "../../../pwned.py", "content": "pass"}),
        ("write_file", {"path": "solution/gold.patch", "content": ""}),
        ("write_file", {"path": "task.yaml", "content": "id: x"}),
        ("edit_file", {"path": f"tests/{HIDDEN_TEST}", "old_str": "assert", "new_str": "pass  #"}),
        (
            "replace_lines",
            {
                "path": f"tests/{HIDDEN_TEST}",
                "start_line": 1,
                "end_line": 1,
                "new_lines": ["pass"],
            },
        ),
        ("read_file", {"path": f"tests/{HIDDEN_TEST}"}),
    ]
    for name, args in attacks:
        assert not box.call(name, args).ok, f"{name}({args}) should have been refused"
    assert _digest(task.path) == before
    assert {GATE_TESTS_MODIFIED, GATE_SANDBOX_ESCAPE} <= box.gates
