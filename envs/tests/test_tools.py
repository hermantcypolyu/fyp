"""Tests for the six agent tools (spec 5.5).

Two things are being pinned down here. The obvious one is that each tool does what it
says. The important one is the *refusal* behaviour: every path that a reward-hacking or
confused policy could take must come back as one short sentence, and the two paths that
are reward gates - writing to a forbidden glob, writing outside the sandbox - must also
raise their gate. A tool that silently succeeded where it should have refused would not
show up as a test failure anywhere else in the project; it would show up as a suspiciously
high reward.
"""

from __future__ import annotations

import pytest

from envs.coding_env.tools import (
    BUDGET_SHORT,
    CHARS_PER_TOKEN,
    GATE_SANDBOX_ESCAPE,
    GATE_TESTS_MODIFIED,
    MAX_READ_LINES,
    TOOL_DEFINITIONS,
    Toolbox,
    truncate,
)

ALLOWED = ["repo/pkg/**"]
FORBIDDEN = ["tests/**", "solution/**", "**/conftest.py", "task.yaml"]


@pytest.fixture
def box(tmp_path):
    """A sandbox shaped like a real materialised task, minus the hidden tests."""
    (tmp_path / "repo" / "pkg").mkdir(parents=True)
    (tmp_path / "repo" / "pkg" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "repo" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "repo" / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
    (tmp_path / "repo" / "README.md").write_text("hi\n", encoding="utf-8")
    (tmp_path / "smoke").mkdir()
    return Toolbox(root=tmp_path, allowed_write_globs=ALLOWED, forbidden_write_globs=FORBIDDEN)


# --------------------------------------------------------------------------- truncation


def test_truncate_keeps_head_and_tail():
    text = "A" * 100 + "B" * 5000 + "Z" * 100
    out = truncate(text, 100)
    assert out.startswith("AAA")
    assert out.endswith("ZZZ")
    assert "truncated" in out
    assert len(out) < len(text)


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_every_error_fits_the_short_budget(box):
    """Errors are one sentence by construction - a long one would crowd out the code."""
    for result in [
        box.read_file("nope.py"),
        box.edit_file("repo/pkg/mod.py", "absent", "x"),
        box.write_file("tests/test_x.py", "x"),
        box.list_files("../.."),
    ]:
        assert not result.ok
        assert len(result.output) <= BUDGET_SHORT * CHARS_PER_TOKEN


# --------------------------------------------------------------------------- list_files


def test_list_files_is_recursive_and_sandbox_relative(box):
    out = box.list_files().output
    assert "repo/pkg/mod.py" in out
    assert "repo/README.md" in out
    assert "repo/pkg/" in out


def test_list_files_hides_pycache(box):
    assert "__pycache__" not in box.list_files().output


def test_list_files_rejects_absolute_paths(box):
    assert not box.list_files(str(box.root)).ok


def test_list_files_missing_path(box):
    result = box.list_files("repo/nowhere")
    assert not result.ok
    assert "no such path" in result.output


# --------------------------------------------------------------------------- read_file


def test_read_file_numbers_lines(box):
    out = box.read_file("repo/pkg/mod.py").output
    assert "1 | def f():" in out
    assert "lines 1-2 of 2" in out


def test_read_file_slices_inclusively(box):
    (box.root / "repo" / "pkg" / "big.py").write_text("\n".join(f"line{n}" for n in range(1, 21)), encoding="utf-8")
    out = box.read_file("repo/pkg/big.py", start=5, end=7).output
    assert "line5" in out and "line7" in out
    assert "line4" not in out and "line8" not in out


def test_read_file_caps_at_max_lines(box):
    (box.root / "repo" / "pkg" / "huge.py").write_text("\n".join(["x = 1"] * 500), encoding="utf-8")
    out = box.read_file("repo/pkg/huge.py").output
    assert f"lines 1-{MAX_READ_LINES} of 500" in out
    assert f"continue with start={MAX_READ_LINES + 1}" in out


def test_read_file_start_past_end_of_file(box):
    result = box.read_file("repo/pkg/mod.py", start=99)
    assert not result.ok
    assert "only 2 lines" in result.output


def test_read_file_on_a_directory_points_at_list_files(box):
    result = box.read_file("repo/pkg")
    assert not result.ok
    assert "list_files" in result.output


# --------------------------------------------------------------------------- write_file


def test_write_file_inside_the_allow_list(box):
    assert box.write_file("repo/pkg/new.py", "x = 1\n").ok
    assert (box.root / "repo" / "pkg" / "new.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not box.gates


def test_write_file_creates_parent_directories(box):
    assert box.write_file("repo/pkg/sub/deep.py", "x = 1\n").ok
    assert (box.root / "repo" / "pkg" / "sub" / "deep.py").is_file()


def test_write_file_to_a_forbidden_glob_raises_the_gate(box):
    """The #1 reward hack: rewrite the tests instead of the code."""
    (box.root / "tests").mkdir()
    result = box.write_file("tests/test_secret.py", "assert True")
    assert not result.ok
    assert GATE_TESTS_MODIFIED in box.gates
    assert not (box.root / "tests" / "test_secret.py").exists()


def test_write_file_outside_the_allow_list_is_refused_without_a_gate(box):
    """Writing to repo/README.md is not cheating, just out of scope - refuse, don't zero."""
    result = box.write_file("repo/README.md", "nope")
    assert not result.ok
    assert "only write to" in result.output
    assert not box.gates
    assert (box.root / "repo" / "README.md").read_text(encoding="utf-8") == "hi\n"


def test_write_file_cannot_escape_the_sandbox(box):
    result = box.write_file("../escaped.py", "x = 1")
    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in box.gates
    assert not (box.root.parent / "escaped.py").exists()


def test_deny_list_beats_allow_list(box):
    """`repo/pkg/conftest.py` matches both lists. Deny has to win, or the gate is bypassable."""
    result = box.write_file("repo/pkg/conftest.py", "x = 1")
    assert not result.ok
    assert GATE_TESTS_MODIFIED in box.gates


# --------------------------------------------------------------------------- edit_file


def test_edit_file_replaces_the_unique_match(box):
    assert box.edit_file("repo/pkg/mod.py", "return 1", "return 2").ok
    assert "return 2" in (box.root / "repo" / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_edit_file_reports_zero_matches_by_name(box):
    result = box.edit_file("repo/pkg/mod.py", "return 42", "return 2")
    assert not result.ok
    assert result.output == "Error: No match for old_str in repo/pkg/mod.py. Read the file and copy the text exactly."


def test_edit_file_refuses_an_ambiguous_match(box):
    (box.root / "repo" / "pkg" / "dup.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
    result = box.edit_file("repo/pkg/dup.py", "= 1", "= 2")
    assert not result.ok
    assert "matches 2 times" in result.output
    assert (box.root / "repo" / "pkg" / "dup.py").read_text(encoding="utf-8") == "a = 1\nb = 1\n"


def test_edit_file_honours_the_write_globs(box):
    (box.root / "tests").mkdir()
    (box.root / "tests" / "test_x.py").write_text("assert False\n", encoding="utf-8")
    assert not box.edit_file("tests/test_x.py", "False", "True").ok
    assert GATE_TESTS_MODIFIED in box.gates
    assert "False" in (box.root / "tests" / "test_x.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- run_tests


def test_run_tests_refuses_if_hidden_tests_are_on_disk(box):
    """A `tests/` directory in the agent's sandbox means the sandbox was built wrong."""
    (box.root / "tests").mkdir()
    result = box.run_tests()
    assert not result.ok
    assert "misbuilt" in result.output


def test_run_tests_without_smoke_tests_says_so(box):
    result = box.run_tests()
    assert not result.ok
    assert "no smoke tests" in result.output


@pytest.mark.slow
def test_run_tests_runs_the_smoke_suite(box):
    (box.root / "smoke" / "test_smoke.py").write_text(
        "from pkg.mod import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )
    result = box.run_tests()
    assert result.ok
    assert "1/1 passed" in result.output
    assert "not the grading tests" in result.output


@pytest.mark.slow
def test_run_tests_shows_the_failure_tail(box):
    (box.root / "smoke" / "test_smoke.py").write_text(
        "from pkg.mod import f\n\n\ndef test_f():\n    assert f() == 99\n", encoding="utf-8"
    )
    result = box.run_tests()
    assert result.ok  # the tool worked; the tests did not
    assert "0/1 passed" in result.output


# --------------------------------------------------------------------------- dispatch


def test_submit_ends_the_episode(box):
    assert not box.submitted
    assert box.submit().ok
    assert box.submitted


def test_call_dispatches_and_logs(box):
    box.call("read_file", {"path": "repo/pkg/mod.py"})
    assert [c.name for c in box.log] == ["read_file"]
    assert box.log[0].ok


def test_call_rejects_an_unknown_tool(box):
    result = box.call("bash", {"command": "rm -rf /"})
    assert not result.ok
    assert "unknown tool" in result.output


def test_call_survives_bad_arguments(box):
    result = box.call("read_file", {"file_path": "repo/pkg/mod.py"})
    assert not result.ok
    assert "bad arguments" in result.output


def test_call_enforces_the_tool_budget(box):
    box.max_tool_calls = 2
    box.call("list_files")
    box.call("list_files")
    result = box.call("list_files")
    assert not result.ok
    assert "submit()" in result.output


# --------------------------------------------------------------------------- schemas


def test_schema_matches_the_implementation():
    """Exactly six tools, and every advertised name is really callable."""
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert names == ["list_files", "read_file", "write_file", "edit_file", "run_tests", "submit"]
    assert all(hasattr(Toolbox, n) for n in names)
