"""Tests for the line-based source editing tool.

replace_lines offers a more grounded editing interface for smaller coding
models. It replaces an inclusive range of 1-based source lines while retaining
the same sandbox and write-permission guarantees as write_file and edit_file.
"""

from __future__ import annotations

import pytest

from envs.coding_env.tools import (
    GATE_SANDBOX_ESCAPE,
    GATE_TESTS_MODIFIED,
    TOOL_DEFINITIONS,
    Toolbox,
)


ALLOWED = ["repo/pkg/**"]
FORBIDDEN = [
    "tests/**",
    "solution/**",
    "**/conftest.py",
    "task.yaml",
]


@pytest.fixture
def box(tmp_path):
    """A minimal materialised coding sandbox."""
    source_dir = tmp_path / "repo" / "pkg"
    source_dir.mkdir(parents=True)

    (source_dir / "mod.py").write_text(
        "alpha\n"
        "beta\n"
        "gamma\n"
        "delta\n",
        encoding="utf-8",
    )

    (tmp_path / "repo" / "README.md").write_text(
        "read only\n",
        encoding="utf-8",
    )

    (tmp_path / "smoke").mkdir()

    return Toolbox(
        root=tmp_path,
        allowed_write_globs=ALLOWED,
        forbidden_write_globs=FORBIDDEN,
    )


def test_replace_lines_replaces_inclusive_range(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=3,
        new_lines=["new_beta", "new_gamma"],
    )

    assert result.ok

    text = (
        box.root
        / "repo"
        / "pkg"
        / "mod.py"
    ).read_text(encoding="utf-8")

    assert text == (
        "alpha\n"
        "new_beta\n"
        "new_gamma\n"
        "delta\n"
    )

    assert "Replaced lines 2-3" in result.output
    assert "repo/pkg/mod.py" in result.output


def test_replace_lines_replaces_one_line(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=2,
        new_lines=["replacement"],
    )

    assert result.ok

    text = (
        box.root
        / "repo"
        / "pkg"
        / "mod.py"
    ).read_text(encoding="utf-8")

    assert text == (
        "alpha\n"
        "replacement\n"
        "gamma\n"
        "delta\n"
    )


def test_replace_lines_can_delete_lines(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=3,
        new_lines=[],
    )

    assert result.ok

    text = (
        box.root
        / "repo"
        / "pkg"
        / "mod.py"
    ).read_text(encoding="utf-8")

    assert text == (
        "alpha\n"
        "delta\n"
    )


def test_replace_lines_can_insert_multiple_lines(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=2,
        new_lines=["first", "second", "third"],
    )

    assert result.ok

    text = (
        box.root
        / "repo"
        / "pkg"
        / "mod.py"
    ).read_text(encoding="utf-8")

    assert text == (
        "alpha\n"
        "first\n"
        "second\n"
        "third\n"
        "gamma\n"
        "delta\n"
    )


def test_replace_lines_rejects_start_below_one(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=0,
        end_line=1,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "start_line must be at least 1" in result.output


def test_replace_lines_rejects_end_before_start(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=3,
        end_line=2,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "end_line=2 is before start_line=3" in result.output


def test_replace_lines_rejects_range_past_end(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=3,
        end_line=99,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "has only 4 lines" in result.output

    text = (
        box.root
        / "repo"
        / "pkg"
        / "mod.py"
    ).read_text(encoding="utf-8")

    assert text == (
        "alpha\n"
        "beta\n"
        "gamma\n"
        "delta\n"
    )


def test_replace_lines_rejects_more_than_12_source_lines(
    box,
):
    path = box.root / "repo" / "pkg" / "large.py"

    path.write_text(
        "\n".join(
            f"line_{number}"
            for number in range(1, 31)
        )
        + "\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/large.py",
        start_line=1,
        end_line=13,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "at most 12 lines" in result.output


def test_replace_lines_rejects_non_integer_line_numbers(
    box,
):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line="one",
        end_line=2,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "start_line and end_line must be integers" in (
        result.output
    )


def test_replace_lines_rejects_non_string_line(
    box,
):
    path = box.root / "repo" / "pkg" / "mod.py"
    before = path.read_bytes()

    result = box.call(
        "replace_lines",
        {
            "path": "repo/pkg/mod.py",
            "start_line": 1,
            "end_line": 1,
            "new_lines": [123],
        },
    )

    assert not result.ok
    assert "new_lines must contain only strings" in result.output
    assert path.read_bytes() == before


def test_replace_lines_rejects_non_list_new_lines(box):
    result = box.call(
        "replace_lines",
        {
            "path": "repo/pkg/mod.py",
            "start_line": 1,
            "end_line": 1,
            "new_lines": "replacement",
        },
    )

    assert not result.ok
    assert "new_lines must be a list of strings" in result.output


def test_replace_lines_accepts_legacy_new_str(box):
    path = box.root / "repo" / "pkg" / "legacy.py"
    path.write_text(
        "def value():\n"
        "    return 1\n"
        "gamma\n"
        "delta\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/legacy.py",
        start_line=1,
        end_line=2,
        new_str="def changed():\n    return 2",
    )

    assert result.ok

    assert path.read_text(encoding="utf-8") == (
        "def changed():\n"
        "    return 2\n"
        "gamma\n"
        "delta\n"
    )


def test_replace_lines_rejects_both_replacement_arguments(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=1,
        end_line=1,
        new_str="legacy",
        new_lines=["structured"],
    )

    assert not result.ok
    assert "provide either new_str or new_lines" in result.output


def test_replace_lines_requires_replacement_arguments(box):
    result = box.call(
        "replace_lines",
        {
            "path": "repo/pkg/mod.py",
            "start_line": 1,
            "end_line": 1,
        },
    )

    assert not result.ok
    assert "provide new_lines" in result.output


def test_replace_lines_tool_schema_uses_array_of_strings():
    tool = next(
        tool
        for tool in TOOL_DEFINITIONS
        if tool["name"] == "replace_lines"
    )
    new_lines_schema = tool["parameters"]["properties"]["new_lines"]

    assert new_lines_schema["type"] == "array"
    assert new_lines_schema["items"] == {"type": "string"}
    assert "new_lines" in tool["parameters"]["required"]
    assert "new_str" not in tool["parameters"]["properties"]


def test_replace_lines_rejects_missing_file(box):
    result = box.replace_lines(
        path="repo/pkg/missing.py",
        start_line=1,
        end_line=1,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "no such file" in result.output


def test_replace_lines_rejects_directory(box):
    result = box.replace_lines(
        path="repo/pkg",
        start_line=1,
        end_line=1,
        new_lines=["replacement"],
    )

    assert not result.ok
    assert "no such file" in result.output


def test_replace_lines_honours_allowed_write_globs(box):
    result = box.replace_lines(
        path="repo/README.md",
        start_line=1,
        end_line=1,
        new_lines=["changed"],
    )

    assert not result.ok
    assert "only write to" in result.output

    text = (
        box.root
        / "repo"
        / "README.md"
    ).read_text(encoding="utf-8")

    assert text == "read only\n"
    assert not box.gates


def test_replace_lines_forbidden_path_raises_gate(
    box,
):
    tests_dir = box.root / "tests"
    tests_dir.mkdir()

    hidden_test = tests_dir / "test_secret.py"
    hidden_test.write_text(
        "assert False\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="tests/test_secret.py",
        start_line=1,
        end_line=1,
        new_lines=["assert True"],
    )

    assert not result.ok
    assert GATE_TESTS_MODIFIED in box.gates
    assert hidden_test.read_text(
        encoding="utf-8"
    ) == "assert False\n"


def test_replace_lines_cannot_escape_sandbox(box):
    escaped = box.root.parent / f"{box.root.name}-escaped.py"
    escaped.write_text(
        "original\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="../escaped.py",
        start_line=1,
        end_line=1,
        new_lines=["changed"],
    )

    assert not result.ok
    assert GATE_SANDBOX_ESCAPE in box.gates
    assert escaped.read_text(
        encoding="utf-8"
    ) == "original\n"


def test_replace_lines_dispatches_and_logs(box):
    result = box.call(
        "replace_lines",
        {
            "path": "repo/pkg/mod.py",
            "start_line": 2,
            "end_line": 2,
            "new_lines": ["replacement"],
        },
    )

    assert result.ok
    assert box.log[-1].name == "replace_lines"
    assert box.log[-1].ok


def test_replace_lines_returns_updated_preview(box):
    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=3,
        new_lines=["new_beta", "new_gamma"],
    )

    assert result.ok
    assert "Updated preview:" in result.output
    assert "1 | alpha" in result.output
    assert "2 | new_beta" in result.output
    assert "3 | new_gamma" in result.output
    assert "4 | delta" in result.output


def test_replace_lines_accepts_valid_python(box):
    path = box.root / "repo" / "pkg" / "valid.py"

    path.write_text(
        "def value():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/valid.py",
        start_line=2,
        end_line=2,
        new_lines=["    return 2"],
    )

    assert result.ok

    assert path.read_text(encoding="utf-8") == (
        "def value():\n"
        "    return 2\n"
    )


def test_replace_lines_rejects_invalid_python(box):
    path = box.root / "repo" / "pkg" / "invalid.py"

    original = (
        "def value():\n"
        "    return 1\n"
    )

    path.write_text(
        original,
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/invalid.py",
        start_line=1,
        end_line=1,
        new_lines=["    broken indentation"],
    )

    assert not result.ok
    assert "invalid Python" in result.output
    assert "No changes were written" in result.output

    assert path.read_text(encoding="utf-8") == original


def test_invalid_python_preserves_original_bytes(box):
    path = box.root / "repo" / "pkg" / "preserved.py"

    original = (
        "import os\n"
        "\n"
        "def value():\n"
        "    return os.name\n"
    )

    path.write_text(
        original,
        encoding="utf-8",
    )

    before = path.read_bytes()

    result = box.replace_lines(
        path="repo/pkg/preserved.py",
        start_line=1,
        end_line=2,
        new_lines=["def broken("],
    )

    assert not result.ok
    assert path.read_bytes() == before


def test_python_syntax_error_reports_line_number(box):
    path = box.root / "repo" / "pkg" / "line_error.py"

    original = (
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )

    path.write_text(
        original,
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/line_error.py",
        start_line=4,
        end_line=4,
        new_lines=["def second("],
    )

    assert not result.ok
    assert "line" in result.output
    assert "No changes were written" in result.output
    assert path.read_text(encoding="utf-8") == original


def test_replace_lines_does_not_parse_non_python_text(box):
    path = box.root / "repo" / "pkg" / "notes.txt"

    path.write_text(
        "alpha\n"
        "beta\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/notes.txt",
        start_line=2,
        end_line=2,
        new_lines=["this is not Python: !!!"],
    )

    assert result.ok

    assert path.read_text(encoding="utf-8") == (
        "alpha\n"
        "this is not Python: !!!\n"
    )


def test_replace_lines_accepts_structured_new_lines(box):
    path = box.root / "repo" / "pkg" / "mod.py"
    path.write_text(
        "alpha\n"
        "beta\n",
        encoding="utf-8",
    )

    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=1,
        end_line=2,
        new_lines=[
            "def changed():",
            "    return 2",
        ],
    )

    assert result.ok

    assert path.read_text(encoding="utf-8") == (
        "def changed():\n"
        "    return 2\n"
    )


def test_replace_lines_rejects_no_op_replacement(box):
    path = box.root / "repo" / "pkg" / "mod.py"
    before = path.read_bytes()

    result = box.replace_lines(
        path="repo/pkg/mod.py",
        start_line=2,
        end_line=2,
        new_lines=[
            "beta",
        ],
    )

    assert not result.ok
    assert "makes no changes" in result.output
    assert path.read_bytes() == before
