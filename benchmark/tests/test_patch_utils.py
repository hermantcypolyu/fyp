"""Unit tests for the unified-diff engine.

We do not shell out to `git apply`: the benchmark has to behave identically on a Windows
laptop with no git on PATH and on a Linux training node. That means we own the diff code,
which means it needs tests of its own - a silently wrong patch applier would corrupt every
gold patch in the corpus at once, and `verify_gold_patches.py` would happily report the
damage as a legitimate task failure.
"""

from __future__ import annotations

import pytest

from benchmark.tools.patch_utils import (
    PatchError,
    apply_patch,
    make_patch,
    parse_unified_diff,
    patched_files,
)

BEFORE = """\
def total_pages(total_items, per_page):
    return total_items // per_page


def page_of(items, page, per_page):
    start = (page - 1) * per_page
    return items[start:start + per_page]
"""

AFTER = """\
def total_pages(total_items, per_page):
    if total_items == 0:
        return 1
    return (total_items + per_page - 1) // per_page


def page_of(items, page, per_page):
    start = (page - 1) * per_page
    return items[start:start + per_page]
"""


def _tree(tmp_path, **files):
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so the literal line endings in `content` reach the disk untouched -
        # the default would translate "\n" to "\r\n" on Windows and turn the CRLF fixture
        # below into doubled carriage returns.
        path.write_text(content, encoding="utf-8", newline="")
    return tmp_path


def _patch(tmp_path, rel, before, after):
    a = _tree(tmp_path / "a", **{rel: before})
    b = _tree(tmp_path / "b", **{rel: after})
    return make_patch(a, b)


def test_roundtrip_apply(tmp_path):
    patch = _patch(tmp_path, "repo/src/pagination.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"repo/src/pagination.py": BEFORE})

    changed = apply_patch(root, patch)

    assert changed == ["repo/src/pagination.py"]
    assert (root / "repo/src/pagination.py").read_text(encoding="utf-8") == AFTER


def test_reverse_undoes_the_patch(tmp_path):
    patch = _patch(tmp_path, "repo/src/pagination.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"repo/src/pagination.py": AFTER})

    apply_patch(root, patch, reverse=True)

    assert (root / "repo/src/pagination.py").read_text(encoding="utf-8") == BEFORE


def test_dry_run_leaves_the_tree_untouched(tmp_path):
    patch = _patch(tmp_path, "repo/src/pagination.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"repo/src/pagination.py": BEFORE})

    changed = apply_patch(root, patch, dry_run=True)

    assert changed == ["repo/src/pagination.py"]
    assert (root / "repo/src/pagination.py").read_text(encoding="utf-8") == BEFORE


def test_trailing_newline_does_not_leak_a_phantom_context_line(tmp_path):
    """Regression: `text.split("\\n")` on a newline-terminated patch yields a trailing
    empty string. The parser used to absorb that as one *extra* blank context line on the
    final hunk, so any patch whose last hunk ran to the end of the file failed to apply
    with a bogus "context not found" error. Hunks now stop at the line counts declared in
    the `@@` header, so no hunk can ever be over-filled.

    The patch below has two hunks precisely so the last one is not also the first.
    """
    before = "\n".join(["h1", "a", "b", "c"] + [f"pad{i}" for i in range(12)] + ["x", "y", "z"]) + "\n"
    after = before.replace("b\n", "B\n").replace("y\n", "Y\n")

    patch = _patch(tmp_path, "f.py", before, after)
    assert patch.endswith("\n")

    parsed = parse_unified_diff(patch)
    hunks = parsed[0].hunks
    assert len(hunks) == 2, "fixture should produce two separate hunks"
    for hunk in hunks:
        assert len(hunk.old_block) == hunk.old_count
        assert len(hunk.new_block) == hunk.new_count

    root = _tree(tmp_path / "work", **{"f.py": before})
    apply_patch(root, patch)
    assert (root / "f.py").read_text(encoding="utf-8") == after


def test_crlf_input_is_normalised(tmp_path):
    """PowerShell writes CRLF. A patch generated from LF sources must still apply."""
    patch = _patch(tmp_path, "f.py", "a\nb\nc\n", "a\nB\nc\n")
    root = _tree(tmp_path / "work", **{"f.py": "a\r\nb\r\nc\r\n"})

    apply_patch(root, patch)

    assert (root / "f.py").read_text(encoding="utf-8").replace("\r\n", "\n") == "a\nB\nc\n"


def test_shifted_context_still_applies(tmp_path):
    """Hunks locate themselves by content, not by absolute line number, so inserting
    unrelated lines above the change does not break the patch."""
    patch = _patch(tmp_path, "f.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"f.py": "# a new header comment\n# and another\n" + BEFORE})

    apply_patch(root, patch)

    assert (root / "f.py").read_text(encoding="utf-8").endswith(AFTER)


def test_context_mismatch_is_a_clear_error(tmp_path):
    patch = _patch(tmp_path, "f.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"f.py": "something else entirely\n"})

    with pytest.raises(PatchError) as exc:
        apply_patch(root, patch)
    assert "context" in str(exc.value).lower()


def test_missing_file_is_a_clear_error(tmp_path):
    patch = _patch(tmp_path, "f.py", BEFORE, AFTER)
    root = _tree(tmp_path / "work", **{"other.py": BEFORE})

    with pytest.raises(PatchError):
        apply_patch(root, patch)


def test_multi_file_patch(tmp_path):
    a = _tree(tmp_path / "a", **{"one.py": "x = 1\n", "two.py": "y = 1\n"})
    b = _tree(tmp_path / "b", **{"one.py": "x = 2\n", "two.py": "y = 2\n"})
    patch = make_patch(a, b)

    assert patched_files(patch) == ["one.py", "two.py"]

    root = _tree(tmp_path / "work", **{"one.py": "x = 1\n", "two.py": "y = 1\n"})
    assert apply_patch(root, patch) == ["one.py", "two.py"]
    assert (root / "one.py").read_text(encoding="utf-8") == "x = 2\n"
    assert (root / "two.py").read_text(encoding="utf-8") == "y = 2\n"


def test_identical_trees_produce_an_empty_patch(tmp_path):
    a = _tree(tmp_path / "a", **{"one.py": "x = 1\n"})
    b = _tree(tmp_path / "b", **{"one.py": "x = 1\n"})
    assert make_patch(a, b).strip() == ""


def test_non_python_files_are_patched_too(tmp_path):
    """config_fix tasks turn on editing pyproject.toml, so the differ must not be
    restricted to *.py."""
    patch = _patch(tmp_path, "pyproject.toml", 'version = "0.3.0"\n', 'version = "0.4.1"\n')
    root = _tree(tmp_path / "work", **{"pyproject.toml": 'version = "0.3.0"\n'})

    apply_patch(root, patch)

    assert (root / "pyproject.toml").read_text(encoding="utf-8") == 'version = "0.4.1"\n'
