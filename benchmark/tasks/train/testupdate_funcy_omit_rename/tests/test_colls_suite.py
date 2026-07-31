"""Hidden verification for `testupdate_funcy_omit_rename`.

Upstream renamed a public helper and updated its own suite in one commit
(Suor/funcy @ a4c0a262419788a550cb854c0f5e5ee1dedf1604, "Rename throw() to omit()"). The
library here is taken from *after* that commit and the project's suite from *before*, so
the code is right and the suite is stale.

Two things are graded, and they pull against each other on purpose:

1. **The project's suite must pass.** Easy to achieve badly - delete the offending test.
2. **It must still be worth having.** So the surviving test count is checked, and the
   updated source must actually exercise the new name and stop mentioning the old one.

Separately, the library's own contract is re-asserted here. That blocks the other cheap
answer: re-adding the removed name to `funcy/colls.py` so the stale suite passes
untouched. `test_the_old_name_is_really_gone` fails if the agent does that, and
`repo/funcy/**` is outside `allowed_write_globs` anyway.

The project's suite is run in a *subprocess*, not imported, so the agent is free to use
fixtures, parametrisation or any other ordinary pytest feature. The surviving test count
comes from JUnit XML rather than from scraping terminal output, which varies by version.
"""

import ast
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pytest

from funcy import colls

REPO = Path.cwd()
AGENT_SUITE = REPO / "tests" / "test_colls.py"

#: The stale suite ships with this many tests. Updating it must not shrink it.
MIN_TESTS = 23


@pytest.fixture(scope="module")
def suite_run() -> tuple[int, str, int]:
    """Run the project's own suite once; return (returncode, output, tests collected)."""
    if not AGENT_SUITE.is_file():
        pytest.fail("tests/test_colls.py is gone - the task was to update the suite, not delete it")
    with tempfile.TemporaryDirectory() as tmp:
        ini = Path(tmp) / "inner.ini"
        ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
        junit = Path(tmp) / "inner.xml"
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable, "-m", "pytest", "-c", str(ini),
                "-p", "no:cacheprovider", "--tb=short", "-q", "--no-header",
                f"--junit-xml={junit}", str(AGENT_SUITE),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        n_tests = len(ET.parse(junit).getroot().findall(".//testcase")) if junit.is_file() else 0
    return proc.returncode, proc.stdout + proc.stderr, n_tests


@pytest.fixture(scope="module")
def suite_source() -> str:
    if not AGENT_SUITE.is_file():
        pytest.fail("tests/test_colls.py is gone - the task was to update the suite, not delete it")
    return AGENT_SUITE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def suite_tree(suite_source) -> ast.Module:
    try:
        return ast.parse(suite_source)
    except SyntaxError as exc:
        pytest.fail(f"tests/test_colls.py does not parse: {exc}")


# --- the suite must pass, without being gutted -----------------------------------------


def test_project_test_suite_passes(suite_run):
    returncode, output, _ = suite_run
    assert returncode == 0, f"The project's own suite still fails.\n--- pytest output ---\n{output[-2500:]}"


def test_project_test_suite_still_has_enough_tests(suite_run):
    _, _, n_tests = suite_run
    assert n_tests >= MIN_TESTS, (
        f"only {n_tests} test(s) left in tests/test_colls.py; at least {MIN_TESTS} are expected. "
        "Update the stale test rather than removing it."
    )


def test_the_new_name_is_exercised(suite_source):
    assert "omit(" in suite_source, "no test calls the renamed helper"


def test_the_old_name_is_no_longer_referenced(suite_source):
    assert "throw" not in suite_source, "the suite still refers to the removed name"


def test_the_renamed_helper_still_has_its_own_test(suite_tree):
    names = [n.name for n in ast.walk(suite_tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    assert any("omit" in name for name in names), (
        f"no test function is named for the renamed helper; found {sorted(names)[:8]}..."
    )


def test_the_renamed_helpers_test_still_covers_type_preservation(suite_tree):
    """The stale test made two assertions: a plain dict and a defaultdict. Both must survive
    the rename, because preserving the collection type is the interesting half."""
    for node in ast.walk(suite_tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and "omit" in node.name:
            body = ast.unparse(node)
            assert body.count("omit(") >= 2, "the renamed test dropped one of its two cases"
            assert "defaultdict" in body, "the renamed test no longer checks that the collection type is preserved"
            return
    pytest.fail("no test function is named for the renamed helper")


# --- and the library must not have been dragged backwards -------------------------------


def test_the_library_still_provides_the_new_name():
    assert colls.omit({"a": 1, "b": 2, "c": 3}, "ac") == {"b": 2}


def test_the_library_still_preserves_the_collection_type():
    dd = defaultdict(int, {"a": 1, "b": 2, "c": 3})
    result = colls.omit(dd, "ac")
    assert type(result) is defaultdict
    assert result == defaultdict(int, {"b": 2})
    assert result.default_factory == dd.default_factory


def test_the_old_name_is_really_gone():
    assert not hasattr(colls, "throw"), (
        "funcy/colls.py has had the removed name put back. The library is correct as shipped; "
        "the suite is what needed updating."
    )


def test_the_sibling_helpers_are_untouched():
    assert colls.project({"a": 1, "b": 2, "c": 3}, "ac") == {"a": 1, "c": 3}
    assert colls.compact({"a": None, "b": 0, "c": 1}) == {"c": 1}
    assert colls.flip({"a": 1, "b": 2}) == {1: "a", 2: "b"}
