"""Hidden verification for `configfix_tallyman_command_paths`.

The library under `tallyman/` is correct and is exercised directly here, so the packaging
metadata can be graded independently of it. Every console script declared in
`[project.scripts]` is resolved exactly the way a generated wrapper resolves it: import the
module half, look up the attribute half, then call it.

Two defects are graded, and they are independent. The entry points still name the
pre-refactor `tallyman.cli` module, which no longer exists; and `[tool.setuptools].packages`
still lists only the top-level package, so the `tallyman.commands` subpackage that now holds
both implementations would be missing from an installed wheel. Fixing either one alone moves
the score, which is the point - an agent that repoints the scripts but leaves the subpackage
unshipped has done most, but not all, of the job.

Each test resolves only what it needs, so one broken entry point takes down the tests for
that entry point and nothing else.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

REPO = Path.cwd()
PYPROJECT = REPO / "pyproject.toml"

LEDGER = """\
# march
food,12.50
transport,3.20
food,7.00
not a row
transport,1.80
"""

#: command name -> the module that README.md says backs it.
DOCUMENTED = {"tally", "tally-report"}


def load_config() -> dict:
    if not PYPROJECT.is_file():
        pytest.fail("pyproject.toml is missing from the project root")
    try:
        return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"pyproject.toml is not valid TOML: {exc}")


def scripts() -> dict:
    return load_config().get("project", {}).get("scripts", {})


def target(command: str) -> tuple[str, str]:
    table = scripts()
    if command not in table:
        pytest.fail(f"[project.scripts] declares no {command!r} command; it declares {sorted(table)}")
    spec = table[command]
    if ":" not in spec:
        pytest.fail(f"entry point for {command!r} is {spec!r}; expected 'module:callable'")
    module_path, _, attribute = spec.partition(":")
    return module_path, attribute


def resolve(command: str):
    module_path, attribute = target(command)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"entry point {command!r} names module {module_path!r}, which cannot be imported: {exc}")
    func = getattr(module, attribute, None)
    if func is None:
        pytest.fail(f"entry point {command!r} names {module_path}:{attribute}, but that module has no {attribute!r}")
    return func


def write_ledger(tmp_path: Path) -> str:
    path = tmp_path / "march.csv"
    path.write_text(LEDGER, encoding="utf-8")
    return str(path)


# --- the library itself, which the agent must not need to touch -------------------------


def test_pyproject_is_valid_and_names_the_project():
    assert load_config()["project"]["name"] == "tallyman"


def test_library_parses_a_ledger_line():
    from tallyman.reader import parse_line

    row = parse_line("food,12.50")
    assert (row.category, row.amount) == ("food", 12.50)
    assert parse_line("not a row") is None
    assert parse_line("# march") is None


def test_library_totals_by_category():
    from tallyman.reader import parse_lines
    from tallyman.stats import totals

    assert totals(parse_lines(LEDGER.splitlines())) == {"food": 19.5, "transport": 5.0}


def test_library_reports_the_leading_category():
    from tallyman.reader import parse_lines
    from tallyman.stats import grand_total, top_category

    rows = list(parse_lines(LEDGER.splitlines()))
    assert top_category(rows) == "food"
    assert grand_total(rows) == pytest.approx(24.5)


def test_command_implementations_run_when_imported_directly(tmp_path, capsys):
    from tallyman.commands import count, report

    assert count.main([write_ledger(tmp_path)]) == 0
    assert "food" in capsys.readouterr().out
    assert report.main([write_ledger(tmp_path)]) == 0
    assert "top: food" in capsys.readouterr().out


def test_library_filters_rows_and_totals():
    from tallyman.filters import apply_filters, by_amount, by_category, drop_below
    from tallyman.reader import parse_lines
    from tallyman.stats import totals

    rows = list(parse_lines(LEDGER.splitlines()))
    assert totals(apply_filters(rows, [by_category("food")])) == {"food": 19.5}
    assert len(apply_filters(rows, [by_amount(minimum=5.0)])) == 2
    assert drop_below(totals(rows), 10.0) == {"food": 19.5}


# --- both documented commands must be declared and must resolve -------------------------


def test_scripts_table_declares_both_documented_commands():
    declared = set(scripts())
    assert declared == DOCUMENTED, (
        f"[project.scripts] declares {sorted(declared)}; README documents {sorted(DOCUMENTED)}"
    )


def test_tally_entry_point_module_is_importable():
    module_path, _ = target("tally")
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`tally` is declared against module {module_path!r}, which does not import: {exc}")


def test_tally_entry_point_attribute_exists():
    assert callable(resolve("tally"))


def test_tally_entry_point_runs_end_to_end(tmp_path, capsys):
    assert resolve("tally")([write_ledger(tmp_path)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 2
    assert printed[0].startswith("food")


def test_tally_report_entry_point_module_is_importable():
    module_path, _ = target("tally-report")
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`tally-report` is declared against module {module_path!r}, which does not import: {exc}")


def test_tally_report_entry_point_attribute_exists():
    assert callable(resolve("tally-report"))


def test_tally_report_entry_point_runs_end_to_end(tmp_path, capsys):
    assert resolve("tally-report")([write_ledger(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "total" in printed
    assert "top: food" in printed


# --- the subpackage holding those commands must actually ship ---------------------------


def test_declared_packages_include_the_commands_subpackage():
    declared = load_config().get("tool", {}).get("setuptools", {}).get("packages")
    assert declared is not None, "[tool.setuptools].packages is missing, so nothing declares what ships"
    assert "tallyman.commands" in declared, (
        f"packages lists {declared}; tallyman/commands/ holds both command modules and would not be "
        "installed, so the console scripts would fail on a real install"
    )
