"""Hidden verification for `configfix_console_entrypoints`.

The library under `logtidy/` is correct and is exercised directly here, so that the
console-script metadata can be graded independently of it. Every entry point declared in
`[project.scripts]` is resolved the way a generated wrapper resolves it: import the
module half, look up the attribute half, then call it.

Each test imports what it needs inside its own body. A broken entry point takes down the
tests that resolve that entry point and nothing else.
"""

import importlib
import inspect
import tomllib
from pathlib import Path

import pytest

REPO = Path.cwd()
PYPROJECT = REPO / "pyproject.toml"

SAMPLE_LOG = """\
2024-03-11T08:15:04 INFO started
banner text that is not a log line
2024-03-11T08:15:07 WARNING disk usage at 91%
2024-03-11T08:16:00 ERROR upload failed
2024-03-11T08:16:01 WARNING retrying
"""

#: command name -> (module, attribute) that README.md documents.
DOCUMENTED = {"logtidy", "logtidy-summary"}


def load_config() -> dict:
    if not PYPROJECT.is_file():
        pytest.fail("pyproject.toml is missing from the project root")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def scripts() -> dict[str, str]:
    return load_config().get("project", {}).get("scripts", {})


def target(command: str) -> tuple[str, str]:
    """The `module:attr` halves of a declared console script."""
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


def write_log(tmp_path: Path) -> str:
    path = tmp_path / "app.log"
    path.write_text(SAMPLE_LOG, encoding="utf-8")
    return str(path)


# --- the project metadata and the library itself ----------------------------------------

def test_pyproject_is_valid_toml():
    try:
        config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"pyproject.toml is not valid TOML: {exc}")
    assert config["project"]["name"] == "logtidy"


def test_scripts_table_declares_both_documented_commands():
    declared = set(scripts())
    assert declared == DOCUMENTED, (
        f"[project.scripts] declares {sorted(declared)}; README documents {sorted(DOCUMENTED)}"
    )


def test_library_parses_a_log_line():
    from logtidy.parser import parse_line

    record = parse_line("2024-03-11T08:15:07 WARNING disk usage at 91%")
    assert record is not None
    assert (record.stamp, record.level, record.message) == (
        "2024-03-11T08:15:07",
        "WARNING",
        "disk usage at 91%",
    )
    assert parse_line("banner text that is not a log line") is None


def test_library_counts_levels():
    from logtidy.parser import parse_lines
    from logtidy.report import count_levels

    counts = count_levels(parse_lines(SAMPLE_LOG.splitlines()))
    assert counts == {"INFO": 1, "WARNING": 2, "ERROR": 1}


def test_cli_function_runs_end_to_end(tmp_path, capsys):
    from logtidy.cli import cli

    assert cli([write_log(tmp_path), "--min-level", "WARNING"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3
    assert printed[0].startswith("2024-03-11T08:15:07 WARNING ")


def test_summarise_function_runs_end_to_end(tmp_path, capsys):
    from logtidy.summarise import main

    assert main([write_log(tmp_path), "--total"]) == 0
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "total" in printed


# --- every declared console script must actually resolve --------------------------------

def test_logtidy_entry_point_module_is_importable():
    module_path, _ = target("logtidy")
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`logtidy` is declared against module {module_path!r}, which does not import: {exc}")


def test_logtidy_entry_point_attribute_exists():
    module_path, attribute = target("logtidy")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`logtidy` names module {module_path!r}, which does not import: {exc}")
    assert hasattr(module, attribute), (
        f"`logtidy` is declared as {module_path}:{attribute}, but {module_path} defines no {attribute!r} "
        f"(it defines {sorted(n for n in vars(module) if not n.startswith('_'))})"
    )


def test_logtidy_entry_point_has_the_documented_signature():
    func = resolve("logtidy")
    assert callable(func), "the `logtidy` entry point does not resolve to something callable"
    params = list(inspect.signature(func).parameters.values())
    assert len(params) == 1, f"expected a single `argv` parameter, got {[p.name for p in params]}"
    assert params[0].name == "argv"
    assert params[0].default is not inspect.Parameter.empty, "`argv` must be optional"


def test_logtidy_entry_point_runs_end_to_end(tmp_path, capsys):
    func = resolve("logtidy")
    assert func([write_log(tmp_path), "--min-level", "ERROR"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == ["2024-03-11T08:16:00 ERROR    upload failed"]


def test_summary_entry_point_module_is_importable():
    module_path, _ = target("logtidy-summary")
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`logtidy-summary` is declared against module {module_path!r}, which does not import: {exc}")


def test_summary_entry_point_attribute_exists():
    module_path, attribute = target("logtidy-summary")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(f"`logtidy-summary` names module {module_path!r}, which does not import: {exc}")
    assert hasattr(module, attribute), (
        f"`logtidy-summary` is declared as {module_path}:{attribute}, but {module_path} defines no {attribute!r}"
    )


def test_summary_entry_point_has_the_documented_signature():
    func = resolve("logtidy-summary")
    assert callable(func), "the `logtidy-summary` entry point does not resolve to something callable"
    params = list(inspect.signature(func).parameters.values())
    assert len(params) == 1, f"expected a single `argv` parameter, got {[p.name for p in params]}"
    assert params[0].name == "argv"
    assert params[0].default is not inspect.Parameter.empty, "`argv` must be optional"


def test_summary_entry_point_runs_end_to_end(tmp_path, capsys):
    func = resolve("logtidy-summary")
    assert func([write_log(tmp_path)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == ["INFO     1", "WARNING  2", "ERROR    1"]
