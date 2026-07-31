"""Hidden verification for `configfix_src_layout_migration`.

The three library modules are correct; they are loaded straight off disk here, by path,
so that they can be exercised without going through the package's `__init__`. Everything
else graded here is the migration: where the packaging metadata says the package lives,
where the test configuration says to look for it, what name the distribution claims, and
whether any module still reaches for the pre-move top-level package.

The package import is performed the way `pytest` would perform it - by honouring the
`pythonpath` the project declares for itself - so a wrong path and a stale import are
graded as the separate faults they are.
"""

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path.cwd()
PYPROJECT = REPO / "pyproject.toml"

IMPORT_NAME = "dataspec"
PACKAGE_DIR = Path("src") / IMPORT_NAME
MODULES = ("fields.py", "rules.py", "validate.py", "__init__.py")
PRE_MOVE_PACKAGE = "spectool"


def load_config() -> dict:
    if not PYPROJECT.is_file():
        pytest.fail("pyproject.toml is missing from the project root")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def declared_pythonpath() -> list[Path]:
    ini = load_config().get("tool", {}).get("pytest", {}).get("ini_options", {})
    return [(REPO / entry).resolve() for entry in ini.get("pythonpath", [])]


def load_by_path(filename: str):
    """Load one library module directly off disk, bypassing the package."""
    path = REPO / PACKAGE_DIR / filename
    if not path.is_file():
        pytest.fail(f"{PACKAGE_DIR.as_posix()}/{filename} is not on disk")
    spec = importlib.util.spec_from_file_location(f"_under_test_{filename[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_package():
    """Import the package using only the import path the project declares for itself."""
    for name in [n for n in sys.modules if n == IMPORT_NAME or n.startswith(IMPORT_NAME + ".")]:
        del sys.modules[name]
    saved = list(sys.path)
    sys.path[:0] = [str(entry) for entry in declared_pythonpath()]
    try:
        return importlib.import_module(IMPORT_NAME)
    except ImportError as exc:
        pytest.fail(
            f"`import {IMPORT_NAME}` fails with the project's declared pythonpath "
            f"{[str(p) for p in declared_pythonpath()]}: {exc}"
        )
    finally:
        sys.path[:] = saved


def normalise(name: str) -> str:
    """PEP 503 normalisation."""
    out = []
    previous_separator = False
    for char in name.lower():
        if char in "-_.":
            if not previous_separator:
                out.append("-")
            previous_separator = True
        else:
            out.append(char)
            previous_separator = False
    return "".join(out)


# --- what the move already got right ----------------------------------------------------

def test_pyproject_is_valid_toml():
    try:
        config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"pyproject.toml is not valid TOML: {exc}")
    assert config["project"]["version"] == "2.0.0"


def test_the_package_has_moved_into_src():
    for filename in MODULES:
        assert (REPO / PACKAGE_DIR / filename).is_file(), (
            f"{(PACKAGE_DIR / filename).as_posix()} is missing"
        )
    assert not (REPO / IMPORT_NAME).exists(), "a flat-layout copy of the package is still on disk"


def test_every_module_is_syntactically_valid():
    for path in sorted((REPO / PACKAGE_DIR).rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"{path.name} does not parse: {exc}")


def test_field_specs_parse():
    fields = load_by_path("fields.py")

    schema = fields.parse_schema(["name:str!", "age:int"])
    assert [(f.name, f.kind, f.required) for f in schema] == [
        ("name", "str", True),
        ("age", "int", False),
    ]
    with pytest.raises(ValueError):
        fields.parse_field("name")


def test_rules_return_none_or_a_message():
    rules = load_by_path("rules.py")

    assert rules.min_length(3)("abc") is None
    assert rules.min_length(3)("ab") == "is shorter than 3"
    assert rules.one_of("red", "blue")("green") == "is not one of red, blue"
    assert rules.matches(r"\d+")("2024") is None


def test_the_build_backend_is_declared():
    config = load_config()
    assert config["build-system"]["build-backend"] == "hatchling.build"
    assert config["build-system"]["requires"]


# --- what the move left behind ----------------------------------------------------------

def test_declared_pythonpath_reaches_the_package():
    entries = declared_pythonpath()
    assert entries, "[tool.pytest.ini_options] declares no pythonpath"
    reachable = [entry for entry in entries if (entry / IMPORT_NAME / "__init__.py").is_file()]
    assert reachable, (
        f"none of the declared pythonpath entries {[str(e) for e in entries]} contains "
        f"{IMPORT_NAME}/__init__.py; pytest would collect nothing but import errors"
    )


def test_no_module_reaches_for_the_pre_move_package():
    offenders = []
    for path in sorted((REPO / PACKAGE_DIR).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root == PRE_MOVE_PACKAGE:
                    offenders.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == PRE_MOVE_PACKAGE:
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, f"still importing the pre-move package: {offenders}"


def test_package_imports():
    package = import_package()
    assert package.__version__ == "2.0.0"


def test_public_api_is_re_exported():
    package = import_package()
    expected = {"Field", "matches", "min_length", "one_of", "parse_field", "parse_schema", "validate"}
    missing = sorted(name for name in expected if not hasattr(package, name))
    assert not missing, f"the package does not re-export {missing}"


def test_validate_accepts_a_good_record():
    package = import_package()
    schema = package.parse_schema(["name:str!", "age:int"])
    assert package.validate({"name": "ada", "age": 36}, schema) == []


def test_validate_reports_every_problem_at_once():
    package = import_package()
    schema = package.parse_schema(["name:str!", "age:int"])
    problems = package.validate(
        {"age": "old", "colour": "red"}, schema, {"name": [package.min_length(2)]}
    )
    assert problems == ["name: missing", "age: expected int, got str", "colour: not in the schema"]


def test_distribution_name_normalises_to_the_import_name():
    name = load_config()["project"]["name"]
    assert normalise(name) == IMPORT_NAME, (
        f"distribution name {name!r} normalises to {normalise(name)!r}, "
        f"but the importable package is {IMPORT_NAME!r}"
    )


def test_wheel_packages_paths_exist_on_disk():
    wheel = load_config().get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    packages = wheel.get("packages")
    assert packages, "no [tool.hatch.build.targets.wheel] packages entry"
    for relative in packages:
        assert (REPO / relative).is_dir(), (
            f"wheel packages names {relative!r}, which is not a directory in the source tree; "
            "the built wheel would be empty"
        )
