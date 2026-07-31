"""Hidden verification for `configfix_optional_dependency_markers`.

`reportkit/features.py` is the specification: its `SUPPORTED_EXTRAS` keys are the extra
names the library accepts and prints in install hints. Everything here cross-references
`[project.optional-dependencies]` against that, and checks the two house rules the README
states about markers and about re-pinning core dependencies.

Requirements and markers are picked apart by hand with the standard library only - no
`packaging`, which the sandbox does not have.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path.cwd()
PYPROJECT = REPO / "pyproject.toml"

_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?P<spec>.*)$"
)
_PY_CLAUSE = re.compile(r"""python_version\s*(==|!=|<=|>=|<|>)\s*['"]([0-9]+(?:\.[0-9]+)*)['"]""")

_COMPARE = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}


def load_config() -> dict:
    if not PYPROJECT.is_file():
        pytest.fail("pyproject.toml is missing from the project root")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def extras() -> dict[str, list[str]]:
    return load_config().get("project", {}).get("optional-dependencies", {})


def core_dependencies() -> list[str]:
    return load_config().get("project", {}).get("dependencies", [])


def normalise(name: str) -> str:
    """PEP 503 normalisation, so `PyYAML` and `pyyaml` are one distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(text: str) -> tuple[str, list[str], str]:
    """`name`, its version specifiers, and its environment marker (empty if none)."""
    body, _, marker = text.partition(";")
    match = _REQUIREMENT.match(body.strip())
    if match is None:
        pytest.fail(f"requirement {text!r} does not look like 'name[extras]<specifiers>'")
    spec = match.group("spec").strip()
    specifiers = [part.strip() for part in spec.split(",") if part.strip()]
    return normalise(match.group("name")), specifiers, marker.strip()


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def python_floor() -> tuple[int, ...]:
    spec = load_config().get("project", {}).get("requires-python", "")
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", spec)
    if match is None:
        pytest.fail(f"requires-python {spec!r} has no '>=<major>.<minor>' lower bound")
    return version_tuple(match.group(1))


def unsatisfiable_clauses(marker: str, interpreter: tuple[int, ...]) -> list[str]:
    """The `python_version` clauses of `marker` that are false on `interpreter`."""
    dead = []
    for operator, literal in _PY_CLAUSE.findall(marker):
        if not _COMPARE[operator](interpreter, version_tuple(literal)):
            dead.append(f"python_version {operator} {literal!r}")
    return dead


def conflicting(specifiers: list[str]) -> bool:
    """Whether a set of range specifiers leaves no version that satisfies all of them."""
    lower = upper = None
    lower_open = upper_open = False
    for specifier in specifiers:
        match = re.match(r"(>=|>|<=|<)\s*([0-9]+(?:\.[0-9]+)*)", specifier)
        if match is None:
            continue
        operator, version = match.group(1), version_tuple(match.group(2))
        if operator in (">=", ">") and (lower is None or version >= lower):
            lower, lower_open = version, operator == ">"
        if operator in ("<=", "<") and (upper is None or version <= upper):
            upper, upper_open = version, operator == "<"
    if lower is None or upper is None:
        return False
    return lower > upper or (lower == upper and (lower_open or upper_open))


# --- what the declarations already get right --------------------------------------------

def test_pyproject_is_valid_toml():
    try:
        config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"pyproject.toml is not valid TOML: {exc}")
    assert config["project"]["name"] == "reportkit"


def test_core_dependencies_are_named_and_pinned():
    deps = core_dependencies()
    assert deps, "[project] declares no core dependencies"
    for requirement in deps:
        name, specifiers, _ = split_requirement(requirement)
        assert name, f"{requirement!r} has no distribution name"
        assert specifiers, f"core dependency {requirement!r} carries no version specifier"


def test_optional_dependencies_table_is_well_formed():
    table = extras()
    assert table, "[project.optional-dependencies] is missing or empty"
    for extra, requirements in table.items():
        assert isinstance(requirements, list) and requirements, f"extra {extra!r} declares nothing"
        assert extra == extra.lower(), f"extra name {extra!r} is not lower case"


def test_available_only_accepts_extras_the_library_knows():
    from reportkit import features

    for extra in features.SUPPORTED_EXTRAS:
        assert isinstance(features.available(extra), bool)
    with pytest.raises(ValueError):
        features.available("definitely-not-an-extra")


def test_install_hint_is_the_command_a_user_should_run():
    from reportkit import features

    for extra in features.SUPPORTED_EXTRAS:
        assert features.install_hint(extra) == f"pip install reportkit[{extra}]"


def test_plain_text_report_renders_without_any_extra():
    from reportkit import render_report

    text = render_report("Weekly", ["job", "runs"], [["build", "41"], ["deploy", "7"]])
    assert text.splitlines() == [
        "Weekly",
        "======",
        "",
        "job     runs",
        "------  ----",
        "build   41",
        "deploy  7",
    ]


# --- what the declarations get wrong ----------------------------------------------------

def test_every_declared_extra_is_one_the_library_supports():
    from reportkit.features import SUPPORTED_EXTRAS

    unknown = sorted(set(extras()) - set(SUPPORTED_EXTRAS))
    assert not unknown, (
        f"declared extras {unknown} are not in SUPPORTED_EXTRAS "
        f"({sorted(SUPPORTED_EXTRAS)}); nothing in the library can ever ask for them"
    )


def test_every_extra_the_library_supports_is_declared():
    from reportkit.features import SUPPORTED_EXTRAS

    undeclared = sorted(set(SUPPORTED_EXTRAS) - set(extras()))
    assert not undeclared, (
        f"the library supports {undeclared}, but [project.optional-dependencies] does not "
        "declare them, so pip rejects the install"
    )


def test_the_install_hints_name_extras_pip_can_resolve():
    from reportkit import features

    declared = set(extras())
    broken = []
    for extra in features.SUPPORTED_EXTRAS:
        hint = features.install_hint(extra)
        requested = hint.partition("[")[2].rstrip("]")
        if requested not in declared:
            broken.append(hint)
    assert not broken, f"these printed install commands would fail: {broken}"


def test_every_environment_marker_is_satisfiable_on_the_oldest_supported_python():
    floor = python_floor()
    dead = []
    for extra, requirements in extras().items():
        for requirement in requirements:
            _, _, marker = split_requirement(requirement)
            if marker:
                for clause in unsatisfiable_clauses(marker, floor):
                    dead.append(f"{extra}: {requirement!r} requires {clause}")
    for requirement in core_dependencies():
        _, _, marker = split_requirement(requirement)
        if marker:
            for clause in unsatisfiable_clauses(marker, floor):
                dead.append(f"core: {requirement!r} requires {clause}")
    assert not dead, (
        f"these markers can never be true on Python {'.'.join(map(str, floor))}, "
        f"the oldest interpreter requires-python admits: {dead}"
    )


def test_no_extra_installs_nothing_at_all():
    floor = python_floor()
    empty = []
    for extra, requirements in extras().items():
        live = [
            requirement
            for requirement in requirements
            if not unsatisfiable_clauses(split_requirement(requirement)[2], floor)
        ]
        if not live:
            empty.append(extra)
    assert not empty, (
        f"installing {empty} pulls in nothing on the oldest supported interpreter; "
        "the feature stays off with no error"
    )


def test_no_extra_redeclares_a_core_dependency():
    core = {split_requirement(requirement)[0] for requirement in core_dependencies()}
    clashes = []
    for extra, requirements in extras().items():
        for requirement in requirements:
            name, _, _ = split_requirement(requirement)
            if name in core:
                clashes.append(f"{extra}: {requirement!r} is already a core dependency")
    assert not clashes, f"an extra must not re-pin a core dependency: {clashes}"


def test_no_distribution_is_pinned_two_incompatible_ways():
    collected: dict[str, list[str]] = {}
    for requirement in core_dependencies():
        name, specifiers, _ = split_requirement(requirement)
        collected.setdefault(name, []).extend(specifiers)
    for requirements in extras().values():
        for requirement in requirements:
            name, specifiers, _ = split_requirement(requirement)
            collected.setdefault(name, []).extend(specifiers)

    conflicts = [
        f"{name}: {sorted(set(specifiers))} cannot all hold at once"
        for name, specifiers in collected.items()
        if conflicting(specifiers)
    ]
    assert not conflicts, f"an install would fail to resolve: {conflicts}"
