"""Hidden verification for `configfix_textkit_packaging`.

The Python source is correct. Everything graded here is packaging metadata or the
package's own re-export surface: the things that make `import textkit` work and make a
built wheel contain what it claims to.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path.cwd()
PYPROJECT = REPO / "pyproject.toml"
MIN_PYTHON = (3, 10)


@pytest.fixture(scope="module")
def config() -> dict:
    if not PYPROJECT.is_file():
        pytest.fail("pyproject.toml is missing from the project root")
    try:
        return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"pyproject.toml is not valid TOML: {exc}")


def _normalise(name: str) -> str:
    """PEP 503 normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


# --- the package must import and expose what it advertises ------------------------------

def test_package_imports():
    import textkit  # noqa: F401


def test_slugify_is_re_exported_and_works():
    import textkit

    assert textkit.slugify("Crème Brûlée, please!") == "creme-brulee-please"
    assert textkit.slugify("the quick brown fox", max_length=12) == "the-quick"


def test_case_helpers_are_re_exported_and_work():
    import textkit

    assert textkit.title_case("the wind in the willows") == "The Wind in the Willows"
    assert textkit.snake_case("HTTPResponseCode") == "http_response_code"
    assert textkit.camel_case("http response code") == "httpResponseCode"


def test_dunder_all_only_names_things_that_exist():
    import textkit

    missing = [name for name in textkit.__all__ if not hasattr(textkit, name)]
    assert not missing, f"__all__ promises names the package does not define: {missing}"


def test_dunder_all_covers_every_public_helper():
    import textkit

    expected = {"slugify", "title_case", "snake_case", "camel_case"}
    assert expected <= set(textkit.__all__), (
        f"__all__ is missing {sorted(expected - set(textkit.__all__))}"
    )


# --- the metadata must describe the package that actually exists ------------------------

def test_distribution_name_matches_the_import_name(config):
    name = config["project"]["name"]
    assert _normalise(name) == "textkit", (
        f"distribution name {name!r} normalises to {_normalise(name)!r}, "
        "but the importable package is 'textkit'"
    )


def test_declared_version_matches_the_package_version(config):
    import textkit

    assert config["project"]["version"] == textkit.__version__, (
        f"pyproject declares {config['project']['version']!r} but "
        f"textkit.__version__ is {textkit.__version__!r}; the release job refuses to publish "
        "on a mismatch"
    )


def test_requires_python_floor_is_honest(config):
    spec = config["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert match, f"requires-python {spec!r} has no '>=<major>.<minor>' lower bound"
    floor = (int(match.group(1)), int(match.group(2)))
    assert floor >= MIN_PYTHON, (
        f"requires-python floor is {floor[0]}.{floor[1]}, but the source needs at least "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}; installing on an older interpreter would fail at import"
    )


def test_wheel_package_path_exists_on_disk(config):
    packages = config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}) \
        .get("wheel", {}).get("packages")
    assert packages, "no [tool.hatch.build.targets.wheel] packages entry"
    for rel in packages:
        assert (REPO / rel).is_dir(), (
            f"wheel packages points at {rel!r}, which does not exist; "
            "the built wheel would be empty"
        )
