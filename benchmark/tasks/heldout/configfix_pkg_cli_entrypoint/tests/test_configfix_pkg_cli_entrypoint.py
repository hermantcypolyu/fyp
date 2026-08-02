import tomllib
from pathlib import Path


def _load_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "repo" / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def test_console_script_entrypoint_is_declared():
    pyproject = _load_pyproject()
    assert pyproject["project"]["scripts"]["demo"] == "demo.commands:main"


def test_pyproject_still_has_project_metadata():
    pyproject = _load_pyproject()
    assert pyproject["project"]["name"] == "demo-pkg"


def test_project_version_is_present():
    pyproject = _load_pyproject()
    assert pyproject["project"]["version"] == "0.1.0"
