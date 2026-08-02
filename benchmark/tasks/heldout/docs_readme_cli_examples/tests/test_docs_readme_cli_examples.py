from pathlib import Path


def _readme_text() -> str:
    return (Path(__file__).resolve().parents[1] / "repo" / "README.md").read_text(encoding="utf-8")


def test_readme_mentions_the_dry_run_example():
    assert "--dry-run" in _readme_text()


def test_readme_mentions_the_verbose_example():
    assert "--verbose" in _readme_text()


def test_readme_includes_an_examples_section():
    assert "## Examples" in _readme_text()
