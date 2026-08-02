from pathlib import Path
from textwrap import dedent
from datetime import date
import shutil

root = Path('benchmark/tasks/heldout')
root.mkdir(parents=True, exist_ok=True)

for task_id in [
    'bugfix_slugify_empty_string',
    'configfix_pkg_cli_entrypoint',
    'feature_thefuzz_normalize',
    'bugfix_moreitertools_window',
    'testupdate_thefuzz_api_rename',
    'feature_inflect_plural_forms',
    'docs_readme_cli_examples',
]:
    path = root / task_id
    if path.exists():
        shutil.rmtree(path)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding='utf-8')


def task_yaml(task_id, category, difficulty, instruction, allowed_glob, entrypoint_hint, n_tests, tags, *, extra_forbidden=None):
    extra = ''
    if extra_forbidden:
        extra = '\n' + '\n'.join(f'    - "{g}"' for g in extra_forbidden)
    return dedent(f'''\
id: {task_id}
category: {category}
difficulty: {difficulty}
split: heldout

instruction: >
  {instruction}

repo:
  root: repo/
  loc: 0
  entrypoint_hint: {entrypoint_hint}

verification:
  runner: pytest
  test_files: [tests/test_{task_id}.py]
  timeout_s: 60
  n_tests: 0

constraints:
  allowed_write_globs:
    - "repo/{allowed_glob}"
  forbidden_write_globs:
    - "tests/**"
    - "solution/**"
    - "**/conftest.py"
    - "task.yaml"
    - "repo/smoke/**"{extra}
  max_turns: 20
  max_tool_calls: 40

provenance:
  source: authored
  license: MIT
  notes: >
    Authored to fit the benchmark's category mix without reusing a library already present in train.

metadata:
  expected_files_changed: 1
  created: {date.today().isoformat()}
  tags: [{', '.join(tags)}]
''').strip() + '\n'

# Batch 1: bug_fix, config_fix, small_feature
path = root / 'bugfix_slugify_empty_string'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""Simple slugify helpers for the benchmark."""\n')
write(path / 'repo/slugify_utils.py', '''
import re


def slugify(value):
    text = str(value).strip().lower()
    if not text:
        raise ValueError("value cannot be empty")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
''')
write(path / 'solution/fixed/repo/slugify_utils.py', '''
import re


def slugify(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
''')
write(path / 'tests/test_bugfix_slugify_empty_string.py', '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from slugify_utils import slugify


def test_empty_string_returns_empty_slug():
    assert slugify("") == ""


def test_whitespace_only_returns_empty_slug():
    assert slugify("   ") == ""


def test_none_returns_empty_slug():
    assert slugify(None) == ""


def test_regular_text_is_normalized():
    assert slugify("Hello World") == "hello-world"
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("slugify_utils")
''')
write(path / 'task.yaml', task_yaml(
    'bugfix_slugify_empty_string',
    'bug_fix',
    'easy',
    'Empty and whitespace-only input should not raise; the helper should return an empty slug instead and keep normal text normalization unchanged.',
    'slugify_utils.py',
    'slugify_utils.py',
    4,
    ['slugify', 'empty-input', 'bug-fix'],
))

path = root / 'configfix_pkg_cli_entrypoint'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""Demo package."""\n')
write(path / 'repo/demo/__init__.py', '"""Demo package namespace."""\n')
write(path / 'repo/demo/commands.py', '''
def main() -> str:
    return "demo ready"
''')
write(path / 'solution/fixed/repo/demo/commands.py', '''
def main() -> str:
    return "demo ready"
''')
write(path / 'repo/pyproject.toml', '''
[project]
name = "demo-pkg"
version = "0.1.0"
description = "A tiny demo package"
requires-python = ">=3.10"
''')
write(path / 'solution/fixed/repo/pyproject.toml', '''
[project]
name = "demo-pkg"
version = "0.1.0"
description = "A tiny demo package"
requires-python = ">=3.10"

[project.scripts]
demo = "demo.commands:main"
''')
write(path / 'tests/test_configfix_pkg_cli_entrypoint.py', '''
import tomllib
from pathlib import Path


def test_console_script_entrypoint_is_declared():
    pyproject = tomllib.loads(Path(__file__).resolve().parents[1] / "repo" / "pyproject.toml")
    assert pyproject["project"]["scripts"]["demo"] == "demo.commands:main"


def test_pyproject_still_has_project_metadata():
    pyproject = tomllib.loads(Path(__file__).resolve().parents[1] / "repo" / "pyproject.toml")
    assert pyproject["project"]["name"] == "demo-pkg"
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("demo.commands")
''')
write(path / 'task.yaml', task_yaml(
    'configfix_pkg_cli_entrypoint',
    'config_fix',
    'easy',
    'The package metadata exposes the project name and version, but the console entry point for the demo command is missing. Add the packaging declaration so the CLI can be discovered from the project metadata.',
    'pyproject.toml',
    'pyproject.toml',
    2,
    ['packaging', 'entry-point', 'configuration'],
))

path = root / 'feature_thefuzz_normalize'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""Tiny normalization helpers."""\n')
write(path / 'repo/normalizer.py', '''
def normalize(text: str) -> str:
    return text.lower()
''')
write(path / 'solution/fixed/repo/normalizer.py', '''
def normalize(text: str) -> str:
    if not text:
        return ""
    words = [piece for piece in text.lower().replace("-", " ").split() if piece]
    return " ".join(words)
''')
write(path / 'tests/test_feature_thefuzz_normalize.py', '''
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from normalizer import normalize


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Hello   WORLD  ") == "hello world"


def test_normalize_replaces_hyphens_with_spaces():
    assert normalize("Hello-World") == "hello world"


def test_normalize_handles_empty_input():
    assert normalize("") == ""
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("normalizer")
''')
write(path / 'task.yaml', task_yaml(
    'feature_thefuzz_normalize',
    'small_feature',
    'easy',
    'The text normalizer should collapse whitespace and turn hyphenated tokens into plain word boundaries. The current helper only lowercases the text, so it misses the intended normalization behavior.',
    'normalizer.py',
    'normalizer.py',
    3,
    ['normalizer', 'whitespace', 'small-feature'],
))

# Batch 2: bug_fix, test_update, small_feature
path = root / 'bugfix_moreitertools_window'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""Windowing helpers."""\n')
write(path / 'repo/windowing.py', '''
def window(values, size):
    if size <= 0:
        raise ValueError("size must be positive")
    return [tuple(values[i:i + size]) for i in range(len(values) - size + 1)]
''')
write(path / 'solution/fixed/repo/windowing.py', '''
def window(values, size):
    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(values):
        return []
    return [tuple(values[i:i + size]) for i in range(len(values) - size + 1)]
''')
write(path / 'tests/test_bugfix_moreitertools_window.py', '''
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from windowing import window


def test_window_returns_empty_for_too_large_size():
    assert window([1, 2], 3) == []


def test_window_returns_singletons_for_size_one():
    assert window([1, 2, 3], 1) == [(1,), (2,), (3,)]


def test_window_returns_expected_sliding_tuples():
    assert window([1, 2, 3, 4], 2) == [(1, 2), (2, 3), (3, 4)]
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("windowing")
''')
write(path / 'task.yaml', task_yaml(
    'bugfix_moreitertools_window',
    'bug_fix',
    'hard',
    'The sliding-window helper should return an empty result when the requested size exceeds the input length, instead of producing a malformed slice. Keep the public contract intact and preserve the existing behavior for valid windows.',
    'windowing.py',
    'windowing.py',
    3,
    ['windowing', 'sliding-window', 'edge-case'],
))

path = root / 'testupdate_thefuzz_api_rename'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""A tiny string comparison helper."""\n')
write(path / 'repo/compare.py', '''
def compare(a: str, b: str) -> int:
    return 1 if a.strip().lower() == b.strip().lower() else 0
''')
write(path / 'repo/tests/test_strings.py', '''
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from compare import compare_text


def test_compare_is_case_insensitive():
    assert compare_text("Alpha", "alpha") == 1


def test_compare_detects_difference():
    assert compare_text("Alpha", "Beta") == 0
''')
write(path / 'solution/fixed/repo/tests/test_strings.py', '''
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from compare import compare


def test_compare_is_case_insensitive():
    assert compare("Alpha", "alpha") == 1


def test_compare_detects_difference():
    assert compare("Alpha", "Beta") == 0
''')
write(path / 'tests/test_testupdate_thefuzz_api_rename.py', '''
from pathlib import Path
import ast


def test_visible_suite_calls_the_current_compare_symbol():
    test_file = Path(__file__).resolve().parents[1] / "repo" / "tests" / "test_strings.py"
    tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
    call_names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "compare" in call_names
    assert "compare_text" not in call_names
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("compare")
''')
write(path / 'task.yaml', task_yaml(
    'testupdate_thefuzz_api_rename',
    'test_update',
    'medium',
    'The project’s own visible test suite still calls the old helper name from a renamed API. Update the tests so they exercise the current public symbol without changing the expected behavior or the library implementation.',
    'tests/**',
    'tests/test_strings.py',
    1,
    ['api-rename', 'test-maintenance', 'visible-suite'],
    extra_forbidden=['repo/compare.py'],
))

path = root / 'feature_inflect_plural_forms'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/__init__.py', '"""Tiny pluralization utilities."""\n')
write(path / 'repo/pluralize.py', '''
def pluralize(word: str) -> str:
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word + "es"
    return word + "s"
''')
write(path / 'solution/fixed/repo/pluralize.py', '''
def pluralize(word: str) -> str:
    irregular = {
        "child": "children",
        "person": "people",
        "mouse": "mice",
        "leaf": "leaves",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word + "es"
    return word + "s"
''')
write(path / 'tests/test_feature_inflect_plural_forms.py', '''
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from pluralize import pluralize


def test_irregular_forms_are_supported():
    assert pluralize("child") == "children"
    assert pluralize("person") == "people"


def test_regular_suffix_rules_still_work():
    assert pluralize("party") == "parties"
    assert pluralize("bus") == "buses"
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("pluralize")
''')
write(path / 'task.yaml', task_yaml(
    'feature_inflect_plural_forms',
    'small_feature',
    'medium',
    'The pluralization helper should support common irregular forms in addition to the existing suffix-based rules. Extend the public helper rather than changing unrelated behavior.',
    'pluralize.py',
    'pluralize.py',
    2,
    ['pluralization', 'irregular-forms', 'small-feature'],
))

# Batch 3: docs_improve
path = root / 'docs_readme_cli_examples'
(path / 'repo').mkdir(parents=True, exist_ok=True)
(path / 'tests').mkdir(parents=True, exist_ok=True)
(path / 'smoke').mkdir(parents=True, exist_ok=True)
(path / 'solution/fixed').mkdir(parents=True, exist_ok=True)
write(path / 'repo/README.md', '''
# Demo CLI

This tool exposes a `demo` command.

Use it to print a confirmation message.
''')
write(path / 'solution/fixed/repo/README.md', '''
# Demo CLI

This tool exposes a `demo` command.

Use it to print a confirmation message.

## Examples

- `demo --dry-run` prints the planned action without executing it.
- `demo --verbose` prints the full action details.
''')
write(path / 'repo/cli.py', '''
def build_help() -> str:
    return "demo"
''')
write(path / 'solution/fixed/repo/cli.py', '''
def build_help() -> str:
    return "demo"
''')
write(path / 'tests/test_docs_readme_cli_examples.py', '''
from pathlib import Path


def test_readme_mentions_the_dry_run_example():
    readme = Path(__file__).resolve().parents[1] / "repo" / "README.md"
    assert "--dry-run" in readme.read_text(encoding="utf-8")


def test_readme_mentions_the_verbose_example():
    readme = Path(__file__).resolve().parents[1] / "repo" / "README.md"
    assert "--verbose" in readme.read_text(encoding="utf-8")
''')
write(path / 'smoke/test_smoke.py', '''
def test_smoke_import():
    import importlib
    assert importlib.import_module("cli")
''')
write(path / 'task.yaml', task_yaml(
    'docs_readme_cli_examples',
    'docs_improve',
    'hard',
    'The project README is missing the documented CLI examples that users expect from the current command surface. Update the docs so the examples match the documented flags and are discoverable from the repository entry point.',
    'README.md',
    'README.md',
    2,
    ['documentation', 'cli-examples', 'readme'],
))
