from pathlib import Path
import ast


def _parse_test_file():
    test_file = Path(__file__).resolve().parents[1] / "repo" / "tests" / "test_strings.py"
    return ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))


def test_visible_suite_calls_the_current_compare_symbol():
    tree = _parse_test_file()
    call_names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "compare" in call_names
    assert "compare_text" not in call_names


def test_visible_suite_imports_the_current_compare_symbol():
    tree = _parse_test_file()
    import_names = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "compare" in import_names


def test_visible_suite_uses_compare_in_assertions():
    tree = _parse_test_file()
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "compare" for node in ast.walk(tree))
