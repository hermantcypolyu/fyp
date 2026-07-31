"""Visible smoke checks: the two leaf modules still load and still behave.

They are loaded straight off disk by path, so they say nothing about the package layout,
the import path or the packaging metadata.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(filename: str):
    path = ROOT / "src" / "dataspec" / filename
    spec = importlib.util.spec_from_file_location(f"_smoke_{filename[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rules_module_loads():
    assert _load("rules.py").min_length(1)("a") is None


def test_fields_module_loads():
    assert _load("fields.py").parse_field("age:int").kind == "int"
