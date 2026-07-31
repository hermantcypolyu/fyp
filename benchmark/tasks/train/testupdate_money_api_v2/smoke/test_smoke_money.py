"""Visible smoke checks: the module imports and renders. Deliberately shallow -
these say nothing about whether the repo's own suite has been brought up to date.
"""

from src.money import Money


def test_module_imports_and_renders():
    assert Money.from_cents(1234).format() == "USD 12.34"


def test_addition_works():
    assert (Money.from_cents(100) + Money.from_cents(1)).cents == 101
