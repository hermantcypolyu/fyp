"""Visible smoke checks. Shallow by design - every mapping here carries every key named
in the condition, so nothing is revealed about ragged input.
"""

import funcy


def test_module_imports_and_selects():
    data = [{'a': 1, 'b': 2}, {'a': 10, 'b': 2}]
    assert funcy.where(data, a=1) == [{'a': 1, 'b': 2}]


def test_pluck_reads_a_column():
    assert funcy.pluck('a', [{'a': 1}, {'a': 10}]) == [1, 10]
