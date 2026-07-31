"""Test suite for src/money.py.

Updated for the 2.0 API described in CHANGELOG.md.
"""

import pytest

from src.money import CurrencyMismatch, Money


def test_construct_from_whole_cents():
    price = Money.from_cents(1234)
    assert price.cents == 1234
    assert price.currency == "USD"


def test_parse_decimal_string():
    assert Money.from_string("12.34") == Money.from_cents(1234)
    assert Money.from_string("-0.05") == Money.from_cents(-5)


def test_format_uses_the_currency_code():
    assert Money.from_cents(1234).format() == "USD 12.34"
    assert Money.from_cents(5).format() == "USD 0.05"
    assert Money.from_cents(-5).format() == "USD -0.05"


def test_addition_and_subtraction():
    a = Money.from_cents(150)
    b = Money.from_cents(75)
    assert (a + b).format() == "USD 2.25"
    assert (a - b).format() == "USD 0.75"


def test_multiplication_by_a_whole_number():
    assert (Money.from_cents(125) * 4).format() == "USD 5.00"


def test_mixing_currencies_is_rejected():
    with pytest.raises(CurrencyMismatch):
        Money.from_cents(100, "USD") + Money.from_cents(100, "EUR")


def test_allocate_never_loses_a_cent():
    parts = Money.from_cents(100).allocate([1, 1, 1])
    assert [p.cents for p in parts] == [34, 33, 33]
    assert sum(p.cents for p in parts) == 100


def test_allocate_respects_weights():
    parts = Money.from_cents(1000).allocate([3, 1])
    assert [p.cents for p in parts] == [750, 250]


def test_amounts_are_hashable_and_comparable():
    assert Money.from_cents(100) < Money.from_cents(200)
    assert len({Money.from_cents(100), Money.from_cents(100)}) == 1
