"""Test suite for src/money.py.

NOTE: still written against the 1.x API. See CHANGELOG.md.
"""

import pytest

from src.money import CurrencyMismatch, Money


def test_construct_from_dollars_and_cents():
    price = Money(dollars=12, cents=34)
    assert price.dollars == 12
    assert price.cents == 34


def test_parse_decimal_string():
    assert Money.parse("12.34") == Money(dollars=12, cents=34)
    assert Money.parse("-0.05") == Money(dollars=0, cents=-5)


def test_format_uses_a_dollar_sign():
    assert Money(dollars=12, cents=34).format() == "$12.34"
    assert Money(dollars=0, cents=5).format() == "$0.05"


def test_addition_and_subtraction():
    a = Money(dollars=1, cents=50)
    b = Money(dollars=0, cents=75)
    assert (a + b).format() == "$2.25"
    assert (a - b).format() == "$0.75"


def test_multiplication_by_a_whole_number():
    assert (Money(dollars=1, cents=25) * 4).format() == "$5.00"


def test_mixing_currencies_is_rejected():
    with pytest.raises(CurrencyMismatch):
        Money(dollars=1, cents=0, currency="USD") + Money(dollars=1, cents=0, currency="EUR")


def test_split_never_loses_a_cent():
    parts = Money(dollars=1, cents=0).split(3)
    assert [p.cents for p in parts] == [34, 33, 33]
    assert sum(p.cents for p in parts) == 100


def test_amounts_are_hashable_and_comparable():
    assert Money(dollars=1, cents=0) < Money(dollars=2, cents=0)
    assert len({Money(dollars=1, cents=0), Money(dollars=1, cents=0)}) == 1
