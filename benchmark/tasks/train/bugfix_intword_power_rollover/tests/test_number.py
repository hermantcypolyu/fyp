"""Hidden verification for `bugfix_intword_power_rollover`.

Assertions are taken from the upstream suite at the fixing commit
(python-humanize/humanize @ 8705e2967ee610a36143e2379bf091d75a1ed9b7,
tests/test_number.py), regrouped from one large `@parametrize` into behaviour-sized tests
so that partial credit is meaningful, and extended with the neighbouring values that
trigger the same code path.

Four tests pass against the shipped defect and four do not. One of the passing four
matters more than the others: `test_carry_at_ordinary_boundaries`. The carry logic being
fixed here is not useless - it is what turns 999999999 into "1.0 billion" rather than
"1000.0 million" - so deleting the branch outright makes the four failing tests pass and
breaks that one. The fix has to make the branch aware of the real ratio between adjacent
scales instead of assuming it is always a thousand.
"""

import math

import humanize

ONE_THOUSAND_DECILLION = 1_000_000_000_000_000_000_000_000_000_000_000_000


def test_ordinary_scales():
    assert humanize.intword("0") == "0"
    assert humanize.intword("100") == "100"
    assert humanize.intword("1000") == "1.0 thousand"
    assert humanize.intword("12490") == "12.5 thousand"
    assert humanize.intword("1200000") == "1.2 million"
    assert humanize.intword("6000000000000") == "6.0 trillion"
    assert humanize.intword("3500000000000000000000") == "3.5 sextillion"
    assert humanize.intword("8100000000000000000000000000000000") == "8.1 decillion"


def test_carry_at_ordinary_boundaries():
    # Rounding pushes the mantissa to a full thousand, so the result must step up a scale.
    assert humanize.intword("999999999") == "1.0 billion"
    assert humanize.intword("999999999999") == "1.0 trillion"
    assert humanize.intword("999999999999999") == "1.0 quadrillion"


def test_negative_values_and_custom_formats():
    assert humanize.intword("-100") == "-100"
    assert humanize.intword("-1000000") == "-1.0 million"
    assert humanize.intword("-1300000000000000") == "-1.3 quadrillion"
    assert humanize.intword("1230000", "%0.2f") == "1.23 million"


def test_non_finite_and_unparseable_values():
    assert humanize.intword(None) == "None"
    assert humanize.intword(math.nan) == "NaN"
    assert humanize.intword(math.inf) == "+Inf"
    assert humanize.intword(-math.inf) == "-Inf"
    assert humanize.intword("nan") == "NaN"
    assert humanize.intword("-inf") == "-Inf"


# --- the defect ------------------------------------------------------------------------
# Decillion is the last named scale, and the gap from it to the next table entry is not a
# factor of a thousand. A mantissa that rounds to exactly 1000.0 must therefore be printed
# as it stands rather than carried.


def test_exactly_one_thousand_decillion():
    assert humanize.intword(ONE_THOUSAND_DECILLION) == "1000.0 decillion"


def test_just_under_one_thousand_decillion():
    assert humanize.intword(ONE_THOUSAND_DECILLION - 1) == "1000.0 decillion"


def test_a_mantissa_that_rounds_up_to_one_thousand_decillion():
    assert humanize.intword(999_950_000_000_000_000_000_000_000_000_000_000) == "1000.0 decillion"


def test_custom_format_at_the_top_of_the_table():
    assert humanize.intword(ONE_THOUSAND_DECILLION, "%0.2f") == "1000.00 decillion"


def test_larger_multiples_of_decillion_are_unaffected():
    # These never round to exactly a thousand, so they already work - and must keep working.
    assert humanize.intword(1_100_000_000_000_000_000_000_000_000_000_000_000) == "1100.0 decillion"
    assert humanize.intword(2_100_000_000_000_000_000_000_000_000_000_000_000) == "2100.0 decillion"
