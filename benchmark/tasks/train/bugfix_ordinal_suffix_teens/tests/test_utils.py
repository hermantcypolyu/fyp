"""Hidden verification for `bugfix_ordinal_suffix_teens`.

Assertions are drawn from the upstream regression test added in
jd/tenacity @ 0bdf1d9e5f42b7179e7aae52cb70d26283a4c5ed (tests/test_utils.py), extended
with the same pattern at larger magnitudes. Upstream writes them as one test function;
they are split by behaviour here so that a partly-correct fix scores between 0 and 1.

Six of these pass against the shipped defect and five do not. The passing six are not
padding - they are what stops the obvious wrong fix, "return 'th' whenever the number
ends in 1, 2 or 3 above 100", from scoring well.
"""

from tenacity import _utils


def test_first_second_third():
    assert _utils.find_ordinal(1) == "st"
    assert _utils.find_ordinal(2) == "nd"
    assert _utils.find_ordinal(3) == "rd"


def test_zero_and_the_plain_th_range():
    assert _utils.find_ordinal(0) == "th"
    assert _utils.find_ordinal(4) == "th"
    assert _utils.find_ordinal(9) == "th"
    assert _utils.find_ordinal(10) == "th"


def test_teens_below_twenty():
    assert _utils.find_ordinal(11) == "th"
    assert _utils.find_ordinal(12) == "th"
    assert _utils.find_ordinal(13) == "th"
    assert _utils.find_ordinal(19) == "th"


def test_twenty_first_second_third():
    assert _utils.find_ordinal(21) == "st"
    assert _utils.find_ordinal(22) == "nd"
    assert _utils.find_ordinal(23) == "rd"


def test_hundreds_ending_in_one_two_three_keep_their_suffix():
    assert _utils.find_ordinal(101) == "st"
    assert _utils.find_ordinal(102) == "nd"
    assert _utils.find_ordinal(103) == "rd"
    assert _utils.find_ordinal(121) == "st"


def test_to_ordinal_joins_number_and_suffix():
    assert _utils.to_ordinal(1) == "1st"
    assert _utils.to_ordinal(4) == "4th"
    assert _utils.to_ordinal(22) == "22nd"


# --- the defect ------------------------------------------------------------------------
# English gives every number whose last two digits are 11, 12 or 13 a "th" suffix, at any
# magnitude. Recursing on `pos_num % 10` throws away exactly the information needed to
# tell 111 from 101.


def test_one_hundred_teens():
    assert _utils.find_ordinal(111) == "th"
    assert _utils.find_ordinal(112) == "th"
    assert _utils.find_ordinal(113) == "th"


def test_two_hundred_teens():
    assert _utils.find_ordinal(211) == "th"
    assert _utils.find_ordinal(212) == "th"
    assert _utils.find_ordinal(213) == "th"


def test_thousand_teens():
    assert _utils.find_ordinal(1011) == "th"
    assert _utils.find_ordinal(1012) == "th"
    assert _utils.find_ordinal(1013) == "th"


def test_arbitrarily_large_teens():
    assert _utils.find_ordinal(12311) == "th"
    assert _utils.find_ordinal(45012) == "th"
    assert _utils.find_ordinal(99913) == "th"


def test_to_ordinal_on_teen_hundreds():
    assert _utils.to_ordinal(111) == "111th"
    assert _utils.to_ordinal(212) == "212th"
