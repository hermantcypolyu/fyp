"""Hidden verification for `bugfix_intcomma_zero_ndigits`.

Assertions are taken from the upstream suite at the fixing commit
(python-humanize/humanize @ 03863fed6c2ba02cfd13d5b1e0b161f680c7b966,
tests/test_number.py), which expresses them as one large `@parametrize`. They are
regrouped here into behaviour-sized tests because reward_v1 is the pass fraction, and a
single parametrised test would collapse the whole signal into one bit.

Six tests cover behaviour that already works; five cover the defect. The six are load
bearing - they are what stops the lazy "always format, never fall back" fix from scoring
well, since that would change the output for every value given no digit count at all.
"""

import humanize


def test_integers_get_thousands_separators():
    assert humanize.intcomma(100) == "100"
    assert humanize.intcomma(1000) == "1,000"
    assert humanize.intcomma(10123) == "10,123"
    assert humanize.intcomma(1_000_000) == "1,000,000"


def test_numeric_strings_are_accepted():
    assert humanize.intcomma("1000") == "1,000"
    assert humanize.intcomma("10311") == "10,311"
    assert humanize.intcomma("1234567.1234567") == "1,234,567.1234567"


def test_non_numeric_input_is_returned_unchanged():
    assert humanize.intcomma(None) == "None"


def test_floats_keep_full_precision_when_no_rounding_is_asked_for():
    assert humanize.intcomma(14308.40) == "14,308.4"
    assert humanize.intcomma(1234.5454545) == "1,234.5454545"
    assert humanize.intcomma(1_234_567.25) == "1,234,567.25"


def test_explicit_none_behaves_the_same_as_omitting_the_argument():
    assert humanize.intcomma(14308.40, None) == "14,308.4"
    assert humanize.intcomma(1234.5454545, None) == "1,234.5454545"


def test_positive_digit_counts_round_and_pad():
    assert humanize.intcomma(1234.5454545, 1) == "1,234.5"
    assert humanize.intcomma(1234.5454545, 2) == "1,234.55"
    assert humanize.intcomma(14308.40, 3) == "14,308.400"
    assert humanize.intcomma(1234.5454545, 10) == "1,234.5454545000"


# --- the defect ------------------------------------------------------------------------
# Zero is a perfectly good number of decimal places, but it is also falsy.


def test_zero_digits_rounds_to_a_whole_number():
    assert humanize.intcomma(1234.5454545, 0) == "1,235"


def test_zero_digits_rounds_down_when_it_should():
    assert humanize.intcomma(14308.40, 0) == "14,308"


def test_zero_digits_rounds_up_when_it_should():
    assert humanize.intcomma(1000.6, 0) == "1,001"


def test_zero_digits_on_a_larger_value():
    assert humanize.intcomma(1_234_567.25, 0) == "1,234,567"


def test_zero_digits_on_a_negative_value():
    assert humanize.intcomma(-1234.5454545, 0) == "-1,235"
