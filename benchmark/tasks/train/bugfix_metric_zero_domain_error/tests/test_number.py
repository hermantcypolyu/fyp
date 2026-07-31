"""Hidden verification for `bugfix_metric_zero_domain_error`.

Assertions are taken from the upstream test suite at the fixing commit
(https://github.com/python-humanize/humanize @
b1ee6875820839a59f1b6f4a5fedc7cad8883834, tests/test_number.py). Upstream expresses them
as one large `@parametrize`; they are regrouped here into behaviour-sized tests, because
reward_v1 is the pass fraction and a single parametrised test would collapse the whole
signal into one bit.

Six tests cover the defect - zero in each of the ways the function can be asked to render
it - and six cover behaviour that already works and must not regress. The zero tests vary
the unit and the precision on purpose: they are what stops a hard-coded return value from
scoring.
"""

import humanize


def test_metric_of_zero_needs_no_prefix():
    assert humanize.metric(0) == "0.00"


def test_metric_of_zero_still_appends_the_unit():
    assert humanize.metric(0, "Hz") == "0.00 Hz"
    assert humanize.metric(0, "W") == "0.00 W"


def test_metric_of_zero_honours_a_wider_precision():
    assert humanize.metric(0, "W", 5) == "0.0000 W"
    assert humanize.metric(0, "", 6) == "0.00000"


def test_metric_of_zero_honours_a_narrower_precision():
    assert humanize.metric(0, precision=1) == "0"
    assert humanize.metric(0, unit="V", precision=1) == "0 V"


def test_metric_of_floating_point_zero_matches_integer_zero():
    assert humanize.metric(0.0) == "0.00"
    assert humanize.metric(0.0, "V") == "0.00 V"


def test_metric_omits_the_space_before_a_degree_sign():
    assert humanize.metric(0, "\u00b0") == "0.00\u00b0"
    assert humanize.metric(1, "\u00b0") == "1.00\u00b0"
    assert humanize.metric(0.1, "\u00b0") == "100m\u00b0"


# --- behaviour that already works and must survive the fix -------------------------------


def test_metric_keeps_three_significant_digits_by_default():
    assert humanize.metric(1.23456) == "1.23"
    assert humanize.metric(12.3456) == "12.3"
    assert humanize.metric(123.456) == "123"
    assert humanize.metric(100) == "100"


def test_metric_separates_a_unit_with_a_space():
    assert humanize.metric(1, "Hz") == "1.00 Hz"
    assert humanize.metric(1.0, "W") == "1.00 W"
    assert humanize.metric(3, "C") == "3.00 C"
    assert humanize.metric(-1500, "V") == "-1.50 kV"


def test_metric_precision_argument_widens_the_mantissa():
    assert humanize.metric(3, "W", 5) == "3.0000 W"
    assert humanize.metric(12345, "", 6) == "12.3450 k"


def test_metric_picks_the_prefix_that_avoids_leading_zeros():
    assert humanize.metric(1234.56) == "1.23 k"
    assert humanize.metric(200_000) == "200 k"
    assert humanize.metric(1e25, "m") == "10.0 Ym"
    assert humanize.metric(1e26, "m") == "100 Ym"


def test_metric_uses_submultiple_prefixes_below_one():
    assert humanize.metric(0.1) == "100 m"
    assert humanize.metric(0.12) == "120 m"
    assert humanize.metric(0.012) == "12.0 m"
    assert humanize.metric(0.0012) == "1.20 m"
    assert humanize.metric(0.00012) == "120 \u03bc"
    assert humanize.metric(1e-23) == "10.0 y"
    assert humanize.metric(1e-24) == "1.00 y"


def test_metric_falls_back_to_scientific_notation_outside_the_prefix_range():
    assert humanize.metric(1e27, "A") == "1.00 x 10\u00b2\u2077A"
    assert humanize.metric(1.234e28, "A") == "1.23 x 10\u00b2\u2078A"
    assert humanize.metric(1e-25) == "1.00 x 10\u207b\u00b2\u2075"
    assert humanize.metric(1e-26) == "1.00 x 10\u207b\u00b2\u2076"
