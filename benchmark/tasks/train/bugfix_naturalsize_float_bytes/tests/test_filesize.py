"""Hidden verification for `bugfix_naturalsize_float_bytes`.

Assertions are taken from the upstream test suite at the fixing commit
(https://github.com/python-humanize/humanize @
11e62eeedb8990298c3612af82376dc2344d24f8, tests/test_filesize.py). Upstream expresses
them as one large `@parametrize` that also re-runs every case with the sign flipped;
they are regrouped here into behaviour-sized tests, because reward_v1 is the pass
fraction and a single parametrised test would collapse the whole signal into one bit.

Six tests cover the defect itself and six cover behaviour that already works and must
not regress; two of the six failing ones double as guards against rounding rather than
truncating, and the custom-format test guards against coercing the input on the way in.
"""

import humanize


def test_fractional_byte_count_is_truncated_to_a_whole_number():
    assert humanize.naturalsize(1.123456789) == "1 Bytes"


def test_fractional_byte_count_in_gnu_mode():
    assert humanize.naturalsize(1.123456789, False, True) == "1B"


def test_fractional_byte_count_in_binary_mode():
    assert humanize.naturalsize(1.123456789, True) == "1 Bytes"


def test_a_fraction_just_below_the_first_unit_is_not_rounded_up():
    assert humanize.naturalsize(999.5) == "999 Bytes"
    assert humanize.naturalsize(0.5) == "0 Bytes"


def test_negative_fractional_byte_count_keeps_its_sign():
    assert humanize.naturalsize(-1.9) == "-1 Bytes"
    assert humanize.naturalsize(-1.9, False, True) == "-1B"


def test_numeric_strings_below_the_first_unit():
    assert humanize.naturalsize("300") == "300 Bytes"
    assert humanize.naturalsize("-1.123456789") == "-1 Bytes"


# --- behaviour that already works and must survive the fix -------------------------------


def test_integer_byte_counts_are_unchanged():
    assert humanize.naturalsize(300) == "300 Bytes"
    assert humanize.naturalsize(0) == "0 Bytes"
    assert humanize.naturalsize(1) == "1 Byte"
    assert humanize.naturalsize(-300) == "-300 Bytes"


def test_decimal_suffixes():
    assert humanize.naturalsize(3000) == "3.0 kB"
    assert humanize.naturalsize(3000000) == "3.0 MB"
    assert humanize.naturalsize(10**28) == "10.0 RB"


def test_binary_suffixes():
    assert humanize.naturalsize(3000, True) == "2.9 KiB"
    assert humanize.naturalsize(3000000, True) == "2.9 MiB"
    assert humanize.naturalsize(-4096, True) == "-4.0 KiB"


def test_gnu_suffixes():
    assert humanize.naturalsize(300, False, True) == "300B"
    assert humanize.naturalsize(3000, False, True) == "2.9K"
    assert humanize.naturalsize(3000000, False, True) == "2.9M"
    assert humanize.naturalsize(1024, False, True) == "1.0K"


def test_fractional_values_above_the_first_unit_keep_their_decimals():
    assert humanize.naturalsize(1.123456789 * 10**3) == "1.1 kB"
    assert humanize.naturalsize(1.123456789 * 10**6) == "1.1 MB"
    assert humanize.naturalsize(1.123456789 * 10**3, False, True) == "1.1K"
    assert humanize.naturalsize(1.123456789 * 10**6, False, True) == "1.1M"


def test_custom_format_strings_see_the_unrounded_value():
    assert humanize.naturalsize(3000, False, True, "%.3f") == "2.930K"
    assert humanize.naturalsize(3141592, False, False, "%.2f") == "3.14 MB"
    assert humanize.naturalsize(1999.9, False, False, "%.3f") == "2.000 kB"
