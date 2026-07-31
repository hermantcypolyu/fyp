"""Hidden verification for `bugfix_filesize_unit_rollover`.

Assertions are taken from the upstream test suite at the fixing commit
(python-humanize/humanize @ 823ad6096e1e5ba82ea876ce761fc2efebd76157,
tests/test_filesize.py). Upstream expresses them as one large `@parametrize`; they are
regrouped here into behaviour-sized tests, because reward_v1 is the pass fraction and a
single parametrised test would collapse the whole signal into one bit.

The split is deliberate: seven tests cover behaviour that already works and must not
regress, six cover the rollover defect itself, one guards against over-correcting.
"""

import humanize


def test_decimal_suffixes():
    assert humanize.naturalsize(3000000) == "3.0 MB"
    assert humanize.naturalsize(3000) == "3.0 kB"
    assert humanize.naturalsize(300) == "300 Bytes"


def test_binary_suffixes():
    assert humanize.naturalsize(3000, True) == "2.9 KiB"
    assert humanize.naturalsize(3000000, True) == "2.9 MiB"


def test_gnu_suffixes():
    assert humanize.naturalsize(300, False, True) == "300B"
    assert humanize.naturalsize(3000, False, True) == "2.9K"
    assert humanize.naturalsize(3000000, False, True) == "2.9M"


def test_single_byte_is_singular():
    assert humanize.naturalsize(1) == "1 Byte"
    assert humanize.naturalsize(0) == "0 Bytes"


def test_negative_values_keep_their_sign():
    assert humanize.naturalsize(-4096, True) == "-4.0 KiB"
    assert humanize.naturalsize(-3000000) == "-3.0 MB"


def test_custom_format_string():
    assert humanize.naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_very_large_values_stay_on_the_largest_suffix():
    assert humanize.naturalsize(10**28) == "10.0 RB"
    assert humanize.naturalsize(10**34 * 3) == "30000.0 QB"


# --- the defect ------------------------------------------------------------------------
# The suffix is chosen from the unrounded byte count, but the mantissa is rounded
# afterwards. When rounding pushes the mantissa up to the base, the result must carry into
# the next suffix rather than read "1000.0 kB".


def test_rounding_carries_from_kilobytes_to_megabytes():
    assert humanize.naturalsize(999999) == "1.0 MB"


def test_rounding_carries_from_megabytes_to_gigabytes():
    assert humanize.naturalsize(999999999) == "1.0 GB"


def test_rounding_carries_from_gigabytes_to_terabytes():
    assert humanize.naturalsize(999999999999) == "1.0 TB"


def test_rounding_carries_between_binary_suffixes():
    assert humanize.naturalsize(1024**2 - 1, True) == "1.0 MiB"


def test_rounding_carries_between_larger_binary_suffixes():
    assert humanize.naturalsize(1024**3 - 1, True) == "1.0 GiB"


def test_rounding_carries_in_gnu_mode():
    assert humanize.naturalsize(1024**2 - 1, False, True) == "1.0M"


# --- and the correction must not fire when it should not --------------------------------


def test_no_carry_when_the_rounded_mantissa_stays_below_the_base():
    assert humanize.naturalsize(999499) == "999.5 kB"
    assert humanize.naturalsize(1000) == "1.0 kB"
    assert humanize.naturalsize(1024, True) == "1.0 KiB"
