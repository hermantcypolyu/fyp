"""Humanizing functions for numbers."""

from __future__ import annotations
import math
import re
import sys
from fractions import Fraction
from typing import TYPE_CHECKING
from .i18n import _gettext as _
from .i18n import _ngettext
from .i18n import _ngettext_noop as NS_
from .i18n import _pgettext as P_
from .i18n import decimal_separator, thousands_separator

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

NumberOrString: TypeAlias = "float | str"


def _format_not_finite(value: float) -> str:
    """Utility function to handle infinite and nan cases."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value) and value < 0:
        return "-Inf"
    if math.isinf(value) and value > 0:
        return "+Inf"
    return ""


def ordinal(value: NumberOrString, gender: str = "male") -> str:
    """Converts an integer to its ordinal as a string.

    For example, 1 is "1st", 2 is "2nd", 3 is "3rd", etc. Works for any integer or
    anything `int()` will turn into an integer. Anything else will return the output
    of str(value).

    Examples:
        ```pycon
        >>> ordinal(1)
        '1st'
        >>> ordinal(1002)
        '1002nd'
        >>> ordinal(103)
        '103rd'
        >>> ordinal(4)
        '4th'
        >>> ordinal(12)
        '12th'
        >>> ordinal(101)
        '101st'
        >>> ordinal(111)
        '111th'
        >>> ordinal("something else")
        'something else'
        >>> ordinal([1, 2, 3]) == "[1, 2, 3]"
        True

        ```
    Args:
        value (int, str, float): Integer to convert.
        gender (str): Gender for translations. Accepts either "male" or "female".

    Returns:
        str: Ordinal string.
    """
    try:
        if not math.isfinite(float(value)):
            return _format_not_finite(float(value))
        value = int(value)
    except (TypeError, ValueError):
        return str(value)
    if gender == "male":
        t = (
            P_("0 (male)", "th"),
            P_("1 (male)", "st"),
            P_("2 (male)", "nd"),
            P_("3 (male)", "rd"),
            P_("4 (male)", "th"),
            P_("5 (male)", "th"),
            P_("6 (male)", "th"),
            P_("7 (male)", "th"),
            P_("8 (male)", "th"),
            P_("9 (male)", "th"),
        )
    else:
        t = (
            P_("0 (female)", "th"),
            P_("1 (female)", "st"),
            P_("2 (female)", "nd"),
            P_("3 (female)", "rd"),
            P_("4 (female)", "th"),
            P_("5 (female)", "th"),
            P_("6 (female)", "th"),
            P_("7 (female)", "th"),
            P_("8 (female)", "th"),
            P_("9 (female)", "th"),
        )
    if value % 100 in (11, 12, 13):  # special case
        return f"{value}{t[0]}"
    return f"{value}{t[value % 10]}"


def intcomma(value: NumberOrString, ndigits: int | None = None) -> str:
    """Converts an integer to a string containing commas every three digits.

    For example, 3000 becomes "3,000" and 45000 becomes "45,000". To maintain some
    compatibility with Django's `intcomma`, this function also accepts floats.

    Examples:
        ```pycon
        >>> intcomma(100)
        '100'
        >>> intcomma("1000")
        '1,000'
        >>> intcomma(1_000_000)
        '1,000,000'
        >>> intcomma(1_234_567.25)
        '1,234,567.25'
        >>> intcomma(1234.5454545, 2)
        '1,234.55'
        >>> intcomma(14308.40, 1)
        '14,308.4'
        >>> intcomma("14308.40", 1)
        '14,308.4'
        >>> intcomma(None)
        'None'

        ```
    Args:
        value (int, float, str): Integer or float to convert.
        ndigits (int, None): Digits of precision for rounding after the decimal point.

    Returns:
        str: String containing commas every three digits.
    """
    thousands_sep = thousands_separator()
    decimal_sep = decimal_separator()
    try:
        if isinstance(value, str):
            value = value.replace(thousands_sep, "").replace(decimal_sep, ".")
            if not math.isfinite(float(value)):
                return _format_not_finite(float(value))
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        else:
            if not math.isfinite(float(value)):
                return _format_not_finite(float(value))
            float(value)
    except (TypeError, ValueError):
        return str(value)

    if ndigits is not None:
        orig = "{0:.{1}f}".format(value, ndigits)
    else:
        orig = str(value)
    orig = orig.replace(".", decimal_sep)
    while True:
        new = re.sub(r"^(-?\d+)(\d{3})", rf"\g<1>{thousands_sep}\g<2>", orig)
        if orig == new:
            return new
        orig = new

powers = [10**x for x in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]

human_powers = (
    NS_("thousand", "thousand"),
    NS_("million", "million"),
    NS_("billion", "billion"),
    NS_("trillion", "trillion"),
    NS_("quadrillion", "quadrillion"),
    NS_("quintillion", "quintillion"),
    NS_("sextillion", "sextillion"),
    NS_("septillion", "septillion"),
    NS_("octillion", "octillion"),
    NS_("nonillion", "nonillion"),
    NS_("decillion", "decillion"),
    NS_("googol", "googol"),
)


def intword(value: NumberOrString, format: str = "%.1f") -> str:
    """Converts a large integer to a friendly text representation.

    Works best for numbers over 1 million. For example, 1_000_000 becomes "1.0 million",
    1200000 becomes "1.2 million" and "1_200_000_000" becomes "1.2 billion". Supports up
    to decillion (33 digits) and googol (100 digits).

    Examples:
        ```pycon
        >>> intword("100")
        '100'
        >>> intword("12400")
        '12.4 thousand'
        >>> intword("1000000")
        '1.0 million'
        >>> intword(1_200_000_000)
        '1.2 billion'
        >>> intword(8100000000000000000000000000000000)
        '8.1 decillion'
        >>> intword(None)
        'None'
        >>> intword("1234000", "%0.3f")
        '1.234 million'

        ```
    Args:
        value (int, float, str): Integer to convert.
        format (str): To change the number of decimal or general format of the number
            portion.

    Returns:
        str: Friendly text representation as a string, unless the value passed could not
        be coaxed into an `int`.
    """
    try:
        if not math.isfinite(float(value)):
            return _format_not_finite(float(value))
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if value < 0:
        value *= -1
        negative_prefix = "-"
    else:
        negative_prefix = ""

    if value < powers[0]:
        return negative_prefix + str(value)

    for ordinal_, power in enumerate(powers[1:], 1):
        if value < power:
            chopped = value / float(powers[ordinal_ - 1])
            if float(format % chopped) == float(10**3):
                chopped = value / float(powers[ordinal_])
                singular, plural = human_powers[ordinal_]
                return (
                    negative_prefix
                    + " ".join(
                        [format, _ngettext(singular, plural, math.ceil(chopped))]
                    )
                ) % chopped

            singular, plural = human_powers[ordinal_ - 1]
            return (
                negative_prefix
                + " ".join([format, _ngettext(singular, plural, math.ceil(chopped))])
            ) % chopped

    return negative_prefix + str(value)
