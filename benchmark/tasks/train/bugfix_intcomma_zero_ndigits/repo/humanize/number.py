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
from .i18n import thousands_separator

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

NumberOrString: TypeAlias = "float | str"


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
        >>> intcomma(None)
        'None'

        ```
    Args:
        value (int, float, str): Integer or float to convert.
        ndigits (int, None): Digits of precision for rounding after the decimal point.

    Returns:
        str: string containing commas every three digits.
    """
    sep = thousands_separator()
    try:
        if isinstance(value, str):
            float(value.replace(sep, ""))
        else:
            float(value)
    except (TypeError, ValueError):
        return str(value)

    if ndigits:
        orig = "{0:.{1}f}".format(value, ndigits)
    else:
        orig = str(value)

    new = re.sub(r"^(-?\d+)(\d{3})", rf"\g<1>{sep}\g<2>", orig)
    if orig == new:
        return new
    else:
        return intcomma(new)


def apnumber(value: NumberOrString) -> str:
    """Converts an integer to Associated Press style.

    Examples:
      ```pycon
      >>> apnumber(0)
      'zero'
      >>> apnumber(5)
      'five'
      >>> apnumber(10)
      '10'
      >>> apnumber("7")
      'seven'
      >>> apnumber("foo")
      'foo'
      >>> apnumber(None)
      'None'

      ```
    Args:
        value (int, float, str): Integer to convert.

    Returns:
        str: For numbers 0-9, the number spelled out. Otherwise, the number. This always
        returns a string unless the value was not `int`-able, then `str(value)`
        is returned.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)
    if not 0 <= value < 10:
        return str(value)
    return (
        _("zero"),
        _("one"),
        _("two"),
        _("three"),
        _("four"),
        _("five"),
        _("six"),
        _("seven"),
        _("eight"),
        _("nine"),
    )[value]
