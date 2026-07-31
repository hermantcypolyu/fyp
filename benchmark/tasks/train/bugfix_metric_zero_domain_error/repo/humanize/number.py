#!/usr/bin/env python

"""Humanizing functions for numbers."""
from __future__ import annotations

import math
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

# This type can be better defined by typing.SupportsInt, typing.SupportsFloat
# but that's a Python 3.8 only typing option.
NumberOrString: TypeAlias = "float | str"


def scientific(value: NumberOrString, precision: int = 2) -> str:
    """Return number in string scientific notation z.wq x 10ⁿ.

    Examples:
        ```pycon
        >>> scientific(float(0.3))
        '3.00 x 10⁻¹'
        >>> scientific(int(500))
        '5.00 x 10²'
        >>> scientific(-1000)
        '-1.00 x 10³'
        >>> scientific(1000, 1)
        '1.0 x 10³'
        >>> scientific(1000, 3)
        '1.000 x 10³'
        >>> scientific("99")
        '9.90 x 10¹'
        >>> scientific("foo")
        'foo'
        >>> scientific(None)
        'None'

        ```

    Args:
        value (int, float, str): Input number.
        precision (int): Number of decimal for first part of the number.

    Returns:
        str: Number in scientific notation z.wq x 10ⁿ.
    """
    exponents = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
    }
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    fmt = "{:.%se}" % str(int(precision))
    n = fmt.format(value)
    part1, part2 = n.split("e")
    # Remove redundant leading '+' or '0's (preserving the last '0' for 10⁰).
    part2 = re.sub(r"^\+?(\-?)0*(.+)$", r"\1\2", part2)

    new_part2 = []
    for char in part2:
        new_part2.append(exponents[char])

    final_str = part1 + " x 10" + "".join(new_part2)

    return final_str


def metric(value: float, unit: str = "", precision: int = 3) -> str:
    """Return a value with a metric SI unit-prefix appended.

    Examples:
        ```pycon
        >>> metric(1500, "V")
        '1.50 kV'
        >>> metric(2e8, "W")
        '200 MW'
        >>> metric(220e-6, "F")
        '220 μF'
        >>> metric(1e-14, precision=4)
        '10.00 f'

        ```

    The unit prefix is always chosen so that non-significant zero digits are required.
    i.e. `123,000` will become `123k` instead of `0.123M` and `1,230,000` will become
    `1.23M` instead of `1230K`. For numbers that are either too huge or too tiny to
    represent without resorting to either leading or trailing zeroes, it falls back to
    `scientific()`.
    ```pycon
    >>> metric(1e40)
    '1.00 x 10⁴⁰'

    ```

    Args:
        value (int, float): Input number.
        unit (str): Optional base unit.
        precision (int): The number of digits the output should contain.

    Returns:
        str:
    """
    exponent = int(math.floor(math.log10(abs(value))))

    if exponent >= 27 or exponent < -24:
        return scientific(value, precision - 1) + unit

    value /= 10 ** (exponent // 3 * 3)
    if exponent >= 3:
        ordinal = "kMGTPEZY"[exponent // 3 - 1]
    elif exponent < 0:
        ordinal = "mμnpfazy"[(-exponent - 1) // 3]
    else:
        ordinal = ""
    value_ = format(value, ".%if" % (precision - (exponent % 3) - 1))
    if not (unit or ordinal) or unit in ("°", "′", "″"):
        space = ""
    else:
        space = " "

    return f"{value_}{space}{ordinal}{unit}"
