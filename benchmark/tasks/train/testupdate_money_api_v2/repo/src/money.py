"""Fixed-point money arithmetic.

Amounts are held as whole cents to keep addition exact - binary floats cannot
represent 0.10, and a cent lost per transaction is a cent lost forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

__all__ = ["Money", "CurrencyMismatch"]


class CurrencyMismatch(ValueError):
    """Raised when two amounts in different currencies are combined."""


@dataclass(frozen=True, order=True)
class Money:
    """An exact amount in a single currency, stored as whole cents."""

    cents: int
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not isinstance(self.cents, int) or isinstance(self.cents, bool):
            raise TypeError("cents must be an int - use Money.from_string for decimal input")
        if len(self.currency) != 3 or not self.currency.isalpha():
            raise ValueError(f"currency must be a 3-letter code, got {self.currency!r}")

    @classmethod
    def from_cents(cls, cents: int, currency: str = "USD") -> "Money":
        """Build an amount from a whole number of cents."""
        return cls(cents=int(cents), currency=currency.upper())

    @classmethod
    def from_string(cls, text: str, currency: str = "USD") -> "Money":
        """Parse a decimal string such as '12.34' or '-0.05' into an exact amount."""
        try:
            value = Decimal(str(text).strip())
        except InvalidOperation as exc:
            raise ValueError(f"cannot parse {text!r} as an amount") from exc
        cents = int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return cls(cents=cents, currency=currency.upper())

    @property
    def amount(self) -> Decimal:
        """The value as an exact Decimal, e.g. Decimal('12.34')."""
        return Decimal(self.cents) / 100

    def format(self) -> str:
        """Render as '<CODE> <amount>', e.g. 'USD 12.34' or 'USD -0.05'."""
        sign = "-" if self.cents < 0 else ""
        whole, frac = divmod(abs(self.cents), 100)
        return f"{self.currency} {sign}{whole}.{frac:02d}"

    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(f"cannot combine {self.currency} and {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents - other.cents, self.currency)

    def __mul__(self, factor: int) -> "Money":
        if not isinstance(factor, int) or isinstance(factor, bool):
            raise TypeError("Money can only be multiplied by a whole number")
        return Money(self.cents * factor, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.cents, self.currency)

    def __str__(self) -> str:
        return self.format()

    def allocate(self, ratios: list[int]) -> list["Money"]:
        """Split the amount across `ratios`, distributing leftover cents from the left.

        No cent is created or destroyed: sum(result) == self, always.

        >>> [m.cents for m in Money.from_cents(100).allocate([1, 1, 1])]
        [34, 33, 33]
        """
        if not ratios or any(r < 0 for r in ratios) or sum(ratios) == 0:
            raise ValueError("ratios must be a non-empty list of non-negative ints with a positive sum")
        total = sum(ratios)
        shares = [self.cents * r // total for r in ratios]
        remainder = self.cents - sum(shares)
        for i in range(abs(remainder)):
            shares[i % len(shares)] += 1 if remainder > 0 else -1
        return [Money(c, self.currency) for c in shares]
