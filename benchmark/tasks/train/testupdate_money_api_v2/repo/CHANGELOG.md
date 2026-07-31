# Changelog

## 2.0.0 — breaking

`src/money.py` was reworked so that every amount is stored as whole cents. The old
dollars-plus-cents constructor allowed two different objects to represent the same
amount, which broke equality and dict lookups.

| Removed (1.x) | Replacement (2.0) |
| --- | --- |
| `Money(dollars=12, cents=34)` | `Money.from_cents(1234)` |
| `Money.parse("12.34")` | `Money.from_string("12.34")` |
| `money.dollars` → `12` | `money.amount` → `Decimal("12.34")` |
| `money.format()` → `"$12.34"` | `money.format()` → `"USD 12.34"` |
| `money.split(3)` → equal parts | `money.allocate([1, 1, 1])` → weighted parts |

Behaviour that did **not** change: amounts remain immutable and hashable, `+`, `-` and
`*` still work the same way, mixing currencies still raises, and splitting an amount
still never creates or destroys a cent.

## 1.4.0

Added `Money.split()`.
