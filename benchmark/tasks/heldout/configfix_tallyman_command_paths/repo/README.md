# tallyman

Tally simple `category,amount` ledgers.

```
# groceries.csv
food,12.50
transport,3.20
food,7.00
```

## Commands

Two console scripts are installed:

- `tally groceries.csv` prints the total booked against each category.
  `--min` hides categories below a threshold.
- `tally-report groceries.csv` prints the same block, then the grand total and
  the leading category.

## Layout

The command implementations live one module per script under `tallyman/commands/`:
`tallyman/commands/count.py` backs `tally`, and `tallyman/commands/report.py`
backs `tally-report`. Both expose `main(argv=None) -> int`.

The library itself is importable directly:

```python
from tallyman import parse_lines, totals

totals(parse_lines(["food,12.50", "food,7.00"]))
# {'food': 19.5}
```
