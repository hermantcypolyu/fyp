# tablesum

A reporting helper for tables that someone else has already parsed. `src/reader.py`
turns a CSV document into a list of dictionaries; `src/summarise.py` counts them.

```python
from src import count_by, read_rows

rows = read_rows(open("sightings.csv").read())
count_by(rows, "city")   # {'leeds': 2, 'york': 1}
```

## What a row is

A row is a mapping from column name to cell value, and the table is allowed to be
ragged: not every row has to carry every column. Two things follow, and both are
already implemented:

* `column_names(rows)` is the union of the keys, in the order they were first seen -
  not sorted, because report columns should come out in the order the source produced
  them.
* A cell is **filled in** unless it is `None` or the empty string. `values_of` and
  `count_by` both skip the ones that are not; a blank cell is missing data, not a
  category called "".

Asking about a column that appears in no row at all raises `KeyError`. That is
deliberate: it is nearly always a typo in a report definition, and returning an empty
tally would hide it until someone noticed the numbers were wrong.

## Not implemented yet: per-column profiles

The reporting job wants more than one tally at a time. The agreed shape is a single
`describe(rows, columns=None)` in `src/summarise.py` that returns a mapping from column
name to that column's statistics, covering a caller-chosen subset of the columns and
telling numeric columns apart from categorical ones. Numeric columns should report
count, min, max and mean; everything else should report count and the most common
value. See the issue for the exact contract.
