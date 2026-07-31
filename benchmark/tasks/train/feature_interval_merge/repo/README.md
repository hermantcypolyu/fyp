# slotwork

Interval arithmetic for the room-booking service. Times are whole minutes since
midnight; intervals are **half-open**, `[start, end)`, so two bookings that share an
endpoint sit next to each other rather than clashing.

```python
from src import Interval, span, day_window, total_length

held = [span(9, 11), span(10, 12)]
total_length(held)          # 240 - hours sold
```

## What is already here

* `Interval(start, end)` is a frozen value object. `end <= start` raises `ValueError`,
  so no empty or reversed interval can exist at all.
* `length` is `end - start`; `contains(point)` includes the start and excludes the end.
* `overlaps(other)` is true only when the two share at least one point, so `[0, 10)` and
  `[10, 20)` do **not** overlap.
* `normalise(intervals)` returns a new list sorted by start and then by end, leaving the
  sequence it was given alone.
* `total_length(intervals)` adds the lengths up as given, double-counting anything
  booked twice.
* `src/day.py` turns clock times into minutes: `at(9, 30)`, `span(9, 17)`, `day_window()`.

## Wanted: coverage, and free time

`total_length` answers "hours sold". Nobody can currently ask "hours occupied", and
nobody can ask the question the front desk actually asks, which is when a room is free.
Two functions in `src/intervals.py`, following these rules exactly:

* `merge(intervals)` returns a **new list**, sorted ascending, holding the fewest
  intervals that cover exactly the same points as the input.
  * Overlapping intervals join, and so do **touching** ones: `[0, 10)` and `[10, 20)`
    become `[0, 20)`, because under a half-open convention they already are.
  * The input may arrive in any order and may contain nested or duplicate intervals.
  * The sequence passed in must not be modified.
  * An empty input returns an empty list.
  * No two intervals in the result overlap or touch.
* `complement(intervals, window)` returns the parts of `window` that none of
  `intervals` covers, as a list of intervals in ascending order.
  * Anything outside `window` is ignored, and anything crossing its edge is clipped
    to it.
  * The complement of no intervals at all is `[window]`.
  * An interval spanning the whole window gives `[]`, and so do two touching intervals
    that between them span it - a shared endpoint leaves no gap, not a zero-length one.
  * As with `merge`, no result is empty, and none of them touch or overlap.

None of the existing functions may change their signatures or their behaviour.
