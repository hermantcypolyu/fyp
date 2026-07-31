# throttle

Rate limiters for our API edge. Each one is handed a **clock** — a zero-argument
callable returning seconds as a float — instead of reading the time itself, so the whole
package is testable without sleeping and a replay can run a day of traffic instantly.

```python
from time import monotonic
from src import FixedWindowLimiter, KeyedLimiter

per_tenant = KeyedLimiter(lambda: FixedWindowLimiter(limit=100, window_s=60.0, clock=monotonic))
if not per_tenant.allow(tenant_id):
    raise TooManyRequests
```

## What `FixedWindowLimiter` already does

* Grants at most `limit` units per window of `window_s` seconds. `remaining` is what is
  left of the current window, and reading it rolls the window over if it has expired.
* The window rolls when `now - window_start >= window_s`, and the new window starts at
  `now` — the windows are not aligned to any external grid.
* A refused request spends nothing.
* `allow(n)` with `n < 1` or `n > limit` raises `ValueError`; so does construction with
  `limit < 1` or `window_s <= 0`. A request that could never be granted is a caller bug,
  not backpressure.

`KeyedLimiter` gives each key its own limiter from a factory, `for_key` exposes it,
`forget(key)` drops it, and `len()` counts the keys seen.

## Wanted: `TokenBucket`, in `src/limiter.py`

The fixed window is wrong at the boundary and cannot say "slow on average, bursty in the
moment". We want a token bucket next to it, constructed as
`TokenBucket(capacity, refill_per_s, clock)` and following these rules exactly:

* A new bucket starts **full**: `tokens` is `float(capacity)`.
* `tokens` is a **read-only property** returning the current level as a `float`, exact
  and unrounded. Assigning to it must raise `AttributeError`.
* Time is only ever read through the injected clock, and only when `allow()` is called
  or `tokens` is read. Both apply any pending refill *before* answering.
* **Refill is continuous, not stepped.** A refill sets
  `tokens = min(capacity, tokens + elapsed * refill_per_s)`, where `elapsed` is the
  current clock reading minus the reading at the previous refill. **Fractions carry**:
  two 0.25 s gaps at 2 tokens/s must leave the same 1.0 token as one 0.5 s gap, because
  the stored reading advances to now on every refill and nothing is truncated away.
* The bucket never holds more than `capacity`, however long it idles.
* `allow(n=1)` is all-or-nothing: if at least `n` tokens are available it removes exactly
  `n` and returns `True`; otherwise it removes nothing and returns `False`.
* `allow(n)` with `n < 1` or `n > capacity` raises `ValueError`, matching the fixed
  window. Construction with `capacity < 1` or `refill_per_s < 0` raises `ValueError`
  (`refill_per_s == 0` is legal — a one-shot budget that never refills).
* A clock reading **below** the previous one counts as zero elapsed time and is ignored
  entirely: it neither adds nor removes tokens, and the bucket keeps the highest reading
  it has seen, so a later forward reading is measured from that high-water mark.

`FixedWindowLimiter` and `KeyedLimiter` are used by the gateway and by two other
services; their signatures, behaviour and exception types must not change.
