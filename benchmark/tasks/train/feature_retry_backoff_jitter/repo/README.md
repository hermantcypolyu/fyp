# tinyretry

A retry helper small enough to vendor. Two entry points, both in `src/retry.py`:

| | |
| --- | --- |
| `call_with_retry(fn, *, attempts, delay, on, sleeper)` | call a zero-argument callable until it succeeds |
| `retry(*, attempts, delay, on, sleeper)` | the same policy as a decorator |

`src/client.py` shows the intended usage: a `ServiceClient` that retries only the
transport failures its own code classifies as transient.

## Why the sleep function is a parameter

Every entry point takes a `sleeper`, defaulting to `time.sleep`. Nothing in the library
imports a clock at the call site, so a test can pass `waits.append` and assert on the
exact sequence of pauses a policy asked for instead of waiting for them. Keep it that
way - a policy you cannot observe is a policy nobody will change.

```python
waits: list[float] = []
call_with_retry(fetch, attempts=3, delay=0.5, sleeper=waits.append)
assert waits == [0.5, 0.5]
```

## Semantics worth stating

* `attempts` is the **total** number of calls, not the number of retries. `attempts=1`
  never sleeps.
* Only the exception types in `on` are retried. Everything else propagates immediately.
* Once the budget is exhausted the exception from the final attempt is re-raised
  unchanged; the helper never wraps it.
* `attempts < 1` and `delay < 0` raise `ValueError` before the first call is attempted.

## Planned: exponential backoff, a ceiling, and full jitter

**Not implemented yet.** Today every wait is the same length, which is exactly what we
do not want during an upstream brown-out: every client in the fleet retries in lockstep
and keeps the dependency down. The agreed design adds four keyword-only options to
`call_with_retry` *and* to `retry`, with defaults that reproduce today's behaviour
exactly:

| option | default | meaning |
| --- | --- | --- |
| `backoff` | `1.0` | multiplier applied once per elapsed attempt |
| `max_delay` | `None` | ceiling on the wait, in seconds; `None` means no ceiling |
| `jitter` | `False` | whether to spread the wait randomly |
| `rng` | `None` | the random source to spread it with |

The wait after the *n*-th failed attempt, with *n* starting at 1, is computed in three
steps and in this order:

1. `wait = delay * backoff ** (n - 1)`
2. if `max_delay is not None`, `wait = min(wait, max_delay)`
3. if `jitter`, `wait = rng.uniform(0.0, wait)` - the *capped* value is the upper bound

Step 3 is "full jitter": draw uniformly from zero up to the planned wait rather than
perturbing it slightly, because only the wide draw actually de-synchronises a fleet.

`rng` is injectable for the same reason `sleeper` is: it accepts any object exposing
`uniform(a, b)`, and falls back to a module-level `random.Random` when left as `None`,
so a test can hand in a stub and get a deterministic schedule.

`backoff < 1.0` and a negative `max_delay` are configuration errors and must raise
`ValueError` alongside the existing checks.
