# holdfast

A small LRU cache and a memoising decorator built on it.

```python
from src import LRUCache, memoise

sessions = LRUCache(capacity=1000)
sessions.put(token, user)
sessions.get(token)          # -> user, and now the most recently used entry

@memoise(capacity=256)
def resolve(hostname: str) -> str: ...
resolve.cache.clear()
```

## The rules the cache already follows

* **Recency is read-based.** `get` on a key that is present moves it to the most
  recently used position. `put` inserts, or re-inserts, at the same position.
* **`in` is a peek.** Membership testing deliberately does not disturb recency, so
  that instrumentation and debugging cannot change which entry survives.
* **Overwriting never displaces.** Writing a key that is already stored replaces its
  value and leaves the size unchanged, so nothing is dropped to make room for it.
* **Eviction is pressure only.** Entries are dropped, oldest first, only while the
  cache is over capacity. `clear()` empties it, but that is the caller emptying it, not
  the cache displacing anything.
* **A miss is not an error.** `get` returns `default` for a key it does not hold.
* `capacity < 1` raises `ValueError`.

## Wanted: instrumentation

We size these caches by guesswork. A cache that never evicts is wasted memory and a
cache that never hits is wasted indirection, and today an instance will not tell you
which of the two you have. What is missing is per-instance counters and a way to zero
them without throwing away the warm entries - see the issue for the exact shape.
