"""Hidden verification suite for feature_lru_cache_eviction_stats.

Never mounted into the sandbox. Fourteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first six pin the caching behaviour that already exists and must not regress; the
remaining eight cover the counters. The statistics object is only ever reached through
`cache.stats`, never imported by name, so that its absence shows up as eight failures
rather than as a collection error.
"""

import pytest

from src.cache import LRUCache


# --- existing behaviour: these pass before and after the feature lands ---------------


def test_a_stored_value_comes_back_out():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert len(cache) == 1


def test_a_missing_key_returns_the_default():
    cache = LRUCache(capacity=2)
    assert cache.get("nope") is None
    assert cache.get("nope", "fallback") == "fallback"


def test_the_least_recently_used_entry_goes_first():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_reading_an_entry_makes_it_the_most_recently_used():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_overwriting_a_key_replaces_the_value_without_growing():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert cache.get("a") == 2
    assert len(cache) == 1


def test_a_capacity_below_one_is_rejected():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
    with pytest.raises(ValueError):
        LRUCache(capacity=-3)


# --- the feature ---------------------------------------------------------------------


def test_a_fresh_cache_reports_zeroed_statistics():
    stats = LRUCache(capacity=2).stats
    assert (stats.hits, stats.misses, stats.evictions, stats.size) == (0, 0, 0, 0)
    assert stats.hit_rate == 0.0


def test_hits_and_misses_are_counted_separately():
    cache = LRUCache(capacity=4)
    cache.put("a", 1)
    cache.get("a")
    cache.get("a")
    cache.get("elsewhere")
    assert (cache.stats.hits, cache.stats.misses) == (2, 1)


def test_writing_and_peeking_are_neither_hits_nor_misses():
    cache = LRUCache(capacity=4)
    cache.put("a", 1)
    cache.put("b", 2)
    assert "a" in cache
    assert "elsewhere" not in cache
    stats = cache.stats
    assert (stats.hits, stats.misses, stats.evictions, stats.size) == (0, 0, 0, 2)


def test_overwriting_an_existing_key_is_not_an_eviction():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("a", 2)
    cache.put("a", 3)
    assert cache.stats.evictions == 0
    assert cache.stats.size == 1


def test_one_eviction_is_counted_per_discarded_entry():
    cache = LRUCache(capacity=2)
    for key in ("a", "b", "c", "d"):
        cache.put(key, key.upper())
    assert cache.stats.evictions == 2
    assert cache.stats.size == 2


def test_the_hit_rate_is_rounded_to_four_decimal_places():
    cache = LRUCache(capacity=4)
    cache.put("a", 1)
    cache.get("a")
    cache.get("a")
    cache.get("elsewhere")
    assert cache.stats.hit_rate == 0.6667


def test_resetting_the_counters_keeps_the_cached_entries():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.get("elsewhere")
    cache.put("c", 3)
    assert cache.reset_stats() is None
    stats = cache.stats
    assert (stats.hits, stats.misses, stats.evictions) == (0, 0, 0)
    assert stats.size == 2
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_the_statistics_object_is_an_immutable_snapshot():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    snapshot = cache.stats
    cache.get("a")
    cache.get("elsewhere")
    assert (snapshot.hits, snapshot.misses, snapshot.size) == (0, 0, 1)
    with pytest.raises(AttributeError):
        snapshot.hits = 99
