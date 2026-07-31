"""Build an index from a set of documents in one call."""

from __future__ import annotations

from .textindex import InvertedIndex, merge_indexes


def index_all(documents: dict[str, str]) -> InvertedIndex:
    """Index every ``doc_id -> text`` pair and return the finished index."""
    index = InvertedIndex()
    for doc_id, text in documents.items():
        index.add(doc_id, text)
    return index


def index_shards(shards: list[dict[str, str]]) -> InvertedIndex:
    """Index each shard separately, then fold the results into one index."""
    merged = InvertedIndex()
    for shard in shards:
        merged = merge_indexes(merged, index_all(shard))
    return merged
