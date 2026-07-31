"""The inverted index behind the handbook search box."""

from .loader import index_all, index_shards
from .textindex import InvertedIndex, merge_indexes, tokenise

__all__ = ["InvertedIndex", "index_all", "index_shards", "merge_indexes", "tokenise"]
