"""A small inverted index: words to the documents that contain them.

There is no ranking model here beyond raw term frequency. It is enough for the internal
search box over the handbook, which is the only thing that uses it, and it fits in one
file that a new maintainer can read in a sitting.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

__all__ = ["InvertedIndex", "merge_indexes", "tokenise"]

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOP_WORDS = frozenset({"a", "an", "and", "of", "or", "the", "to"})


def tokenise(text: str) -> list[str]:
    """Lower-case ``text``, split it into words and drop the stop words."""
    return [word for word in _WORD_RE.findall(text.lower()) if word not in _STOP_WORDS]


class InvertedIndex:
    """Maps every indexed word to the documents it appears in, and how often."""

    def __init__(self) -> None:
        self._postings: dict[str, Counter[str]] = defaultdict(Counter)
        self._lengths: dict[str, int] = {}

    def add(self, doc_id: str, text: str) -> None:
        """Index ``text`` under ``doc_id``, replacing whatever was stored there before."""
        self.remove(doc_id)
        words = tokenise(text)
        for word in words:
            self._postings[word][doc_id] += 1
        self._lengths[doc_id] = len(words)

    def remove(self, doc_id: str) -> None:
        """Drop ``doc_id`` and every posting that mentions it. A no-op if it is absent."""
        if doc_id not in self._lengths:
            return
        del self._lengths[doc_id]
        for word in list(self._postings):
            counts = self._postings[word]
            counts.pop(doc_id, None)
            if not counts:
                del self._postings[word]

    def postings(self, word: str) -> dict[str, int]:
        """How often ``word`` appears in each document that contains it."""
        return dict(self._postings.get(word, {}))

    def documents(self) -> list[str]:
        """Every indexed document id, in the order it was first added."""
        return list(self._lengths)

    def search(self, query: str) -> list[tuple[str, int]]:
        """Score documents against ``query`` and return the ones that matched.

        A document's score is how many times the query's words occur in it. Results come
        back best first, and ties are broken by document id so the order is stable.
        Documents that match nothing are left out.
        """
        scores: Counter[str] = Counter()
        for word in tokenise(query):
            scores.update(self._postings.get(word, {}))
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def __len__(self) -> int:
        return len(self._lengths)


def merge_indexes(left: InvertedIndex, right: InvertedIndex) -> InvertedIndex:
    """Combine two indexes into a third, leaving both originals untouched.

    Where the same document id appears in both, the postings are added together, so
    merging overlapping indexes double-counts. Nothing in the handbook pipeline does
    that: the shards are built from disjoint directories.
    """
    merged = InvertedIndex()
    for source in (left, right):
        for word, postings in source._postings.items():
            merged._postings[word].update(postings)
        merged._lengths.update(source._lengths)
    return merged
