# textindex

The inverted index behind the handbook search box. One word maps to the documents that
contain it and how often; scoring is raw term frequency and nothing more, which is as
much as an internal search box over a few hundred pages needs.

## Usage

```python
>>> from src.textindex import InvertedIndex
>>> index = InvertedIndex()
>>> index.add_document("d1", "the quick brown fox", weight=2)
>>> index.add_document("d2", "a quick quick red fox")
>>> index.search("quick fox")
['d2', 'd1']
```

Results come back best first, and ties break on document id, so the same corpus always
answers the same way.

## Merging indexes

Shards are indexed separately and folded together afterwards, which is how the nightly
rebuild parallelises over the handbook's directories:

```python
>>> from src.textindex import InvertedIndex, merge_indexes
>>> left = InvertedIndex()
>>> left.add_document("d1", "the quick brown fox")
>>> right = InvertedIndex()
>>> right.add_document("d2", "a quick red fox")
>>> both = merge_indexes(left, right)
>>> both.documents()
['d1', 'd2']
>>> both.search("brown")
['d1']
```

## Public API

* `InvertedIndex` — the index itself: `add`, `remove`, `postings`, `documents`, `search`
* `tokenise` — lower-case, split, drop stop words; useful for checking what will be indexed
* `merge_indexes` — fold two indexes into a third without touching either

## House rules

* The sessions above are the documentation people copy, so they are expected to run as
  written and to print exactly what they print here.
* `src/textindex.py` keeps `__all__` equal to the public classes and functions it
  defines — the same surface this section lists. Anything the module does not mean to
  export is named with a leading underscore instead.
* `src/loader.py` is a convenience layer over the index and documents itself.
