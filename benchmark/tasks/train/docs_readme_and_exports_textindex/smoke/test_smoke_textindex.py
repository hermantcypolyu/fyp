"""Visible smoke checks - this is what run_tests() shows the agent.

Deliberately weak: they confirm the package still imports and that indexing and
searching still work end to end. They say nothing about the README or the export list,
because run_tests() must never hand over the grading signal (AGENT_BUILD_SPEC 5.5).
"""

from src import InvertedIndex, index_all


def test_an_indexed_word_can_be_found_again():
    index = InvertedIndex()
    index.add("d1", "the quick brown fox")
    assert [doc_id for doc_id, _score in index.search("fox")] == ["d1"]


def test_indexing_a_batch_keeps_every_document():
    index = index_all({"d1": "quick brown fox", "d2": "slow green turtle"})
    assert len(index) == 2
