"""Hidden verification suite for docs_readme_and_exports_textindex.

Never mounted into the sandbox. Fourteen independent assertions, no parametrisation, so
that a partly-corrected repository earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

``__all__`` lives in the module, so the agent has to be able to write the module, and
that leaves the obvious cheat: rename ``add`` back to ``add_document``, give it a
``weight`` argument, make ``search`` return bare ids, and the old README becomes true.
The first seven tests close it by pinning what the index really does. The last seven
read the documentation: the two README sessions are extracted from their fenced blocks
and executed, once graded on whether anything raised and once on whether the printed
results matched, and ``__all__`` is compared against both a star-import and the public
surface the README lists.

Nothing here grades prose. Every assertion is a property of the repository as executed.
"""

from __future__ import annotations

import doctest
import inspect
import re
from pathlib import Path

import src.textindex as textindex
from src.textindex import InvertedIndex, merge_indexes, tokenise

BLOCK_RE = re.compile(r"^```(?:[a-z]*)\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
BULLET_RE = re.compile(r"^\s*[-*]\s+`([A-Za-z_][A-Za-z0-9_]*)`", re.MULTILINE)


def _readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")


def _sessions() -> list[str]:
    """The fenced code blocks of README.md that are interactive sessions."""
    return [m.group(1) for m in BLOCK_RE.finditer(_readme()) if ">>>" in m.group(1)]


def _documented_surface() -> set[str]:
    """The names the README's public API section presents as exported."""
    for chunk in re.split(r"^##\s+", _readme(), flags=re.MULTILINE)[1:]:
        if chunk.lower().startswith("public api"):
            return set(BULLET_RE.findall(chunk))
    return set()


def _defined_surface() -> set[str]:
    """The public classes and functions the module itself defines."""
    return {
        name
        for name, obj in vars(textindex).items()
        if not name.startswith("_")
        and (inspect.isclass(obj) or inspect.isfunction(obj))
        and obj.__module__ == textindex.__name__
    }


class _CountingRunner(doctest.DocTestRunner):
    """Separates "that blew up" from "that printed something else"."""

    def __init__(self) -> None:
        super().__init__(verbose=False)
        self.raised = 0
        self.mismatched = 0

    def report_unexpected_exception(self, out, test, example, exc_info):
        self.raised += 1
        super().report_unexpected_exception(out, test, example, exc_info)

    def report_failure(self, out, test, example, got):
        self.mismatched += 1
        super().report_failure(out, test, example, got)


def _run_session(number: int) -> tuple[_CountingRunner, str]:
    sessions = _sessions()
    assert len(sessions) > number, f"README.md no longer has {number + 1} runnable sessions"
    runner = _CountingRunner()
    test = doctest.DocTestParser().get_doctest(sessions[number], {}, f"README session {number}", "README.md", 0)
    report: list[str] = []
    runner.run(test, out=report.append)
    return runner, "".join(report)


# --- what the index really does, and must go on doing ---------------------------------


def test_tokenising_lower_cases_splits_and_drops_stop_words():
    assert tokenise("The Quick, brown FOX!") == ["quick", "brown", "fox"]
    assert tokenise("of and the") == []


def test_search_scores_by_term_frequency_and_orders_best_first():
    index = InvertedIndex()
    index.add("d1", "the quick brown fox")
    index.add("d2", "a quick quick red fox")
    assert index.search("quick fox") == [("d2", 3), ("d1", 2)]


def test_a_query_that_matches_nothing_comes_back_empty():
    index = InvertedIndex()
    index.add("d1", "the quick brown fox")
    assert index.search("aardvark") == []
    assert index.search("") == []


def test_adding_the_same_id_twice_replaces_the_document():
    index = InvertedIndex()
    index.add("d1", "the quick brown fox")
    index.add("d1", "slow green turtle")
    assert len(index) == 1
    assert index.search("quick") == []
    assert index.search("turtle") == [("d1", 1)]


def test_postings_and_documents_report_what_was_indexed():
    index = InvertedIndex()
    index.add("d1", "the quick brown fox")
    index.add("d2", "a quick red fox")
    assert index.documents() == ["d1", "d2"]
    assert index.postings("quick") == {"d1": 1, "d2": 1}
    assert index.postings("brown") == {"d1": 1}
    assert index.postings("aardvark") == {}


def test_merging_two_indexes_leaves_both_of_them_alone():
    left, right = InvertedIndex(), InvertedIndex()
    left.add("d1", "the quick brown fox")
    right.add("d2", "a quick red fox")
    both = merge_indexes(left, right)
    assert both.documents() == ["d1", "d2"]
    assert both.search("quick") == [("d1", 1), ("d2", 1)]
    assert left.documents() == ["d1"]
    assert right.documents() == ["d2"]


def test_the_readme_still_carries_both_sessions_and_a_public_api_list():
    sessions = _sessions()
    assert len(sessions) >= 2, "an example was deleted rather than corrected"
    assert sum(session.count(">>>") for session in sessions) >= 8
    assert _documented_surface(), "the README no longer says what the module exports"


# --- what the documentation claims ----------------------------------------------------


def test_the_usage_session_runs_without_raising():
    runner, report = _run_session(0)
    assert runner.raised == 0, report


def test_the_usage_session_prints_what_it_claims():
    runner, report = _run_session(0)
    assert runner.mismatched == 0, report


def test_the_merge_session_runs_without_raising():
    runner, report = _run_session(1)
    assert runner.raised == 0, report


def test_the_merge_session_prints_what_it_claims():
    runner, report = _run_session(1)
    assert runner.mismatched == 0, report


def test_a_star_import_brings_in_every_name_the_module_defines():
    namespace: dict[str, object] = {}
    exec("from src.textindex import *", namespace)  # noqa: S102 - the export list is what is under test
    assert _defined_surface() <= set(namespace)


def test_all_lists_exactly_the_public_names_the_module_defines():
    assert set(textindex.__all__) == _defined_surface()


def test_all_agrees_with_the_public_api_the_readme_presents():
    assert set(textindex.__all__) == _documented_surface()
