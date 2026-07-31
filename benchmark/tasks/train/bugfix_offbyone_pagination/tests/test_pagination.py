"""Hidden verification suite for bugfix_offbyone_pagination.

Never mounted into the sandbox. Eleven independent assertions so that a partial fix
earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5) rather than an all-or-nothing 0/1.

Against the untouched repo, 6 of these pass and 5 fail.
"""

import pytest

from src.pagination import Page, paginate, total_pages

LETTERS = list("abcdefghij")  # 10 items - deliberately not a multiple of 3


# --- these pass before and after the fix: they pin behaviour that must not regress ---


def test_first_page_returns_leading_items():
    assert paginate(LETTERS, page=1, per_page=3).items == ["a", "b", "c"]


def test_middle_page_returns_correct_slice():
    assert paginate(LETTERS, page=2, per_page=3).items == ["d", "e", "f"]


def test_page_echoes_its_own_parameters():
    page = paginate(LETTERS, page=2, per_page=4)
    assert (page.page, page.per_page, page.total_items) == (2, 4, 10)


def test_total_pages_when_it_divides_evenly():
    assert total_pages(10, 5) == 2


def test_invalid_arguments_raise_value_error():
    with pytest.raises(ValueError):
        total_pages(10, 0)
    with pytest.raises(ValueError):
        paginate(LETTERS, page=0)


def test_start_and_end_indices_are_one_based():
    page = paginate(LETTERS, page=2, per_page=3)
    assert (page.start_index, page.end_index) == (4, 6)


# --- these fail against the untouched repo ---


def test_total_pages_rounds_up_when_there_is_a_remainder():
    # 10 items at 3 per page needs 4 pages, not 3 - the tenth item has to live somewhere.
    assert total_pages(10, 3) == 4


def test_final_partial_page_is_reachable():
    assert paginate(LETTERS, page=4, per_page=3).items == ["j"]


def test_every_item_appears_exactly_once_across_all_pages():
    seen = []
    for number in range(1, total_pages(len(LETTERS), 3) + 1):
        seen.extend(paginate(LETTERS, page=number, per_page=3).items)
    assert seen == LETTERS


def test_has_next_is_false_only_on_the_last_page():
    flags = [paginate(LETTERS, page=n, per_page=3).has_next for n in (1, 2, 3, 4)]
    assert flags == [True, True, True, False]


def test_empty_collection_still_has_exactly_one_page():
    assert total_pages(0, 10) == 1
    page = paginate([], page=1, per_page=10)
    assert isinstance(page, Page)
    assert page.items == []
    assert page.total_pages == 1
    assert page.has_next is False
