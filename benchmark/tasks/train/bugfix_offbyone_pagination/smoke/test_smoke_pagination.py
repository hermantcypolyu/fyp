"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that the common path
still works, so the agent can catch itself breaking something. They do NOT test the
reported bug, because run_tests() must never hand over the grading signal
(AGENT_BUILD_SPEC 5.5).
"""

from src import Query
from src.pagination import paginate


def test_package_imports_and_exposes_the_public_api():
    from src import Page, total_pages  # noqa: F401

    assert callable(paginate)


def test_first_page_of_a_short_list():
    page = paginate(["x", "y"], page=1, per_page=10)
    assert page.items == ["x", "y"]
    assert page.has_prev is False


def test_query_chaining_still_works():
    rows = [{"n": 3}, {"n": 1}, {"n": 2}]
    assert Query(rows).order_by("n").page(1, per_page=2).items == [{"n": 1}, {"n": 2}]
