"""Visible smoke test: the renderer slice imports and renders a plain code span."""

from mistune import BlockState, MarkdownRenderer


def test_plain_codespan_round_trips():
    assert MarkdownRenderer().codespan({"raw": "abc"}, BlockState()) == "`abc`"
