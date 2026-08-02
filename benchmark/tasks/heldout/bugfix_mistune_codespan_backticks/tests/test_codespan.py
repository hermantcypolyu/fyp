"""Hidden verification for `bugfix_mistune_codespan_backticks`.

Upstream: lepture/mistune @ f6827e82fb0b, "Fix MarkdownRenderer re-rendering of code spans
containing backticks". The repo carries `renderers/markdown.py` from that commit's *parent*,
so the defect is live.

`MarkdownRenderer.codespan` always wrapped its content in exactly one backtick pair. A code
span whose content contains a backtick therefore re-parsed wrong: the inner backtick closed
the span early, and a lone backtick came back as an empty fenced code block. That breaks the
renderer's defining invariant - re-formatting valid Markdown must not change its meaning.

The cases are separated by *consequence*, because they are independently reachable:

- sizing the fence to the longest run fixes the single-backtick case but not a `` `` `` run;
- sizing without the space padding fixes interior backticks but still merges the delimiter
  with a leading or trailing one;
- padding without sizing fixes the lone backtick but leaves interior runs closing early.

So an agent that implements only one half of the upstream fix scores strictly between the
floor and 1.0, which is the point.

The floor tests below pass against the untouched repo on purpose: they pin the behaviour of
content with no backticks at all, and of the renderer's other methods, none of which the fix
may disturb.
"""

from mistune import BlockState, MarkdownRenderer


def render_codespan(raw: str) -> str:
    return MarkdownRenderer().codespan({"raw": raw}, BlockState())


# --- the defect: the fence must clear every backtick run in the content ----------------


def test_single_interior_backtick_widens_the_fence():
    assert render_codespan("a`b") == "``a`b``"


def test_double_interior_backtick_run_widens_the_fence_further():
    assert render_codespan("a``b") == "```a``b```"


def test_the_longest_run_decides_the_fence_width():
    assert render_codespan("a`b``c") == "```a`b``c```"


def test_leading_backtick_is_padded_away_from_the_fence():
    assert render_codespan("`x") == "`` `x ``"


def test_trailing_backtick_is_padded_away_from_the_fence():
    assert render_codespan("x`") == "`` x` ``"


def test_content_that_both_starts_and_ends_with_a_backtick_is_padded():
    assert render_codespan("`a`") == "`` `a` ``"


def test_a_lone_backtick_does_not_become_an_empty_fence():
    assert render_codespan("`") == "`` ` ``"


# --- floor: content without backticks, and the renderer's other methods ----------------


def test_plain_content_keeps_a_single_pair():
    assert render_codespan("abc") == "`abc`"


def test_content_with_spaces_is_not_padded():
    assert render_codespan("a b") == "`a b`"


def test_empty_content_is_unchanged():
    assert render_codespan("") == "``"


def test_text_still_escapes_a_literal_backtick():
    assert MarkdownRenderer().text({"raw": "a`b"}, BlockState()) == "a\\`b"


def test_linebreak_is_unchanged():
    assert MarkdownRenderer().linebreak({}, BlockState()) == "  \n"


def test_inline_html_is_passed_through():
    assert MarkdownRenderer().inline_html({"raw": "<b>"}, BlockState()) == "<b>"
