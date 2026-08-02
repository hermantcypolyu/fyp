"""Hidden verification for `bugfix_mistune_angle_link_escape`.

Upstream: lepture/mistune @ e12f0d38d028a297ffd6c8bec9b5480627fe1903,
"fix: keep escaped backslashes in angle-bracket link destinations". The repo carries
`helpers.py` from that commit's *parent*, so the defect is live.

`_parse_angle_link_href` listed "\\" in its reject set, so a single backslash anywhere
inside `<...>` aborted the whole destination. The sibling bare-destination branch in
`parse_link_href` already consumed an escape as a unit, so the two forms disagreed.

The cases below are separated by *consequence*, not by input shape, because they are
independently reachable: a fix that consumes the escape but forgets to clamp the final
`pos` crashes only on a trailing backslash; a fix applied to `parse_link_href` rather
than the angle branch leaves reference definitions broken; a fix that strips the
backslash instead of preserving it passes the destination tests but corrupts the URL
that `parse_link` derives through `unescape_char`.

The floor tests below pass against the untouched repo on purpose. They pin the
behaviour the fix must *not* regress - an unescaped "<", a line ending or a NUL still
terminates the destination, and the bare form is unaffected.
"""

from mistune.helpers import parse_link, parse_link_href

# --- the defect: an escape inside <...> must be consumed, not rejected ----------------


def test_escaped_punctuation_is_kept_in_an_angle_destination():
    assert parse_link_href(r"<foo\*bar>", 0) == (r"foo\*bar", 10)


def test_escaped_closing_angle_does_not_terminate_the_destination():
    assert parse_link_href(r"<a\>b>", 0) == (r"a\>b", 6)


def test_escaped_opening_angle_is_permitted():
    assert parse_link_href(r"<a\<b>", 0) == (r"a\<b", 6)


def test_escaped_backslash_is_consumed_as_one_unit():
    assert parse_link_href(r"<a\\b>", 0) == (r"a\\b", 6)


def test_escaped_space_is_kept():
    assert parse_link_href(r"<a\ b>", 0) == (r"a\ b", 6)


def test_reference_definitions_accept_the_same_escape():
    assert parse_link_href(r"<a\*b>", 0, block=True) == (r"a\*b", 6)


def test_parse_link_unescapes_the_destination_into_the_url():
    attrs, pos = parse_link(r"<foo\*bar>)", 0)
    assert attrs == {"url": "foo*bar"}
    assert pos == 11


def test_parse_link_still_reads_a_title_after_an_escaped_destination():
    attrs, _ = parse_link(r'<a\*b> "T")', 0)
    assert attrs == {"url": "a*b", "title": "T"}


# --- floor: behaviour the fix must not regress ---------------------------------------


def test_plain_angle_destination_is_unchanged():
    assert parse_link_href("<foo>", 0) == ("foo", 5)


def test_angle_destination_may_contain_spaces():
    assert parse_link_href("<foo bar>", 0) == ("foo bar", 9)


def test_bare_destination_is_unchanged():
    assert parse_link_href("/foo", 0) == ("/foo", 4)


def test_bare_destination_already_accepted_escapes():
    assert parse_link_href(r"foo\*bar", 0) == (r"foo\*bar", 8)


def test_unescaped_opening_angle_still_terminates():
    assert parse_link_href("<a<b>", 0) == (None, None)


def test_line_ending_still_terminates():
    assert parse_link_href("<a\nb>", 0) == (None, None)


def test_nul_still_terminates():
    assert parse_link_href("<a\x00b>", 0) == (None, None)


def test_unclosed_angle_destination_is_rejected():
    assert parse_link_href("<afoo", 0) == (None, None)


def test_trailing_backslash_is_rejected_without_running_off_the_end():
    assert parse_link_href("<a\\", 0) == (None, None)
