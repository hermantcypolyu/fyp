"""Hidden verification for `bugfix_image_alt_double_escape`.

The behaviour graded here is the one fixed upstream in mistune commit
0d6f3d850283d51e9c60e5a1b3c9343a18df3722 ("fix: remove double-encoding of image alt
text"). Upstream expresses it as three assertions inside one end-to-end test that parses
markdown; the parser is not vendored here, so the same consequences are exercised against
`HTMLRenderer` directly, with the pre-escaped strings the rendering pipeline would hand
to `image()`.

Eight tests cover the defect - each a separate way the double encoding shows up - and
five cover behaviour that already works and must not regress. The regression tests exist
because the cheapest wrong answers (drop the tag stripping, or decode the text back to
raw markup, or stop escaping in `text()`) all produce an `alt` that is wrong in some
other way.
"""

import html as stdlib_html

from mistune.renderers.html import HTMLRenderer


def render() -> HTMLRenderer:
    return HTMLRenderer()


# --- the defect ------------------------------------------------------------------


def test_ampersand_in_alt_is_not_encoded_twice():
    out = render().image("dogs &amp; cats", "dogs.png")
    assert out == '<img src="dogs.png" alt="dogs &amp; cats" />'


def test_angle_bracket_entities_in_alt_are_not_encoded_twice():
    out = render().image("a &lt; b &gt; c", "cmp.png")
    assert out == '<img src="cmp.png" alt="a &lt; b &gt; c" />'


def test_quote_entities_in_alt_are_not_encoded_twice():
    out = render().image("&quot;quoted&quot;", "q.png")
    assert out == '<img src="q.png" alt="&quot;quoted&quot;" />'


def test_named_entity_written_by_the_author_survives_unchanged():
    # the author wrote `&copy;`, which the pipeline already escaped to `&amp;copy;`
    out = render().image("&amp;copy; 2026", "c.png")
    assert out == '<img src="c.png" alt="&amp;copy; 2026" />'


def test_alt_is_left_alone_when_the_text_holds_no_markup():
    text = "A &amp; B &lt;tag&gt; &quot;q&quot;"
    out = render().image(text, "mix.png")
    assert out == '<img src="mix.png" alt="' + text + '" />'


def test_alt_decodes_back_to_what_the_author_typed():
    out = render().image("dogs &amp; cats", "dogs.png")
    alt = out.split('alt="', 1)[1].split('"', 1)[0]
    assert stdlib_html.unescape(alt) == "dogs & cats"


def test_alt_from_emphasised_text_is_not_encoded_twice():
    out = render().image("<em>a &amp; b</em>", "em.png")
    assert out == '<img src="em.png" alt="a &amp; b" />'


def test_full_tag_with_a_title_is_correct_in_every_attribute():
    out = render().image("dogs &amp; cats", "dogs.png", "Tom & Jerry")
    assert out == '<img src="dogs.png" alt="dogs &amp; cats" title="Tom &amp; Jerry" />'


# --- behaviour that already works and must not regress ---------------------------


def test_inline_tags_are_still_stripped_from_alt():
    out = render().image("<em>hi</em>", "em.png")
    assert out == '<img src="em.png" alt="hi" />'


def test_html_comments_are_still_stripped_from_alt():
    out = render().image("before<!-- note -->after", "c.png")
    assert out == '<img src="c.png" alt="beforeafter" />'


def test_alt_never_contains_raw_markup_characters():
    out = render().image("5 &lt; 6 &quot;x&quot;", "raw.png")
    alt = out.split('alt="', 1)[1].split('"', 1)[0]
    assert "<" not in alt
    assert ">" not in alt
    assert '"' not in alt


def test_ordinary_text_rendering_still_escapes():
    assert render().text("a & b <c>") == "a &amp; b &lt;c&gt;"


def test_harmful_image_urls_are_still_refused():
    out = render().image("logo", "javascript:alert(1)")
    assert out == '<img src="#harmful-link" alt="logo" />'
