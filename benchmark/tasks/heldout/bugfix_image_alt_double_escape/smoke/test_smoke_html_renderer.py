"""Visible smoke checks. Shallow by design - they confirm the package imports and that
the renderer still produces the tags it is supposed to, and say nothing about how any
attribute is encoded.
"""

from mistune.core import BlockState
from mistune.renderers.html import HTMLRenderer


def test_renderer_constructs_and_emits_an_img_tag():
    out = HTMLRenderer().image("logo", "logo.png")
    assert out.startswith("<img ")
    assert out.endswith(" />")
    assert 'src="logo.png"' in out


def test_other_inline_renderers_still_work():
    renderer = HTMLRenderer()
    assert renderer.emphasis("hi") == "<em>hi</em>"
    assert renderer.link("home", "index.html") == '<a href="index.html">home</a>'


def test_token_rendering_pipeline_is_wired_up():
    state = BlockState()
    tokens = [{"type": "paragraph", "children": [{"type": "text", "raw": "hello"}]}]
    assert HTMLRenderer().render_tokens(tokens, state) == "<p>hello</p>\n"
