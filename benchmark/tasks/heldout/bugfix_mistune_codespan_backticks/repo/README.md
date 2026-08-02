# mistune (Markdown renderer slice)

A reduced extract of [mistune](https://github.com/lepture/mistune) (BSD-3-Clause) carrying
`MarkdownRenderer`, the renderer that turns a parsed token stream back into Markdown.

Its defining invariant is a round trip: re-formatting valid Markdown must not change what
the document means. Every method therefore has to emit text that re-parses to the token it
came from - which is why `text()` escapes a literal backtick, and why a fenced block sizes
its fence to clear anything inside it.

    >>> MarkdownRenderer().text({"raw": "a`b"}, BlockState())
    'a\\`b'
