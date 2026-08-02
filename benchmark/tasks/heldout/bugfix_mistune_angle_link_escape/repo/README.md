# mistune (link-destination slice)

A reduced extract of [mistune](https://github.com/lepture/mistune) (BSD-3-Clause),
containing only the helpers that parse a CommonMark link destination.

`parse_link_href` handles the two destination forms:

- **angle-bracket**, `<...>` — delimited, may contain spaces
- **bare**, everything up to the first unbalanced `)` or whitespace

Both forms accept backslash escapes, which `parse_link` resolves via `unescape_char`
before the destination is URL-escaped.

    >>> parse_link_href("[a](/foo)", 3)
    ('/foo', 8)

Reference definitions use the same destination grammar with `block=True`.
