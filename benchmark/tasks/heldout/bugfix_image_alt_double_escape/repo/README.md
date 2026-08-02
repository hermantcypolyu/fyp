# mistune (renderer extract)

A cut-down copy of [mistune](https://github.com/lepture/mistune)'s HTML renderer,
kept here for exercising the escaping rules of the markup it generates. Only the
renderer layer is vendored — the block and inline parsers are not.

```
mistune/
├── core.py               BlockState and BaseRenderer
├── util.py               escaping helpers
└── renderers/html.py     HTMLRenderer
```

## How rendering works

`BaseRenderer.render_tokens()` walks a token tree and dispatches each token to the
method named after its type. A token carrying a `raw` string goes to `text()`; a
token carrying `children` has those children rendered first, and the resulting
**already-escaped HTML string** is passed as the `text` argument of the parent's
method.

So by the time `emphasis()`, `link()` or `image()` sees its `text`, escaping has
already happened: `&` is `&amp;`, `<` is `&lt;`, and a literal `"` is `&quot;`.
Those methods wrap that string; they do not encode it a second time.

`image()` differs in shape rather than in kind: because the `alt` attribute cannot
hold markup, it runs `striptags()` over the rendered children to drop any
`<em>`/`<code>` wrappers before putting the remainder in the attribute.
`striptags()` removes tags only — entities in its input come through untouched.

## Licence

BSD-3-Clause, as upstream.
