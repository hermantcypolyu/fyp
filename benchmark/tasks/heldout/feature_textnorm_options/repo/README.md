# textnorm

Normalise text so that two strings that mean the same thing compare equal.

```python
from textnorm import normalise

normalise("  Hello,   World! ")
# 'hello world'
```

`normalise` lowercases, drops punctuation, and joins the surviving word tokens
with a single space. `textnorm.compare` builds on it for set-style similarity.

## Planned: normalisation options

Callers need to vary three things about that pipeline without reimplementing it.
The next release adds three keyword-only options to `normalise`, all defaulting
to today's behaviour so existing calls are unaffected:

| option | default | effect |
| --- | --- | --- |
| `fold` | `False` | replace accented characters with unaccented equivalents |
| `keep_case` | `False` | leave the original case alone instead of lowercasing |
| `separator` | `" "` | join the surviving tokens with this string |

```python
normalise("Café Crème", fold=True)
# 'cafe creme'
normalise("Café Crème", fold=True, keep_case=True)
# 'Cafe Creme'
normalise("Hello, World!", separator="-")
# 'hello-world'
```

The folding table and its helper already exist as `textnorm.accents.fold_accents`,
which preserves case and expands the ligatures (`ß` becomes `ss`, `æ` becomes
`ae`). The options compose: folding happens after the optional lowercasing, and
the separator is applied last.
