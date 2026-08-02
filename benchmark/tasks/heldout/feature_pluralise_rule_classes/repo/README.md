# plural

Turn english nouns into their plural form.

```python
from plural import pluralise

pluralise("party")   # 'parties'
pluralise("bus")     # 'buses'
pluralise("cat")     # 'cats'
```

Today the engine knows only the generic suffix rules in `plural/rules.py`, tried
in order, with a plain trailing `s` as the fallback.

## Planned: the ending rules that suffixes cannot express

Three classes of word are currently wrong, and none of them can be expressed as
a `SuffixRule`. They are resolved in `plural/engine.py`, ahead of the generic
rules, in this order:

**1. Uncountable nouns** are their own plural and must come back untouched:

    sheep, series, species, fish, information, equipment

**2. Irregular forms** are looked up rather than derived:

| singular | plural | | singular | plural |
| --- | --- | --- | --- | --- |
| child | children | | tooth | teeth |
| person | people | | foot | feet |
| mouse | mice | | man | men |
| goose | geese | | woman | women |

**3. `-f` and `-fe` endings** become `-ves`: `leaf` → `leaves`, `knife` →
`knives`, `wolf` → `wolves`. Four words keep the plain `s` instead and must be
excluded: `roof`, `chief`, `belief`, `cliff`.

Finally, a **leading capital is preserved** across all of the above, so
`Child` → `Children` and `Leaf` → `Leaves`. The generic suffix rules already
carry case through on their own.
