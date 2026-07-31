# reportdates

Date arithmetic for the weekly reporting pipeline. Two modules:

| module | what it does |
| --- | --- |
| `src/dateparse.py` | the primitives: parsing, formatting, day counting, month bounds |
| `src/report.py` | the summary line built on top of them |

Nothing here reads the system clock. A caller that needs "today" passes it in, which is
what lets an old window be re-rendered from the archive and come out byte-identical.

## House style

Every helper in `src/dateparse.py` carries at least one runnable example in its
docstring, written as an interactive session. Those examples are the module's real
documentation — people copy them — so they are expected to be executable and accurate:

```
python -m doctest src/dateparse.py
```

should print nothing at all. `src/report.py` is exempt: its functions need a whole
reporting window before they say anything useful, which is more setup than reads well
inline, so it documents itself in prose and leans on `dateparse` for examples.

Examples are corrected in place when behaviour moves. Deleting one, or muting it with a
`# doctest:` directive, loses the only check we have that the documentation is true.
