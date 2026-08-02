# funcy (reduced extract)

A cut-down copy of [funcy](https://github.com/Suor/funcy)'s function-manipulation
helpers, kept small enough to read in one sitting.

```
funcy/
  simple_funcs.py   identity, constantly, caller, partial-application and currying
  funcs.py          composition on top of simple_funcs: compose, complement, juxt
  funcmakers.py     turns a callable/None/index/mapping/set into a function
  cross.py          Python 2/3 compatibility shims
  primitives.py     tiny predicates plus the EMPTY sentinel
tests/
  test_funcs.py     the project's own suite
```

Run the suite with `python -m pytest tests/`.

The two helpers that bind or curry arguments from the *right* rather than the left were
renamed in the last release; `simple_funcs.py` is the release's version.
