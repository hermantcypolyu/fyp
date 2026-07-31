# reportkit

Plain-text reports, with optional renderers behind extras.

```python
from reportkit import render_report

print(render_report("Weekly", ["job", "runs"], [["build", "41"], ["deploy", "7"]]))
```

## Optional features

Everything beyond the plain-text renderer is optional and sits behind an extra. The
authority on which extras exist is `SUPPORTED_EXTRAS` in `reportkit/features.py`: those
keys are the only names `available()`, `install_hint()` and `require()` accept, and they
are the names printed in the install hints users see when a feature is missing.

`[project.optional-dependencies]` has to declare exactly those extras, under exactly
those names, or the hint the library prints is a command that pip rejects.

## House rules for dependency declarations

* Every extra the library supports is declared, and no extra is declared that the library
  does not support. The two lists are the same set.
* An extra never re-declares or re-pins something already in the core `dependencies`
  list. The core pin is the only pin.
* Every environment marker has to be satisfiable on the oldest interpreter
  `requires-python` admits. A marker that can never be true makes an extra install
  silently empty, which is worse than an error: the feature just stays off.

## Layout

    pyproject.toml
    reportkit/
      __init__.py   re-exports the public API
      features.py   SUPPORTED_EXTRAS, available(), install_hint(), require()
      table.py      column widths and row rendering
      render.py     the renderers

The Python is correct and settled. Nothing under `reportkit/` needs to change - in
particular, `SUPPORTED_EXTRAS` is the specification, not the thing that is wrong.
