# textkit

Small, dependency-free text helpers.

```python
>>> import textkit
>>> textkit.slugify("Crème Brûlée, please!")
'creme-brulee-please'
>>> textkit.title_case("the wind in the willows")
'The Wind in the Willows'
>>> textkit.snake_case("HTTPResponseCode")
'http_response_code'
```

## Layout

The package is a plain flat layout - `textkit/` sits directly in the project root, next
to `pyproject.toml`. There is no `src/` directory.

## Supported Python

The code uses PEP 604 unions (`int | None`) at runtime in annotations that are evaluated
by `typing.get_type_hints`, and `frozenset[str]` subscripting, so the real floor is
Python 3.10.

## Release checklist

* `textkit.__version__` and the `version` field in `pyproject.toml` must agree - the CI
  release job compares them and refuses to publish on a mismatch.
* The distribution name and the import name should match after PEP 503 normalisation.
