# dataspec

Tiny record validation against a compact schema. No dependencies.

```python
from dataspec import min_length, parse_schema, validate

schema = parse_schema(["name:str!", "age:int"])
validate({"name": "a", "age": 36}, schema, {"name": [min_length(2)]})
# ["name: is shorter than 2"]
```

## The 2.0.0 move, which is only half done

Up to 1.x this project was distributed as `spectool`, laid out flat, with the importable
package sitting in `spectool/` beside `pyproject.toml`. 2.0.0 does two things at once:

* the package is renamed to `dataspec`, and
* it moves into a `src/` layout, so the importable package is now `src/dataspec/`.

`src/dataspec/` is where the code actually lives now. The packaging metadata, the test
configuration and one import statement were never updated to follow it.

## House rules this project holds itself to

* The distribution name normalises (PEP 503) to the import name. One project, one name.
* Intra-package imports are relative. A module under `src/dataspec/` never spells out a
  top-level package name to reach its own siblings.
* The wheel `packages` entry is a path that exists in the source tree, so a build cannot
  quietly produce an empty wheel.
* `pytest` is run from the project root with no arguments, and the pytest configuration
  in `pyproject.toml` is what puts the package on the import path.

## Layout

    pyproject.toml
    src/
      dataspec/
        __init__.py     re-exports the public API
        fields.py       compact field specs -> Field
        rules.py        reusable value rules
        validate.py     record + schema -> list of problems
    tests/              the project's own suite

The Python is correct and settled. `fields.py`, `rules.py` and `validate.py` do not need
to change.
