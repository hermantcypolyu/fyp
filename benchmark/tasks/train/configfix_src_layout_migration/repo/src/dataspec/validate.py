"""Check a record against a parsed schema.

Problems come back as a list of `"<field>: <what is wrong>"` strings, in schema order
followed by any keys the schema does not declare. An empty list means the record is good.
"""

from __future__ import annotations

from .fields import Field, python_type


def validate(record: dict, schema: list[Field], rules: dict | None = None) -> list[str]:
    """Every problem with `record`, not just the first."""
    rules = rules or {}
    problems: list[str] = []

    for field in schema:
        if field.name not in record:
            if field.required:
                problems.append(f"{field.name}: missing")
            continue
        value = record[field.name]
        if field.kind == "int" and isinstance(value, bool):
            problems.append(f"{field.name}: expected int, got bool")
            continue
        if not isinstance(value, python_type(field)):
            problems.append(f"{field.name}: expected {field.kind}, got {type(value).__name__}")
            continue
        for rule in rules.get(field.name, ()):
            message = rule(value)
            if message is not None:
                problems.append(f"{field.name}: {message}")

    declared = {field.name for field in schema}
    for name in record:
        if name not in declared:
            problems.append(f"{name}: not in the schema")
    return problems
