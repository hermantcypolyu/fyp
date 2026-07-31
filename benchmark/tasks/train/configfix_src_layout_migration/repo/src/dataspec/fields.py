"""Schema fields, parsed from the compact string form used in config files.

    "name:str!"   required string
    "age:int"     optional integer
    "tags:list"   optional list

Self-contained on purpose: this module is the bottom of the package and imports nothing
from its siblings.
"""

from __future__ import annotations

from typing import NamedTuple

TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool, "list": list}


class Field(NamedTuple):
    """One declared field of a schema."""

    name: str
    kind: str
    required: bool


def parse_field(spec: str) -> Field:
    """Parse one compact field spec. A trailing `!` marks the field required."""
    text = spec.strip()
    required = text.endswith("!")
    if required:
        text = text[:-1]
    name, sep, kind = text.partition(":")
    if not sep:
        raise ValueError(f"field spec {spec!r} is missing its ':<type>' part")
    if kind not in TYPES:
        raise ValueError(f"unknown field type {kind!r} in {spec!r}; expected one of {', '.join(sorted(TYPES))}")
    return Field(name.strip(), kind, required)


def parse_schema(specs) -> list[Field]:
    """Parse an iterable of field specs, in order."""
    return [parse_field(spec) for spec in specs]


def python_type(field: Field) -> type:
    """The runtime type a value for this field must have."""
    return TYPES[field.kind]
