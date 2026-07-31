"""dataspec - tiny record validation against a compact schema."""

from .fields import Field, parse_field, parse_schema
from .rules import matches, min_length, one_of
from .validate import validate

__version__ = "2.0.0"

__all__ = [
    "Field",
    "matches",
    "min_length",
    "one_of",
    "parse_field",
    "parse_schema",
    "validate",
]
