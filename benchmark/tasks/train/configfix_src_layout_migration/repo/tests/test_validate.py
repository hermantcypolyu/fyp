"""The project's own suite. Run it with `pytest` from the project root."""

from dataspec import parse_schema, validate


def test_missing_required_field_is_reported():
    schema = parse_schema(["name:str!", "age:int"])
    assert validate({"age": 36}, schema) == ["name: missing"]


def test_a_good_record_has_no_problems():
    schema = parse_schema(["name:str!", "age:int"])
    assert validate({"name": "ada", "age": 36}, schema) == []
