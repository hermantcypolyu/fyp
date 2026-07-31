"""Hidden verification for `bugfix_dotenv_bom_prefix`.

Assertions follow the upstream regression test added in
theskumar/python-dotenv @ bca6644d9aedbe287b792b756b3ae3d650cd0d3a (tests/test_parser.py),
regrouped from one large `@parametrize` into behaviour-sized tests so partial credit is
meaningful, and extended with the neighbouring cases the same code path reaches.

Seven tests pass against the shipped defect and five do not. Two of the seven are guards:
`test_ordinary_parsing_is_unaffected` catches a fix that strips too eagerly, and
`test_a_mark_in_the_middle_of_a_file_is_not_stripped` catches the tempting
`.replace(...)` over the whole document, which would silently corrupt a value that
legitimately contains that character.
"""

import io

from dotenv.parser import parse_stream

BOM = "\ufeff"


def parse(text):
    return list(parse_stream(io.StringIO(text)))


def test_a_single_assignment():
    (binding,) = parse("a=b")
    assert (binding.key, binding.value, binding.error) == ("a", "b", False)


def test_several_lines():
    bindings = parse("a=b\nc=d\ne=f")
    assert [b.key for b in bindings] == ["a", "c", "e"]
    assert [b.value for b in bindings] == ["b", "d", "f"]


def test_quoted_values():
    (single,) = parse("a='b c'")
    assert (single.key, single.value) == ("a", "b c")
    (double,) = parse('a="b c"')
    assert (double.key, double.value) == ("a", "b c")


def test_export_prefix_is_accepted():
    (binding,) = parse("export a=b")
    assert (binding.key, binding.value, binding.error) == ("a", "b", False)


def test_comments_and_blank_lines_are_skipped():
    bindings = [b for b in parse("# a comment\n\na=b\n") if b.key is not None]
    assert [(b.key, b.value) for b in bindings] == [("a", "b")]


def test_ordinary_parsing_is_unaffected():
    # A fix that strips more than a leading marker would break these.
    (binding,) = parse("a=b c")
    assert binding.value == "b c"
    (quoted,) = parse("a=' b '")
    assert quoted.value == " b "


def test_a_mark_in_the_middle_of_a_file_is_not_stripped():
    # Only a leading marker is meaningful. One that appears later is ordinary data and
    # must survive - a blanket replace over the whole document would eat it.
    bindings = parse(f"a=b\n{BOM}c=d")
    assert [b.key for b in bindings] == ["a", f"{BOM}c"]


# --- the defect ------------------------------------------------------------------------
# Editors on Windows often save .env files with a leading byte-order mark. It is invisible,
# it is not whitespace, and nothing downstream strips it - so it ends up glued to the front
# of the first key and that variable silently disappears.


def test_a_leading_mark_does_not_become_part_of_the_first_key():
    (binding,) = parse(f"{BOM}a=b")
    assert binding.key == "a"
    assert binding.value == "b"
    assert binding.error is False


def test_a_leading_mark_does_not_break_the_rest_of_the_file():
    bindings = parse(f"{BOM}a=b\nc=d")
    assert [b.key for b in bindings] == ["a", "c"]
    assert [b.value for b in bindings] == ["b", "d"]


def test_a_leading_mark_before_an_export_statement():
    (binding,) = parse(f"{BOM}export a=b")
    assert binding.key == "a"
    assert binding.error is False


def test_a_leading_mark_before_a_quoted_value():
    (binding,) = parse(f"{BOM}a='b c'")
    assert binding.key == "a"
    assert binding.value == "b c"


def test_the_recorded_original_text_omits_the_mark():
    (binding,) = parse(f"{BOM}a=b")
    assert binding.original.string == "a=b"
    assert binding.original.line == 1
