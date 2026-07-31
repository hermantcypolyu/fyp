"""Visible smoke checks. Shallow by design - they parse an ordinary file and say nothing
about what happens at the very start of one.
"""

import io

from dotenv.parser import parse_stream


def test_module_imports_and_parses_a_pair():
    (binding,) = list(parse_stream(io.StringIO("a=b")))
    assert (binding.key, binding.value) == ("a", "b")


def test_several_pairs():
    bindings = list(parse_stream(io.StringIO("a=b\nc=d")))
    assert len(bindings) == 2
