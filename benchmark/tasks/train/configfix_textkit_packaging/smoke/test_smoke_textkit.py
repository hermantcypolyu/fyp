"""Visible smoke checks. These import the submodules directly, so they keep passing even
while the package's own __init__ is broken - they say nothing about packaging metadata.
"""

from textkit.case import title_case
from textkit.slug import slugify


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_title_case_basic():
    assert title_case("the wind in the willows") == "The Wind in the Willows"
