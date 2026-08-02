"""Hidden verification for `feature_textnorm_options`.

Three keyword-only options are added to `normalise`, and they are graded one at a time
because they are independently implementable: an agent can land `separator` without
`fold`, or `fold` without `keep_case`, and score accordingly.

The folding table and `fold_accents` already exist and are already correct, so this is an
API extension rather than an algorithm - the work is threading the options through the
pipeline in the documented order (optional lowercasing, then optional folding, then the
join) without disturbing the default path.

The floor tests below pass against the untouched repo on purpose: every existing call must
keep behaving exactly as it does now, which is what makes the defaults load-bearing.
"""

import inspect

import pytest

from textnorm import compare, keys, normalise
from textnorm.accents import fold_accents
from textnorm.tokens import tokenise

# --- the new options --------------------------------------------------------------------


def test_fold_replaces_accented_characters():
    assert normalise("Café Crème", fold=True) == "cafe creme"


def test_fold_defaults_to_leaving_accents_alone():
    assert normalise("Café Crème", fold=False) == "café crème"


def test_fold_expands_the_ligatures():
    assert normalise("Straße Æon", fold=True) == "strasse aeon"


def test_keep_case_preserves_the_original_case():
    assert normalise("Hello, World!", keep_case=True) == "Hello World"


def test_separator_joins_the_tokens():
    assert normalise("Hello, World!", separator="-") == "hello-world"


def test_separator_may_be_empty():
    assert normalise("Hello, World!", separator="") == "helloworld"


def test_fold_and_keep_case_compose():
    assert normalise("Café Crème", fold=True, keep_case=True) == "Cafe Creme"


def test_all_three_options_compose():
    assert normalise("Café Crème", fold=True, keep_case=True, separator="_") == "Cafe_Creme"


def test_the_options_are_keyword_only():
    parameters = inspect.signature(normalise).parameters
    for name in ("fold", "keep_case", "separator"):
        assert name in parameters, f"normalise() takes no {name!r} option"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name!r} must be keyword-only so positional calls keep their meaning"
        )


# --- floor: the existing behaviour these options default to -----------------------------


def test_default_call_is_unchanged():
    assert normalise("  Hello,   World! ") == "hello world"


def test_punctuation_and_underscores_are_still_separators():
    assert normalise("a_b, c!!d") == "a b c d"


def test_digits_survive_normalisation():
    assert normalise("Route 66") == "route 66"


def test_tokeniser_is_unchanged():
    assert tokenise("Hello, World!") == ["Hello", "World"]


def test_fold_accents_helper_is_unchanged():
    assert fold_accents("Café") == "Cafe"
    assert fold_accents("Straße") == "Strasse"


def test_comparisons_still_work_on_the_default_pipeline():
    assert compare.same_normalised("Hello, World!", "  hello world  ")
    assert compare.overlap("a b", "b c") == pytest.approx(1 / 3)


def test_derived_keys_still_build_on_the_default_pipeline():
    assert keys.sort_key("World, hello!") == "hello world"
    assert keys.dedupe_tokens("hello hello world") == ["hello", "world"]
    assert keys.content_key("the Hello of a World") == "hello world"
