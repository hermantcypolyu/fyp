"""Hidden verification for `bugfix_where_missing_keys`.

Assertions are taken from the upstream suite at the fixing commit
(Suor/funcy @ e068b64005b0f3c432311a7584b82d4e08420a67, tests/test_colls.py) - the
pre-existing `test_where`, `test_pluck`, `test_pluck_attr` and `test_invoke`, plus the
`test_where_nonexistent_key` the commit added - split into behaviour-sized tests so
partial credit is meaningful, and extended with the neighbouring shapes of ragged input
that run through the same predicate.

Two of the passing tests exist to block plausible wrong fixes:

* `test_where_returns_a_list` - the shortest way to make the failing cases pass is to
  rewrite `where` as a generator expression, or to have it delegate to the lazy variant.
  Either would silently change `where` from a list-returning function into something that
  is empty on second use. This test pins the eager return type.
* `test_iwhere_stays_lazy` - the mirror-image mistake: making the lazy variant eager, so
  that it consumes its whole input up front. This test pins laziness by counting how much
  of a generator has been read.

`test_where_distinguishes_a_missing_key_from_a_none_value` additionally rules out
`m.get(k) == v`, which is the obvious one-token repair and quietly treats an absent key as
a key whose value is None.
"""

from collections import namedtuple

from funcy import invoke, iwhere, pluck, pluck_attr, where


# --- behaviour that already works, and must keep working --------------------------------


def test_where_selects_matching_mappings():
    data = [{'a': 1, 'b': 2}, {'a': 10, 'b': 2}]
    assert where(data, a=1) == [{'a': 1, 'b': 2}]
    assert where(data, a=1, b=2) == [{'a': 1, 'b': 2}]
    assert where(data, b=2) == data


def test_where_with_no_conditions_returns_everything():
    data = [{'a': 1}, {'b': 2}]
    assert where(data) == data


def test_where_returns_a_list():
    data = [{'a': 1, 'b': 2}, {'a': 10, 'b': 2}]
    result = where(data, b=2)
    assert isinstance(result, list)
    # A list can be walked twice; a one-shot iterator cannot.
    assert list(result) == list(result) == data


def test_iwhere_stays_lazy():
    read = []

    def records():
        for m in ({'a': 1}, {'a': 2}, {'a': 1}):
            read.append(m)
            yield m

    result = iwhere(records(), a=1)
    assert read == []
    assert next(iter(result)) == {'a': 1}
    assert read == [{'a': 1}]


def test_pluck_and_invoke_are_unaffected():
    data = [{'a': 1, 'b': 2}, {'a': 10, 'b': 2}]
    assert pluck('a', data) == [1, 10]

    Obj = namedtuple('Obj', ('id', 'name'))
    assert pluck_attr('id', [Obj(1, 'test1'), Obj(5, 'test2'), Obj(10, 'test3')]) == [1, 5, 10]

    assert invoke(['abc', 'def', 'b'], 'find', 'b') == [1, -1, 0]


# --- the defect -------------------------------------------------------------------------
# A condition names a key. A mapping that does not carry that key simply does not match;
# it is not an error.


def test_where_skips_mappings_missing_the_key():
    data = [{'a': 1}, {'b': 2}]
    assert where(data, a=1) == [{'a': 1}]


def test_where_skips_mappings_missing_a_later_key():
    data = [{'a': 1}, {'b': 2}]
    assert where(data, b=2) == [{'b': 2}]


def test_iwhere_skips_mappings_missing_the_key():
    data = [{'a': 1}, {'b': 2}]
    assert list(iwhere(data, a=1)) == [{'a': 1}]
    assert list(iwhere(data, b=2)) == [{'b': 2}]


def test_where_with_a_key_no_mapping_has():
    data = [{'a': 1}, {'a': 2}]
    assert where(data, zzz=1) == []


def test_where_distinguishes_a_missing_key_from_a_none_value():
    data = [{'a': None}, {'b': 2}]
    assert where(data, a=None) == [{'a': None}]


def test_where_with_several_conditions_over_ragged_records():
    data = [{'a': 1, 'b': 2}, {'a': 1}, {'b': 2}, {'a': 1, 'b': 3}]
    assert where(data, a=1, b=2) == [{'a': 1, 'b': 2}]


def test_where_over_a_generator_of_ragged_mappings():
    def records():
        yield {'a': 1}
        yield {'b': 2}
        yield {'a': 1, 'b': 2}

    assert where(records(), a=1) == [{'a': 1}, {'a': 1, 'b': 2}]
