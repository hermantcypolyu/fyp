import pytest
from collections import defaultdict
from collections.abc import Iterator
from itertools import chain, count

from funcy.colls import *


# Utilities
def eq(a, b):
    return type(a) is type(b) and a == b \
       and (a.default_factory == b.default_factory if isinstance(a, defaultdict) else True)

def S(*args):
    """"Set literal" for the poor python 2.6"""
    return set(args)

def inc(x):
    return x + 1

def hinc(xs):
    return map(inc, xs)


def test_empty():
    assert eq(empty({'a': 1}), {})
    assert eq(empty(defaultdict(int)), defaultdict(int))
    assert empty(defaultdict(int)).default_factory == defaultdict(int).default_factory

def test_empty_iter():
    it = empty(iter([]))
    assert isinstance(it, Iterator)
    assert list(it) == []

def test_empty_quirks():
    class A:
        FLAG = 1
    assert empty(A.__dict__) == {}
    assert empty({}.keys()) == []
    assert empty({}.values()) == []
    assert empty({}.items()) == []


def test_iteritems():
    assert list(iteritems([1,2])) == [1,2]
    assert list(iteritems((1,2))) == [1,2]
    assert list(iteritems({'a': 1})) == [('a', 1)]

def test_itervalues():
    assert list(itervalues([1,2])) == [1,2]
    assert list(itervalues((1,2))) == [1,2]
    assert list(itervalues({'a': 1})) == [1]


def test_merge():
    assert eq(merge({1: 2}, {3: 4}), {1: 2, 3: 4})

def test_join():
    assert join([]) is None
    with pytest.raises(TypeError): join([1])
    assert eq(join(['ab', '', 'cd']), 'abcd')
    assert eq(join([['a', 'b'], 'c']), list('abc'))
    assert eq(join([('a', 'b'), ('c',)]), tuple('abc'))
    assert eq(join([{'a': 1}, {'b': 2}]), {'a': 1, 'b': 2})
    assert eq(join([{'a': 1}, {'a': 2}]), {'a': 2})
    assert eq(join([S(1,2), S(3)]), S(1,2,3))

    it1 = (x for x in range(2))
    it2 = (x for x in range(5, 7))
    joined = join([it1, it2])
    assert isinstance(joined, Iterator) and list(joined) == [0,1,5,6]


def test_walk():
    assert eq(walk(inc, [1,2,3]), [2,3,4])
    assert eq(walk(inc, (1,2,3)), (2,3,4))
    assert eq(walk(inc, S(1,2,3)), S(2,3,4))
    assert eq(walk(hinc, {1:1,2:2,3:3}), {2:2,3:3,4:4})

def test_walk_iter():
    it = walk(inc, chain([0], [1, 2]))
    assert isinstance(it, Iterator) and list(it) == [1,2,3]

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b':2}) == {'A': 1, 'B': 2}

def test_walk_values():
    assert walk_values(lambda x: x * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_values_defaultdict():
    dd = defaultdict(lambda: 'hey', {1: 'a', 2: 'ab'})
    walked_dd = walk_values(len, dd)
    assert walked_dd == {1: 1, 2: 2}
    # resulting default factory should be compose(len, lambda: 'hey')
    assert walked_dd[0] == 3


def test_select():
    assert eq(select(lambda x: x > 1, [1,2,3]), [2,3])
    assert eq(select(lambda x: x > 1, (1,2,3)), (2,3))
    assert eq(select(lambda x: x > 1, S(1,2,3)), S(2,3))
    assert eq(select(lambda p: p[1] > 1, {'a':1,'b':2,'c':3}), {'b':2,'c':3})

def test_select_keys():
    assert select_keys(lambda k: k[0] == 'a', {'a':1, 'b':2, 'ab':3}) == {'a': 1, 'ab':3}

def test_select_values():
    assert select_values(lambda v: v % 2, {'a': 1, 'b': 2}) == {'a': 1}


def test_compact():
    assert eq(compact([0, 1, None, 3]), [1, 3])
    assert eq(compact((0, 1, None, 3)), (1, 3))
    assert eq(compact({'a': None, 'b': 0, 'c': 1}), {'c': 1})


def test_zipdict():
    assert zipdict([1, 2], 'ab') == {1: 'a', 2:'b'}
    assert zipdict('ab', count()) == {'a': 0, 'b': 1}

def test_flip():
    assert flip({'a':1, 'b':2}) == {1:'a', 2:'b'}

def test_project():
    assert project({'a':1, 'b':2, 'c': 3}, 'ac') == {'a':1, 'c': 3}
    dd = defaultdict(int, {'a':1, 'b':2, 'c': 3})
    assert eq(project(dd, 'ac'), defaultdict(int, {'a':1, 'c': 3}))

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, 'ac') == {'b': 2}
    dd = defaultdict(int, {'a': 1, 'b': 2, 'c': 3})
    assert eq(omit(dd, 'ac'), defaultdict(int, {'b': 2}))


def test_get_in():
    d = {"a": {"b": "c", "f": {"g": "h"}}, "i": "j"}
    assert get_in(d, ["m"]) is None
    assert get_in(d, ["m", "n"], "foo") == "foo"
    assert get_in(d, ["a", "b"]) == "c"
    assert get_in(d, ["a", "f", "g"]) == "h"

def test_set_in():
    d = {'a': {'b': 1, 'c': 2}, 'd': 5}
    d2 = set_in(d, ['a', 'c'], 7)
    assert d == {'a': {'b': 1, 'c': 2}, 'd': 5}
    assert d2 == {'a': {'b': 1, 'c': 7}, 'd': 5}

def test_update_in():
    d = {'c': []}
    d2 = update_in(d, ['c'], len)
    assert d == {'c': []}
    assert d2 == {'c': 0}
