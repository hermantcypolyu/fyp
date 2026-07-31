"""Minimal predicate/function-maker shim.

Upstream funcy lets a "function" argument be given as a regex, a string, an int, a slice,
a mapping or a set, and `funcmakers.py` turns any of those into a callable by leaning on
`funcy.strings`. The regex branch and the `strings` module it needs are not involved in
the defect under test, so they are dropped here; what remains handles callables, None,
integers/slices, mappings and sets, which is everything the reduced repo passes in. The
import sites in `colls.py`, `funcs.py` and `seqs.py` are unchanged.
"""

from collections import abc
from operator import itemgetter

__all__ = ('make_func', 'make_pred')


def make_func(f, builtin=False, test=False):
    if callable(f):
        return f
    elif f is None:
        # pass None to builtin as predicate or mapping function for speed
        return None if builtin else \
               bool if test else lambda x: x
    elif isinstance(f, (int, slice)):
        return itemgetter(f)
    elif isinstance(f, abc.Mapping):
        return f.__getitem__
    elif isinstance(f, abc.Set):
        return f.__contains__
    else:
        raise TypeError("Can't make a func from %s" % f.__class__.__name__)


def make_pred(pred, builtin=False):
    return make_func(pred, builtin=builtin, test=True)
