"""Minimal function-composition shim.

Upstream funcy's `funcs.py` carries currying, autocurrying, `juxt`, `iffy` and the
argument-spec inspection module they depend on, none of which is involved here; `colls.py`
imports only `partial`, `compose` and `complement` from it. `compose` and `complement` are
upstream's, and `partial` is the `functools` one that upstream also re-exports. The import
site in `colls.py` is unchanged.
"""

from functools import partial, reduce  # noqa: F401  (partial is re-exported, as upstream)
from operator import __not__

from .cross import imap
from .funcmakers import make_func

__all__ = ['identity', 'partial', 'compose', 'complement']


def identity(x):
    """Returns its argument."""
    return x


def compose(*fs):
    """Composes passed functions."""
    if fs:
        pair = lambda f, g: lambda *a, **kw: f(g(*a, **kw))  # noqa: E731
        return reduce(pair, imap(make_func, fs))
    else:
        return identity


def complement(pred):
    return compose(__not__, pred)
