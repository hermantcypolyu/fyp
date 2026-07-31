"""Minimal sequence-helper shim.

Upstream funcy's `seqs.py` is ~380 lines of lazy sequence tooling - windowing, chunking,
grouping, interleaving, the `l*`/`i*` eager/lazy pairs and their Python-2 variants - of
which `colls.py` imports exactly three names. Only those three are provided here, with
`take` and `ifilter` copied from upstream and `ximap` bound the way upstream binds it on
Python 3. The import site in `colls.py` is unchanged.
"""

from itertools import islice

from .cross import PY2, imap
from .cross import ifilter as _ifilter
from .funcmakers import make_pred

__all__ = ['take', 'ximap', 'ifilter']


def take(n, seq):
    """Returns a list of first n items in the sequence,
       or all items if there are fewer than n."""
    return list(islice(seq, n))


def ifilter(pred, seq):
    """An extended version of builtin ifilter().
       Derives a predicate from string, int, slice, dict or set."""
    return _ifilter(make_pred(pred, builtin=PY2), seq)


ximap = imap
