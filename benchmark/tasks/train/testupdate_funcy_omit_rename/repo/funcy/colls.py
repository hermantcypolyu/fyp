from operator import itemgetter, methodcaller, attrgetter
from collections import defaultdict
from collections.abc import Mapping, Set, Iterable, Iterator
from itertools import chain, tee
from .cross import basestring, xrange, izip, map, filter, imap, PY2
from .primitives import EMPTY
from .funcs import partial, compose, complement
from .funcmakers import make_func, make_pred
from .seqs import take, ximap, ifilter

FACTORY_REPLACE = {
    type(object.__dict__): dict,
    type({}.keys()): list,
    type({}.values()): list,
    type({}.items()): list,
}


def _factory(coll, mapper=None):
    # Hack for defaultdicts overriden constructor
    if isinstance(coll, defaultdict):
        item_factory = compose(mapper, coll.default_factory) if mapper else coll.default_factory
        return partial(defaultdict, item_factory)
    elif isinstance(coll, Iterator):
        return iter
    elif isinstance(coll, basestring):
        return ''.join
    elif type(coll) in FACTORY_REPLACE:
        return FACTORY_REPLACE[type(coll)]
    else:
        return coll.__class__


def empty(coll):
    if isinstance(coll, Iterator):
        return iter([])
    return _factory(coll)()

if PY2:
    def iteritems(coll):
        return coll.iteritems() if hasattr(coll, 'iteritems') else coll

    def itervalues(coll):
        return coll.itervalues() if hasattr(coll, 'itervalues') else coll
else:
    def iteritems(coll):
        return coll.items() if hasattr(coll, 'items') else coll

    def itervalues(coll):
        return coll.values() if hasattr(coll, 'values') else coll


def join(colls):
    colls, colls_copy = tee(colls)
    it = iter(colls_copy)
    try:
        dest = next(it)
    except StopIteration:
        return None
    cls = dest.__class__

    if isinstance(dest, basestring):
        return ''.join(colls)
    elif isinstance(dest, Mapping):
        result = dest.copy()
        for d in it:
            result.update(d)
        return result
    elif isinstance(dest, Set):
        return dest.union(*it)
    elif isinstance(dest, (Iterator, xrange)):
        return chain.from_iterable(colls)
    elif isinstance(dest, Iterable):
        # NOTE: this could be reduce(concat, ...),
        #       more effective for low count
        return cls(chain.from_iterable(colls))
    else:
        raise TypeError("Don't know how to join %s" % cls.__name__)


def merge(*colls):
    return join(colls)


def walk(f, coll):
    """
    Walk coll transforming it's elements with f.
    Same as map, but preserves coll type.
    """
    return _factory(coll)(ximap(f, iteritems(coll)))


def walk_keys(f, coll):
    f = make_func(f)
    # NOTE: we use this awkward construct instead of lambda to be Python 3 compatible
    def pair_f(pair):
        k, v = pair
        return f(k), v

    return walk(pair_f, coll)


def walk_values(f, coll):
    f = make_func(f)
    # NOTE: we use this awkward construct instead of lambda to be Python 3 compatible
    def pair_f(pair):
        k, v = pair
        return k, f(v)

    return _factory(coll, mapper=f)(ximap(pair_f, iteritems(coll)))


def select(pred, coll):
    """Same as filter but preserves coll type."""
    return _factory(coll)(ifilter(pred, iteritems(coll)))


def select_keys(pred, coll):
    pred = make_pred(pred)
    return select(lambda pair: pred(pair[0]), coll)


def select_values(pred, coll):
    pred = make_pred(pred)
    return select(lambda pair: pred(pair[1]), coll)


def compact(coll):
    if isinstance(coll, Mapping):
        return select_values(bool, coll)
    else:
        return select(bool, coll)


def zipdict(keys, vals):
    return dict(izip(keys, vals))


def flip(mapping):
    def flip_pair(pair):
        k, v = pair
        return v, k
    return walk(flip_pair, mapping)


def project(mapping, keys):
    return _factory(mapping)((k, mapping[k]) for k in keys if k in mapping)


def omit(mapping, keys):
    return _factory(mapping)((k, v) for k, v in iteritems(mapping) if k not in keys)


def get_in(coll, path, default=None):
    for key in path:
        try:
            coll = coll[key]
        except (KeyError, IndexError):
            return default
    return coll


def set_in(coll, path, value):
    return update_in(coll, path, lambda _: value)


def update_in(coll, path, update, default=None):
    if not path:
        return update(coll)
    elif isinstance(coll, list):
        copy = coll[:]
        # NOTE: there is no auto-vivication for lists
        copy[path[0]] = update_in(copy[path[0]], path[1:], update, default)
        return copy
    else:
        copy = coll.copy()
        current_default = {} if len(path) > 1 else default
        copy[path[0]] = update_in(copy.get(path[0], current_default), path[1:], update, default)
        return copy
