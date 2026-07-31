from operator import itemgetter, methodcaller, attrgetter
from collections import defaultdict
from collections.abc import Mapping, Set, Iterable, Iterator
from itertools import chain, tee
from .cross import basestring, xrange, izip, map, filter, imap, PY2
from .primitives import EMPTY
from .funcs import partial, compose
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
    """Creates an empty collection of the same type."""
    if isinstance(coll, Iterator):
        return iter([])
    return _factory(coll)()


def zipdict(keys, vals):
    """Creates a dict with keys mapped to the corresponding vals."""
    return dict(izip(keys, vals))


def project(mapping, keys):
    """Leaves only given keys in mapping."""
    return _factory(mapping)((k, mapping[k]) for k in keys if k in mapping)


def where(mappings, **cond):
    """Selects mappings containing all pairs in cond."""
    match = lambda m: all(k in m and m[k] == v for k, v in cond.items())
    return filter(match, mappings)


def pluck(key, mappings):
    """Lists values for key in each mapping."""
    return map(itemgetter(key), mappings)


def pluck_attr(attr, objects):
    """Lists values of given attribute of each object."""
    return map(attrgetter(attr), objects)


def invoke(objects, name, *args, **kwargs):
    """Makes a list of results of the obj.name(*args, **kwargs)
       for each object in objects."""
    return map(methodcaller(name, *args, **kwargs), objects)


def iwhere(mappings, **cond):
    """Iterates over mappings containing all pairs in cond."""
    match = lambda m: all(k in m and m[k] == v for k, v in cond.items())
    return ifilter(match, mappings)


def ipluck(key, mappings):
    """Iterates over values for key in mappings."""
    return imap(itemgetter(key), mappings)


def ipluck_attr(attr, objects):
    """Iterates over values of given attribute of given objects."""
    return imap(attrgetter(attr), objects)


def iinvoke(objects, name, *args, **kwargs):
    """Yields results of the obj.name(*args, **kwargs)
       for each object in objects."""
    return imap(methodcaller(name, *args, **kwargs), objects)
