"""Minimal decorator-plumbing shim.

Upstream funcy builds its own decorator machinery in this module: `@decorator` turns a
flat wrapper into a decorator whose call arguments are introspectable, `arggetter` and
`get_argnames` do the argument-name bookkeeping that makes that possible, `wraps` is a
hardened `functools.wraps` that additionally records `__original__`, and there is a
Python-2 backport of `contextlib.contextmanager` and `ContextDecorator`. None of that is
involved in the defect under test, and carrying it verbatim would have added ~310 lines
of argument inspection and version shims to this task repo, so it is reduced here to
working equivalents of just the five names `flow.py` imports. Of those, only `wraps` is
exercised by the code kept in `flow.py`. The import sites in `flow.py` are unchanged.
"""

import inspect
from contextlib import contextmanager  # noqa: F401  (upstream re-exports this on Python 3)
from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES

__all__ = ['decorator', 'wraps', 'get_argnames', 'arggetter', 'contextmanager']


def unwrap(func):
    """Returns the innermost function behind a chain of wrappers."""
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    return func


def wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):
    """Passes function metadata from a wrapped function to its wrapper.

    Like `functools.wraps`, but tolerant of callables missing some of the attributes and
    additionally setting `__original__` to the innermost wrapped function.
    """
    def _wrap(wrapper):
        for attr in assigned:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                pass
            else:
                setattr(wrapper, attr, value)
        for attr in updated:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))

        wrapper.__wrapped__ = wrapped
        wrapper.__original__ = getattr(wrapped, '__original__', None) or unwrap(wrapped)
        return wrapper
    return _wrap


def get_argnames(func):
    """Names of the positional parameters the function declares."""
    func = getattr(func, '__original__', None) or unwrap(func)
    return func.__code__.co_varnames[:func.__code__.co_argcount]


def arggetter(func):
    """Builds `get_arg(name, args, kwargs)` for the given function."""
    signature = inspect.signature(getattr(func, '__original__', None) or unwrap(func))

    def get_arg(name, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if name not in bound.arguments:
            raise TypeError("%s() doesn't have argument named %s" % (func.__name__, name))
        return bound.arguments[name]

    return get_arg


class Call(object):
    """A proxy for a decorated call, with its arguments reachable by name."""

    def __init__(self, func, args, kwargs):
        self._func, self._args, self._kwargs = func, args, kwargs

    def __call__(self, *a, **kw):
        if not a and not kw:
            return self._func(*self._args, **self._kwargs)
        return self._func(*(self._args + a), **dict(self._kwargs, **kw))

    def __getattr__(self, name):
        try:
            res = self.__dict__[name] = arggetter(self._func)(name, self._args, self._kwargs)
            return res
        except TypeError as e:
            raise AttributeError(*e.args)


def decorator(deco):
    """Transforms a flat wrapper `deco(call, *dargs, **dkwargs)` into a decorator."""
    def decorator_fab(*dargs, **dkwargs):
        def _decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return deco(Call(func, args, kwargs), *dargs, **dkwargs)
            return wrapper
        return _decorator
    return wraps(deco)(decorator_fab)
