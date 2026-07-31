from datetime import datetime, timedelta
import time
import threading
from .compat import map, range, raise_from, Hashable
from .decorators import decorator, wraps, get_argnames, arggetter, contextmanager


def raiser(exception_or_class=Exception, *args, **kwargs):
    """Constructs function that raises the given exception
       with given arguments on any invocation."""
    if isinstance(exception_or_class, str):
        exception_or_class = Exception(exception_or_class)

    def _raiser(*a, **kw):
        if args or kwargs:
            raise exception_or_class(*args, **kwargs)
        else:
            raise exception_or_class
    return _raiser


def ignore(errors, default=None):
    """Alters function to ignore given errors, returning default instead."""
    errors = _ensure_exceptable(errors)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors:
                return default
        return wrapper
    return decorator


def silent(func):
    """Alters function to ignore all exceptions."""
    return ignore(Exception)(func)


def _ensure_exceptable(errors):
    """Ensures that errors are passable to except clause.
       I.e. should be BaseException subclass or a tuple."""
    return errors if _is_exception_type(errors) else tuple(errors)


def _is_exception_type(value):
    return isinstance(value, type) and issubclass(value, BaseException)


class ErrorRateExceeded(Exception):
    pass


def limit_error_rate(fails, timeout, exception=ErrorRateExceeded):
    """If function fails to complete fails times in a row,
       calls to it will be intercepted for timeout with exception raised instead."""
    if isinstance(timeout, int):
        timeout = timedelta(seconds=timeout)

    def decorator(func):
        func.fails = 0
        func.blocked = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            if func.blocked:
                if datetime.now() - func.blocked < timeout:
                    raise exception
                else:
                    func.blocked = None

            try:
                result = func(*args, **kwargs)
            except:  # noqa
                func.fails += 1
                if func.fails >= fails:
                    func.blocked = datetime.now()
                raise
            else:
                func.fails = 0
                return result
        return wrapper
    return decorator


def throttle(period):
    """Allows only one run in a period, the rest is skipped"""
    if isinstance(period, timedelta):
        period = period.total_seconds()

    def decorator(func):
        func.blocked_until = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if func.blocked_until and func.blocked_until > now:
                return
            func.blocked_until = now + period

            return func(*args, **kwargs)

        return wrapper

    return decorator
