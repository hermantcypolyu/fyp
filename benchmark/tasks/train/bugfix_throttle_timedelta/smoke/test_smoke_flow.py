"""Visible smoke checks. Shallow by design - they use a plain numeric period only, and
say nothing about how a period is interpreted.
"""

import funcy


def test_module_imports_and_decorates():
    @funcy.throttle(1)
    def noop():
        return 'ok'

    assert callable(noop)


def test_first_call_runs():
    @funcy.throttle(1)
    def once(x):
        return x

    assert once(7) == 7
