"""funcy - a collection of fancy functional tools.

This is a reduced extract: only the function-manipulation helpers relevant to this task
are here.
"""

from funcy.funcs import (
    autocurry,
    caller,
    complement,
    compose,
    constantly,
    curry,
    func_partial,
    identity,
    iffy,
    ijuxt,
    juxt,
    partial,
    rcurry,
    rpartial,
)

__all__ = ['identity', 'constantly', 'caller',
           'partial', 'rpartial', 'func_partial',
           'curry', 'rcurry', 'autocurry',
           'iffy', 'compose', 'complement', 'juxt', 'ijuxt']
