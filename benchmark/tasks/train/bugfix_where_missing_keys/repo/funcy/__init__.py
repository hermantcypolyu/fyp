"""funcy - a collection of fancy functional tools.

This is a reduced extract: only the collection helpers relevant to this task are here.
"""

from funcy.colls import (
    empty, invoke, iinvoke, ipluck, ipluck_attr, iwhere, pluck, pluck_attr, project,
    where, zipdict,
)

__all__ = ['empty', 'zipdict', 'project',
           'where', 'pluck', 'pluck_attr', 'invoke',
           'iwhere', 'ipluck', 'ipluck_attr', 'iinvoke']
