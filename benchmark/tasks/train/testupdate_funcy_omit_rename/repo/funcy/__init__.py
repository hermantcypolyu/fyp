"""funcy - a collection of fancy functional tools.

This is a reduced extract: only the collection helpers relevant to this task are here.
"""

from funcy.colls import (
    compact,
    empty,
    flip,
    get_in,
    join,
    merge,
    omit,
    project,
    select,
    select_keys,
    select_values,
    set_in,
    update_in,
    walk,
    walk_keys,
    walk_values,
    zipdict,
)

__all__ = ['empty', 'join', 'merge',
           'walk', 'walk_keys', 'walk_values', 'select', 'select_keys', 'select_values', 'compact',
           'zipdict', 'flip', 'project', 'omit',
           'get_in', 'set_in', 'update_in']
