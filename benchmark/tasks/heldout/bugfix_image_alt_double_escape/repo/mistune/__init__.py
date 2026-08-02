"""A trimmed extract of mistune's renderer layer.

Only the pieces the HTML renderer needs are vendored here: the token state and
renderer base class from ``core``, the escaping helpers from ``util``, and the
``HTMLRenderer`` itself. The block and inline parsers are not included.
"""

from .core import BaseRenderer, BlockState
from .renderers.html import HTMLRenderer

__all__ = ["BaseRenderer", "BlockState", "HTMLRenderer"]
