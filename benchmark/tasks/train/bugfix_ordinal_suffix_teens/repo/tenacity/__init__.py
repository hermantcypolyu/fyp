"""tenacity - general-purpose retrying.

This is a reduced extract: only the internal utility module is present. Upstream's
package `__init__` is several hundred lines of retry machinery that has no bearing on the
defect under test, so it is replaced here by a plain namespace.
"""

from tenacity import _utils

__all__ = ["_utils"]
