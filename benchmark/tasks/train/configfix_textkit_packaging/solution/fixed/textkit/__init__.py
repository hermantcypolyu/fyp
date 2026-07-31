"""textkit - small, dependency-free text helpers."""

from .case import camel_case, snake_case, title_case
from .slug import slugify

__version__ = "0.4.1"

__all__ = ["slugify", "title_case", "snake_case", "camel_case"]
