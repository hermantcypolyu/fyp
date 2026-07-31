"""textkit - small, dependency-free text helpers."""

from .slugify import slugify
from .case import camel_case, snake_case, title_case

__version__ = "0.4.1"

__all__ = ["slugify", "titlecase", "snake_case", "camel_case"]
