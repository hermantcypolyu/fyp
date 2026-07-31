"""reportkit - plain-text reports, with optional renderers behind extras."""

from .features import SUPPORTED_EXTRAS, available, install_hint, require
from .render import render_report, render_table
from .table import divider, format_row, widths

__version__ = "0.8.2"

__all__ = [
    "SUPPORTED_EXTRAS",
    "available",
    "divider",
    "format_row",
    "install_hint",
    "render_report",
    "render_table",
    "require",
    "widths",
]
