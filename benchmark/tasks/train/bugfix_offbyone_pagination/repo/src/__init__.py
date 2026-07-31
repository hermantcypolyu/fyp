"""A tiny in-memory record store with offset-based pagination."""

from .pagination import Page, paginate, total_pages
from .query import Query

__all__ = ["Page", "Query", "paginate", "total_pages"]
