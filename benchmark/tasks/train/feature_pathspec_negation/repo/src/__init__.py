"""siftpath - gitignore-style path matching over plain strings."""

from .listing import filter_paths, partition
from .pathspec import PathSpec, Rule

__all__ = ["PathSpec", "Rule", "filter_paths", "partition"]
