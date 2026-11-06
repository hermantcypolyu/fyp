"""Repository scripts, importable as `scripts.<module>`.

This file exists for one reason: to make `scripts` a **regular package** rather than an
implicit namespace package.

Without it, `scripts` is only a namespace package, and Python's import system gives a
regular package on `sys.path` priority over a namespace package *regardless of path order*.
The serving virtualenv (`~/.venvs/qwen-server`) ships
`site-packages/scripts/__init__.py` - llama-cpp-python's GGUF helper scripts, which have
nothing to do with this project - so every `import scripts.<anything>` resolved to that
directory and failed with `ModuleNotFoundError`, no matter which `PYTHONPATH` came first.
That is why Phase 6E needed an undocumented `/tmp/phase6e_shim` symlink farm to run the
collector at all.

A regular package in the same directory ties with the other regular package, and the tie is
then broken by `sys.path` order - which is controllable, documented, and what
`PYTHONPATH=<repo>` already does. Declaring the package explicitly turns a silent
environment-dependent failure into an ordinary, resolvable name clash.

The directory intentionally exports nothing. Its modules are entry points and libraries
that import heavy optional dependencies (`openai`, tokenizers, the gym stack); re-exporting
them here would make `import scripts` pull in all of that for every caller.
"""

from __future__ import annotations

__all__: list[str] = []
