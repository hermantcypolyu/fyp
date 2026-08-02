"""Execution limits for the graded pytest run.

The threat model is not a hostile attacker - it is a 1.5B policy that writes
`while True:`, allocates a list until the box swaps, or (having noticed there is a network)
`pip install`s the upstream package that contains the answer. Each of those corrupts the
reward signal, so each gets a cap.

**Be honest about what is actually enforced.** Three of the four caps below rely on POSIX
`setrlimit`, which does not exist on Windows, and this project is developed on Windows
(ADR 0001). Rather than pretend, `enforced()` reports per-limit truth for the running
platform and the verifier records that dict on every result. A number in the write-up that
claims "memory-capped" on a run where it was not is exactly the fabrication 1.2 forbids.

What holds on **every** platform, including Windows:
- the **wall-clock timeout**, via `subprocess.run(timeout=...)`, which is the cap that
  actually matters for reward integrity (a timeout is a scored failure, not an error); and
- the **network guard**, which is a `sitecustomize.py` injected on PYTHONPATH rather than a
  kernel facility - see `network_guard_dir`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NETWORK_GUARD_SOURCE", "ResourceLimits", "network_guard_dir", "no_network_env"]

_POSIX = os.name == "posix"


@dataclass(frozen=True)
class ResourceLimits:
    """Caps applied to one graded pytest subprocess."""

    wall_clock_s: int = 60
    cpu_s: int = 60
    memory_mb: int = 2048
    max_file_mb: int = 64
    max_processes: int = 64

    def enforced(self) -> dict[str, bool]:
        """Which caps this platform really applies. Recorded on every verify result."""
        return {
            "wall_clock_s": True,
            "network": True,
            "cpu_s": _POSIX,
            "memory_mb": _POSIX,
            "max_file_mb": _POSIX,
            "max_processes": _POSIX,
        }

    def preexec_fn(self) -> Callable[[], None] | None:
        """A `subprocess` preexec callback applying the rlimits, or None off POSIX.

        `cpu_s` is set one second above the wall clock so that the wall clock is what
        normally fires: a wall-clock timeout is observable and scoreable, whereas SIGXCPU
        arrives as an opaque negative return code. The CPU cap is the backstop for a
        process that stops responding to the timer, not the primary control.
        """
        if not _POSIX:
            return None
        import resource  # POSIX-only; imported lazily so this module imports on Windows

        cpu = self.cpu_s
        mem = self.memory_mb * 1024 * 1024
        fsize = self.max_file_mb * 1024 * 1024
        nproc = self.max_processes

        def _apply() -> None:  # pragma: no cover - runs in the child, after fork
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return _apply


# --------------------------------------------------------------------------- network

#: Injected as `sitecustomize.py` on the graded run's PYTHONPATH. CPython imports
#: `sitecustomize` automatically during interpreter start-up, before any test code, so the
#: patch is in place for everything that follows.
#:
#: This blocks *outbound connections from Python*. It is not a firewall: a subprocess
#: spawned with a different interpreter, or a compiled extension calling connect(2)
#: directly, walks straight past it. That is acceptable here because the failure being
#: prevented is a confused agent reaching for the internet, not an adversary evading a
#: sandbox - and because the alternative (a container) does not exist on this machine
#: (ADR 0001). `socketpair` is left alone: it is local-only and asyncio needs it.
NETWORK_GUARD_SOURCE = '''\
"""Auto-loaded by CPython at start-up. Blocks outbound network access during grading."""

import socket


class NetworkBlocked(OSError):
    """Raised instead of reaching the network. Grading runs offline by construction."""


def _blocked(*_args, **_kwargs):
    raise NetworkBlocked("network access is disabled during grading")


# Patch the *connecting* operations, not the socket class itself: `ssl` does
# `class SSLSocket(socket)` at import time, so replacing the class turns any later
# `import ssl` - which pdb, asyncio and therefore pytest all reach - into a TypeError.
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
for _name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "create_server"):
    if hasattr(socket, _name):
        setattr(socket, _name, _blocked)
'''


def network_guard_dir(parent: Path) -> Path:
    """Write the guard to `parent/_guard/sitecustomize.py` and return the directory.

    It lives outside the graded working copy on purpose: putting it under `repo/` would
    make it show up in the agent's diff and in `count_loc`.
    """
    guard = Path(parent) / "_guard"
    guard.mkdir(parents=True, exist_ok=True)
    (guard / "sitecustomize.py").write_text(NETWORK_GUARD_SOURCE, encoding="utf-8")
    return guard


def no_network_env() -> dict[str, str]:
    """Environment overrides that stop the usual suspects before the socket patch has to.

    `PIP_NO_INDEX` and the proxy variables are belt-and-braces: they turn a network attempt
    into a fast, legible failure instead of a 30-second DNS timeout inside the graded run.
    """
    return {
        "no_proxy": "*",
        "NO_PROXY": "*",
        "http_proxy": "",
        "https_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
    }
