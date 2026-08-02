"""Pinned versions of the third-party stacks, and what is actually on disk.

Two numbers matter for reproducibility: the commit this project was *written against*, and
the commit currently checked out in `_vendor/`. Recording only the first lets a vendored
clone drift silently; recording only the second makes an old result impossible to interpret.
Every rollout trace carries both, plus a boolean saying whether they agree.

The clones live outside the repo (`../_vendor/`) and are gitignored, so `resolve()` returning
`None` on a fresh machine is normal, not an error.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PINNED", "VENDOR_ROOT", "Pin", "resolve", "resolve_all"]

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT.parent / "_vendor"

#: Commit -> what the code in this repo was written and tested against.
#: Both are pre-1.0 and move fast; re-read the source before the training phase (spec 1.3).
PINNED: dict[str, dict[str, str]] = {
    "nemo_gym": {
        "url": "https://github.com/NVIDIA-NeMo/Gym.git",
        "commit": "ec5a8136046aa0dd0c1525313654502f2fb7c42e",
        "dir": "nemo-gym",
        "requires_python": ">=3.12",
    },
    "nemo_rl": {
        "url": "https://github.com/NVIDIA-NeMo/RL.git",
        "commit": "55296257c96d49cd95c7d77613cb0f36bd4a4dc7",
        "dir": "nemo-rl",
        # Not a typo, and not the same as NeMo Gym's: NeMo RL will not install on 3.12.
        "requires_python": ">=3.13.13,<3.14",
    },
}


@dataclass(frozen=True)
class Pin:
    name: str
    pinned: str
    actual: str | None

    @property
    def matches(self) -> bool:
        return self.actual is not None and self.actual == self.pinned

    def as_dict(self) -> dict[str, object]:
        return {"pinned": self.pinned, "actual": self.actual, "matches": self.matches}


def _head(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def resolve(name: str, vendor_root: Path = VENDOR_ROOT) -> Pin:
    spec = PINNED[name]
    return Pin(name=name, pinned=spec["commit"], actual=_head(Path(vendor_root) / spec["dir"]))


def resolve_all(vendor_root: Path = VENDOR_ROOT) -> dict[str, dict[str, object]]:
    return {name: resolve(name, vendor_root).as_dict() for name in PINNED}


def repo_commit() -> str | None:
    """This repository's HEAD, so a trace can be tied to the code that produced it."""
    return _head(REPO_ROOT)
