"""The agent's sandbox, and the guarantee that the hidden tests are not in it.

Every container-backed provider NeMo Gym ships - docker, apptainer, enroot, daytona,
ecs_fargate - is unavailable on this machine (ADR 0001: no Docker, no WSL, no GPU). So this
is a local-filesystem provider wearing the same lifecycle interface, and the isolation it
offers has to come from somewhere other than a kernel namespace.

**It comes from absence.** The hidden suite is never copied into the agent's world:
`materialise(task, root, include_tests=False)` simply does not write it. That is a much
stronger guarantee than file permissions, because it does not depend on a check being
correct, on `chmod` meaning anything on NTFS, or on the tool layer having no holes. There
is no `tests/` directory to read, no matter what path the agent constructs.

The three defences, in the order they would have to be defeated:

1. **Absence** - the hidden tests are not on disk here (this module).
2. **Containment** - `Toolbox._resolve` rejects absolute paths and, after `resolve()`,
   anything outside the sandbox root, which is what stops `../` and symlinks reaching the
   task source directory that *does* hold them (tools.py).
3. **Refusal** - `run_tests` will not run at all if a `tests/` directory appears anyway,
   converting a construction bug here into a loud failure rather than an inflated score
   (tools.py).

# Pattern adapted from: nemo-gym/nemo_gym/sandbox/api.py (Sandbox.start/stop/__enter__,
# L258) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Any

from benchmark.tools.task_loader import Task, materialise
from envs.coding_env.tools import Toolbox

__all__ = ["Sandbox", "open_sandbox"]


@dataclass
class Sandbox:
    """One episode's disposable working copy, plus the toolbox bound to it.

    Use it as a context manager; `stop()` removes the directory.
    """

    task: Task
    root: Path
    toolbox: Toolbox
    base: Path
    _owns_dir: bool = field(default=True, repr=False)
    _stopped: bool = field(default=False, init=False, repr=False)

    @classmethod
    def start(cls, task: Task, parent: Path | None = None) -> Sandbox:
        """Materialise `task` without its hidden tests and return a live sandbox.

        `parent` is for tests that want a predictable location; production uses a temp dir.
        """
        owns = parent is None
        base = Path(parent) if parent is not None else Path(tempfile.mkdtemp(prefix=f"sandbox_{task.id}_"))
        root = base / "work"
        materialise(task, root, include_tests=False, include_smoke=True)
        if (root / "tests").exists():  # cannot happen; assert it anyway - this is the guarantee
            raise RuntimeError(f"{task.id}: hidden tests leaked into the sandbox at {root / 'tests'}")
        return cls(
            task=task,
            root=root.resolve(),
            toolbox=Toolbox.for_task(task, root),
            base=base.resolve(),
            _owns_dir=owns,
        )

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._owns_dir:
            rmtree(self.base, ignore_errors=True)

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def open_sandbox(task: Task, parent: Path | None = None) -> Sandbox:
    """`with open_sandbox(task) as sb: ...`"""
    return Sandbox.start(task, parent)
