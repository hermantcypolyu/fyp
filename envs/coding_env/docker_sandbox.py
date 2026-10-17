"""Docker-backed agent sandbox for SWE-Gym Lite pilot tasks.

The official instance image is used only to materialise a clean repository.
The agent works on a disposable host copy through the existing Toolbox.

Verifier-only patches and hidden test identifiers are never copied into the
agent workspace.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envs.coding_env.tools import Toolbox


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AGENT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "swe_gym_lite"
    / "pilot_5_agent_manifest.json"
)


class DockerSandboxError(RuntimeError):
    """Raised when a Docker-backed sandbox cannot be prepared safely."""


@dataclass(frozen=True)
class DockerTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    image: str


@dataclass
class DockerSandbox:
    """One disposable SWE-Gym agent workspace."""

    task: DockerTask
    root: Path
    toolbox: Toolbox
    base: Path
    _owns_dir: bool = field(default=True, repr=False)
    _stopped: bool = field(default=False, init=False, repr=False)

    @classmethod
    def start(
        cls,
        task: DockerTask,
        parent: Path | None = None,
    ) -> "DockerSandbox":
        owns_dir = parent is None

        base = (
            Path(parent)
            if parent is not None
            else Path(
                tempfile.mkdtemp(
                    prefix=f"swegym_{task.instance_id}_"
                )
            )
        ).resolve()

        if base.exists() and owns_dir is False:
            shutil.rmtree(base)

        base.mkdir(parents=True, exist_ok=True)
        root = base / "work"
        repo_dir = root / "repo"
        root.mkdir()

        container_name = (
            "hermanyeung-swegym-materialise-"
            + task.instance_id.replace("/", "-")
        )

        try:
            _run(
                [
                    "docker",
                    "rm",
                    "-f",
                    container_name,
                ],
                check=False,
                quiet=True,
            )

            _run(
                [
                    "docker",
                    "create",
                    "--name",
                    container_name,
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "256",
                    "--memory",
                    "2g",
                    "--cpus",
                    "2",
                    task.image,
                    "/bin/bash",
                    "-lc",
                    "sleep infinity",
                ]
            )

            _run(
                [
                    "docker",
                    "cp",
                    f"{container_name}:/testbed",
                    str(repo_dir),
                ]
            )

        finally:
            _run(
                [
                    "docker",
                    "rm",
                    "-f",
                    container_name,
                ],
                check=False,
                quiet=True,
            )

        if not repo_dir.is_dir():
            raise DockerSandboxError(
                f"{task.instance_id}: Docker image did not "
                "materialise /testbed."
            )

        actual_commit = _git(
            repo_dir,
            "rev-parse",
            "HEAD",
        ).strip()

        if actual_commit != task.base_commit:
            raise DockerSandboxError(
                f"{task.instance_id}: expected base commit "
                f"{task.base_commit}, got {actual_commit}."
            )

        # Some official instance images contain tracked setup-time
        # modifications, for example an added pip pin in a requirements
        # file. The installed environment already exists, so the agent
        # repository must be restored to the dataset base commit before
        # the episode begins. Ignored generated files are deliberately
        # retained; do not use git clean -fdx here.
        _git(
            repo_dir,
            "reset",
            "--hard",
            task.base_commit,
        )

        status = _git(
            repo_dir,
            "status",
            "--short",
        ).strip()

        if status:
            status_preview = status[:2000]

            raise DockerSandboxError(
                f"{task.instance_id}: materialised repository "
                "is not clean after restoring the base commit.\n"
                f"{status_preview}"
            )

        forbidden_names = {
            ".swegym_test.patch",
            ".swegym_gold.patch",
        }

        leaked = [
            path
            for path in repo_dir.rglob("*")
            if path.name in forbidden_names
        ]

        if leaked:
            raise DockerSandboxError(
                f"{task.instance_id}: verifier-only files "
                "leaked into the agent workspace."
            )

        toolbox = Toolbox(
            root=root,
            allowed_write_globs=["repo/**"],
            forbidden_write_globs=[
                "tests/**",
                "solution/**",
                "**/conftest.py",
                "task.yaml",
            ],
            timeout_s=120,
            max_tool_calls=40,
        )

        return cls(
            task=task,
            root=root,
            toolbox=toolbox,
            base=base,
            _owns_dir=owns_dir,
        )

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    def export_patch(self) -> str:
        """Return the agent's candidate Git patch."""

        return _git(
            self.repo_dir,
            "diff",
            "--binary",
            "--",
            ".",
        )

    def stop(self) -> None:
        if self._stopped:
            return

        self._stopped = True

        if self._owns_dir:
            shutil.rmtree(
                self.base,
                ignore_errors=True,
            )

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def load_docker_tasks(
    manifest_path: Path = DEFAULT_AGENT_MANIFEST,
) -> dict[str, DockerTask]:
    rows = json.loads(
        Path(manifest_path).read_text(
            encoding="utf-8"
        )
    )

    tasks = {}

    for row in rows:
        task = DockerTask(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            image=row["image"],
        )

        if task.instance_id in tasks:
            raise DockerSandboxError(
                f"duplicate instance: {task.instance_id}"
            )

        tasks[task.instance_id] = task

    return tasks


def open_docker_sandbox(
    task: DockerTask,
    parent: Path | None = None,
) -> DockerSandbox:
    return DockerSandbox.start(
        task,
        parent,
    )


def _git(repo_dir: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            *arguments,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return result.stdout


def _run(
    command: list[str],
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=(
            subprocess.DEVNULL
            if quiet
            else None
        ),
        stderr=(
            subprocess.DEVNULL
            if quiet
            else None
        ),
    )
