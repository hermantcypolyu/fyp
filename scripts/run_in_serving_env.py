#!/usr/bin/env python
"""Run a repository script under the *serving* virtualenv, safely.

Why this exists
---------------
`scripts/collect_baseline_rollouts.py` drives a local vLLM server. vLLM and its pinned
`transformers` live in the serving virtualenv (`~/.venvs/qwen-server`), while the project's
own dependencies live in `.venv`. Neither venv contains the other's packages, so the
collector cannot simply be run with `.venv/bin/python`.

Adding the serving venv to `PYTHONPATH` used to fail anyway, for a reason that had nothing
to do with this project: that venv's `site-packages` contains llama-cpp-python's
`scripts/__init__.py`, which shadowed this repository's `scripts/` namespace package, so
every `import scripts.<module>` raised `ModuleNotFoundError`. Phase 6E worked around it with
an undocumented `/tmp/phase6e_shim` symlink farm.

The repository now declares `scripts/__init__.py`, making `scripts` a regular package. Two
regular packages tie, and the tie is broken by `sys.path` order, so putting the repository
first in `PYTHONPATH` selects ours deterministically. This launcher does exactly that, and
verifies it happened rather than assuming it.

What it checks before handing over
----------------------------------
* the repository's `scripts` package is the one that resolves, not a foreign one;
* `transformers` is importable and its recorded version satisfies its own requirements
  (Phase 6E's metadata failure was a missing `regex` requirement, not a missing transform);
* the target script exists.

It deliberately does not install, upgrade or remove anything, and does not touch the GPU.
Whatever the script does with CUDA is still governed by `CUDA_VISIBLE_DEVICES`, which this
launcher never sets (docs/PHASE5_GPU_RESOURCE_RULES.md rules 1-2: a script that quietly
overrides the environment makes the rule unenforceable).

Usage
-----
    python scripts/run_in_serving_env.py scripts/collect_baseline_rollouts.py --dry-run
    python scripts/run_in_serving_env.py scripts/collect_baseline_rollouts.py --json

Exit codes: 0 the script ran / 3 environment check failed / 4 interpreter not found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default serving interpreter. Overridable so a moved venv does not require editing code.
DEFAULT_SERVING_PYTHON = Path.home() / "venvs" / "qwen-server" / "bin" / "python"

#: Packages the collector needs from the serving venv. `transformers` is checked by import
#: because its metadata validation is what failed in Phase 6E.
SERVING_REQUIREMENTS = ("transformers", "tokenizers", "safetensors", "regex")

_PROBE = r"""
import importlib.util
import json
import sys

out = {"scripts_path": None, "repo_scripts_ok": False, "packages": {}, "error": None}
try:
    import scripts

    out["scripts_path"] = getattr(scripts, "__file__", None)
    out["repo_scripts_ok"] = bool(out["scripts_path"]) and str(sys.argv[1]) in str(out["scripts_path"])
except Exception as exc:  # noqa: BLE001 - reported, not raised: the caller decides
    out["error"] = f"{type(exc).__name__}: {exc}"

for name in sys.argv[2].split(","):
    if not name:
        continue
    try:
        module = __import__(name)
        out["packages"][name] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        out["packages"][name] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"

print(json.dumps(out))
"""


def _serving_python() -> Path:
    override = os.environ.get("SERVING_PYTHON")
    return Path(override).expanduser() if override else DEFAULT_SERVING_PYTHON


def _check_environment(python: Path, repo_root: Path) -> tuple[bool, dict]:
    """Import-probe the serving interpreter with the repository first on `sys.path`.

    Run in a subprocess on purpose: the packages under test must be imported by the
    *serving* interpreter, and importing them into this one would check the wrong venv.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))
    proc = subprocess.run(
        [str(python), "-c", _PROBE, str(repo_root), ",".join(SERVING_REQUIREMENTS)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        return False, {"error": proc.stderr.strip() or f"probe exited {proc.returncode}"}
    import json

    try:
        return True, json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, {"error": f"probe produced no parseable output: {proc.stdout!r}"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if argv else 2

    script, extra = argv[0], argv[1:]
    script_path = Path(script)
    if not script_path.exists():
        candidate = REPO_ROOT / script
        if not candidate.exists():
            print(f"ERROR: no such script: {script}", file=sys.stderr)
            return 2
        script_path = candidate

    python = _serving_python()
    if not python.exists():
        print(
            f"ERROR: serving interpreter not found at {python}.\n"
            "Point SERVING_PYTHON at the venv that has vLLM installed.",
            file=sys.stderr,
        )
        return 4

    ok, report = _check_environment(python, REPO_ROOT)
    if not ok or not report.get("repo_scripts_ok"):
        print("ERROR: the serving environment is not usable for this script.", file=sys.stderr)
        print(f"  scripts resolved to: {report.get('scripts_path')}", file=sys.stderr)
        print(f"  expected under:      {REPO_ROOT / 'scripts'}", file=sys.stderr)
        if report.get("error"):
            print(f"  error: {report['error']}", file=sys.stderr)
        print(
            "\nIf `scripts` resolved outside the repository, a foreign `scripts` package is "
            "still winning the import. Check that scripts/__init__.py exists.",
            file=sys.stderr,
        )
        return 3

    bad = {name: v for name, v in report["packages"].items() if str(v).startswith("IMPORT FAILED")}
    if bad:
        print("ERROR: serving environment is missing required packages:", file=sys.stderr)
        for name, detail in bad.items():
            print(f"  {name}: {detail}", file=sys.stderr)
        return 3

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    versions = " ".join(f"{name}={version}" for name, version in report["packages"].items())
    print(
        f"[run_in_serving_env] {python}\n"
        f"[run_in_serving_env] scripts={report['scripts_path']}\n"
        f"[run_in_serving_env] {versions}",
        file=sys.stderr,
    )
    return subprocess.call([str(python), str(script_path), *extra], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
