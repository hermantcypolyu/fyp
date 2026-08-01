"""The six agent tools (spec 5.5).

Fewer tools means a smaller action space, which is what a 1.5B policy needs. There are
exactly six and there will not be a seventh: `list_files`, `read_file`, `write_file`,
`edit_file`, `run_tests`, `submit`.

Three rules govern everything below.

1. **Paths are sandbox-relative, always.** `repo/humanize/number.py`, never `C:\\...` and
   never `../../etc/passwd`. This is the same namespace `task.yaml`'s
   `allowed_write_globs` uses, so a permission decision is a pure string match against
   `benchmark.tools.globs` - no path arithmetic at the decision point.
2. **Errors are short and instructive.** A 1.5B model recovers from
   `No match for old_str in repo/foo.py` and does not recover from a traceback. Every
   failure path here returns one sentence naming the file and the fix. Nothing raises out
   of a tool.
3. **Every output is hard-truncated.** Runaway pytest output eats the context window and
   produces failures that have nothing to do with coding ability, which would silently
   corrupt the whole study.

Two of the failures are *reward gates*, not just errors: writing to a
`forbidden_write_globs` path (`tests_modified`, the #1 reward hack) and writing outside
the sandbox root (`sandbox_escape`). Those are recorded on the toolbox and zero the reward
outright in 5.7 - the tool still refuses politely, but the episode is already lost.

Grounded in NeMo Gym built-ins; see the per-function citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.tools import task_loader
from benchmark.tools.globs import normalise, path_matches_any

__all__ = [
    "GATE_SANDBOX_ESCAPE",
    "GATE_TESTS_MODIFIED",
    "MAX_READ_LINES",
    "TOOL_DEFINITIONS",
    "ToolCall",
    "ToolResult",
    "Toolbox",
    "truncate",
]

#: Rough chars-per-token for English + Python under a BPE tokeniser. Deliberately an
#: estimate: the budget only has to be *a* hard ceiling, and counting real tokens here
#: would drag a tokenizer dependency into the environment core for no benefit.
CHARS_PER_TOKEN = 4

#: Per-tool output budgets, in tokens.
BUDGET_LIST_FILES = 700
BUDGET_READ_FILE = 1200
BUDGET_RUN_TESTS = 900
BUDGET_SHORT = 120  # write_file / edit_file / submit / any error

#: A single `read_file` never returns more than this many lines, whatever `end` says.
MAX_READ_LINES = 200

#: `list_files` never returns more than this many entries.
MAX_LIST_ENTRIES = 200

#: Reward gates raised from here. Consumed by rewards/ in 5.7.
GATE_TESTS_MODIFIED = "tests_modified"
GATE_SANDBOX_ESCAPE = "sandbox_escape"

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".ruff_cache"}


# --------------------------------------------------------------------------- truncation


def truncate(text: str, max_tokens: int) -> str:
    """Head-and-tail truncation with an explicit marker.

    Head *and* tail rather than head-only because the two informative parts of a pytest
    run are the first failure and the final summary line; keeping only the head throws the
    count away, keeping only the tail throws the diagnosis away.

    # Pattern adapted from: nemo-gym/resources_servers/browsecomp_advanced_harness/app.py
    # (_bash_truncate, L389) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
    """
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[...truncated {omitted} chars; head above, tail below...]\n{text[-tail:]}"


# --------------------------------------------------------------------------- results


@dataclass
class ToolResult:
    """What one tool call returns. `output` is already truncated and agent-visible."""

    ok: bool
    output: str

    def __str__(self) -> str:  # so a harness can drop it straight into a tool message
        return self.output


@dataclass
class ToolCall:
    """One entry of the episode's tool log. Feeds analysis/trace_schema.json in B4."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    output: str


def _ok(text: str, budget: int = BUDGET_SHORT) -> ToolResult:
    return ToolResult(ok=True, output=truncate(text, budget))


def _err(text: str) -> ToolResult:
    """Errors are always short: one sentence, naming the file and the fix."""
    return ToolResult(ok=False, output=truncate(f"Error: {text}", BUDGET_SHORT))


# --------------------------------------------------------------------------- toolbox


@dataclass
class Toolbox:
    """The six tools, bound to one sandbox directory and one task's constraints.

    `root` is the materialised working copy: it contains `repo/` and `smoke/`, and - in
    the agent's world - deliberately no `tests/` (see `task_loader.materialise`).
    """

    root: Path
    allowed_write_globs: list[str] = field(default_factory=list)
    forbidden_write_globs: list[str] = field(default_factory=list)
    timeout_s: int = 60
    max_tool_calls: int = 40

    submitted: bool = field(default=False, init=False)
    gates: set[str] = field(default_factory=set, init=False)
    log: list[ToolCall] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    @classmethod
    def for_task(cls, task: task_loader.Task, root: Path) -> Toolbox:
        """Bind the tools to a task's `constraints:` block and an already-materialised copy."""
        constraints = task.data.get("constraints", {}) or {}
        return cls(
            root=root,
            allowed_write_globs=list(constraints.get("allowed_write_globs", [])),
            forbidden_write_globs=list(constraints.get("forbidden_write_globs", [])),
            timeout_s=task.timeout_s,
            max_tool_calls=int(constraints.get("max_tool_calls", 40)),
        )

    # ------------------------------------------------------------------ dispatch

    @property
    def n_calls(self) -> int:
        return len(self.log)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Dispatch by name. The harness calls this; the tools below are also public.

        # Pattern adapted from: nemo-gym/resources_servers/example_multi_step/app.py
        # (SimpleResourcesServer.setup_webserver tool routing) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
        """
        arguments = dict(arguments or {})
        method = getattr(self, name, None)
        if name not in _DISPATCHABLE or method is None:
            result = _err(f"unknown tool {name!r}. Available: {', '.join(sorted(_DISPATCHABLE))}.")
        elif self.n_calls >= self.max_tool_calls:
            result = _err(f"tool call budget of {self.max_tool_calls} is exhausted. Call submit() now.")
        else:
            try:
                result = method(**arguments)
            except TypeError as e:  # wrong/missing argument names from the model
                result = _err(f"bad arguments for {name}: {e}")
        self.log.append(ToolCall(name=name, arguments=arguments, ok=result.ok, output=result.output))
        return result

    # ------------------------------------------------------------------ paths

    def _resolve(self, raw: str) -> tuple[str, Path]:
        """Sandbox-relative string -> (normalised relative path, absolute path).

        Raises `_PathError` with a one-line message. Escapes are checked after
        `resolve()`, so a symlink pointing out of the sandbox is caught too.

        # Pattern adapted from: nemo-gym/resources_servers/legal_agent_bench/vendor/
        # harvey_labs/lab_harbor/container_tool_runner.py (_ensure_allowed, L101)
        # @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
        """
        if not isinstance(raw, str):
            raise _PathError(f"path must be a string, got {type(raw).__name__}.")
        if Path(raw).is_absolute() or (len(raw) > 1 and raw[1] == ":"):
            raise _PathError(f"{raw!r} is absolute. Use a path relative to the sandbox, e.g. repo/module.py.")
        rel = normalise(raw)
        absolute = (self.root / rel).resolve() if rel else self.root
        if absolute != self.root and self.root not in absolute.parents:
            self.gates.add(GATE_SANDBOX_ESCAPE)
            raise _PathError(f"{raw!r} points outside the sandbox. Stay under repo/.")
        return rel, absolute

    def _check_writable(self, rel: str) -> str | None:
        """None if `rel` may be written, else a one-line refusal. Deny wins over allow."""
        if path_matches_any(rel, self.forbidden_write_globs):
            self.gates.add(GATE_TESTS_MODIFIED)
            return f"{rel} is read-only. Fix the source under repo/ instead of changing the tests."
        if self.allowed_write_globs and not path_matches_any(rel, self.allowed_write_globs):
            allowed = ", ".join(self.allowed_write_globs)
            return f"{rel} is not writable. You may only write to: {allowed}."
        return None

    # ------------------------------------------------------------------ the six tools

    def list_files(self, path: str = ".") -> ToolResult:
        """List files under `path`, recursively. Directories end with `/`.

        Recursive because these repos are under ~500 LOC by construction (5.4 rule 6): one
        call shows the whole tree, which spends far fewer turns than a 1.5B model walking
        directories one level at a time.
        """
        try:
            rel, target = self._resolve(path)
        except _PathError as e:
            return _err(str(e))
        if not target.exists():
            return _err(f"no such path: {rel or '.'}")
        if target.is_file():
            return _ok(rel, BUDGET_LIST_FILES)

        entries: list[str] = []
        for p in sorted(target.rglob("*"), key=lambda q: (q.parent.as_posix(), q.name)):
            if _SKIP_DIRS & set(p.relative_to(target).parts):
                continue
            shown = p.relative_to(self.root).as_posix()
            entries.append(f"{shown}/" if p.is_dir() else shown)
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"[...more than {MAX_LIST_ENTRIES} entries; listing stopped]")
                break
        if not entries:
            return _ok(f"{rel or '.'} is empty", BUDGET_LIST_FILES)
        return _ok("\n".join(entries), BUDGET_LIST_FILES)

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> ToolResult:
        """Read `path` with 1-based line numbers. `start`/`end` are inclusive.

        Line numbers are returned so that a later `edit_file` can quote an exact,
        unambiguous `old_str` - without them the model guesses at whitespace.

        # Pattern adapted from: nemo-gym/resources_servers/legal_agent_bench/vendor/
        # harvey_labs/lab_harbor/container_tool_runner.py (_read / _slice_read_content, L139)
        # @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
        """
        try:
            rel, target = self._resolve(path)
        except _PathError as e:
            return _err(str(e))
        if not target.exists():
            return _err(f"no such file: {rel}")
        if target.is_dir():
            return _err(f"{rel} is a directory. Use list_files to see what is in it.")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return _err(f"{rel} is not UTF-8 text.")

        total = len(lines)
        first = max(int(start), 1) if start is not None else 1
        last = min(int(end), total) if end is not None else total
        if first > total:
            return _err(f"{rel} has only {total} lines; start={first} is past the end.")
        last = min(last, first + MAX_READ_LINES - 1)
        if last < first:
            return _err(f"end={end} is before start={first} in {rel}.")

        body = "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(first, last + 1))
        header = f"{rel} lines {first}-{last} of {total}"
        if last < total:
            header += f" (continue with start={last + 1})"
        return _ok(f"{header}\n{body}", BUDGET_READ_FILE)

    def write_file(self, path: str, content: str) -> ToolResult:
        """Overwrite (or create) `path`. Refused unless the write globs allow it.

        # Pattern adapted from: nemo-gym/resources_servers/legal_agent_bench/vendor/
        # harvey_labs/lab_harbor/container_tool_runner.py (_write, L323)
        # @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
        """
        try:
            rel, target = self._resolve(path)
        except _PathError as e:
            return _err(str(e))
        if not rel:
            return _err("path is required, e.g. repo/module.py.")
        if refusal := self._check_writable(rel):
            return _err(refusal)
        if not isinstance(content, str):
            return _err(f"content must be a string, got {type(content).__name__}.")
        if target.is_dir():
            return _err(f"{rel} is a directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return _ok(f"Wrote {len(content.splitlines())} lines to {rel}.")

    def edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """Replace the single exact occurrence of `old_str` in `path` with `new_str`.

        Zero matches and more than one match are both errors, and both name the count -
        an ambiguous edit silently applied to the wrong site is the worst failure mode
        available to a small model, because nothing in the transcript reveals it.

        # Pattern adapted from: nemo-gym/resources_servers/legal_agent_bench/vendor/
        # harvey_labs/lab_harbor/container_tool_runner.py (_edit, L393)
        # @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
        """
        try:
            rel, target = self._resolve(path)
        except _PathError as e:
            return _err(str(e))
        if refusal := self._check_writable(rel):
            return _err(refusal)
        if not target.is_file():
            return _err(f"no such file: {rel}")
        if not isinstance(old_str, str) or not isinstance(new_str, str):
            return _err("old_str and new_str must both be strings.")
        if not old_str:
            return _err("old_str is empty. Give the exact text to replace.")

        text = target.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return _err(f"No match for old_str in {rel}. Read the file and copy the text exactly.")
        if count > 1:
            return _err(f"old_str matches {count} times in {rel}. Add surrounding lines to make it unique.")
        target.write_text(text.replace(old_str, new_str, 1), encoding="utf-8", newline="\n")
        return _ok(f"Edited {rel}.")

    def run_tests(self) -> ToolResult:
        """Run the **visible smoke tests only** - never the hidden suite.

        The hidden suite is not on disk in the agent's sandbox at all
        (`materialise(..., include_tests=False)`), so this cannot leak the grading signal
        even if the guard below were removed. The guard is belt-and-braces: if a `tests/`
        directory ever appears here, that is a sandbox bug and this refuses to run.
        """
        if (self.root / "tests").exists():
            return _err("sandbox is misbuilt (hidden tests present); refusing to run. Report this.")
        smoke_dir = self.root / "smoke"
        if not smoke_dir.is_dir():
            return _err("this task has no smoke tests. Reason about the code and submit when done.")
        files = [f"smoke/{p.name}" for p in sorted(smoke_dir.glob("*.py")) if p.name != "__init__.py"]
        if not files:
            return _err("this task has no smoke tests. Reason about the code and submit when done.")

        result = task_loader.run_tests(self.root, files, timeout_s=self.timeout_s, tail_chars=4000)
        if result.timed_out:
            return _err(f"smoke tests exceeded {self.timeout_s}s. Look for an infinite loop.")
        if result.collection_error:
            return _err(
                f"smoke tests could not be collected - the repo probably no longer imports.\n{result.stdout_tail}"
            )
        head = f"{result.summary()} (smoke tests only; these are not the grading tests)"
        if result.all_passed:
            return _ok(head, BUDGET_RUN_TESTS)
        return _ok(f"{head}\n{result.stdout_tail}", BUDGET_RUN_TESTS)

    def submit(self) -> ToolResult:
        """End the episode. Hidden verification runs after this; there is no going back."""
        self.submitted = True
        return _ok("Submitted. The episode is over.")


_DISPATCHABLE = frozenset({"list_files", "read_file", "write_file", "edit_file", "run_tests", "submit"})


class _PathError(Exception):
    """Internal: carries a one-line, model-readable message out of `_resolve`."""


# --------------------------------------------------------------------- tool schemas

# The agent-visible JSON schema for the six tools, in OpenAI function-calling shape.
# Descriptions are written for a 1.5B reader: short, concrete, and stating the constraint
# up front rather than in a trailing clause.
#
# Pattern adapted from: nemo-gym/resources_servers/legal_agent_bench/vendor/harvey_labs/
# lab_harbor/tools.py (TOOL_DEFINITIONS) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files in the sandbox, recursively. Start here to see the repo layout.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Sandbox-relative directory, e.g. 'repo'. Defaults to '.'.",
                    "default": ".",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            f"Read a file with line numbers. Returns at most {MAX_READ_LINES} lines per call; "
            "use start/end to page through longer files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Sandbox-relative file, e.g. 'repo/pkg/mod.py'."},
                "start": {"type": "integer", "description": "First line to show, 1-based. Optional."},
                "end": {"type": "integer", "description": "Last line to show, inclusive. Optional."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a whole file, replacing any existing content. Prefer edit_file for small changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Sandbox-relative file to write."},
                "content": {"type": "string", "description": "The complete new contents of the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact snippet in a file. old_str must appear exactly once - "
            "include surrounding lines to make it unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Sandbox-relative file to edit."},
                "old_str": {"type": "string", "description": "Exact text to replace, copied from read_file."},
                "new_str": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run the visible smoke tests. These are a sanity check, not the grading tests - "
            "passing them does not mean the task is solved."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit",
        "description": "Finish the task. Call this once the fix is in place; the episode ends immediately.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]
