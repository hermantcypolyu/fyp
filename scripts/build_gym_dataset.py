from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.tools.task_loader import iter_tasks
from envs.coding_env.session import episode_metadata
from envs.coding_env.tools import TOOL_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmark" / "data"


CODING_AGENT_INSTRUCTION = (
    "You are operating a real coding sandbox. Solve the task by interacting "
    "with the repository through the provided tools. "
    "\n\n"
    "Every response must be a tool call until the task is finished. Do not "
    "respond with ordinary assistant text, explanations, or code blocks. "
    "Text-only responses do not modify the repository and do not complete "
    "the task. "
    "\n\n"
    "Start by inspecting the repository and reading the relevant source file. "
    "After every tool call, carefully inspect the returned result before "
    "choosing the next action. The result may report success, failure, invalid "
    "arguments, a missing source match, test output, or another condition that "
    "requires a revised action. "
    "\n\n"
    "Make the smallest correct code change that solves the task. Prefer "
    "replace_lines when read_file provides clear line numbers. Use start_line "
    "and end_line exactly as shown by read_file, and replace only the smallest "
    "necessary range. For replace_lines, provide new_lines as a JSON array "
    "with exactly one source-code line per item. Do not write escaped newline "
    "sequences such as backslash-n inside an item. Preserve Python indentation "
    "exactly. When using edit_file, copy old_str exactly from the latest "
    "read_file output and include enough surrounding source text to make the "
    "match unique. Do not assume a modification succeeded without inspecting "
    "the tool result. "
    "\n\n"
    "If edit_file, replace_lines, or write_file fails, do not continue as if "
    "the repository was modified. Inspect the error, read the file again when "
    "necessary, revise the tool arguments, and retry the modification. You may "
    "call read_file, edit_file, replace_lines, and write_file multiple times. "
    "\n\n"
    "After a successful edit, line replacement, or file write, run tests. "
    "Inspect the test result carefully. If tests fail, revise the "
    "implementation and run tests again. You may repeat the inspect, modify, "
    "and test cycle until the repair is complete or the environment limit is "
    "reached. "
    "\n\n"
    "Call submit only after making a genuine repair attempt and running tests. "
    "The episode is complete only when submit succeeds. Do not stop with a "
    "text-only response before submitting."
)


def gym_tools(task: Any) -> list[dict[str, Any]]:
    """Convert environment tools into strict Responses API function tools.

    Task-specific path enums keep the small policy grounded in the repository
    entrypoint and prevent it from inventing unrelated paths.
    """
    converted: list[dict[str, Any]] = []

    repo_data = task.data.get("repo", {}) or {}
    hint = repo_data.get("entrypoint_hint")
    entrypoint = f"repo/{hint}" if hint else None

    for tool in TOOL_DEFINITIONS:
        parameters = dict(tool["parameters"])
        parameters["additionalProperties"] = False

        # The first agent turn is forced to list_files. Restrict its optional
        # path to the repository sandbox root.
        if tool["name"] == "list_files":
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": ["."],
                        "description": "Repository sandbox root.",
                        "default": ".",
                    }
                },
                "required": [],
                "additionalProperties": False,
            }

        # When the task supplies an entrypoint hint, force source inspection to
        # use that task-specific file.
        elif tool["name"] == "read_file" and entrypoint:
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": [entrypoint],
                        "description": "Task entrypoint file.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            }

        # Keep edits grounded in the same task-specific entrypoint. old_str
        # still has to be copied exactly from read_file output.
        elif tool["name"] == "edit_file" and entrypoint:
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": [entrypoint],
                        "description": "Task entrypoint file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "Exact source text copied from the latest read_file "
                            "output. Include surrounding lines when needed to "
                            "make the match unique."
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": (
                            "Replacement source text implementing the smallest "
                            "correct repair."
                        ),
                    },
                },
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
            }

        elif tool["name"] == "replace_lines" and entrypoint:
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": [entrypoint],
                        "description": "Task entrypoint file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "First existing source line to replace, "
                            "using the 1-based numbers from read_file."
                        ),
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Last existing source line to replace, "
                            "inclusive."
                        ),
                    },
                    "new_lines": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": (
                            "Replacement source code, one source line per "
                            "array item. Copy indentation exactly. Use an "
                            "empty array to delete the selected lines."
                        ),
                    },
                },
                "required": [
                    "path",
                    "start_line",
                    "end_line",
                    "new_lines",
                ],
                "additionalProperties": False,
            }

        converted.append(
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": parameters,
                "strict": True,
            }
        )

    return converted


def build_row(task: Any) -> dict[str, Any]:
    """Build one NeMo Gym training row from one benchmark task."""
    metadata = episode_metadata(task)

    return {
        "task_id": task.id,
        "split": "train",
        "responses_create_params": {
            "input": [
                {
                    "role": "developer",
                    "content": CODING_AGENT_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": task.data["instruction"],
                },
            ],
            "tools": gym_tools(task),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "metadata": metadata,
        },
        "metadata": metadata,
    }


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write materialized dataset rows as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main() -> None:
    """Build train and single-row integration datasets.

    Held-out tasks are deliberately excluded.
    """
    tasks = list(iter_tasks(("train",)))

    if not tasks:
        raise SystemExit("No train tasks found.")

    rows = [build_row(task) for task in tasks]

    train_path = OUTPUT_DIR / "train.jsonl"
    example_path = OUTPUT_DIR / "example.jsonl"

    write_jsonl(rows, train_path)
    write_jsonl(rows[:1], example_path)

    print(f"Wrote {len(rows)} train rows to {train_path}")
    print(f"Wrote 1 integration row to {example_path}")
    print("Held-out tasks were not read.")


if __name__ == "__main__":
    main()
