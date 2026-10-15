from __future__ import annotations

import json
from pathlib import Path

from benchmark.tools.task_loader import iter_tasks
from envs.coding_env.session import episode_metadata
from envs.coding_env.tools import TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmark" / "data"


def gym_tools(task) -> list[dict]:
    converted = []

    for tool in TOOL_DEFINITIONS:
        parameters = dict(tool["parameters"])
        parameters["additionalProperties"] = False

        # The first agent turn forces list_files. Constrain its optional path
        # so the small model cannot invent an absolute Windows path or treat
        # the .env file mentioned in the task as a directory.
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

        if tool["name"] == "read_file":
            hint = (task.data.get("repo", {}) or {}).get("entrypoint_hint")
            if hint:
                entrypoint = f"repo/{hint}"
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

        if tool["name"] == "edit_file":
            hint = (task.data.get("repo", {}) or {}).get("entrypoint_hint")
            if hint:
                entrypoint = f"repo/{hint}"
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
                            "description": "Exact source text to replace.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement source text.",
                        },
                    },
                    "required": ["path", "old_str", "new_str"],
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


def build_row(task) -> dict:
    return {
        "task_id": task.id,
        "split": "train",
        "responses_create_params": {
            "input": [
                {
                    "role": "developer",
                    "content": (
                        "You are operating a real coding sandbox. You must solve the task "
                        "by calling the provided tools. Do not only explain a solution and do "
                        "not output a code block instead of editing files. Your first action "
                        "must be a list_files tool call. Then inspect the relevant files, make "
                        "the smallest correct edit with edit_file or write_file, run_tests, "
                        "and finally call submit. Ordinary assistant text does not modify the "
                        "repository and will receive a failing reward."
                    ),
                },
                {
                    "role": "user",
                    "content": task.data["instruction"],
                },
            ],
            "tools": gym_tools(task),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "metadata": episode_metadata(task),
        },
        "metadata": episode_metadata(task),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
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
