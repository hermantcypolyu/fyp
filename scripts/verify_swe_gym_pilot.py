"""Verify SWE-Gym Lite pilot instances in isolated Docker containers.

The verifier-only test and gold patches remain outside the agent workspace.
Each invocation creates one temporary container and removes it in a finally
block.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "swe_gym_lite"
    / "pilot_5_verification_manifest.json"
)

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "swe_gym_lite"
    / "verification"
)

PYTEST_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"

PYTEST_SUMMARY_PATTERN = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<status>passed|failed|error|errors|skipped)"
)



def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def run_command(
    command: list[str],
    *,
    check: bool = False,
    capture_output: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=check,
        text=input_bytes is None,
        input=input_bytes,
        capture_output=capture_output,
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError("Verification manifest must contain a JSON list.")

    return rows


def select_instance(
    rows: list[dict[str, Any]],
    instance_id: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("instance_id") == instance_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest row for {instance_id!r}, "
            f"found {len(matches)}."
        )

    return matches[0]


def safe_container_suffix(instance_id: str) -> str:
    suffix = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "-",
        instance_id,
    )
    return suffix.lower()[:80]


def docker_exec(
    container_name: str,
    arguments: list[str],
    *,
    workdir: str = "/testbed",
    check: bool = False,
    capture_output: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    command = [
        "docker",
        "exec",
    ]

    if input_bytes is not None:
        command.append("-i")

    command.extend(
        [
            "-w",
            workdir,
            container_name,
            *arguments,
        ]
    )

    return run_command(
        command,
        check=check,
        capture_output=capture_output,
        input_bytes=input_bytes,
    )


def stream_private_file(
    *,
    container_name: str,
    source: Path,
    destination: str,
) -> None:
    docker_exec(
        container_name,
        [
            "/bin/bash",
            "-c",
            f"umask 077; cat > {destination}",
        ],
        check=True,
        input_bytes=source.read_bytes(),
    )


def apply_patch(
    *,
    container_name: str,
    patch_path: str,
) -> None:
    docker_exec(
        container_name,
        [
            "git",
            "apply",
            "--check",
            patch_path,
        ],
        check=True,
    )

    docker_exec(
        container_name,
        [
            "git",
            "apply",
            patch_path,
        ],
        check=True,
    )


def run_pytest(
    *,
    container_name: str,
    tests: list[str],
    label: str,
    pytest_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    print(f"\nRunning {label}:")

    for test in tests:
        print(" ", test)

    if not tests:
        print("  No tests in this group.")

        return {
            "label": label,
            "tests": [],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "summary_counts": {},
        }

    extra_args = pytest_extra_args or []

    result = docker_exec(
        container_name,
        [
            PYTEST_PYTHON,
            "-m",
            "pytest",
            "-q",
            *extra_args,
            *tests,
        ],
        capture_output=True,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n")

    combined = f"{stdout}\n{stderr}"

    summary_counts: dict[str, int] = {}

    for match in PYTEST_SUMMARY_PATTERN.finditer(combined):
        status = match.group("status")
        count = int(match.group("count"))
        summary_counts[status] = max(
            summary_counts.get(status, 0),
            count,
        )

    print(f"{label} exit code:", result.returncode)

    return {
        "label": label,
        "tests": tests,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "summary_counts": summary_counts,
    }


def verify_instance(
    *,
    row: dict[str, Any],
    mode: str,
    pull: bool,
    candidate_patch_path: Path | None = None,
) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    pytest_extra_args = (
        ["-p", "no:snail"]
        if instance_id == "facebookresearch__hydra-1741"
        else []
    )
    image = str(row["image"])
    base_commit = str(row["base_commit"])

    fail_to_pass = list(row.get("fail_to_pass") or [])
    pass_to_pass = list(row.get("pass_to_pass") or [])

    test_patch_path = resolve_project_path(
        row["test_patch_path"]
    )

    gold_patch_path = resolve_project_path(
        row["gold_patch_path"]
    )

    if candidate_patch_path is not None:
        candidate_patch_path = Path(
            candidate_patch_path
        ).resolve()

    if mode == "candidate":
        if candidate_patch_path is None:
            raise ValueError(
                "candidate mode requires "
                "--candidate-patch."
            )

        if not candidate_patch_path.is_file():
            raise FileNotFoundError(
                candidate_patch_path
            )

    if not test_patch_path.is_file():
        raise FileNotFoundError(test_patch_path)

    if not gold_patch_path.is_file():
        raise FileNotFoundError(gold_patch_path)

    container_name = (
        "hermanyeung-swegym-verify-"
        f"{safe_container_suffix(instance_id)}"
    )

    result: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": row.get("repo"),
        "mode": mode,
        "image": image,
        "base_commit_expected": base_commit,
        "base_commit_actual": None,
        "base_commit_match": False,
        "test_patch_applied": False,
        "gold_patch_applied": False,
        "candidate_patch_applied": False,
        "candidate_patch_empty": False,
        "candidate_patch_path": (
            str(candidate_patch_path)
            if candidate_patch_path is not None
            else None
        ),
        "tests_passed": 0,
        "tests_total": (
            len(fail_to_pass)
            + len(pass_to_pass)
        ),
        "floor_tests": len(pass_to_pass),
        "tests_above_floor": 0,
        "candidate_solved": False,
        "fail_to_pass_result": None,
        "pass_to_pass_result": None,
        "gold_result": None,
        "validation_status": "error",
        "container_cleaned": False,
        "error": None,
    }

    if pull:
        print("Pulling image:", image)
        run_command(
            ["docker", "pull", image],
            check=True,
        )

    run_command(
        [
            "docker",
            "rm",
            "-f",
            container_name,
        ],
        check=False,
        capture_output=True,
    )

    create_command = [
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
        "512",
        "--memory",
        "4g",
        "--cpus",
        "4",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g",
        image,
        "/bin/bash",
        "-lc",
        "sleep infinity",
    ]

    try:
        print("Creating container:", container_name)

        run_command(
            create_command,
            check=True,
        )

        run_command(
            [
                "docker",
                "start",
                container_name,
            ],
            check=True,
        )

        commit_result = docker_exec(
            container_name,
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            check=True,
        )

        actual_commit = (
            commit_result.stdout or ""
        ).strip()

        result["base_commit_actual"] = actual_commit
        result["base_commit_match"] = (
            actual_commit == base_commit
        )

        print("Expected commit:", base_commit)
        print("Actual commit:  ", actual_commit)

        if not result["base_commit_match"]:
            raise RuntimeError(
                "Docker image Git HEAD does not match "
                "the dataset base_commit."
            )

        stream_private_file(
            container_name=container_name,
            source=test_patch_path,
            destination=".swegym_test.patch",
        )

        print("Applying verifier-only test patch...")

        apply_patch(
            container_name=container_name,
            patch_path=(
                "/testbed/.swegym_test.patch"
            ),
        )

        result["test_patch_applied"] = True

        if mode == "baseline":
            fail_result = run_pytest(
                container_name=container_name,
                tests=fail_to_pass,
                label="FAIL_TO_PASS baseline tests",
                pytest_extra_args=pytest_extra_args,
            )

            pass_result = run_pytest(
                container_name=container_name,
                tests=pass_to_pass,
                label="PASS_TO_PASS baseline tests",
                pytest_extra_args=pytest_extra_args,
            )

            result["fail_to_pass_result"] = (
                fail_result
            )
            result["pass_to_pass_result"] = (
                pass_result
            )

            fail_group_failed = (
                fail_result["exit_code"] != 0
            )
            pass_group_passed = (
                pass_result["exit_code"] == 0
            )

            if (
                fail_group_failed
                and pass_group_passed
            ):
                result["validation_status"] = (
                    "baseline_valid"
                )
            else:
                result["validation_status"] = (
                    "baseline_invalid"
                )

        elif mode == "gold":
            stream_private_file(
                container_name=container_name,
                source=gold_patch_path,
                destination=".swegym_gold.patch",
            )

            print("Applying verifier-only gold patch...")

            apply_patch(
                container_name=container_name,
                patch_path=(
                    "/testbed/.swegym_gold.patch"
                ),
            )

            result["gold_patch_applied"] = True

            gold_result = run_pytest(
                container_name=container_name,
                tests=[
                    *fail_to_pass,
                    *pass_to_pass,
                ],
                label="gold verification tests",
                pytest_extra_args=pytest_extra_args,
            )

            result["gold_result"] = gold_result

            if gold_result["exit_code"] == 0:
                result["validation_status"] = (
                    "gold_valid"
                )
            else:
                result["validation_status"] = (
                    "gold_invalid"
                )

        elif mode == "candidate":
            assert candidate_patch_path is not None

            candidate_is_empty = (
                candidate_patch_path.stat().st_size
                == 0
            )

            result["candidate_patch_empty"] = (
                candidate_is_empty
            )

            if candidate_is_empty:
                print(
                    "Candidate patch is empty; "
                    "verifying the untouched repository."
                )

            else:
                stream_private_file(
                    container_name=container_name,
                    source=candidate_patch_path,
                    destination=(
                        ".swegym_candidate.patch"
                    ),
                )

                print("Applying candidate patch...")

                apply_patch(
                    container_name=container_name,
                    patch_path=(
                        "/testbed/"
                        ".swegym_candidate.patch"
                    ),
                )

                result[
                    "candidate_patch_applied"
                ] = True

            fail_result = run_pytest(
                container_name=container_name,
                tests=fail_to_pass,
                label="FAIL_TO_PASS candidate tests",
                pytest_extra_args=pytest_extra_args,
            )

            pass_result = run_pytest(
                container_name=container_name,
                tests=pass_to_pass,
                label="PASS_TO_PASS candidate tests",
                pytest_extra_args=pytest_extra_args,
            )

            result["fail_to_pass_result"] = (
                fail_result
            )
            result["pass_to_pass_result"] = (
                pass_result
            )

            fail_passed = fail_result[
                "summary_counts"
            ].get("passed", 0)

            pass_passed = pass_result[
                "summary_counts"
            ].get("passed", 0)

            tests_passed = (
                fail_passed
                + pass_passed
            )

            floor_tests = len(pass_to_pass)

            result["tests_passed"] = tests_passed
            result["floor_tests"] = floor_tests
            result["tests_above_floor"] = max(
                tests_passed - floor_tests,
                0,
            )

            candidate_solved = (
                fail_result["exit_code"] == 0
                and pass_result["exit_code"] == 0
            )

            result["candidate_solved"] = (
                candidate_solved
            )

            result["validation_status"] = (
                "candidate_valid"
            )

        else:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "\nVerification error:",
            result["error"],
            file=sys.stderr,
        )

    finally:
        cleanup = run_command(
            [
                "docker",
                "rm",
                "-f",
                container_name,
            ],
            check=False,
            capture_output=True,
        )

        result["container_cleaned"] = (
            cleanup.returncode == 0
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one SWE-Gym Lite pilot instance "
            "using an isolated Docker container."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Pilot verification manifest.",
    )

    parser.add_argument(
        "--instance-id",
        required=True,
        help="SWE-Gym instance ID to verify.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "gold",
            "candidate",
        ],
        required=True,
    )

    parser.add_argument(
        "--candidate-patch",
        type=Path,
        default=None,
        help=(
            "Agent-generated Git patch. Required "
            "when --mode candidate is selected."
        ),
    )

    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull the instance image before verification.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rows = load_manifest(args.manifest)

    row = select_instance(
        rows,
        args.instance_id,
    )

    result = verify_instance(
        row=row,
        mode=args.mode,
        pull=args.pull,
        candidate_patch_path=(
            args.candidate_patch
        ),
    )

    output_path = args.output

    if output_path is None:
        output_path = (
            DEFAULT_RESULTS_DIR
            / f"{args.instance_id}.{args.mode}.json"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nWrote:", output_path)
    print(
        "Validation status:",
        result["validation_status"],
    )
    print(
        "Container cleaned:",
        result["container_cleaned"],
    )

    valid_status = {
        "baseline": "baseline_valid",
        "gold": "gold_valid",
        "candidate": "candidate_valid",
    }[args.mode]

    return int(
        result["validation_status"]
        != valid_status
    )


if __name__ == "__main__":
    raise SystemExit(main())
