#!/usr/bin/env python
"""Capture the real hardware of whatever machine this runs on, as pasteable Markdown.

AGENT_BUILD_SPEC §1.2 forbids writing a number that no command produced. This script
is how `docs/vram_notes.md` gets its numbers: run it on the RTX 4070 Ti box and paste
the output into the "Measured" section.

    uv run python scripts/capture_hardware.py                 # print Markdown
    uv run python scripts/capture_hardware.py --json          # machine-readable
    uv run python scripts/capture_hardware.py --out docs/_hw.md

It never fails the build: if `nvidia-smi` or `torch` is missing it records that fact
rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NOT_FOUND = "[NOT AVAILABLE ON THIS MACHINE]"

NVIDIA_SMI_FIELDS = [
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "driver_version",
    "compute_cap",
]


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def probe_nvidia() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"present": False, "reason": "nvidia-smi not on PATH"}
    csv = _run(["nvidia-smi", f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}", "--format=csv,noheader"])
    if csv is None:
        return {"present": False, "reason": "nvidia-smi found but returned a non-zero exit code"}
    gpus = []
    for line in csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(NVIDIA_SMI_FIELDS):
            gpus.append(dict(zip(NVIDIA_SMI_FIELDS, parts, strict=True)))
    return {
        "present": bool(gpus),
        "gpus": gpus,
        "full_output": _run(["nvidia-smi"]) or "",
    }


def probe_non_nvidia_gpu() -> list[str]:
    """Best-effort display-adapter list, so 'no NVIDIA GPU' is backed by what IS there."""
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ]
        )
    elif sys.platform == "darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
    else:
        out = _run(["lspci"])
        if out:
            out = "\n".join(line for line in out.splitlines() if "VGA" in line or "3D controller" in line)
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def probe_torch() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"installed": False, "note": "torch is not a dependency of this repo (no training here)"}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if info["cuda_available"]:
        info["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def probe_memory_gib() -> float | str:
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        return round(int(out) / 1024**3, 1) if out and out.isdigit() else NOT_FOUND
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return NOT_FOUND


def collect() -> dict[str, Any]:
    total, used, free = shutil.disk_usage(Path.cwd().anchor or "/")
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname_hash": f"{abs(hash(platform.node())) % 10**8:08d}",  # stable, not identifying
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": platform.processor() or NOT_FOUND,
        "cpu_logical_cores": os.cpu_count(),
        "ram_gib": probe_memory_gib(),
        "disk_free_gib": round(free / 1024**3, 1),
        "python": sys.version.split()[0],
        "nvidia": probe_nvidia(),
        "display_adapters": probe_non_nvidia_gpu(),
        "torch": probe_torch(),
    }


def to_markdown(d: dict[str, Any]) -> str:
    lines = [
        f"<!-- generated by scripts/capture_hardware.py at {d['captured_at']} -->",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| OS | {d['os']} |",
        f"| Arch | {d['machine']} |",
        f"| CPU | {d['cpu']} |",
        f"| Logical cores | {d['cpu_logical_cores']} |",
        f"| RAM | {d['ram_gib']} GiB |",
        f"| Free disk | {d['disk_free_gib']} GiB |",
        f"| Python | {d['python']} |",
    ]
    nv = d["nvidia"]
    if nv.get("present"):
        for i, gpu in enumerate(nv["gpus"]):
            lines += [
                f"| GPU {i} | {gpu['name']} |",
                f"| GPU {i} VRAM total | {gpu['memory.total']} |",
                f"| GPU {i} VRAM free | {gpu['memory.free']} |",
                f"| GPU {i} driver | {gpu['driver_version']} |",
                f"| GPU {i} compute capability | {gpu['compute_cap']} |",
            ]
    else:
        lines.append(f"| NVIDIA GPU | **none detected** ({nv.get('reason', 'unknown')}) |")
        for a in d["display_adapters"]:
            lines.append(f"| Display adapter | {a} |")

    t = d["torch"]
    lines.append(
        f"| torch | {t.get('version', 'not installed')}"
        f"{', CUDA ' + str(t.get('cuda_version')) if t.get('cuda_available') else ', no CUDA'} |"
    )

    if nv.get("full_output"):
        lines += ["", "```text", nv["full_output"], "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--out", default=None, help="Also write the output to this path.")
    args = parser.parse_args()

    data = collect()
    text = json.dumps(data, indent=2) + "\n" if args.json else to_markdown(data)
    sys.stdout.write(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
