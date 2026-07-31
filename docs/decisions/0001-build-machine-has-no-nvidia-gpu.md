# 0001 — Build machine is not the target GPU machine

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestone:** B0

## Context

`AGENT_BUILD_SPEC.md` §2 states the local card is an **RTX 4070 Ti, 12 GB GDDR6X**, and §11.2
instructs us to record `nvidia-smi` output in `docs/vram_notes.md`.

The machine this repository was actually built on has **no NVIDIA GPU**:

```
Get-CimInstance Win32_VideoController | Select Name,AdapterRAM,DriverVersion
Name          : Intel(R) Iris(R) Xe Graphics
AdapterRAM    : 2147479552
DriverVersion : 32.0.101.7080
```

`nvidia-smi` is not installed and not on `PATH` (`C:\Windows\System32\nvidia-smi.exe` absent).

## Options considered

1. **Write the expected 4070 Ti `nvidia-smi` table from the spec.**
   Rejected outright. §1.2 forbids fabricating numbers, and a fake `nvidia-smi` dump is exactly
   the kind of unreproducible number that destroys credibility. It would also be the *first*
   number in the repo, setting a terrible precedent.
2. **Skip `docs/vram_notes.md` entirely.**
   Rejected. The file is a B0 deliverable and becomes real evidence for the training-phase config.
3. **Write the file with the real finding, an explicit `[NOT MEASURED]` marker, and a
   copy-pasteable command block for the human to run on the 4070 Ti box.** ✅ Chosen.

## Decision

`docs/vram_notes.md` records what was actually observed on the build machine, marks every
4070 Ti figure as **not measured here**, and carries a ready-to-run capture script
(`scripts/capture_hardware.py`) whose output the human pastes back in one step.

All VRAM budgeting in `configs/training/grpo_lowvram.yaml` and `docs/TRAINING_READINESS.md`
is therefore explicitly labelled **estimated, never executed**.

## Consequences

- ✅ Zero fabricated hardware numbers anywhere in the repo.
- ✅ The gap is visible rather than hidden, so the human cannot accidentally trust an estimate.
- ⚠️ `docs/vram_notes.md` stays incomplete until someone runs the capture script on the real card.
  This is tracked as an open item in `PROGRESS.md` and `docs/TRAINING_READINESS.md` §6.
- ⚠️ No CUDA-dependent code path in this repo has ever been executed. Everything shipped is
  CPU-only by construction, which is consistent with §2 ("~70% of the engineering has no
  compute dependency").
