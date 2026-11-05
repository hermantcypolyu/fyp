# Phase 5.3 — GPU resource rules

**Status: a rulebook being executed for the first time.** The rules below were written on a
CPU-only Windows machine that could not run any of them (see
[docs/decisions/0001-build-machine-has-no-nvidia-gpu.md](decisions/0001-build-machine-has-no-nvidia-gpu.md)).
Phase 6 runs on the Linux host (2 × RTX A6000), so the commands are now executable and §0's
"never executed" caveat no longer holds for the Phase 6 baseline run — see
[docs/PHASE6_BASELINE_EVALUATION.md](PHASE6_BASELINE_EVALUATION.md) for what was actually
measured. The rules themselves are unchanged; pasting them is still the requirement.

It sets out what a session must do to its GPUs, so that a run is reproducible and so that the
other person using this machine is not affected by it.

The nine rules come from the Phase 5.3 checklist in
[docs/CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md). Each is stated as a rule, given a
command that enforces or checks it, and — where it applies — a line saying how the run manifest
records it (`scripts/run_manifest.py`, Phase 5.2).

---

## 0. Why this is a document and not a checklist of good intentions

The execution host is a two-card machine (2 × RTX A6000, ~48 GB each), of which **one card is
ours**. This project was designed against a 24 GB A5000 budget, and the settings that budget
produced — group size 4, `max_seq_length` 8192, `gpu_memory_utilization` 0.4, concurrency 1 —
are deliberately unchanged (see `configs/training/protocol_1p5b.yaml` §hardware). The larger
card is not a licence to raise them mid-experiment; it means the headroom exists and is
unused, which is a different and safer position than the one this rulebook was written for.

Every failure mode below is silent:

- a job on the wrong device does not error — it runs, slowly, and fights the other user for
  memory until one of you gets an OOM an hour in;
- a vLLM server left up holds VRAM indefinitely and looks like an idle process, not a fault;
- an idle server is also *load-bearing* for the next run, so killing it carelessly breaks the
  thing you wanted to keep;
- and a run that ends without `nvidia-smi` being checked leaves no evidence either way.

None of these produce a failing test. That is the argument for a written rulebook: the
mistakes that matter here are the ones no assertion can catch.

---

## 1. Use GPU 0 only

This project trains and serves on **GPU 0**. GPU 1 is treated as not ours.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv
```

Expect GPU 0 to be the card the run uses and GPU 1 to be the other user's. Before starting a
long job, run the query above and record what GPU 0 looked like at *idle* — that number, not
24576 MiB, is the budget (see [docs/vram_notes.md](vram_notes.md) §2).

> A card reporting high `memory.used` with 0 % utilisation is usually somebody else's resident
> process. Do not "reclaim" it. Ask, or use less memory.

---

## 2. Every command sets `CUDA_VISIBLE_DEVICES=0` explicitly

This is the one rule whose reason is not obvious, so it gets the space.

`CUDA_VISIBLE_DEVICES` **remaps** the visible devices; it does not merely filter them. With
`CUDA_VISIBLE_DEVICES=1`, the card that is physically GPU 1 becomes device `0` *inside the
process*. So:

- a process that says `cuda:0` is not necessarily on the physical GPU 0;
- with the variable unset, `cuda:0` **is** physical GPU 0 — until the day it isn't, because
  somebody else's job took it or the ordering changed;
- and a stack that defaults to "all visible devices" (vLLM and most trainers do) will occupy
  both cards if the variable is absent.

Setting it explicitly on every command makes the process's view of the world the same as the
operator's, so a log line saying `cuda:0` can be trusted.

```bash
# every invocation, including short ones
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_one_task.py ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_optimizer_smoke.py ...
CUDA_VISIBLE_DEVICES=0 uv run vllm serve ...
```

Recorded in the manifest as `gpu.cuda_visible_devices`. It is not inferred from the
environment at run time precisely because a manifest that recorded "whatever was set" would
read as `"0"` whether or not anyone had set it.
See `analysis/tests/test_run_manifest.py::test_no_gpu_is_recorded_as_absent_not_omitted`.

---

## 3. GPU 1 processes belonging to other users are never touched

This is already a "Do not do now" item in
[docs/CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md), and it is worth restating in
operational terms. Never:

- `nvidia-smi --gpu-reset -i 1`, or any other reset/ECC/clock operation on GPU 1;
- `kill` a PID you did not start, even one using 20 GB and 0 % utilisation;
- set `CUDA_VISIBLE_DEVICES` to anything other than `0`;
- run a job without the variable and assume it will "pick the free card".

If GPU 0 is occupied by a process you did not start, the answer is to wait or to ask — not to
move to GPU 1.

---

## 4. vLLM is stopped after rollouts

The server exists to serve rollouts. Once a rollout batch is scored, it is pure overhead:
VRAM that the training step needs, and an idle listener.

```bash
# start, and record both facts the manifest wants
CUDA_VISIBLE_DEVICES=0 nohup uv run vllm serve <model> \
    --port 8000 --gpu-memory-utilization 0.4 \
    > logs/vllm-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
echo $! | tee logs/vllm.pid
```

The `0.4` is not a suggestion: `configs/training/grpo_nemo_rl_a5000.yaml` sets
`gpu_memory_utilization: 0.4`, leaving room for the training step's own allocations on the same
card. A server started without that flag will take what it can and the training step will OOM.

Stop it deliberately:

```bash
kill "$(cat logs/vllm.pid)" && rm -f logs/vllm.pid
```

Then confirm the memory actually came back (§6). A `kill` that returns 0 does not mean the
memory was released — the driver can lag, and a wedged process can hold its allocation after
`SIGTERM`.

---

## 5. GPU is freed after training

Training ends the same way rollouts do: the process exits, and the memory must be visibly
returned before the session is left. This matters more than it looks, because the next thing
anyone runs is usually a smoke test that will fail confusingly if 20 GB are still held.

```bash
# after training completes or is stopped
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If the training PID is still listed, it has not exited; if it has exited and memory is still
held, the GPU is in a state that only a proper process kill (or a reboot you have agreed with
the other user) resolves. Do not proceed to a new run on top of it.

---

## 6. `nvidia-smi` is checked before leaving

Before closing a session — and before walking away from a long run — check both what is
running and what is left over:

```bash
nvidia-smi                                          # cards, memory, utilisation, PIDs
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # who is holding what
```

The leaving condition for this project is: **GPU 0 holds no processes you started, and GPU 1
shows nothing of yours.** A run left overnight is acceptable *only* if it is expected to
finish, and its PID and log path are written down (§8) so somebody else can check on it.

---

## 7. No idle server holds VRAM long-term

A vLLM server is a daemon: it will happily sit for days at a few hundred MiB to tens of GiB,
serving nothing. Two people on one card cannot afford that.

The rule: **a server may outlive the command that started it only while rollouts are pending.**
Between batches, stop it (§4). If a server must stay up across a break, its PID and log path go
in the manifest (§8) and the person is told — not left to discover it from `nvidia-smi`.

---

## 8. Server PID and log path are recorded

So that a forgotten process can be identified and killed by someone who did not start it, and
so that a crashed run can be read afterwards. The manifest already has the slots; they are
written as `null` at declaration time and filled in when the run starts:

```jsonc
"artifacts": {
  "run_dir": "results/raw/<run_id>",
  "trace":   "results/raw/<run_id>/trace.jsonl",
  "log":     "logs/vllm-20261104T071359Z.log",
  "server_pid": 123456
}
```

`scripts/run_manifest.py` populates `log` and `server_pid` on the manifest it writes; they are
`null` in a manifest produced by `--finish` that never saw a server. A `null` here is a
statement that no server was involved, which is the useful distinction — not an omission.

---

## 9. Reasonable timeouts are set

Every long command gets a timeout, so that a hung process dies by itself rather than holding
the GPU until someone notices. Hung rollouts and hung training steps look identical from
outside: a PID with high memory and zero utilisation.

The episode-level timeout is enforced in the environment (`timeout` gate, so a hung episode
ends as a *scored row*, not a traceback — see
[docs/RUN_ON_A5000.md](RUN_ON_A5000.md) and `envs/coding_env/`). That covers a hung episode,
not a hung driver or a hung vLLM.

At the shell level:

```bash
timeout 2h  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_optimizer_smoke.py ...
timeout 30m CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_one_task.py ...
```

A timeout that fires is a **failure**, and the manifest records it as one:

```bash
uv run python scripts/run_manifest.py --finish failed "timeout 2h exceeded" -o results/raw/<run_id>
```

`timeout` exits 124, so `$? -eq 124` distinguishes "the guard fired" from "the run failed on its
own" — worth checking before writing a failure reason, since the two imply different fixes.

---

## Summary

| # | Rule | Enforced by | Manifest field |
| --- | --- | --- | --- |
| 1 | GPU 0 only | `nvidia-smi` query before starting | `gpu.devices` |
| 2 | `CUDA_VISIBLE_DEVICES=0` on every command | written into each command line | `gpu.cuda_visible_devices` |
| 3 | Never touch GPU 1 or other users' PIDs | never do it | — |
| 4 | vLLM stopped after rollouts | `kill "$(cat logs/vllm.pid)"` | `artifacts.server_pid` |
| 5 | GPU freed after training | `--query-compute-apps` | `timing.finished_at` |
| 6 | `nvidia-smi` checked before leaving | the two queries in §6 | — |
| 7 | No idle server long-term | stop between batches | `artifacts.server_pid` |
| 8 | PID and log path recorded | `echo $! \| tee logs/vllm.pid` | `artifacts.server_pid`, `artifacts.log` |
| 9 | Timeouts set | `timeout <dur>` + episode gate | `outcome.status`, `outcome.failure_reason` |

Three of the nine are checkable by a test or a command; the rest are session discipline. The
manifest is what makes the discipline visible afterwards, which is the most a static check can
do here.
