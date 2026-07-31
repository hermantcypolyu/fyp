# VRAM notes

**Status: incomplete by design.** Every NVIDIA figure below is marked either **MEASURED**
(a command produced it, output saved under `results/raw/`) or **ESTIMATE** (arithmetic, shown
so you can check it). Nothing here was measured on an RTX 4070 Ti, because the machine that
built this repository does not have one. See
[docs/decisions/0001-build-machine-has-no-nvidia-gpu.md](decisions/0001-build-machine-has-no-nvidia-gpu.md).

---

## 1. Build machine — MEASURED

Produced by `uv run python scripts/capture_hardware.py`; raw output saved at
[results/raw/b0_hardware/build_machine.md](../results/raw/b0_hardware/build_machine.md).

| Field | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200) |
| Arch | AMD64 |
| CPU | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel |
| Logical cores | 12 |
| RAM | 31.4 GiB |
| Free disk | 790.0 GiB |
| Python | 3.14.4 |
| NVIDIA GPU | **none detected** (`nvidia-smi` not on PATH) |
| Display adapter | Intel(R) Iris(R) Xe Graphics |
| torch | not installed, no CUDA |

`nvidia-smi` output: **none — the executable does not exist on this machine.**

This is the honest answer to spec §11.2 ("record `nvidia-smi` output"). Everything in this
repository is CPU-only and was built and tested without touching CUDA.

## 2. Target machine — NOT MEASURED

`AGENT_BUILD_SPEC.md` §2 states the project machine is an **RTX 4070 Ti, 12 GB GDDR6X**.
Fill this in with one command on that box:

```bash
uv run python scripts/capture_hardware.py --out docs/_hw_target.md
# then paste the table here and delete this block
```

| Field | Value |
| --- | --- |
| GPU 0 | `[NOT MEASURED]` |
| VRAM total | `[NOT MEASURED]` — spec says 12 GB |
| VRAM free at idle | `[NOT MEASURED]` ← **this is the number that actually matters** |
| Driver version | `[NOT MEASURED]` |
| CUDA version | `[NOT MEASURED]` |
| Compute capability | `[NOT MEASURED]` — Ada Lovelace should report 8.9 |

> **Why "free at idle" matters more than "total".** On a desktop running a display, the
> compositor, browser and IDE typically hold 0.7–1.5 GiB. A 12 GB card in practice offers
> roughly 10.5–11.3 GiB to a training job. Budget against the *measured free* number, never
> against 12.

## 3. Memory budget — ESTIMATE, never executed

Nothing below has been run. It exists so the training phase starts from arithmetic rather
than from trial and error. **Re-derive it from the real `config.json` before trusting it:**

```bash
python - <<'PY'
import json, urllib.request
url = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/config.json"
print(json.dumps(json.load(urllib.request.urlopen(url)), indent=2))
PY
```

Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, nominally 1.5B parameters.
Let $P = 1.54 \times 10^9$ (**[UNVERIFIED]** — confirm against the model card / `config.json`).

### 3.1 Static cost per training strategy

Standard mixed-precision AdamW holds four things: bf16 weights ($2P$), bf16 gradients ($2P$),
fp32 master weights ($4P$), and Adam's two fp32 moments ($8P$) — i.e. $16P$ bytes.

| Strategy | Formula | Bytes | GiB | Fits in ~11 GiB? |
| --- | --- | --- | --- | --- |
| Full fine-tune, AdamW | $16P$ | 24.6 GB | 22.9 | ❌ no |
| Full fine-tune, 8-bit Adam | $2P + 2P + 4P + 2P$ | 15.4 GB | 14.3 | ❌ no |
| **LoRA (r=16), bf16 base frozen** | $2P + 16P_{\text{lora}}$ | ≈3.3 GB | **≈3.1** | ✅ yes |
| QLoRA (nf4 base) | $0.5P + 16P_{\text{lora}}$ | ≈1.0 GB | ≈0.9 | ✅ yes |

with $P_{\text{lora}} \approx 0.01P$ for r=16 on attention + MLP projections.

**Conclusion: full fine-tuning is arithmetically impossible on 12 GB. LoRA or QLoRA is not a
preference, it is a constraint.** This is recorded as a first-class decision in
`docs/TRAINING_READINESS.md` §4 rather than discovered at run time.

### 3.2 The part people forget: GRPO needs rollouts

GRPO samples $G$ completions per prompt, so the *generation* KV cache — not the weights —
is usually what triggers OOM.

Per token, KV cache bytes $= 2 \times L \times H_{kv} \times d_{head} \times \text{dtype}$.
For Qwen2.5-1.5B (**[UNVERIFIED]** — $L=28$ layers, $H_{kv}=2$ GQA KV heads, $d_{head}=128$):

$$2 \times 28 \times 2 \times 128 \times 2\ \text{bytes} = 28\ \text{KiB per token}$$

| Context | $G$ | KV cache | Note |
| --- | --- | --- | --- |
| 4 096 | 4 | 0.44 GiB | comfortable |
| 8 192 | 8 | 1.75 GiB | the likely default |
| 16 384 | 8 | 3.50 GiB | tight once activations are added |
| 16 384 | 16 | 7.00 GiB | ❌ will not fit alongside anything else |

A multi-turn agent trace (6 tools × file reads) reaches 8k tokens easily, which is exactly why
§5.5 mandates truncating every tool output to a strict budget — that rule is a VRAM decision as
much as a prompt-quality one.

### 3.3 Reference policy

Vanilla GRPO keeps a frozen reference model for the KL term: another $2P$ = 2.9 GiB.
**With LoRA you get it for free** by disabling the adapters on the same base weights. Any
config in `configs/training/` that uses LoRA must therefore also use adapter-disabling for the
reference pass, or it silently wastes ~3 GiB.

## 4. Levers, cheapest first

Apply in this order when you hit OOM. Each is a one-line config change.

| # | Lever | Approx. saving | Cost to you |
| --- | --- | --- | --- |
| 1 | Reference policy via adapter-disable | ~2.9 GiB | none (LoRA only) |
| 2 | Shorten max context 16k → 8k | ~1.8 GiB | truncates long traces |
| 3 | Reduce $G$ 16 → 8 | ~3.5 GiB | noisier advantage estimates |
| 4 | Gradient checkpointing | 40–60% of activations | ~30% slower |
| 5 | Gradient accumulation (micro-batch 1) | activations ÷ batch | slower |
| 6 | 8-bit optimizer | $6P$ ≈ 8.6 GiB (full FT only) | slight quality risk |
| 7 | LoRA r 32 → 16 → 8 | small | less capacity |
| 8 | QLoRA nf4 base | ~2.2 GiB vs LoRA | quantisation noise, slower |
| 9 | Offload optimizer to CPU | large | much slower; needs 31 GiB RAM — the build box has it |
| 10 | Rent an A100/H100 for the sweep | all of it | costs money — **needs human approval** (§0.2) |

## 5. Open items

- [ ] Run `scripts/capture_hardware.py` on the 4070 Ti and replace §2.
- [ ] Confirm $P$, $L$, $H_{kv}$, $d_{head}$ from the real `config.json`; correct §3 if they differ.
- [ ] Measure idle VRAM with the display attached — the real ceiling.
- [ ] After the first (human-approved) smoke run, replace §3 estimates with `torch.cuda.max_memory_allocated()` readings.
