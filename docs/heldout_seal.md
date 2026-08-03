# Held-out split — sealed

**Status: SEALED on 2026-08-03 (B4).**

| | |
| --- | --- |
| Sealed content | `benchmark/tasks/heldout/` — 10 tasks |
| Held-out tree SHA | `318fedc8868bcdd46f24dd78793a5c136be54400` |
| Last commit to modify the split | `ad46f8746a1db72cc925dd656fdb1b7a5ed85d47` |
| Machine-readable record | `benchmark/heldout_seal.json` |
| Enforcement | `scripts/check_heldout_seal.py`, run by `make check` |

## What sealing means

From this point, for the remainder of the project:

- **No edits.** No task added, removed, reworded, retuned or re-scoped. No hidden test
  changed. No gold patch regenerated.
- **No inspection.** The split is not read, browsed, or sampled to see what the tasks look
  like, and per-task held-out results are not examined to form hypotheses.
- **No checkpoint selection.** Held-out reward is never used to choose which checkpoint to
  keep, to early-stop, or to decide a run "worked".
- **No hyperparameter tuning.** Not for learning rate, group size, KL coefficient, turn
  budget, prompt wording, or anything else.
- **Measured once, at the end**, and reported whatever it says.

**Checkpoint selection and all tuning use a validation slice carved from `train` only.**
The 26 train tasks are the entire budget for model selection; if a validation slice is
needed, it is taken from them and documented as such.

## Why it is enforced in code

Contaminating a held-out split is silent. Nothing errors, the tasks still validate, the
rewards still look plausible — and every generalisation claim afterwards is a training-set
claim with a held-out label on it. Discipline alone is not a control, because the failure
leaves no trace to notice.

So the seal records the git **tree hash** of the split. A tree hash covers every byte of
every file under `benchmark/tasks/heldout/` and nothing else: it is unaffected by later
commits to documentation or reward code, and it changes the instant a task is edited, added
or deleted. `scripts/check_heldout_seal.py` compares it on every `make check`, and also
fails if the working copy is dirty — because a graded run against uncommitted edits would
not be scoring the sealed split even though the committed tree still matches.

If that check fails, the fix is to restore the split. **Editing
`benchmark/heldout_seal.json` to make the check pass destroys the only evidence that the
held-out numbers were measured on a split nobody had tuned against.**

## Sealed corpus

10 tasks — 4 easy / 4 medium / 2 hard; bug_fix 4, test_update 2, small_feature 2,
config_fix 1, docs_improve 1; 6/10 `real_repo` (60%); headroom 6–9 on every task; all 10
solvable by gold and failable without it; reward floor mean 0.48.

Full composition and per-task numbers: `PROGRESS.md`, entries dated 2026-08-02 and
2026-08-03.
