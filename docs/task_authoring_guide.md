# Writing a task

This guide is meant to be sufficient on its own. If you follow it end to end you will
produce a task that `make validate && make verify-gold` accepts, without needing to read
any of the tooling.

Read [AGENT_BUILD_SPEC.md](../AGENT_BUILD_SPEC.md) §5 first if you want the rationale;
this document is the procedure.

---

## 1. What a task is

A task is a small, self-contained Python project that is **broken in one specific way**,
plus a **hidden test suite** that measures how thoroughly it has been fixed, plus a
**reference solution** that proves the fix is achievable.

```
benchmark/tasks/<split>/<task_id>/
├── task.yaml            the contract: instruction, constraints, provenance
├── repo/                what the agent sees and edits
├── tests/               HIDDEN. never written into the agent's sandbox
├── smoke/               optional, VISIBLE. what run_tests() shows the agent
└── solution/
    ├── fixed/           your corrected copies of the files under repo/
    └── gold.patch       generated from solution/fixed/ - never hand-written
```

Two properties matter more than anything else, and both are checked mechanically:

* **failable** — the untouched `repo/` must fail at least one hidden test. Otherwise the
  task rewards doing nothing.
* **solvable** — `gold.patch` must make every hidden test pass. Otherwise the reward
  ceiling is below 1 and every number computed from the task is wrong.

The fraction in between is the **reward floor**: what an agent scores for submitting
without touching anything. `make verify-gold` prints it for every task.

```
task                                   cat            diff    untouched    +gold        floor   ok
bugfix_offbyone_pagination             bug_fix        easy    6/11 passed  11/11 passed 0.55    yes
configfix_textkit_packaging            config_fix     hard    0/9 passed   9/9 passed   0.00    yes
testupdate_money_api_v2                test_update    medium  4/10 passed  10/10 passed 0.40    yes
```

---

## 2. The two test directories

Every task has two directories called `tests`. Only one of them is secret. Getting this
straight is the single most load-bearing idea in the format, and it is what makes the
`test_update` category possible at all.

| | `<task>/tests/` | `<task>/repo/tests/` |
| --- | --- | --- |
| what it is | the **hidden grading suite** | the **project's own suite**, ordinary source |
| who wrote it | you, the task author | the fictional (or real) project |
| decides the reward | yes - `reward_v1` is its pass fraction | no |
| agent can read it | **never** | yes |
| agent can write it | **never** | only if `allowed_write_globs` says so |
| exists in every task | yes, mandatory | only where the task needs one |

### What the agent's world actually looks like

`materialise()` builds a disposable copy of the task, and the agent works inside it with
`repo/` as its working directory:

```
work/
├── repo/          <- the agent's cwd and PYTHONPATH. Everything it can see.
│   ├── src/
│   └── tests/     <- the project's own suite, if the task has one. Visible. Editable.
├── smoke/         <- what run_tests() executes for the agent
└── tests/         <- THE HIDDEN SUITE - and only when grading
```

When the sandbox is built for the agent it is built with `include_tests=False`, so that last
directory is **never written to disk inside the agent's world**. Isolation does not depend on
file permissions, on the agent being well behaved, or on a glob being spelled correctly: the
bytes are simply not there. The hidden suite is materialised only in the grading copy, which
the agent never touches.

### Four independent mechanisms, deliberately

Any one of these would mostly work. They are stacked because "mostly" is not good enough for
the quantity that defines the reward.

1. **Absence.** `materialise(..., include_tests=False)` - the agent's copy has no hidden
   suite in it. This is the real guarantee; the rest are belt and braces.
2. **The deny-list.** `tests/**` and `solution/**` are mandatory in every
   `forbidden_write_globs`, and B4 turns a match into the `tests_modified` hard-zero gate.
   The matcher is `benchmark/tools/globs.py`, not `fnmatch`, because `fnmatch` treats `/` as
   an ordinary character and would quietly widen every task's permissions.
   `benchmark/tests/test_globs.py` pins down that `tests/**` matches
   `tests/test_money_suite.py` and does **not** match `repo/tests/test_money.py`.
3. **The grading path.** `run_tests()` is given the hidden files by absolute path and runs
   under a generated pytest ini, so no `conftest.py`, `pytest.ini` or `pyproject.toml` inside
   `repo/` can influence collection, add plugins, or monkeypatch its way into the run.
4. **The instruction check.** `check_leakage` refuses an instruction that names a hidden test
   file (`LEAK_TESTNAME`, an error) and warns about a `tests/` path that does not resolve to
   something real under `repo/` (`LEAK_TESTDIR`). It resolves the path rather than matching
   the word, so an honest `test_update` instruction is silent - see
   `benchmark/tests/test_leakage.py`.

### What this means when you write a `test_update` task

The agent is asked to edit `repo/tests/`, which is why that path - and only that path - goes
in `allowed_write_globs`:

```yaml
constraints:
  allowed_write_globs:
    - "repo/tests/**"        # the project's own suite: the thing to be updated
  forbidden_write_globs:
    - "tests/**"             # the hidden grading suite: mandatory, always
    - "solution/**"
    - "repo/src/**"          # the library is CORRECT; updating the tests is the task
```

Write paths in the instruction exactly as the agent would type them - relative to `repo/`,
so `tests/test_money.py`, not `repo/tests/test_money.py`. That is what the agent's `cwd` is,
and it is what `check_leakage` resolves against.

The hidden suite for such a task cannot simply re-run the project's suite and report pass or
fail; that would collapse the reward to one bit and make deleting every test the shortest
winning move. `testupdate_money_api_v2` shows the shape that works: run the project's suite in
a subprocess, count the surviving tests from the JUnit XML, check the updated source still
mentions the renamed entry points, and **separately re-assert the library's own contract** so
that "fixing" the tests by reverting the library fails.

### The mistake to avoid

Do not give a hidden test file the same name as a visible one. `tests/test_money.py` and
`repo/tests/test_money.py` will both work, and then someone reading a traceback will lose an
hour. Name hidden suites for what they verify - `test_money_suite.py`, `test_packaging.py` -
and keep them distinct from anything under `repo/`.

---

## 3. The procedure

### Step 0 — pick a slot

Check `benchmark/catalogue.md` (or the corpus warnings at the bottom of `make validate`)
for what the mix still needs. Targets from §5.2:

| category | train | held-out |
| --- | --- | --- |
| `bug_fix` | 10 | 4 |
| `test_update` | 6 | 2 |
| `config_fix` | 5 | 1 |
| `small_feature` | 6 | 2 |
| `docs_improve` | 3 | 1 |

Difficulty should land near 40% easy / 40% medium / 20% hard, and at least half the
corpus should be `source: real_repo`.

Held-out tasks must come from the *same* generator as training tasks — same authors, same
patterns, same size. A held-out split that is systematically harder measures nothing
except that it is harder.

### Step 1 — scaffold

```
make new-task ID=bugfix_retry_backoff CATEGORY=bug_fix
# or, for full control:
uv run python benchmark/tools/new_task.py bugfix_retry_backoff \
    --category bug_fix --difficulty medium --split train
```

The id is `snake_case`, starts with the category-ish prefix by convention
(`bugfix_`, `testupdate_`, `configfix_`, `feature_`, `docs_`), and is also the directory
name. The generated `task.yaml` already carries the mandatory deny-list and sensible
per-category budgets.

### Step 2 — write the broken repo

Under `repo/`:

* Keep it **under 500 lines of Python, ideally under 300**. The agent has to read it
  inside a small context window. This is checked.
* Make it look like real code someone would maintain: docstrings, a `README.md`, more than
  one module. A single 20-line file with an obvious typo teaches nothing.
* Put in **exactly one conceptual defect**, but let it have several observable
  consequences. That is what gives partial credit somewhere to land.
* No network, no clocks, no randomness, no absolute paths.

### Step 3 — write the hidden suite

Under `tests/`, named `tests/test_*.py` and listed in `verification.test_files`.

* **3 to 40 tests.** `reward_v1` is the pass fraction, so each test is worth `1/n` of the
  entire learning signal. One monolithic all-or-nothing test destroys the gradient this
  project depends on.
* **Grade consequences separately.** For the pagination example: one test for the page
  count, one for reaching the final partial page, one for the empty-collection edge case,
  one for the round-trip property that every item appears exactly once. An agent that
  fixes the rounding but not the empty case should score meaningfully above zero and
  meaningfully below one.
* **Deterministic.** No `time`, no `random` without a fixed seed, no network, no
  dependence on dict ordering or filesystem ordering. `run_tests` already pins
  `PYTHONHASHSEED=0`, but that is a backstop, not a licence.
* **Only the standard library and pytest.** The sandbox has nothing else installed.
* Tests run with cwd and `PYTHONPATH` both set to the materialised `repo/`, so
  `from src.thing import x` works and `Path.cwd()` is the repo root.

### Step 4 — write the instruction

This is the part most worth slowing down for.

* Describe the **symptom**, the way a bug report or a colleague would. Never the fix.
* **Never name the hidden tests**, their file, their count, or their assertions. The
  validator rejects a task whose instruction mentions a hidden test file name.
* It must be **sufficient**: a competent developer with access only to `repo/` should be
  able to produce a correct fix from the instruction alone. If the only way to pass is to
  guess which of two reasonable behaviours you happened to test, the instruction is
  underspecified and the task is measuring luck.
* Say what must **not** change — public signatures, file layout — if the tests depend on
  it.
* Three to six sentences. Prose, not bullet points.

### Step 5 — write the reference solution

Copy each file you intend to change into `solution/fixed/`, **mirroring its path under
`repo/`**, with the fix applied:

| you are changing | put the fixed copy at |
| --- | --- |
| `repo/src/pagination.py` | `solution/fixed/src/pagination.py` |
| `repo/pyproject.toml` | `solution/fixed/pyproject.toml` |
| `repo/tests/test_money.py` | `solution/fixed/tests/test_money.py` |

Then:

```
make gold-patches
```

which diffs `solution/fixed/` against `repo/` and writes `solution/gold.patch`.

**Never hand-write the patch.** The generator normalises line endings; a diff produced by
a Windows editor is the single most common reason a hand-rolled patch fails to apply.

The solution should be the *minimal* honest fix. If your gold patch also reformats the
file, `expected_files_changed` and the write-glob checks start lying.

### Step 6 — optional smoke tests

`smoke/` is what the agent's `run_tests()` tool executes. Its purpose is to let the agent
confirm the code still imports and still broadly works — **not** to let it read its own
grade.

* Keep them **weak** and keep them **passing in both the broken and the fixed state**.
* A smoke test that is byte-identical to a hidden test is rejected by the validator, but
  the rule is about intent, not bytes: a smoke test that merely paraphrases the hidden
  suite hands over the reward signal just as effectively.
* If you cannot write a useful weak test, delete the directory. It is optional.

### Step 7 — fill in the metadata and validate

```
make validate
```

The validator measures `repo.loc` for you and tells you the number to write down; same
for most other numeric fields. Iterate until it is green, then:

```
make verify-gold
```

Check the reward floor it prints. A floor above `0.75` means the task barely
discriminates — most of the reward is free — and is usually a sign the hidden suite spends
too many tests on behaviour that was never broken.

Finally:

```
make catalogue
```

---

## 4. `task.yaml` field by field

```yaml
id: bugfix_offbyone_pagination      # snake_case; must equal the directory name
category: bug_fix                   # bug_fix|test_update|config_fix|small_feature|docs_improve
difficulty: easy                    # easy|medium|hard - see the rubric below
split: train                        # train|heldout; must equal the parent directory

instruction: >
  Symptom-level prose. Never names the hidden tests.

repo:
  root: repo/
  loc: 96                           # measured; `make validate` tells you the number
  entrypoint_hint: src/pagination.py  # a real path under repo/; omit for hard tasks

verification:
  runner: pytest
  test_files: [tests/test_pagination.py]   # paths relative to the TASK dir, not repo/
  timeout_s: 60                     # generous but finite; raise it if you nest a pytest run
  n_tests: 11                       # must equal what pytest actually collects

constraints:
  allowed_write_globs:              # everything the agent may legitimately edit
    - "repo/src/**"
  forbidden_write_globs:            # `tests/**` and `solution/**` are mandatory
    - "tests/**"
    - "solution/**"
    - "**/conftest.py"
    - "task.yaml"
    - "repo/smoke/**"
  max_turns: 20
  max_tool_calls: 40

provenance:
  source: textbook_pattern          # real_repo|textbook_pattern|synthetic
  license: MIT
  notes: >
    Where this came from and why it is representative.

metadata:
  expected_files_changed: 1
  created: 2026-07-31               # quoting optional; the loader coerces YAML dates
  tags: [off-by-one, integer-division]
```

### Write globs are a security boundary

`forbidden_write_globs` is not advice. In B4 a write that matches it triggers the
`tests_modified` hard-zero gate: the rollout scores exactly `0.0` regardless of how many
tests pass. Two consequences:

* `tests/**` and `solution/**` are mandatory in every task.
* Keep `allowed_write_globs` **as narrow as the task honestly permits**. A
  `config_fix` task that allows `repo/**` lets the agent "fix" the config by rewriting the
  library, which is not the skill being measured.

Remember that `tests/**` and `repo/tests/**` are different places, and that a `test_update`
task needs the second one allowed while the first stays forbidden. Section 2 has the full
picture.

### Difficulty rubric

| | typical shape |
| --- | --- |
| `easy` | one defect, one file, symptom names the module, `entrypoint_hint` given |
| `medium` | one defect with several consequences, or two files; some search required |
| `hard` | the faulty file is not the one named in the symptom, or several coupled files must change; `entrypoint_hint` usually omitted |

### Provenance

| `source` | meaning | extra requirements |
| --- | --- | --- |
| `real_repo` | a real commit from a real project was inverted | `origin: <url> @ <commit-sha>` and `notes` saying which commit and what was inverted |
| `textbook_pattern` | a defect class that genuinely recurs, written fresh here | `notes` explaining why it is representative |
| `synthetic` | invented to fill a gap in the mix | `notes` being honest that it is artificial |

The corpus targets **≥50% `real_repo`**, because invented bugs skew toward the tidy,
textbook errors that a model has already seen a thousand times in training data.

To mine one: clone a small permissively-licensed Python project, find a commit whose
message starts with `fix:` and which touched both source and a test file, **keep the
test, invert the source change**. Record the URL and the exact SHA. Check the licence
permits redistribution — the code is going into this repository.

---

## 5. Category notes

**`bug_fix`** — the reference shape. See `bugfix_offbyone_pagination`.

**`test_update`** — the production code is *correct* and deliberately changed; the repo's
own suite is stale and no longer compiles. The agent updates the tests, not the library.
Two things must be graded together, because they pull against each other: the suite must
pass, *and* it must still be worth having. Otherwise the shortest winning move is to
delete every test. See `testupdate_money_api_v2`, which counts the surviving tests from
JUnit XML, greps the updated source for the renamed entry points, and separately
re-asserts the library's contract so that "fixing" the tests by reverting the library
fails. Put `repo/src/**` in `forbidden_write_globs`. Raise `timeout_s`: the hidden suite
runs pytest in a subprocess.

**`config_fix`** — `pyproject.toml`, `setup.cfg`, imports, `__init__.py` re-exports,
version strings, `requires-python` floors, packaging paths. The Python logic is correct
and off-limits. These tend to be `hard`, because the symptom (`ImportError`) is a long way
from several of the causes. See `configfix_textkit_packaging`.

**`small_feature`** — add a genuinely small capability described precisely enough in the
instruction that a reasonable implementation passes. Beware: this category is the easiest
to underspecify. If the hidden tests assert an exact error message or an exact return
shape, the instruction must say so.

**`docs_improve`** — usually a docstring or `README` that contradicts the code. Grade it on
machine-checkable properties: doctests that actually run, every public function having a
docstring, documented parameter names matching the real signature. Do not grade prose
quality; you cannot do it deterministically.

---

## 6. Checklist

Before you commit a task:

- [ ] `make validate` is green for it
- [ ] `make verify-gold` shows it failable and solvable, with a floor below ~0.75
- [ ] the instruction names no hidden test, and is sufficient without them
- [ ] `repo/` is under 500 lines and reads like real code
- [ ] hidden tests are deterministic and stdlib-only
- [ ] smoke tests pass in both states and reveal nothing
- [ ] `allowed_write_globs` is as narrow as the task permits
- [ ] provenance is honest, and any `real_repo` origin carries a real commit SHA
- [ ] `make catalogue` re-run

Held-out tasks additionally: record the sealing commit SHA in `PROGRESS.md`, and do not
look at held-out results while iterating on the environment.
