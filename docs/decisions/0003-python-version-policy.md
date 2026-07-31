# 0003 — Python 3.12 is the floor; no interpreter version is pinned in the repo

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestone:** B0

## Context

NeMo Gym requires **Python 3.12+** (its README's Requirements table). But the acceptance test
(§0.3) is that a stranger clones this repo and runs `uv sync` successfully within minutes.

Two facts collided during the build:

1. **`ray` — a NeMo Gym dependency — publishes `win_amd64` wheels only up to cp312.** For
   `ray==2.56.1`, PyPI has cp310/cp311/cp312 Windows wheels and nothing for cp313/cp314. So the
   `gym` extra resolves *only* on 3.12 on Windows.
2. **Committing `.python-version` makes `uv` fetch that exact interpreter, and the fetch is
   blocked on some networks.** On this build machine `uv python install 3.12` failed twice:

   ```
   invalid peer certificate: UnknownIssuer          # corporate TLS interception
   connection forcibly closed by the remote host (os error 10054)   # GitHub release downloads blocked
   ```

   A committed `.python-version` would turn "clone and sync" into "clone, fail, debug someone
   else's proxy" — precisely the failure mode §0.3 exists to prevent.

## Options considered

1. **Commit `.python-version = 3.12`.** Reproducible, but hard-fails behind a restrictive proxy
   even when a perfectly good 3.12 is already installed.
2. **Commit `.python-version` and document the proxy workaround.** Still fails first, then asks
   the reader to fix it. Bad first impression, and unfixable if they have no admin rights.
3. **Declare `requires-python = ">=3.12"` and pin nothing.** ✅ Chosen. `uv` uses any local
   interpreter that satisfies the constraint and downloads one only if none exists.

## Decision

- `pyproject.toml` declares `requires-python = ">=3.12"`. No `.python-version` is committed.
- The **core** install is verified to work on more than one interpreter: `uv sync --extra dev`
  succeeded here on **3.14.4**, and the vendored NeMo Gym venv runs on **3.12.10**.
- `docs/TRAINING_READINESS.md` states plainly that the `gym` extra wants **exactly 3.12** on
  Windows, and why.
- For anyone who hits the same proxy wall, the escape hatches are recorded rather than
  rediscovered:

  ```powershell
  $env:UV_SYSTEM_CERTS = "1"      # trust the OS cert store (UV_NATIVE_TLS is deprecated)
  # if GitHub release downloads are blocked, fetch CPython from the NuGet CDN instead:
  curl.exe -sS -L --ssl-no-revoke -o py312.zip `
    https://globalcdn.nuget.org/packages/python.3.12.10.nupkg
  Expand-Archive py312.zip -DestinationPath py312    # interpreter at py312\tools\python.exe
  ```

## Consequences

- ✅ `uv sync` works on any machine with any Python ≥3.12 already present, offline-friendly.
- ✅ Validated on two interpreter versions, which is stronger evidence than a pin.
- ⚠️ Two contributors can run different patch versions. Acceptable: the core depends on
  `pyyaml`/`jsonschema`/`numpy` only, and `uv.lock` still pins every *package*.
- ⚠️ Someone on 3.13/3.14 who runs `uv sync --extra gym` on Windows will get a resolution error.
  Mitigated by the note in `README.md` and `docs/nemo_gym_notes.md` §7.
