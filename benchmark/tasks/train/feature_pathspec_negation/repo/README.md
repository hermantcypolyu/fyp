# siftpath

Gitignore-style matching over plain path strings. Nothing here touches the filesystem,
so the same spec can be applied to a working tree, a tarball listing or a git tree.

```python
from src import PathSpec, filter_paths

spec = PathSpec(open(".gitignore").read().splitlines())
kept = filter_paths(spec, ["src/main.py", "build/app.o", "debug.log"])
```

## What the matcher already does

A path is a repo-relative POSIX string. **A trailing `/` on the path means it is a
directory**; without one, the last component is a file. Leading slashes are stripped.
An empty path raises `ValueError`.

Patterns are read in order. Blank lines and lines whose first non-space character is `#`
are skipped, and every other line is stripped of surrounding whitespace.

* `*` matches any run of characters **within one segment**, `?` matches exactly one
  character within one segment, and neither ever crosses a `/`.
* `**` as a whole segment matches any number of directories including none, so
  `docs/**/notes.md` matches `docs/notes.md` and `docs/a/b/notes.md`. As the final
  segment, `a/**` matches everything inside `a`.
* A pattern containing no `/` (after any trailing one is removed) matches a **basename
  at any depth**: `*.log` matches `var/log/debug.log`.
* A pattern with a `/` anywhere in it, or a leading `/`, is **anchored to the root**:
  `dist/out` matches `dist/out` and not `a/dist/out`.
* A pattern ending in `/` matches **directories only**, so `node_modules/` matches the
  directory and everything beneath it, but not a *file* called `node_modules`.
* Matching walks the path from its outermost directory inwards, which is how excluding a
  directory excludes its contents without listing them.

`spec.rules` exposes the compiled rules in order. `filter_paths` and `partition` apply a
spec to a sequence of paths.

## Wanted: negation, and a way to see which pattern won

Today every rule is an exclusion and the first one to match ends the search, so a spec
cannot say "all of these except that one" and nobody can tell *why* a file went missing.
We want two things, following these rules exactly:

* A leading `!` marks a rule as a **re-inclusion**. A leading `\!` is an escape: the
  rule is an ordinary exclusion whose first literal character is `!`. A backslash
  anywhere else is just a backslash.
* **The last matching rule wins**, not the first. Order in the list is significance:
  a `!` after an exclusion re-includes, and an exclusion after a `!` excludes again.
* The decision is made per candidate as the path is walked from the outside in. At each
  candidate, the *last* rule in the list that matches it decides that candidate; the
  final answer is the decision made at the deepest candidate any rule matched. If no
  rule matched any candidate, the path is not ignored.
* **A directory that ends up ignored stops the walk.** Nothing below it is examined, so
  a `!` on a file cannot rescue it from an excluded parent directory - that whole subtree
  is ignored, and the directory's own rule is what decided it. A directory that ends up
  *re-included* does not stop the walk: the rules below it still apply.
* `explain(path)` returns the deciding rule's pattern **exactly as it was written**,
  including any leading `!` or `\`, and any trailing `/`. It returns `None` when no rule
  matched any candidate. It answers for exclusions and re-inclusions alike - it reports
  which pattern decided, not whether the answer was yes.
* `is_ignored` keeps its signature and its return type, and every behaviour listed above
  under "already does" must survive unchanged.
