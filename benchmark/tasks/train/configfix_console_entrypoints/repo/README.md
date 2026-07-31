# logtidy

Filter and summarise plain-text log files. No dependencies.

## Log format

    2024-03-11T08:15:04 WARNING disk usage at 91%

Anything that does not match is skipped rather than treated as an error.

## Commands

`logtidy` ships exactly two console commands, and both are declared in
`[project.scripts]` in `pyproject.toml`:

| command | implemented by | entry function |
| --- | --- | --- |
| `logtidy` | `logtidy/cli.py` | `cli(argv=None) -> int` |
| `logtidy-summary` | `logtidy/summarise.py` | `main(argv=None) -> int` |

Both entry functions take a single optional `argv` list and return a process exit code,
so they can be called directly as well as through the generated wrappers.

    $ logtidy app.log --min-level WARNING
    $ logtidy-summary app.log --total

## Layout

Flat layout - the importable package is the `logtidy/` directory beside `pyproject.toml`.

    logtidy/
      __init__.py     re-exports the library API
      parser.py       one log line -> Record
      report.py       counting and rendering
      cli.py          the `logtidy` command
      summarise.py    the `logtidy-summary` command

## History

Two renames landed in 1.2.0:

* `logtidy/console.py` became `logtidy/cli.py`, and its `main` became `cli`, so that each
  command's entry function is named after the command it serves.
* `logtidy/stats.py` was deleted; its counting was folded into `logtidy/summarise.py`.

The Python is settled. Nothing under `logtidy/` needs to change.
