# thefuzz (string-preprocessing slice)

A reduced extract of [thefuzz](https://github.com/seatgeek/thefuzz) (MIT), containing the
`utils` and `string_processing` modules that normalise a string before it is scored.

The library dropped end-of-life Python 2 in the release this snapshot is taken from. The
ASCII helper is now a single `utils.ascii_only`, and `StringProcessor` no longer carries the
Python 2 string-module shims.

The project's own suite lives in `tests/test_utils.py`.
