"""Turning a CSV document into the rows the summarising code expects.

Kept separate from :mod:`src.summarise` so that the statistics can be exercised on
hand-written dictionaries without going through a parser.
"""

from __future__ import annotations

import csv
import io


def read_rows(text: str) -> list[dict[str, str]]:
    """Parse ``text`` as CSV with a header line, one dict per data row.

    >>> read_rows("city,temp\\nleeds,12\\n")
    [{'city': 'leeds', 'temp': '12'}]
    """
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]
