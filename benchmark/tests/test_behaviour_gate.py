"""Unit tests for the gate in front of the behaviour check.

`check_behaviour` is the only thing that proves a task is solvable by its gold patch and
failable without it, and it is also what measures `verification.floor_tests`. It is
expensive, so it runs only when the cheap static checks have not already invalidated the
task.

That gate was originally "no findings at all", which quietly conflated advice with
breakage: a task carrying nothing worse than a `LOC_BIG` *warning* had its solvable and
failable checks skipped, and the validator then printed `ok` for a task it had never run.
Two train tasks sat like that, and their floors read 0 when the measured value was 5.

The distinction is what these tests pin down.
"""

from __future__ import annotations

from benchmark.schema.validate_tasks import Report, blocking_errors


def test_a_clean_report_does_not_block():
    assert blocking_errors(Report(), "probe") == []


def test_a_warning_alone_does_not_block():
    report = Report()
    report.warn("LOC_BIG", "probe", "repo/ is 346 LOC")
    assert blocking_errors(report, "probe") == []


def test_an_error_blocks():
    report = Report()
    report.error("NO_GOLD", "probe", "missing solution/gold.patch")
    assert [f.code for f in blocking_errors(report, "probe")] == ["NO_GOLD"]


def test_an_error_still_blocks_when_mixed_with_warnings():
    report = Report()
    report.warn("LOC_BIG", "probe", "repo/ is 346 LOC")
    report.error("NO_GOLD", "probe", "missing solution/gold.patch")
    report.warn("LICENCE", "probe", "unknown licence")
    assert [f.code for f in blocking_errors(report, "probe")] == ["NO_GOLD"]


def test_findings_for_another_task_do_not_block():
    report = Report()
    report.error("NO_GOLD", "other", "missing solution/gold.patch")
    assert blocking_errors(report, "probe") == []
