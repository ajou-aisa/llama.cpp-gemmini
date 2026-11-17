# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
# Run: uv run --with pytest python -m pytest scripts/potal-timeline/test_cpu_parse.py
from dataclasses import replace

import pytest

from cpu_parse import Sample, Slot, TraceError, paired


def sample(event: int) -> Sample:
    return Sample((Slot(0), 15, 1, 2, 3, 4), event, event, event * 100,
                  event * 50, 0, 3, event * 40, event * 80, True, False, True, "riscv_hart_csr")


def test_hart_counters_have_their_own_units() -> None:
    row = paired([sample(1), sample(2)])[0]
    assert row[6:13] == (100, 50, "", 40, 80, 2.0, "valid_hart_counts")


@pytest.mark.parametrize("end", [replace(sample(2), cpu=4), replace(sample(2), counters_valid=False),
                                replace(sample(2), cycles=1), replace(sample(2), source="unavailable")])
def test_untrusted_counters_are_missing_not_zero(end: Sample) -> None:
    row = paired([sample(1), end])[0]
    assert row[9:12] == ("", "", "")
    assert row[6] == 100


@pytest.mark.parametrize("samples", [[sample(1)], [sample(2)], [sample(1), sample(1)],
                                    [sample(1), replace(sample(2), thread_ns=1)]])
def test_corrupt_intervals_are_rejected(samples: list[Sample]) -> None:
    with pytest.raises(TraceError):
        paired(samples)


def test_process_cpu_denominator_only_at_model_boundaries() -> None:
    begin = replace(sample(1), process_valid=True, process_ns=1000)
    end = replace(sample(2), process_valid=True, process_ns=1300)
    assert paired([begin, end])[0][8] == 300


@pytest.mark.parametrize(("stage", "kind"), [(121, "cpu_descriptor"), (122, "cpu_descriptor"),
    (33, "command_api_mixed"), (35, "fence_api_mixed"), (48, "issuer_window_inclusive")])
def test_issue_counters_retain_ipc_without_inventing_wall_time(stage: int, kind: str) -> None:
    samples = [replace(sample(event), key=(Slot(0), stage, 1, 2, 3, 4), mono_ns=0,
                       thread_ns=0, thread_valid=False) for event in (1, 2)]
    row = paired(samples)[0]
    assert row[6:13] == ("", "", "", 40, 80, 2.0, "valid_hart_counts")
    assert row[-1] == kind


def test_missing_wall_time_outside_command_windows_is_rejected() -> None:
    with pytest.raises(TraceError, match="counter-only"):
        paired([replace(sample(1), mono_ns=0), sample(2)])
