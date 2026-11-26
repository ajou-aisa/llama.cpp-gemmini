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


@pytest.mark.parametrize("counters_valid", [True, False])
def test_csr_only_spans_leave_wall_and_cpu_times_missing(counters_valid: bool) -> None:
    samples = [replace(sample(event), mono_ns=0, thread_valid=False, process_valid=False,
                       counters_valid=counters_valid) for event in (1, 2)]
    row = paired(samples)[0]
    assert row[6:9] == ("", "", "")
    assert row[9:13] == ((40, 80, 2.0, "valid_hart_counts") if counters_valid else
                        ("", "", "", "unavailable_or_unpinned"))
    assert row[13] == "riscv_hart_csr"


@pytest.mark.parametrize("begin_mono,end_mono", [(0, 200), (100, 0)])
def test_mixed_monotonic_boundaries_are_rejected(begin_mono: int, end_mono: int) -> None:
    with pytest.raises(TraceError, match="mixed monotonic"):
        paired([replace(sample(1), mono_ns=begin_mono),
                replace(sample(2), mono_ns=end_mono)])


def test_equal_nonzero_monotonic_boundaries_retain_zero_wall_time() -> None:
    assert paired([sample(1), replace(sample(2), mono_ns=100)])[0][6] == 0


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
