from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/eval"), str(ROOT.parent / "IM2P.sim")]
from cycle_accounting import EVENTS, EventAccounting, event_dtype, union_cycles
from eval_common import EvaluationError

CODES = {key: index + 1 for index, key in enumerate(EVENTS)}


def buffer(rows: list[tuple[str, int, int, int]], work: int = 7) -> tuple[memoryview, int]:
    events = np.zeros(len(rows), dtype=event_dtype())
    for index, (kind, identity, cycle, dependency) in enumerate(rows):
        events[index]["type"], events[index]["id"], events[index]["cycle"] = CODES[kind], identity, cycle
        events[index]["dependency"], events[index]["logical_work_id"] = dependency, work
        events[index]["generation"] = 1
    return memoryview(events.tobytes()), len(rows)


def rejected(rows: list[tuple[str, int, int, int]], reason: str, work: int = 7) -> None:
    accounting = EventAccounting(CODES, event_dtype())
    accounting.sink(*buffer(rows, work))
    try:
        accounting.finish(7, 10, 62)
    except EvaluationError as error:
        assert reason in str(error), error
    else:
        raise AssertionError("invalid event stream admitted: " + reason)


def check() -> None:
    # Given: closed intervals that nest, overlap and touch.
    assert union_cycles(np.array([5, 6, 12], dtype=np.int64), np.array([10, 7, 12], dtype=np.int64)) == 7
    assert union_cycles(np.array([1, 3], dtype=np.int64), np.array([2, 4], dtype=np.int64)) == 4
    assert union_cycles(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)) == 0
    # Given: one work's native events split across two drains.
    stream = [("read_request", 1, 10, 0), ("read_request", 2, 12, 0), ("read_response", 3, 14, 1),
              ("read_response", 4, 20, 2), ("write_request", 5, 30, 0), ("write_completion", 6, 33, 0),
              ("scale_request", 7, 40, 0), ("scale_response", 8, 45, 7), *(("scale_lane", 9 + lane, 46 + lane, 0)
              for lane in range(4)), ("scale_release", 20, 60, 0), ("scale_release", 21, 61, 0)]
    accounting = EventAccounting(CODES, event_dtype())
    accounting.sink(*buffer(stream[:7]))
    accounting.sink(*buffer(stream[7:]))
    # When: the work window closes.
    result = accounting.finish(7, 10, 62)
    # Then: every count is an interval union of existing events only.
    assert (result["load_cycles"], result["store_cycles"], result["scale_cycles"]) == (11, 4, 6)
    assert (result["scu_active_cycles"], result["scu_idle_cycles"], result["scu_window_cycles"]) == (12, 40, 52)
    assert (result["scale_request_count"], result["scale_response_count"], result["scale_release_count"]) == (1, 1, 2)
    assert accounting.finish(8, 62, 70)["scu_active_cycles"] == 0
    rejected([("scale_response", 2, 12, 99), ("scale_request", 1, 11, 0)], "scale response pairing")
    rejected([("scale_release", 1, 62, 0)], "outside accepted/resource window")
    rejected([("scale_release", 1, 20, 0)], "another work", work=8)
    rejected([("write_request", 1, 11, 0), ("write_request", 2, 12, 0), ("write_completion", 3, 13, 0),
              ("write_completion", 4, 14, 0)], "single-outstanding")


if __name__ == "__main__":
    check()
    print("PASS: interval union, cross-drain pairing, SCU active/idle conservation, foreign/window/pairing rejection")
