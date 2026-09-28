"""Aggregate native stateful-session events into per-work resource cycle counts.

Pure event aggregation: no timing equation is evaluated or approximated. Every
quantity is the size of a union of closed cycle intervals built from existing
native events, restricted to one work's [accepted, resource_ready) window.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt
from eval_common import Record, require

EVENTS: Final = ("scale_request", "scale_response", "scale_lane", "scale_release", "read_request",
                  "read_response", "write_request", "write_completion")
DEFINITIONS: Final = {
    "load_cycles": "union of [ReadRequest, matching ReadResponse] cycles (backing loads)",
    "store_cycles": "union of [WriteRequest, next WriteCompletion] cycles (single outstanding store)",
    "scale_cycles": "union of [ScaleRequest, matching ScaleResponse] cycles (scale fetch path)",
    "scu_active_cycles": "union of scale fetch intervals, ScaleLane cycles and ScaleRelease cycles",
    "scu_idle_cycles": "resource_ready_cycle - accepted_cycle - scu_active_cycles",
    "scu_cycles": "scu_active_cycles",
    "dense_cycles": "result_ready_cycle - accepted_cycle for dense_main works",
    "residual_cycles": "result_ready_cycle - accepted_cycle for residual works",
}
Int64Array: TypeAlias = npt.NDArray[np.int64]
Events: TypeAlias = npt.NDArray[np.void]


def event_dtype() -> np.dtype[np.void]:
    from sim.cycle.cli import Event
    from sim.cycle.sequence_binding_abi import SequenceEvent

    base = SequenceEvent.event.offset
    fields = {"generation": (SequenceEvent.generation.offset, "<u8"),
              "work_ordinal": (SequenceEvent.work_ordinal.offset, "<u8"),
              "id": (base + Event.id.offset, "<u8"), "cycle": (base + Event.cycle.offset, "<u8"),
              "logical_work_id": (base + Event.logical_work_id.offset, "<u8"),
              "dependency": (base + Event.dependency.offset, "<u8"),
              "type": (base + Event.type.offset, "<u4")}
    return np.dtype({"names": list(fields), "formats": [value[1] for value in fields.values()],
                     "offsets": [value[0] for value in fields.values()], "itemsize": C.sizeof(SequenceEvent)})


def event_codes(library: Path) -> dict[str, int]:
    lib = C.CDLL(str(library))
    lib.im2p_cycle_event_name.argtypes, lib.im2p_cycle_event_name.restype = [C.c_uint32], C.c_char_p
    names: dict[str, int] = {}
    for code in range(256):
        raw: bytes | None = lib.im2p_cycle_event_name(code)
        name = "invalid" if raw is None else raw.decode("utf-8")
        if name == "invalid":
            break
        require(name not in names, "duplicate native event name: " + name)
        names[name] = code
    require(all(name in names for name in EVENTS), "native library lacks scale/load/store event names")
    return {name: names[name] for name in EVENTS}


def union_cycles(starts: Int64Array, ends: Int64Array) -> int:
    require(starts.shape == ends.shape and bool(np.all(ends >= starts)), "invalid cycle interval")
    if starts.size == 0:
        return 0
    order = np.argsort(starts, kind="stable")
    begin, end = starts[order], ends[order]
    covered = np.maximum.accumulate(end)
    previous = np.concatenate((np.array([begin[0] - 1], dtype=np.int64), covered[:-1]))
    return int(np.clip(end - np.maximum(begin, previous + 1) + 1, 0, None).sum())


def _pair_by_dependency(requests: Events, responses: Events, label: str) -> tuple[Int64Array, Int64Array]:
    require(requests.size == responses.size, label + " request/response counts differ")
    order = np.argsort(requests["id"], kind="stable")
    ids = requests["id"][order]
    index = np.searchsorted(ids, responses["dependency"])
    require(bool(np.all(index < ids.size)) and bool(np.all(ids[np.minimum(index, ids.size - 1)] ==
            responses["dependency"])) and np.unique(index).size == index.size, label + " response pairing failed")
    return (requests["cycle"][order][index].astype(np.int64), responses["cycle"].astype(np.int64))


def _pair_in_order(requests: Events, completions: Events, label: str) -> tuple[Int64Array, Int64Array]:
    require(requests.size == completions.size, label + " request/completion counts differ")
    starts = requests["cycle"][np.argsort(requests["id"], kind="stable")].astype(np.int64)
    ends = completions["cycle"][np.argsort(completions["id"], kind="stable")].astype(np.int64)
    require(bool(np.all(ends >= starts)) and bool(np.all(starts[1:] >= ends[:-1])),
            label + " is not single-outstanding in event order")
    return starts, ends


class EventAccounting:
    def __init__(self, codes: dict[str, int], dtype: np.dtype[np.void]) -> None:
        self.codes, self.dtype = codes, dtype
        self._chunks: dict[str, list[Events]] = {key: [] for key in EVENTS}
        self.total_events = 0

    @classmethod
    def for_library(cls, library: Path) -> EventAccounting:
        return cls(event_codes(library), event_dtype())

    def sink(self, data: memoryview, count: int) -> None:
        events = np.frombuffer(data, dtype=self.dtype, count=count)
        self.total_events += count
        for key, code in self.codes.items():
            selected = events[events["type"] == code]
            if selected.size:
                self._chunks[key].append(selected.copy())

    def finish(self, work_id: int, accepted: int, resource_ready: int) -> Record:
        empty = np.empty(0, dtype=self.dtype)
        found = {key: np.concatenate(rows) if rows else empty for key, rows in self._chunks.items()}
        self._chunks = {key: [] for key in EVENTS}
        for rows in found.values():
            require(bool(np.all(rows["logical_work_id"] == work_id)) and bool(np.all(rows["generation"] == 1)),
                    f"work {work_id}: event from another work or generation")
            require(bool(np.all((rows["cycle"] >= accepted) & (rows["cycle"] < resource_ready))),
                    f"work {work_id}: event outside accepted/resource window")
        load = _pair_by_dependency(found["read_request"], found["read_response"], "load")
        scale = _pair_by_dependency(found["scale_request"], found["scale_response"], "scale")
        store = _pair_in_order(found["write_request"], found["write_completion"], "store")
        points = np.concatenate((found["scale_lane"]["cycle"], found["scale_release"]["cycle"])).astype(np.int64)
        active = union_cycles(np.concatenate((scale[0], points)), np.concatenate((scale[1], points)))
        window = resource_ready - accepted
        require(0 <= active <= window, f"work {work_id}: SCU activity exceeds resource window")
        return {"load_cycles": union_cycles(*load), "store_cycles": union_cycles(*store),
                "scale_cycles": union_cycles(*scale), "scu_active_cycles": active,
                "scu_idle_cycles": window - active, "scu_window_cycles": window,
                "scale_request_count": int(found["scale_request"].size),
                "scale_response_count": int(found["scale_response"].size),
                "scale_lane_count": int(found["scale_lane"].size),
                "scale_release_count": int(found["scale_release"].size),
                "load_request_event_count": int(found["read_request"].size),
                "load_response_event_count": int(found["read_response"].size),
                "store_request_event_count": int(found["write_request"].size),
                "store_completion_event_count": int(found["write_completion"].size)}
