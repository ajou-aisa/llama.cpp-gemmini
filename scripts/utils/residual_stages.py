from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Optional

try:
    from .cycle_schema import CycleSchemaError, JsonValue, _optional_integer, _optional_string
except ImportError:
    from cycle_schema import CycleSchemaError, JsonValue, _optional_integer, _optional_string

STAGES: Final = ("event_scan", "weight_dot", "scale_apply")


@dataclass(frozen=True)
class Stage:
    calls: int
    wall_ns: Optional[int]
    thread_cpu_ns: Optional[int]
    cycles: Optional[int]
    cycles_reason: str


def read_stages(tile: Mapping[str, JsonValue], enabled: JsonValue, line: int) -> Mapping[str, Stage]:
    if type(enabled) is not bool:
        raise CycleSchemaError(line, "deep_profile must be boolean")
    if not enabled:
        if "stages" in tile:
            raise CycleSchemaError(line, "tile stages require deep_profile=true")
        return MappingProxyType({})
    raw_stages = tile.get("stages")
    if not isinstance(raw_stages, dict) or set(raw_stages) != set(STAGES):
        raise CycleSchemaError(line, "deep profile requires event_scan, weight_dot and scale_apply stages")
    result: dict[str, Stage] = {}
    for name in STAGES:
        raw = raw_stages[name]
        if not isinstance(raw, dict):
            raise CycleSchemaError(line, "stage must be an object")
        if not {"calls", "wall_ns", "thread_cpu_ns", "cycles", "cycles_valid", "cycles_reason"} <= raw.keys():
            raise CycleSchemaError(line, "stage is missing a measurement field")
        calls, wall, cpu, cycles = (_optional_integer(raw, key, line)
                                    for key in ("calls", "wall_ns", "thread_cpu_ns", "cycles"))
        reason = _optional_string(raw, "cycles_reason", line)
        valid = raw.get("cycles_valid")
        if calls is None or type(valid) is not bool or not reason or not reason.strip():
            raise CycleSchemaError(line, "stage requires calls, boolean cycles_valid and cycles_reason")
        if (valid and (cycles is None or reason != "none")) or (not valid and (cycles is not None or reason == "none")):
            raise CycleSchemaError(line, "stage cycles contradict cycles_valid or cycles_reason")
        if calls == 0 and (any(value is not None for value in (wall, cpu, cycles)) or reason != "no_samples"):
            raise CycleSchemaError(line, "a stage without calls must have null measurements and no_samples reason")
        if calls > 0 and reason == "no_samples":
            raise CycleSchemaError(line, "a sampled stage cannot have no_samples reason")
        result[name] = Stage(calls, wall, cpu, cycles, reason)
    return MappingProxyType(result)
