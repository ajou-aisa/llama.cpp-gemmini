"""Parse residual host profiles using Python 3.9+ and the standard library."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Optional

try:
    from .cycle_schema import (
        CycleSchemaError, JsonValue, _DuplicateKeyError, _InvalidConstantError,
        _optional_integer, _optional_string, _pairs_to_mapping, _reject_constant,
    )
except ImportError:
    from cycle_schema import (
        CycleSchemaError, JsonValue, _DuplicateKeyError, _InvalidConstantError,
        _optional_integer, _optional_string, _pairs_to_mapping, _reject_constant,
    )

PHASES: Final = ("validation", "preparation", "parallel", "finalization")
WORKLOAD: Final = ("event_count", "active_rows", "active_row_blocks", "row_begin", "row_count",
                   "logical_j", "logical_k", "j_tile_count")


# Python 3.9 dataclasses do not support slots=True.
@dataclass(frozen=True)
class HostTiming:
    execution_id: str
    start_ns: int
    end_ns: int
    start_tid: int
    end_tid: int
    thread_id_kind: str

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class Timing:
    host: HostTiming
    cpu_ns: Optional[int]


@dataclass(frozen=True)
class Worker:
    worker_id: int
    tid: int
    timing: Timing
    barrier: Optional[Timing]


@dataclass(frozen=True)
class Tile:
    node_id: int
    worker_id: int
    j_begin: int
    j_end: int
    timing: Timing
    log: Optional[Timing]
    log_calls: Optional[int]
    log_mutex_wait_ns: Optional[int]
    log_io_ns: Optional[int]


@dataclass(frozen=True)
class Profile:
    run_id: Optional[int]
    layer: Optional[str]
    stripe_id: int
    host: HostTiming
    workload: Mapping[str, int]
    phases: Mapping[str, Timing]
    workers: tuple[Worker, ...]
    tiles: tuple[Tile, ...]


def _mapping(value: JsonValue, line: int) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise CycleSchemaError(line, "expected a JSON object")
    return value


def _objects(value: JsonValue, line: int) -> Iterator[Mapping[str, JsonValue]]:
    if not isinstance(value, list):
        raise CycleSchemaError(line, "expected a JSON array")
    return (_mapping(item, line) for item in value)


def _integer(record: Mapping[str, JsonValue], key: str, line: int) -> int:
    value = _optional_integer(record, key, line)
    if value is None:
        raise CycleSchemaError(line, f"field {key!r} must be a non-negative integer")
    return value


def _host(value: JsonValue, line: int) -> Optional[HostTiming]:
    record = _mapping(value, line)
    execution = _optional_string(record, "execution_id", line)
    kind = _optional_string(record, "thread_id_kind", line)
    if (not execution or not execution.strip() or kind is None or kind not in {"os_tid", "process_thread_token"}
            or record.get("clock") != "steady_clock" or record.get("unit") != "nanosecond"):
        raise CycleSchemaError(line, "host_timing has unsupported clock, unit, execution or thread identity")
    if type(record.get("valid")) is not bool:
        raise CycleSchemaError(line, "host_timing.valid must be boolean")
    keys = ("start_ns", "end_ns", "start_tid", "end_tid", "duration_ns")
    values = tuple(_optional_integer(record, key, line) for key in keys)
    if record["valid"] is False:
        if any(value is not None for value in values):
            raise CycleSchemaError(line, "invalid host_timing must have null endpoints and duration")
        return None
    start, end, start_tid, end_tid, duration = values
    if (start is None or end is None or start_tid is None or end_tid is None or duration is None
            or not start_tid or start_tid != end_tid or end < start or duration != end - start):
        raise CycleSchemaError(line, "host_timing has invalid endpoints, thread identity or duration")
    return HostTiming(execution, start, end, start_tid, end_tid, kind)


def _cpu(value: JsonValue, line: int) -> Optional[int]:
    if value is None:
        return None
    record = _mapping(value, line)
    if (record.get("clock") != "thread_cpu" or record.get("unit") != "nanosecond"
            or type(record.get("valid")) is not bool):
        raise CycleSchemaError(line, "thread_cpu_timing has unsupported clock, unit or validity")
    start, end, duration = (_optional_integer(record, key, line)
                            for key in ("start_ns", "end_ns", "duration_ns"))
    if record["valid"] is False:
        if duration is not None:
            raise CycleSchemaError(line, "invalid thread_cpu_timing must have a null duration")
        return None
    if start is None or end is None or duration is None or end < start or duration != end - start:
        raise CycleSchemaError(line, "thread_cpu_timing has inconsistent endpoints or duration")
    return duration


def _timing(record: Mapping[str, JsonValue], line: int, prefix: str = "") -> Optional[Timing]:
    host = _host(record.get(prefix + "host_timing"), line)
    cpu = _cpu(record.get(prefix + "thread_cpu_timing"), line)
    if host is None:
        if cpu is not None:
            raise CycleSchemaError(line, "absent host_timing cannot have valid thread CPU timing")
        return None
    return Timing(host, cpu)


def _required_timing(record: Mapping[str, JsonValue], line: int) -> Timing:
    timing = _timing(record, line)
    if timing is None:
        raise CycleSchemaError(line, "selected profile contains invalid required host_timing")
    return timing


def _inside(child: HostTiming, parent: HostTiming, line: int) -> None:
    if (child.execution_id != parent.execution_id or child.thread_id_kind != parent.thread_id_kind
            or child.start_ns < parent.start_ns or child.end_ns > parent.end_ns):
        raise CycleSchemaError(line, "child host_timing is outside parent or has different execution/thread identity")


def _profile(record: Mapping[str, JsonValue], line: int) -> Profile:
    expected = {"schema": "gemmini.cycle", "version": 2, "source": "steady_clock",
                "unit": "nanosecond", "op": "rmd.cpu_direct.profile", "valid": True}
    if any(type(record.get(key)) is not type(value) or record.get(key) != value
           for key, value in expected.items()):
        raise CycleSchemaError(line, "selected profile has unsupported schema/version/source/unit/op or valid is false")
    run = _optional_integer(record, "run_id", line)
    layer = _optional_string(record, "layer", line)
    stripe = _integer(record, "stripe_id", line)
    host = _required_timing(record, line).host
    workload_raw = _mapping(record.get("workload"), line)
    workload = {key: _integer(workload_raw, key, line) for key in WORKLOAD}
    if (any(workload[key] == 0 for key in ("event_count", "active_rows", "row_count", "logical_j", "logical_k"))
            or workload["active_rows"] > workload["row_count"]
            or workload["active_rows"] > workload["active_row_blocks"]
            or workload["active_row_blocks"] > workload["event_count"]
            or workload["active_row_blocks"] > workload["active_rows"] * ((workload["logical_k"] + 31) // 32)
            or workload["event_count"] > workload["row_count"] * workload["logical_k"]
            or workload["j_tile_count"] != (workload["logical_j"] + 15) // 16):
        raise CycleSchemaError(line, "workload counts or J tile count are inconsistent")
    phase_raw = _mapping(record.get("phases"), line)
    phases = {key: _required_timing(_mapping(phase_raw.get(key), line), line) for key in PHASES}
    previous_end = host.start_ns
    for phase in phases.values():
        _inside(phase.host, host, line)
        if phase.host.start_tid != host.start_tid or phase.host.start_ns < previous_end:
            raise CycleSchemaError(line, "phases have inconsistent thread identity or order")
        previous_end = phase.host.end_ns
    workers: dict[int, Worker] = {}
    for raw in _objects(record.get("workers"), line):
        worker = Worker(_integer(raw, "worker_id", line), _integer(raw, "tid", line),
                        _required_timing(raw, line), _timing(raw, line, "barrier_"))
        _inside(worker.timing.host, phases["parallel"].host, line)
        if worker.worker_id in workers or worker.tid != worker.timing.host.start_tid:
            raise CycleSchemaError(line, "duplicate worker or mismatched worker tid")
        if worker.barrier is not None:
            _inside(worker.barrier.host, phases["parallel"].host, line)
            if worker.barrier.host.start_tid != worker.tid or worker.barrier.host.start_ns != worker.timing.host.end_ns:
                raise CycleSchemaError(line, "worker barrier must follow work on the same thread")
        workers[worker.worker_id] = worker
    if len({worker.tid for worker in workers.values()}) != len(workers) or sorted(workers) != list(range(len(workers))):
        raise CycleSchemaError(line, "workers have duplicate tids or noncontiguous worker IDs")
    tiles: dict[int, Tile] = {}
    for raw in _objects(record.get("tiles"), line):
        tile = Tile(_integer(raw, "node_id", line), _integer(raw, "worker_id", line),
                    _integer(raw, "j_begin", line), _integer(raw, "j_end", line),
                    _required_timing(raw, line), _timing(raw, line, "log_"),
                    *(_optional_integer(raw, key, line) for key in ("log_calls", "log_mutex_wait_ns", "log_io_ns")))
        worker = workers.get(tile.worker_id)
        if (tile.node_id in tiles or worker is None or tile.j_begin != tile.node_id * 16
                or tile.j_end != min(tile.j_begin + 16, workload["logical_j"]) or tile.j_end <= tile.j_begin):
            raise CycleSchemaError(line, "tile has duplicate/unknown identity or invalid J range")
        _inside(tile.timing.host, worker.timing.host, line)
        if tile.timing.host.start_tid != worker.tid:
            raise CycleSchemaError(line, "tile tid does not match its worker")
        if tile.log is not None:
            _inside(tile.log.host, worker.timing.host, line)
            if tile.log.host.start_tid != worker.tid or tile.log.host.start_ns < tile.timing.host.end_ns:
                raise CycleSchemaError(line, "tile logging must follow compute on the same worker")
        calls, wait, io = tile.log_calls, tile.log_mutex_wait_ns, tile.log_io_ns
        if type(raw.get("log_valid")) is not bool:
            raise CycleSchemaError(line, "log_valid must be boolean")
        if raw["log_valid"] is True:
            if tile.log is None or calls is None or wait is None or io is None or calls == 0:
                raise CycleSchemaError(line, "valid logger telemetry requires a span and counters")
            if wait + io > tile.log.host.duration_ns:
                raise CycleSchemaError(line, "logger wait and I/O durations exceed the log span")
        elif any(value is not None for value in (calls, wait, io)):
            raise CycleSchemaError(line, "invalid logger telemetry must have null counters")
        tiles[tile.node_id] = tile
    if len(tiles) != workload["j_tile_count"] or any(node != index for index, node in enumerate(sorted(tiles))):
        raise CycleSchemaError(line, "tiles must cover every J tile exactly once")
    for worker in workers.values():
        end = worker.timing.host.start_ns
        for tile in sorted((tile for tile in tiles.values() if tile.worker_id == worker.worker_id),
                           key=lambda tile: tile.timing.host.start_ns):
            if tile.timing.host.start_ns < end:
                raise CycleSchemaError(line, "tiles overlap on the same worker")
            end = tile.log.host.end_ns if tile.log is not None else tile.timing.host.end_ns
    return Profile(run, layer, stripe, host, MappingProxyType(workload), MappingProxyType(phases),
                   tuple(workers.values()), tuple(tiles[key] for key in sorted(tiles)))


def read_profiles(path: Path, run_id: Optional[int] = None, layer: Optional[str] = None,
                  stripe_id: Optional[int] = None) -> tuple[Profile, ...]:
    """Read selected profiles, retaining separate executions and physical-line errors."""
    profiles = []
    identities = set()
    execution_kinds: dict[str, str] = {}
    seen = False
    filters = (("run_id", run_id), ("layer", layer), ("stripe_id", stripe_id))
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                decoded: JsonValue = json.loads(line, object_pairs_hook=_pairs_to_mapping, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise CycleSchemaError(line_number, f"malformed JSON at column {error.colno}") from None
            except _DuplicateKeyError as error:
                raise CycleSchemaError(line_number, f"duplicate key {error.args[0]!r}") from None
            except _InvalidConstantError as error:
                raise CycleSchemaError(line_number, f"invalid JSON constant {error.args[0]!r}") from None
            record = _mapping(decoded, line_number)
            if record.get("record_type") != "RESIDUAL_HOST_PROFILE":
                continue
            seen = True
            if any(value is not None and (type(record.get(key)) is not type(value) or record.get(key) != value)
                   for key, value in filters):
                continue
            profile = _profile(record, line_number)
            identity = (profile.host.execution_id, profile.run_id, profile.layer, profile.stripe_id)
            kind = execution_kinds.setdefault(profile.host.execution_id, profile.host.thread_id_kind)
            if identity in identities or kind != profile.host.thread_id_kind:
                raise CycleSchemaError(line_number, "duplicate profile identity or inconsistent execution thread_id_kind")
            identities.add(identity)
            profiles.append(profile)
    if not seen:
        raise CycleSchemaError(0, "no RESIDUAL_HOST_PROFILE records; rebuild with LOG_CYCLE && CYCLE_DETAIL "
                               "and collect a new capture")
    if not profiles:
        raise CycleSchemaError(0, "no residual host profiles match the selected filters")
    return tuple(profiles)
