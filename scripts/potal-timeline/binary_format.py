# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: imported by binary_to_csv.py.
"""Bounded binary-v1 records shared with the shutdown writer's documented layout."""

from dataclasses import dataclass
from struct import Struct
from typing import BinaryIO, Final

EVENT: Final = Struct("<QIHHBBBB7IQQ")
CPU: Final = Struct("<5QIiII")
MAGIC: Final = b"# potal_timeline_binary v1\n"
CPU_HEADER: Final = "slot,seq,stage,event,run,layer,stripe,tile,mono_ns,thread_ns,process_ns,cpu,cycles,instructions,thread_valid,process_valid,counters_valid,counter_source\n"


@dataclass(frozen=True, slots=True)
class FormatError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def line(stream: BinaryIO) -> bytes:
    """Read bounded metadata; binary payload is consumed separately."""
    value = stream.readline(65537)
    if len(value) > 65536 or not value.endswith(b"\n"):
        raise FormatError("truncated or oversized metadata line")
    return value


def fields(value: bytes, prefix: bytes) -> dict[str, str]:
    """Parse a small key/value control line without accepting duplicate keys."""
    if not value.startswith(prefix + b","):
        raise FormatError("unexpected control line")
    result: dict[str, str] = {}
    for item in value.rstrip(b"\n").split(b",")[1:]:
        key, sep, val = item.partition(b"=")
        if not sep or key.decode("ascii") in result:
            raise FormatError("malformed or duplicate control field")
        result[key.decode("ascii")] = val.decode("ascii")
    return result


def exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise FormatError("truncated binary payload")
    return data


@dataclass(frozen=True, slots=True)
class Event:
    clock: int
    seq: int
    slot: int
    stage: int
    kind: int
    status: int
    op: int
    phase: int
    run: int
    layer: int
    stripe: int
    tile: int
    buffer: int
    edge: int
    padding: int
    work: int
    mono: int

    def csv(self, run_kind: tuple[int, int]) -> str:
        op, phase = run_kind if self.op == 0 and self.phase == 0 else (self.op, self.phase)
        values = (self.clock, self.seq, self.slot, self.stage, self.kind, self.status,
                  op, phase, self.run, self.layer, self.stripe, self.tile, self.buffer,
                  self.edge, self.work, self.mono)
        return "E," + ",".join(map(str, values)) + "\n"


@dataclass(frozen=True, slots=True)
class Sample:
    mono: int
    thread: int
    process: int
    cycles: int
    instructions: int
    seq: int
    cpu: int
    flags: int
    reserved: int

    def csv(self, event: Event, source: str) -> str:
        values = (event.slot, self.seq, event.stage, event.kind, event.run, event.layer,
                  event.stripe, event.tile, self.mono, self.thread, self.process, self.cpu,
                  self.cycles, self.instructions, self.flags & 1, (self.flags >> 1) & 1,
                  (self.flags >> 2) & 1)
        return ",".join(map(str, values)) + "," + source + "\n"
