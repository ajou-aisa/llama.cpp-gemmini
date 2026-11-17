# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: python3 scripts/potal-timeline/binary_to_csv.py TRACE.bin --out TRACE.csv
"""Convert a completed PoTal binary trace on the host, preserving CSV-v1 semantics."""

import argparse
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, TextIO

from binary_format import CPU, CPU_HEADER, EVENT, MAGIC, Event, FormatError, Sample, exact, fields, line


def transcode(stream: BinaryIO, output: TextIO, cpu: TextIO | None) -> int:
    """Validate all counts and completion before the caller publishes output files."""
    if line(stream) != MAGIC:
        raise FormatError("not a PoTal binary v1 trace")
    config = fields(line(stream), b"#BINARY")
    if (config.get("endian"), config.get("event_bytes"), config.get("cpu_bytes")) != ("little", "64", "56"):
        raise FormatError("unsupported binary layout")
    if config.get("cpu") not in ("0", "1") or config.get("counter_source") not in ("unavailable", "riscv_hart_csr"):
        raise FormatError("unsupported CPU metadata")
    has_cpu = config["cpu"] == "1"
    if has_cpu != (cpu is not None):
        raise FormatError("CPU output does not match binary header")
    if cpu is not None:
        cpu.write(CPU_HEADER)
    expected: dict[int, int] = {}
    seen: dict[int, int] = {}
    runs: dict[int, tuple[int, int]] = {}
    if line(stream) != b"# potal_timeline v1\n":
        raise FormatError("missing CSV metadata header")
    output.write("# potal_timeline v1\n")
    while True:
        raw = line(stream)
        text = raw.decode("ascii")
        output.write(text)
        cols = text.rstrip("\n").split(",")
        if raw.startswith(b"T,"):
            if len(cols) != 19 or int(cols[1]) in expected:
                raise FormatError("invalid thread metadata")
            expected[int(cols[1])] = int(cols[12])
            seen[int(cols[1])] = 0
        if raw.startswith(b"R,"):
            if len(cols) != 8:
                raise FormatError("invalid run metadata")
            if int(cols[5]) != 4294967295 or int(cols[6]) != 4294967295:
                runs[int(cols[1])] = (int(cols[3]), int(cols[4]))
        if raw.startswith(b"#EVENT,"):
            break
    total = 0
    while True:
        raw = line(stream)
        if raw.startswith(b"#END,"):
            end = fields(raw, b"#END")
            if set(end) != {"events"} or int(end["events"]) != total or seen != expected:
                raise FormatError("event count mismatch")
            if stream.read(1):
                raise FormatError("trailing bytes after completion")
            output.write(raw.decode("ascii"))
            return total
        block = fields(raw, b"#BLOCK")
        if set(block) != {"slot", "events", "samples"}:
            raise FormatError("invalid block header")
        slot, count, sampled = (int(block[key]) for key in ("slot", "events", "samples"))
        if slot not in expected or not 1 <= count <= 4096 or not 0 <= sampled <= count or (sampled and not has_cpu):
            raise FormatError("invalid block bounds")
        events = [Event(*parts) for parts in EVENT.iter_unpack(exact(stream, count * EVENT.size))]
        start = seen[slot]
        for index, event in enumerate(events, start):
            if event.slot != slot or event.seq != index or event.padding != 0:
                raise FormatError("event identity mismatch")
            output.write(event.csv(runs.get(event.run, (0, 0))))
        previous = start - 1
        for parts in CPU.iter_unpack(exact(stream, sampled * CPU.size)):
            sample = Sample(*parts)
            if not previous < sample.seq < start + count or sample.seq < start or sample.flags & ~7 or sample.reserved or not sample.mono:
                raise FormatError("invalid CPU sample identity or flags")
            if cpu is None:
                raise FormatError("unexpected CPU payload")
            cpu.write(sample.csv(events[sample.seq - start], config["counter_source"]))
            previous = sample.seq
        if exact(stream, 1) != b"\n":
            raise FormatError("missing block terminator")
        seen[slot] += count
        total += count
        if seen[slot] > expected[slot]:
            raise FormatError("thread event count exceeded")


def convert(source: Path, destination: Path) -> int:
    """Publish only fully validated files; never overwrite an existing analysis input."""
    cpu_path = Path(str(destination) + ".cpu.csv")
    if destination.exists() or cpu_path.exists():
        raise FileExistsError(destination)
    with source.open("rb") as stream:
        if line(stream) != MAGIC:
            raise FormatError("not a PoTal binary v1 trace")
        has_cpu = fields(line(stream), b"#BINARY").get("cpu") == "1"
        stream.seek(0)
        with TemporaryDirectory(prefix="potal-convert-", dir=destination.parent) as folder, ExitStack() as stack:
            tmp = Path(folder)
            out = stack.enter_context((tmp / "timeline.csv").open("w", encoding="ascii", newline=""))
            cpu = stack.enter_context((tmp / "cpu.csv").open("w", encoding="ascii", newline="")) if has_cpu else None
            total = transcode(stream, out, cpu)
            stack.close()
            os.link(tmp / "timeline.csv", destination)
            if has_cpu:
                os.link(tmp / "cpu.csv", cpu_path)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        count = convert(args.source, args.out)
    except (OSError, FormatError, ValueError, UnicodeError) as error:
        parser.exit(1, f"binary conversion failed: {error}\n")
    print(f"Converted {count} events: {args.out}")


if __name__ == "__main__":
    main()
