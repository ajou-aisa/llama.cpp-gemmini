#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# How to run: python3 scripts/utils/capture_cpu_frequency.py --help
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Final, NewType, Sequence

CpuId = NewType("CpuId", int)
WorkerId = NewType("WorkerId", int)
CSV_HEADER: Final = ("sample_index", "monotonic_ns", "elapsed_ns", "cpu_id", "util_percent", "frequency_mhz", "sample_valid", "raw_line")
CPU_BLOCK: Final = re.compile(r"CPU\s*\[([^]]*)\]")
CPU_ENTRY: Final = re.compile(r"\s*(?:(\d+)\s*:\s*)?(\d+)\s*%@\s*(\d+)\s*")
AFFINITY: Final = re.compile(r"\bworker=(\d+)/(\d+)\b.*\bcpus=\{?(\d+)\}?\s*$")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.9 compatibility
class CpuReading:
    cpu_id: CpuId
    utilization: int
    frequency_mhz: int


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.9 compatibility
class CpuSample:
    readings: tuple[CpuReading, ...]
    valid: bool


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.9 compatibility
class CaptureError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.9 compatibility
class Config:
    tegrastats: str
    interval_ms: int
    expected_cpus: frozenset[CpuId]
    raw_output: Path
    csv_output: Path
    affinity_output: Path
    ready_timeout: float
    workload_timeout: float
    workload: tuple[str, ...]


def parse_cpu_sample(line: str, expected: frozenset[CpuId]) -> CpuSample:
    block = CPU_BLOCK.search(line)
    if block is None: return CpuSample((), False)
    readings: list[CpuReading] = []
    explicit = False
    for position, raw_entry in enumerate(block.group(1).split(",")):
        match = CPU_ENTRY.fullmatch(raw_entry)
        if match is None: return CpuSample(tuple(readings), False)
        label, utilization, frequency = match.groups()
        explicit = explicit or label is not None
        readings.append(CpuReading(CpuId(int(label)) if label is not None else CpuId(position), int(utilization), int(frequency)))
    ids = [reading.cpu_id for reading in readings]
    labels_consistent = not explicit or all(":" in entry for entry in block.group(1).split(","))
    valid = labels_consistent and len(ids) == len(set(ids)) and frozenset(ids) == expected
    return CpuSample(tuple(readings), valid)


def parse_affinity(text: str, expected: frozenset[CpuId]) -> dict[WorkerId, CpuId]:
    mapping: dict[WorkerId, CpuId] = {}
    cpu_owners: dict[CpuId, WorkerId] = {}
    for line in text.splitlines():
        if "worker=" not in line and "cpus=" not in line:
            continue
        match = AFFINITY.search(line)
        if match is None:
            raise CaptureError(f"malformed OpenMP affinity: {line}")
        worker_raw, team_raw, cpu_raw = match.groups()
        worker, team, cpu = WorkerId(int(worker_raw)), int(team_raw), CpuId(int(cpu_raw))
        if int(worker) >= team or (worker in mapping and mapping[worker] != cpu):
            raise CaptureError(f"conflicting OpenMP affinity for worker {worker}")
        if (owner := cpu_owners.get(cpu)) is not None and owner != worker:
            raise CaptureError(f"OpenMP affinity CPU {cpu} has multiple workers")
        mapping[worker], cpu_owners[cpu] = cpu, worker
    if frozenset(mapping.values()) != expected:
        raise CaptureError("OpenMP affinity does not map every expected CPU")
    return mapping


def terminate_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    pgid = process.pid
    if pgid <= 1 or pgid == os.getpgrp():
        raise CaptureError(f"refusing unsafe process group {pgid}")
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, requested_signal)
        except ProcessLookupError:
            break
    process.wait(timeout=1)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    raise CaptureError(f"process group {pgid} still exists after SIGKILL")


def run_capture(config: Config) -> int:
    for path in (config.raw_output, config.csv_output, config.affinity_output): path.parent.mkdir(parents=True, exist_ok=True)
    collector: subprocess.Popen[bytes] | None = None
    workload: subprocess.Popen[bytes] | None = None
    cancel_read, cancel_write = os.pipe()
    os.set_blocking(cancel_read, False)
    os.set_blocking(cancel_write, False)
    caught_signal = 0
    selector = selectors.DefaultSelector()
    done_read: int | None = None
    done_write: int | None = None

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal caught_signal
        caught_signal = signum
        try:
            os.write(cancel_write, b"x")
        except BlockingIOError: return

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        collector = subprocess.Popen([config.tegrastats, "--interval", str(config.interval_ms)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        if collector.stdout is None:
            raise CaptureError("tegrastats stdout pipe unavailable")
        os.set_blocking(collector.stdout.fileno(), False)
        selector.register(cancel_read, selectors.EVENT_READ, "cancel")
        selector.register(collector.stdout, selectors.EVENT_READ, "collector")
        start_ns: int | None = None
        sample_index = 0
        collector_buffer = bytearray()
        deadline = time.monotonic() + config.ready_timeout
        with config.raw_output.open("w", encoding="utf-8", newline="") as raw_stream, config.csv_output.open("w", encoding="utf-8", newline="") as csv_stream:
            writer = csv.writer(csv_stream)
            writer.writerow(CSV_HEADER)

            def consume_collector(chunk: bytes) -> bool:
                nonlocal collector_buffer, sample_index, start_ns
                collector_buffer.extend(chunk)
                ready = False
                while b"\n" in collector_buffer:
                    raw, _, remainder = collector_buffer.partition(b"\n")
                    collector_buffer = bytearray(remainder)
                    timestamp = time.monotonic_ns()
                    if start_ns is None:
                        start_ns = timestamp
                    line = raw.decode("utf-8", errors="replace")
                    raw_stream.write(f"{timestamp}\t{line}\n")
                    sample = parse_cpu_sample(line, config.expected_cpus)
                    rows = sample.readings or (CpuReading(CpuId(-1), -1, -1),)
                    for reading in rows:
                        writer.writerow((sample_index, timestamp, timestamp - start_ns, "" if reading.cpu_id < 0 else reading.cpu_id, "" if reading.utilization < 0 else reading.utilization, "" if reading.frequency_mhz < 0 else reading.frequency_mhz, str(sample.valid).lower(), line))
                    sample_index += 1
                    ready = ready or sample.valid
                return ready

            while workload is None:
                events = selector.select(max(0.0, deadline - time.monotonic()))
                if not events:
                    raise CaptureError("tegrastats readiness timeout")
                for key, _mask in events:
                    if key.data == "cancel":
                        return 128 + caught_signal
                    chunk = os.read(collector.stdout.fileno(), 65536)
                    if not chunk:
                        raise CaptureError(f"tegrastats exited before readiness ({collector.wait()})")
                    if consume_collector(chunk):
                        workload = subprocess.Popen(config.workload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                        break
            if workload.stdout is None or workload.stderr is None:
                raise CaptureError("workload output pipes unavailable")
            for stream, label in ((workload.stdout, "stdout"), (workload.stderr, "stderr")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            done_read, done_write = os.pipe()
            os.set_blocking(done_read, False)
            threading.Thread(target=lambda: (workload.wait(), os.write(done_write, b"x")), daemon=True).start()
            selector.register(done_read, selectors.EVENT_READ, "done")
            outputs = {"stdout": bytearray(), "stderr": bytearray()}
            deadline = time.monotonic() + config.workload_timeout
            finished = False
            while not finished:
                events = selector.select(max(0.0, deadline - time.monotonic()))
                if not events:
                    raise CaptureError("workload timeout")
                for key, _mask in events:
                    label = key.data
                    if label == "cancel":
                        return 128 + caught_signal
                    if label == "collector":
                        chunk = os.read(collector.stdout.fileno(), 65536)
                        if not chunk:
                            raise CaptureError("tegrastats exited while workload was running")
                        consume_collector(chunk)
                    elif label == "done":
                        finished = True
                    else:
                        stream = workload.stdout if label == "stdout" else workload.stderr
                        chunk = os.read(stream.fileno(), 65536)
                        outputs[label].extend(chunk)
                        target = sys.stdout.buffer if label == "stdout" else sys.stderr.buffer
                        target.write(chunk); target.flush()
            for label, stream in (("stdout", workload.stdout), ("stderr", workload.stderr)):
                while True:
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    outputs[label].extend(chunk)
                    (sys.stdout.buffer if label == "stdout" else sys.stderr.buffer).write(chunk)
            status = workload.returncode
            affinity = parse_affinity(outputs["stderr"].decode("utf-8", errors="replace"), config.expected_cpus)
            with config.affinity_output.open("w", encoding="utf-8", newline="") as stream:
                affinity_writer = csv.writer(stream)
                affinity_writer.writerow(("worker", "cpu"))
                affinity_writer.writerows(sorted(affinity.items()))
            return status
    finally:
        terminate_group(workload)
        terminate_group(collector)
        selector.close()
        for descriptor in (done_read, done_write):
            if descriptor is not None: os.close(descriptor)
        streams = (collector.stdout if collector is not None else None, workload.stdout if workload is not None else None, workload.stderr if workload is not None else None)
        for stream in streams:
            if stream is not None: stream.close()
        for sig, handler in previous.items(): signal.signal(sig, handler)
        os.close(cancel_read)
        os.close(cancel_write)


def parse_args(argv: Sequence[str]) -> Config:
    parser = argparse.ArgumentParser(description="Capture timestamped tegrastats around a workload")
    parser.add_argument("--tegrastats", required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--expected-cpus", required=True)
    for option in ("raw-output", "csv-output", "affinity-output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--workload-timeout", type=float, required=True)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    workload = tuple(args.workload[1:] if args.workload[:1] == ["--"] else args.workload)
    if not workload or args.interval_ms <= 0 or args.ready_timeout <= 0 or args.workload_timeout <= 0:
        parser.error("positive timeouts, interval, and a workload are required")
    try:
        expected = frozenset(CpuId(int(value.strip())) for value in args.expected_cpus.split(","))
    except ValueError as error:
        parser.error(f"invalid expected CPU list: {error}")
    return Config(args.tegrastats, args.interval_ms, expected, args.raw_output, args.csv_output, args.affinity_output, args.ready_timeout, args.workload_timeout, workload)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_capture(parse_args(sys.argv[1:] if argv is None else argv))
    except (CaptureError, OSError, subprocess.TimeoutExpired) as error:
        print(f"capture error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
