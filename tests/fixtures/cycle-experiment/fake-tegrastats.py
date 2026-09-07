#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


mode = os.environ.get("FAKE_TEGRASTATS_MODE", "happy")
pid_file = os.environ.get("FAKE_TEGRASTATS_PID_FILE")
if pid_file:
    with open(pid_file, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))

if mode == "malformed":
    emit("CPU [broken]")
    raise SystemExit(0)
if mode == "duplicate":
    emit("CPU [0:1%@729, 0:2%@0]")
    raise SystemExit(0)
if mode == "incomplete":
    emit("CPU [1%@729]")
    raise SystemExit(0)
if mode == "exit":
    raise SystemExit(7)
if mode == "exited_leader_descendant":
    ready_read, ready_write = os.pipe()
    descendant = os.fork()
    if descendant == 0:
        os.close(ready_read)
        with open(os.environ["FAKE_TEGRASTATS_DESCENDANT_PID_FILE"], "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        os.write(ready_write, b"x")
        os.close(ready_write)
        os.close(sys.stdout.fileno())
        signal.pause()
    os.close(ready_write)
    os.read(ready_read, 1)
    os.close(ready_read)
    raise SystemExit(17)
if mode == "hang":
    signal.pause()

trace = os.environ.get("FAKE_CAPTURE_TRACE")
if trace:
    with open(trace, "a", encoding="utf-8") as stream:
        stream.write("collector_ready\n")
emit("RAM 1/2MB CPU [ 10%@729 , 0%@0 ] GR3D_FREQ 0%")
signal.pause()
