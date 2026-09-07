#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys


pid_file = os.environ.get("FAKE_WORKLOAD_PID_FILE")
if pid_file:
    with open(pid_file, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))

sentinel = os.environ.get("FAKE_WORKLOAD_SENTINEL")
if sentinel:
    with open(sentinel, "w", encoding="utf-8") as stream:
        stream.write("started\n")
trace = os.environ.get("FAKE_CAPTURE_TRACE")
if trace:
    with open(trace, "a", encoding="utf-8") as stream:
        stream.write("workload_started\n")

lines = ["level=1 worker=0/2 tid=0x1 cpus=0", "level=1 worker=1/2 tid=0x2 cpus=1"]
affinity_mode = os.environ.get("FAKE_AFFINITY_MODE", "normal")
if affinity_mode == "shared_cpu":
    lines[1] = "level=1 worker=1/2 tid=0x2 cpus=0"
if affinity_mode == "remap":
    lines.append("level=1 worker=0/2 tid=0x1 cpus=1")
if affinity_mode == "repeated_regions":
    lines.extend(("level=1 worker=0/2 tid=0x1 cpus=0", "level=1 worker=1/2 tid=0x2 cpus=1"))
for line in lines:
    sys.stderr.write(line + "\n")
sys.stderr.flush()
sys.stdout.write("WORKLOAD_STARTED\n")
sys.stdout.flush()

mode = os.environ.get("FAKE_WORKLOAD_MODE", "success")
if mode == "failure":
    raise SystemExit(9)
if mode == "hang":
    signal.pause()
