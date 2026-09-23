from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Final

from eval_common import Json, Record

READ_LIMIT: Final = 16384
DEVICE_LIMIT: Final = 128
COMMAND_TIMEOUT_SECONDS: Final = 2


def _file(source: Path, unit: str | None) -> Record:
    result: Record = {"source": str(source), "unit": unit}
    try:
        with source.open("rb") as stream:
            raw = stream.read(READ_LIMIT + 1)
    except FileNotFoundError:
        return {**result, "status": "UNAVAILABLE", "raw": None}
    except OSError as error:
        return {**result, "status": "ERROR", "errno": error.errno, "raw": None}
    return {**result, "status": "TRUNCATED" if len(raw) > READ_LIMIT else "OBSERVED",
            "raw": raw[:READ_LIMIT].decode("utf-8", errors="replace").rstrip("\0\n")}


def _command(argv: list[str]) -> Record:
    result: Record = {"source": list(argv), "timeout_seconds": COMMAND_TIMEOUT_SECONDS}
    executable = shutil.which(argv[0])
    if executable is None:
        return {**result, "status": "UNAVAILABLE", "raw": None}
    result["executable"] = executable
    try:
        completed = subprocess.run([executable, *argv[1:]], capture_output=True, text=True,
                                   timeout=COMMAND_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return {**result, "status": "TIMEOUT", "raw": None}
    except OSError as error:
        return {**result, "status": "ERROR", "errno": error.errno, "raw": None}
    status = "OBSERVED" if completed.returncode == 0 else "ERROR"
    if len(completed.stdout) > READ_LIMIT or len(completed.stderr) > READ_LIMIT:
        status = "TRUNCATED"
    return {**result, "status": status, "exit_code": completed.returncode,
            "raw": completed.stdout[:READ_LIMIT].rstrip(), "stderr": completed.stderr[:READ_LIMIT].rstrip()}


def _sysfs(root: Path, pattern: str, fields: tuple[tuple[str, str | None], ...]) -> Record:
    try:
        entries = sorted(islice(root.glob(pattern), DEVICE_LIMIT + 1))
    except OSError as error:
        return {"status": "ERROR", "source": str(root / pattern), "errno": error.errno}
    observations: list[Json] = [
        {name: _file(entry / name, unit) for name, unit in fields}
        for entry in entries[:DEVICE_LIMIT]
    ]
    return {"status": "TRUNCATED" if len(entries) > DEVICE_LIMIT else
            ("OBSERVED" if observations else "UNAVAILABLE"),
            "source": str(root / pattern), "device_limit": DEVICE_LIMIT, "observations": observations}


def _linux_facts(proc_root: Path, sys_root: Path) -> Record:
    memory = _file(proc_root / "meminfo", "per-field native unit")
    selected: Record = {}
    raw = memory.pop("raw", None)
    if isinstance(raw, str):
        for line in raw.splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"}:
                selected[name] = value.strip()
    memory["values"] = selected
    return {
        "memory": memory,
        "cpu_online": _file(sys_root / "devices/system/cpu/online", "CPU index range"),
        "clocks": {
            "cpu": _sysfs(sys_root, "devices/system/cpu/cpufreq/policy*", (
                ("scaling_cur_freq", "kHz"), ("cpuinfo_cur_freq", "kHz"),
                ("scaling_min_freq", "kHz"), ("scaling_max_freq", "kHz"),
                ("scaling_governor", None), ("affected_cpus", "CPU indices"))),
            "device": _sysfs(sys_root, "class/devfreq/*", (
                ("cur_freq", "driver-native; unverified"), ("min_freq", "driver-native; unverified"),
                ("max_freq", "driver-native; unverified"), ("governor", None))),
        },
        "thermal": _sysfs(sys_root, "class/thermal/thermal_zone*", (
            ("type", None), ("temp", "millidegree_C"))),
        "power_mode_observation": _command(["nvpmodel", "-q"]),
        "software": {"nvidia_driver": _file(sys_root / "module/nvidia/version", None)},
    }


def _mac_facts() -> Record:
    unsupported: Record = {"status": "UNSUPPORTED", "source": "Darwin unprivileged collector"}
    return {
        "sysctl": _command(["sysctl", "hw.logicalcpu", "hw.physicalcpu", "hw.memsize",
                            "hw.model", "kern.osproductversion"]),
        "memory": {"status": "SEE_SYSCTL", "source": "sysctl hw.memsize", "unit": "bytes",
                   "available_memory_status": "NOT_COLLECTED"},
        "clocks": unsupported, "thermal": unsupported,
        "power_mode_observation": {**unsupported, "reason": "nvpmodel is a Linux Jetson query"},
    }


def host_facts() -> Record:
    """Read selected host observations; no settings are changed or target clocks inferred."""
    uname = platform.uname()
    identity = hashlib.sha256("\0".join(uname).encode()).hexdigest()
    facts: Record = {
        "host_id": identity, "system": uname.system, "release": uname.release,
        "machine": uname.machine, "python": platform.python_version(), "board": None,
        "power_mode": None, "thermal_clock_memory_status": "PARTIAL_OBSERVATIONS",
        "permanent_host_settings_changed": False,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "logical_cpu_count": os.cpu_count(),
        "scope": "READ_ONLY_HOST_OBSERVATION_NOT_TARGET_FPGA_CLOCK_OR_CAMPAIGN_PROOF",
        "software_versions": {"python": platform.python_version(),
                              "python_implementation": platform.python_implementation(),
                              "kernel_release": uname.release},
    }
    try:
        cpu_indices: list[Json] = []
        cpu_indices.extend(sorted(os.sched_getaffinity(0)))
        affinity: Record = {"status": "OBSERVED", "source": "os.sched_getaffinity(0)",
                            "cpu_indices": cpu_indices}
    except AttributeError:
        affinity = {"status": "UNSUPPORTED", "source": "os.sched_getaffinity(0)"}
    except OSError as error:
        affinity = {"status": "ERROR", "source": "os.sched_getaffinity(0)", "errno": error.errno}
    facts["process_cpu_affinity"] = affinity
    if uname.system == "Linux":
        board = _file(Path("/proc/device-tree/model"), None)
        facts["board"] = board["raw"]
        facts["board_observation"] = board
        facts.update(_linux_facts(Path("/proc"), Path("/sys")))
    if uname.system == "Darwin":
        facts.update(_mac_facts())
    if uname.system not in {"Linux", "Darwin"}:
        facts["thermal_clock_memory_status"] = "UNSUPPORTED_PLATFORM"
        facts["power_mode_observation"] = {"status": "UNSUPPORTED", "source": uname.system}
    power = facts["power_mode_observation"]
    if isinstance(power, dict) and power.get("status") == "OBSERVED":
        facts["power_mode"] = power.get("raw")
    return facts
