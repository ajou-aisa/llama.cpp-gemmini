# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run --no-project --offline tests/test-evaluation-host.py
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))


def main() -> None:
    # Given an evaluation host metadata entry point.
    module = ROOT / "scripts/eval/evaluation_host.py"
    # When looking for the new collector.
    assert module.is_file(), "host metadata collector is not implemented"
    import evaluation_host as host
    from eval_common import record
    with tempfile.TemporaryDirectory(prefix="evaluation-host-") as directory:
        root = Path(directory)
        files = {
            "proc/meminfo": "MemTotal: 123456 kB\nMemAvailable: 65432 kB\nSecret: hidden\n",
            "sys/devices/system/cpu/online": "0-3\n",
            "sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "1234567\n",
            "sys/devices/system/cpu/cpufreq/policy0/scaling_governor": "schedutil\n",
            "sys/class/devfreq/gpu/cur_freq": "987654321\n",
            "sys/class/thermal/thermal_zone0/type": "cpu-thermal\n",
            "sys/class/thermal/thermal_zone0/temp": "42125\n",
            "sys/module/nvidia/version": "550.120\n",
        }
        for relative, raw in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = target.write_text(raw, encoding="utf-8")
        # When sampling a Linux fixture, only external command execution is mocked.
        with patch("evaluation_host.shutil.which", return_value=None):
            facts = host._linux_facts(root / "proc", root / "sys")
        # Then observations preserve raw values, declared units and exact source.
        cpu_rows = record(record(facts["clocks"])["cpu"])["observations"]
        assert isinstance(cpu_rows, list)
        cpu = record(cpu_rows[0])
        assert record(cpu["scaling_cur_freq"])["raw"] == "1234567"
        assert record(cpu["scaling_cur_freq"])["unit"] == "kHz"
        assert record(cpu["scaling_cur_freq"])["source"] == str(
            root / "sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
        assert record(cpu["scaling_governor"])["raw"] == "schedutil"
        device_rows = record(record(facts["clocks"])["device"])["observations"]
        assert isinstance(device_rows, list)
        assert record(record(device_rows[0])["cur_freq"])["unit"] == "driver-native; unverified"
        thermal_rows = record(facts["thermal"])["observations"]
        assert isinstance(thermal_rows, list)
        assert record(record(thermal_rows[0])["temp"])["raw"] == "42125"
        assert record(record(thermal_rows[0])["temp"])["unit"] == "millidegree_C"
        memory = record(record(facts["memory"])["values"])
        assert memory["MemAvailable"] == "65432 kB"
        assert "Secret" not in memory
        assert record(facts["power_mode_observation"])["status"] == "UNAVAILABLE"
        assert record(record(facts["software"])["nvidia_driver"])["raw"] == "550.120"
        assert host._file(root / "absent", None)["status"] == "UNAVAILABLE"
        oversized = root / "oversized"
        _ = oversized.write_text("x" * 20000, encoding="utf-8")
        assert host._file(oversized, None)["status"] == "TRUNCATED"

    # Given a slow installed optional external tool.
    with patch("evaluation_host.shutil.which", return_value="/usr/bin/nvpmodel"), patch(
        "evaluation_host.subprocess.run", side_effect=subprocess.TimeoutExpired("nvpmodel", 2)
    ) as run:
        # When the read-only query times out.
        timed_out = host._command(["nvpmodel", "-q"])
        # Then collection continues, records timeout, and never invokes a shell.
        assert timed_out["status"] == "TIMEOUT"
        assert run.call_args.kwargs["timeout"] == 2
        assert not run.call_args.kwargs.get("shell", False)
        assert run.call_args.args[0] == ["/usr/bin/nvpmodel", "-q"]

    # Given Darwin sysctl output without privileged sensors.
    with patch("evaluation_host.shutil.which", return_value="/usr/sbin/sysctl"), patch(
        "evaluation_host.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "hw.memsize: 17179869184\nhw.logicalcpu: 8\n", ""),
    ) as run:
        # When sampling Darwin metadata.
        facts = host._mac_facts()
        # Then unsupported Jetson and sensor fields are explicit.
        assert record(facts["power_mode_observation"])["status"] == "UNSUPPORTED"
        assert record(facts["thermal"])["status"] == "UNSUPPORTED"
        assert record(facts["sysctl"])["raw"] == "hw.memsize: 17179869184\nhw.logicalcpu: 8"
        assert "-a" not in run.call_args.args[0]
        assert "hw.memsize" in run.call_args.args[0]
    print("PASS: Linux fixture provenance, missing/bounded reads, optional timeout, Darwin unsupported fields")


if __name__ == "__main__":
    main()
