#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python3 scripts/experiment/check-metal-ablation-build.py BUILD_DIR
"""Check the actual production library without enabling instrumented test builds."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check-metal-ablation-build.py BUILD_DIR")
    build = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[2]
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(item for item in entries if item["file"].endswith("/ggml-gemmini.cpp"))
    original = iter(shlex.split(entry["command"]))
    compiler = next(original)
    flags: list[str] = []
    for flag in original:
        if flag in ("-o", "-c"):
            next(original)
        else:
            flags.append(flag)
    for name in ("test-metal-ablation", "test-metal-cpu-exact-int"):
        binary = build / "bin" / name
        command = [compiler, *flags, str(root / "tests" / f"{name}.cpp"),
                   "-I" + str(root / "ggml/src/ggml-gemmini"), "-o", str(binary),
                   "-L" + str(build / "bin"), "-Wl,-rpath," + str(build / "bin"),
                   "-lggml-gemmini", "-lggml", "-lggml-base", "-lggml-gemmini-utils", "-lggml-cpu"]
        subprocess.run(["rtk", "proxy", *command], cwd=entry["directory"], check=True)
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                       "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"}
        subprocess.run(["rtk", "proxy", str(binary)], env=environment, check=True)


if __name__ == "__main__":
    main()
