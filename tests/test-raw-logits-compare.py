#!/usr/bin/env python3
# How to run: python3 -B tests/test-raw-logits-compare.py
"""scripts/eval/compare_raw_logits.py: equal payloads PASS; a differing value (signed zero, NaN payload, ordinary
value), a size, metadata or fingerprint inconsistency FAIL with the first mismatch; a missing payload or missing
metadata is NOT_RUN, never an equality PASS. Per-step dumps (one row per sampled step) compare the same way, locate a
mismatch by step and token, and check the recorded fingerprint against their last row."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

TOOL = Path(__file__).resolve().parents[1] / "scripts/eval/compare_raw_logits.py"
sys.path.insert(0, str(TOOL.parent))
from compare_raw_logits import fnv1a64  # noqa: E402


def native(root: Path, name: str, values: list[int], dump: dict | bool | None = None, write: bool = True,
           fingerprint: str | None = None, steps: int = 0) -> Path:
    """A collection whose dump is one row (steps=0, shape [V]) or one row per step (shape [steps, V])."""
    directory = root / name / "native"
    (directory / "chunk-0").mkdir(parents=True)
    payload = b"".join(struct.pack("<I", v) for v in values)
    if write:
        (directory / "chunk-0/logits-f32.bin").write_bytes(payload)
    vocab = len(values) // steps if steps else len(values)
    meta = {"file": "logits-f32.bin", "bytes": len(payload), "values": len(values), "dtype": "float32",
            "byte_order": "little", "shape": [steps, vocab] if steps else [len(values)],
            "order": "step,token_id" if steps else "token_id", "scope": "every_sample_input" if steps else "last_sample_input"}
    chunk = {"chunk_id": 0, "logits_fingerprint": fingerprint or fnv1a64(payload[-4 * vocab:])}
    if dump is not False:
        chunk["logits_dump"] = {**meta, **(dump if isinstance(dump, dict) else {})}
    (directory / "workload.json").write_text(json.dumps({"chunks": [chunk]}))
    return directory


def run(a: Path, b: Path) -> tuple[int, dict]:
    result = subprocess.run([sys.executable, "-B", str(TOOL), str(a), str(b)], capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


def expect(name: str, a: Path, b: Path, status: str, code: int, check=lambda r: True) -> None:
    rc, result = run(a, b)
    if rc != code or result["status"] != status or not check(result):
        raise SystemExit(f"FAIL {name}: rc={rc} {json.dumps(result)[:400]}")
    print(f"ok {name}: {result['status']} {result.get('first_mismatch_index', '')}")


def main() -> None:
    base = [0x3F800000, 0x00000000, 0xC2C80000, 0x7FC00000]  # 1.0, +0.0, -100.0, quiet NaN
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a = native(root, "a", base)
        expect("identical payloads", a, native(root, "b", base), "PASS", 0, lambda r: r["values_compared"] == 4)
        expect("+0.0 vs -0.0 compares by bits", a, native(root, "c", [base[0], 0x80000000, *base[2:]]), "FAIL", 1,
               lambda r: r["first_mismatch_index"] == 1 and r["first_mismatch_bits"] == ["0x00000000", "0x80000000"])
        expect("NaN payloads are not normalized", a, native(root, "d", [*base[:3], 0x7FC00001]), "FAIL", 1,
               lambda r: r["first_mismatch_index"] == 3)
        expect("ordinary value difference", a, native(root, "e", [base[0], base[1], 0xC2C80001, base[3]]), "FAIL",
               1, lambda r: r["first_mismatch_index"] == 2 and r["differing_values"] == 1)
        expect("missing payload is NOT_RUN", a, native(root, "f", base, write=False), "NOT_RUN", 2)
        expect("missing metadata is NOT_RUN", a, native(root, "g", base, dump=False), "NOT_RUN", 2)
        expect("size inconsistent with metadata", a, native(root, "h", base, dump={"bytes": 12, "values": 3,
               "shape": [3]}), "FAIL", 1)
        expect("dtype differs", a, native(root, "i", base, dump={"dtype": "float16"}), "FAIL", 1)
        expect("scope differs", a, native(root, "j", base, dump={"scope": "last_decode_output"}), "FAIL", 1)
        expect("payload does not match its fingerprint", a, native(root, "k", base, fingerprint="0" * 16),
               "FAIL", 1)
        # One row per sampled step: rows of 4 values, step-major.
        steps = base + [0x40000000, 0x40400000, 0x40800000, 0x80000000] + [0x3F000000] * 4
        m = native(root, "m", steps, steps=3)
        expect("per-step dumps: identical payloads", m, native(root, "n", steps, steps=3), "PASS", 0,
               lambda r: r["steps_compared"] == 3 and r["values_compared"] == 12)
        changed = steps[:5] + [0x40400001] + steps[6:]
        expect("per-step dumps: mismatch located by step and token", m, native(root, "o", changed, steps=3), "FAIL", 1,
               lambda r: r["first_mismatch_index"] == 5 and r["first_mismatch_step"] == 1 and
               r["first_mismatch_token"] == 1 and r["first_mismatch_bits"] == ["0x40400000", "0x40400001"])
        expect("per-step dumps: the fingerprint hashes the last row only", m,
               native(root, "p", steps, steps=3, fingerprint=fnv1a64(b"".join(struct.pack("<I", v) for v in steps))),
               "FAIL", 1, lambda r: "last-row FNV" in r["reasons"][0])
        expect("per-step dumps: shape inconsistent with the values", m,
               native(root, "q", steps, steps=3, dump={"shape": [4, 4]}), "FAIL", 1)
        expect("one-row dump against a per-step dump", a, native(root, "r", base, steps=1), "FAIL", 1,
               lambda r: "metadata differ" in r["reasons"][0])
    print("RAW_LOGITS_COMPARE_PASS")


if __name__ == "__main__":
    main()
