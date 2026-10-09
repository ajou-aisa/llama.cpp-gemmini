#!/usr/bin/env python3
# How to run: python3 -B tests/test-raw-logits-compare.py
"""scripts/eval/compare_raw_logits.py: equal payloads PASS; a differing value (signed zero, NaN payload, ordinary
value), a size, metadata or fingerprint inconsistency FAIL with the first mismatch; a missing payload or missing
metadata is NOT_RUN, never an equality PASS."""
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
           fingerprint: str | None = None) -> Path:
    directory = root / name / "native"
    (directory / "chunk-0").mkdir(parents=True)
    payload = b"".join(struct.pack("<I", v) for v in values)
    if write:
        (directory / "chunk-0/logits-f32.bin").write_bytes(payload)
    meta = {"file": "logits-f32.bin", "bytes": len(payload), "values": len(values), "dtype": "float32",
            "byte_order": "little", "shape": [len(values)], "order": "token_id", "scope": "last_sample_input"}
    chunk = {"chunk_id": 0, "logits_fingerprint": fingerprint or fnv1a64(payload)}
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
    print("RAW_LOGITS_COMPARE_PASS")


if __name__ == "__main__":
    main()
