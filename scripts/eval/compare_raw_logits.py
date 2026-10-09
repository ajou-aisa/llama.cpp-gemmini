#!/usr/bin/env python3
# How to run: python3 -B scripts/eval/compare_raw_logits.py NATIVE_A NATIVE_B [--output RESULT.json]
"""Byte comparison of the raw FP32 logits two native collections recorded at the same capture point.

Each NATIVE directory is a collection's `native/` output: workload.json names, for its single chunk, the
`logits_dump` (file, bytes, values, dtype, byte_order, shape, order, scope) that tools/eval wrote next to the
chunk's logs, together with the FNV-1a64 `logits_fingerprint` of the same bytes. The payloads are compared byte
by byte without any normalization (signed zeros and NaN payloads compare by bits). Status:
  PASS     both payloads present, metadata identical, each payload matches its own FNV fingerprint, bytes equal;
  FAIL     metadata, size or fingerprint inconsistency, or any differing byte (first index and both bit patterns);
  NOT_RUN  a payload or its metadata is missing (never reported as equality).
Exit code 0 PASS, 1 FAIL, 2 NOT_RUN."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

FIELDS = ("bytes", "values", "dtype", "byte_order", "shape", "order", "scope")


def fnv1a64(payload: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in payload:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def side(native: Path) -> dict:
    workload = native / "workload.json"
    if not workload.is_file():
        return {"missing": f"{workload} absent"}
    chunks = json.loads(workload.read_text()).get("chunks") or []
    if len(chunks) != 1:
        return {"missing": f"{workload}: expected one chunk, found {len(chunks)}"}
    chunk = chunks[0]
    dump = chunk.get("logits_dump")
    if not isinstance(dump, dict):
        return {"missing": f"{workload}: chunk {chunk.get('chunk_id')} has no logits_dump"}
    path = native / f"chunk-{chunk.get('chunk_id')}" / str(dump.get("file"))
    if not path.is_file():
        return {"missing": f"{path} absent"}
    payload = path.read_bytes()
    return {"path": str(path), "metadata": {k: dump.get(k) for k in FIELDS}, "payload": payload,
            "sha256": hashlib.sha256(payload).hexdigest(), "fnv_recorded": chunk.get("logits_fingerprint"),
            "fnv_payload": fnv1a64(payload)}


def compare(native_a: Path, native_b: Path) -> dict:
    a, b = side(native_a), side(native_b)
    result: dict = {"a": str(native_a), "b": str(native_b)}
    missing = [s["missing"] for s in (a, b) if "missing" in s]
    if missing:
        return {**result, "status": "NOT_RUN", "reasons": missing}
    result.update({name: {k: v for k, v in s.items() if k != "payload"} for name, s in (("a_side", a), ("b_side", b))})
    reasons = []
    for name, s in (("a", a), ("b", b)):
        meta = s["metadata"]
        if (meta["dtype"] != "float32" or meta["values"] is None or meta["bytes"] != meta["values"] * 4 or
                meta["shape"] != [meta["values"]] or len(s["payload"]) != meta["bytes"]):
            reasons.append(f"{name}: metadata/size inconsistent ({meta}, {len(s['payload'])} bytes on disk)")
        if s["fnv_recorded"] != s["fnv_payload"]:
            reasons.append(f"{name}: payload FNV {s['fnv_payload']} != recorded {s['fnv_recorded']}")
    if a["metadata"] != b["metadata"]:
        reasons.append(f"metadata differ: {a['metadata']} vs {b['metadata']}")
    if reasons:
        return {**result, "status": "FAIL", "reasons": reasons}
    pa, pb = a["payload"], b["payload"]
    if pa == pb:
        return {**result, "status": "PASS", "bytes_compared": len(pa), "values_compared": len(pa) // 4,
                "sha256": a["sha256"]}
    order = "<" if a["metadata"]["byte_order"] == "little" else ">"
    first = next(i for i, (x, y) in enumerate(zip(pa, pb)) if x != y)
    index = first // 4
    bits = [struct.unpack(order + "I", p[index * 4:index * 4 + 4])[0] for p in (pa, pb)]
    differing = sum(1 for i in range(0, len(pa), 4) if pa[i:i + 4] != pb[i:i + 4])
    return {**result, "status": "FAIL", "reasons": ["payload bytes differ"], "first_mismatch_byte": first,
            "first_mismatch_index": index, "first_mismatch_bits": [f"0x{v:08x}" for v in bits],
            "differing_values": differing}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("native_a", type=Path)
    parser.add_argument("native_b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.native_a, args.native_b)
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return {"PASS": 0, "FAIL": 1}.get(result["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
