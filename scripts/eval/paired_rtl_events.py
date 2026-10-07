#!/usr/bin/env python3
# How to run:
#   python3 -B scripts/eval/paired_rtl_events.py capture --build-root HOST_TEST_OUT [...] --out OBSERVE_DIR
#   python3 -B scripts/eval/paired_rtl_events.py diff --base OBSERVE_V0 --candidate OBSERVE_P1 \
#       [--candidate-physical-acc-rows PROFILE=ROWS ...] --out report.json
#   python3 -B scripts/eval/paired_rtl_events.py selftest
"""Passive RTL event capture and slot-normalized V0/P1 comparison for the paired-microtile P1 regression.

capture runs IM2P.sim's current_rtl_certificate.capture() (passive probe, runtime-log identity check) on every
passing profile of a gemmini_build host-test root. It reads no corpus authority and certifies nothing.

diff requires identical captured corpora and identical events.csv rows per profile. The only rewrite is the ACC row
of store_dma addresses (and of load_dma addresses whose ACC bit is set): LoopMatmul rotates output slots by
max_acc_addr/2, so a larger physical accumulator moves slot 1. The row maps to row - slot*(physical/2) +
slot*(compat/2) with slot = row >= physical/2, where physical is that of the build being decoded; the address bits
above the row field are compared unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ACC_BIT = 1 << 31


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paired_rtl_events: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def address_bits(rows: int) -> int:
    return max(1, (rows - 1).bit_length())


def normalize_address(raw: int, scratchpad_rows: int, compat_rows: int, physical_rows: int) -> str:
    bits = max(address_bits(scratchpad_rows), address_bits(physical_rows))
    row = raw & ((1 << bits) - 1)
    require(row < physical_rows, f"ACC row {row} outside {physical_rows} physical rows")
    slot = int(row >= physical_rows // 2)
    return f"acc:{raw >> bits:x}:{row - slot * (physical_rows // 2) + slot * (compat_rows // 2)}"


def normalize_rows(rows: list[list[str]], meta: dict[str, Any], physical_rows: int) -> list[list[str]]:
    result = []
    for row in rows:
        if len(row) == 6 and row[0].isdigit() and (
                row[2] == "store_dma" or (row[2] == "load_dma" and int(row[3]) & ACC_BIT)):
            row = [*row[:3], normalize_address(int(row[3]), meta["scratchpad_rows"], meta["compat_acc_rows"],
                                               physical_rows), *row[4:]]
        result.append(row)
    return result


def compare_rows(base: list[list[str]], candidate: list[list[str]]) -> dict[str, Any]:
    difference = next((index for index, (left, right) in enumerate(zip(base, candidate)) if left != right), None)
    if difference is None and len(base) != len(candidate):
        difference = min(len(base), len(candidate))
    kinds = Counter(row[2] if row and row[0].isdigit() and len(row) > 2 else row[0] for row in base if row)
    report: dict[str, Any] = {"rows": [len(base), len(candidate)], "identical": difference is None,
                              "kinds": dict(sorted(kinds.items()))}
    if difference is not None:
        report["first_difference"] = {
            "index": difference,
            "base": base[difference] if difference < len(base) else None,
            "candidate": candidate[difference] if difference < len(candidate) else None,
        }
    return report


def capture(build_roots: list[Path], out: Path, im2p: Path) -> None:
    sys.path.insert(0, str(im2p))
    from sim.tests.cycle import current_rtl_certificate as certificate

    out.mkdir(parents=True, exist_ok=True)
    for root in build_roots:
        manifest = json.loads((root / "result.json").read_text())
        require(manifest.get("stage") == "host-test", f"{root} is not a host-test root")
        for profile in manifest["profiles"]:
            destination = out / profile["profile"]
            destination.mkdir()
            cases = certificate.capture(profile, destination)
            resolved = json.loads(Path(profile["resolved_profile"]).read_text())
            memory = resolved["memory"]
            events = destination / "capture" / "events.csv"
            (destination / "capture-meta.json").write_text(json.dumps({
                "profile": profile["profile"], "build_root": str(root), "cases": len(cases),
                "compat_acc_rows": memory["accumulator_rows"],
                "scratchpad_rows": memory["bank_count"] * memory["bank_rows"],
                "events_sha256": sha256(events), "events_bytes": events.stat().st_size,
            }, indent=2, sort_keys=True) + "\n")
            print(f"{profile['profile']}: {len(cases)} cases, {events}")


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as stream:
        return [row for row in csv.reader(stream) if row]


def diff(base: Path, candidate: Path, physical: dict[str, int], out: Path) -> bool:
    profiles = sorted(path.name for path in base.iterdir() if (path / "capture-meta.json").is_file())
    require(bool(profiles), f"no captured profiles under {base}")
    report: dict[str, Any] = {"base": str(base), "candidate": str(candidate), "profiles": {}}
    for name in profiles:
        left_meta = json.loads((base / name / "capture-meta.json").read_text())
        if not (candidate / name / "capture-meta.json").is_file():
            report["profiles"][name] = {"status": "MISSING_CANDIDATE"}
            continue
        right_meta = json.loads((candidate / name / "capture-meta.json").read_text())
        candidate_rows = physical[name] if name in physical else int(right_meta["compat_acc_rows"])
        corpus_equal = (json.loads((base / name / "captured-corpus.json").read_text())
                        == json.loads((candidate / name / "captured-corpus.json").read_text()))
        rows = compare_rows(
            normalize_rows(read_rows(base / name / "capture" / "events.csv"), left_meta,
                           left_meta["compat_acc_rows"]),
            normalize_rows(read_rows(candidate / name / "capture" / "events.csv"), right_meta,
                           candidate_rows))
        same_layout = all(left_meta[key] == right_meta[key] for key in ("compat_acc_rows", "scratchpad_rows"))
        identical = corpus_equal and rows["identical"] and same_layout
        report["profiles"][name] = {
            "status": "IDENTICAL" if identical else "DIFFERENT", "corpus_identical": corpus_equal,
            "compat_layout_identical": same_layout, "events": rows,
            "candidate_physical_acc_rows": candidate_rows,
        }
    unused = sorted(set(physical) - set(profiles))
    require(not unused, f"physical rows given for uncaptured profiles: {unused}")
    report["status"] = "PASS" if all(item["status"] == "IDENTICAL" for item in report["profiles"].values()) else "FAIL"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: item["status"] for name, item in report["profiles"].items()} | {"status": report["status"]}))
    return report["status"] == "PASS"


def selftest() -> None:
    sp, compat, physical = 16384, 1024, 2048
    for row in (0, 100, 511):  # slot 0 keeps its row
        assert normalize_address(ACC_BIT | row, sp, compat, physical) == normalize_address(ACC_BIT | row, sp, compat, compat)
    # slot 1 starts at physical/2 in the candidate and at compat/2 in the base
    assert normalize_address(ACC_BIT | 1024 + 7, sp, compat, physical) == normalize_address(ACC_BIT | 512 + 7, sp, compat, compat)
    assert normalize_address(ACC_BIT | 1024 + 7, sp, compat, physical) != normalize_address(ACC_BIT | 7, sp, compat, compat)
    flagged = ACC_BIT | 1 << 30  # accumulate bit stays part of the comparison
    assert normalize_address(flagged | 3, sp, compat, physical) != normalize_address(ACC_BIT | 3, sp, compat, physical)
    meta = {"scratchpad_rows": sp, "compat_acc_rows": compat}
    base = [["CASE", "1", "16"], ["1", "40", "store_dma", str(ACC_BIT | 512 + 1), "16", "0"],
            ["1", "41", "load_dma", "5", "16", "4096"]]
    moved = [["CASE", "1", "16"], ["1", "40", "store_dma", str(ACC_BIT | 1024 + 1), "16", "0"],
             ["1", "41", "load_dma", "5", "16", "4096"]]
    assert compare_rows(normalize_rows(base, meta, compat), normalize_rows(moved, meta, physical))["identical"]
    late = [*moved[:1], ["1", "42", *moved[1][2:]], moved[2]]
    report = compare_rows(normalize_rows(base, meta, compat), normalize_rows(late, meta, physical))
    assert not report["identical"] and report["first_difference"]["index"] == 1
    assert not compare_rows(base, base[:-1])["identical"]
    print("selftest ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Passive RTL event capture and slot-normalized V0/P1 comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--build-root", type=Path, action="append", required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    diff_parser = commands.add_parser("diff")
    diff_parser.add_argument("--base", type=Path, required=True)
    diff_parser.add_argument("--candidate", type=Path, required=True)
    diff_parser.add_argument("--candidate-physical-acc-rows", action="append", default=[],
                             metavar="PROFILE=ROWS")
    diff_parser.add_argument("--out", type=Path, required=True)
    commands.add_parser("selftest")
    arguments = parser.parse_args()
    if arguments.command == "selftest":
        selftest()
        return 0
    if arguments.command == "capture":
        capture([root.resolve() for root in arguments.build_root], arguments.out.resolve(), arguments.im2p.resolve())
        return 0
    physical = {}
    for item in arguments.candidate_physical_acc_rows:
        name, _, rows = item.partition("=")
        require(bool(name) and rows.isdigit() and int(rows) > 0, f"bad PROFILE=ROWS: {item}")
        physical[name] = int(rows)
    return 0 if diff(arguments.base.resolve(), arguments.candidate.resolve(), physical, arguments.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
