#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/campaign_verify.py RESULT_DIRECTORY
from __future__ import annotations

import argparse
from pathlib import Path

from eval_common import read_json, require, sha256


def verify(root: Path) -> None:
    manifest = root / "evaluation_manifest.json"
    digest = sha256(manifest)
    checksums = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for line in checksums:
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and relative not in seen, "unsafe or duplicate checksum entry")
        seen.add(relative)
        require(sha256(path) == expected, "artifact hash changed: " + relative)
    for path in root.glob("*.json"):
        if path.name not in ("evaluation_manifest.json", "dataset_manifest.json", "cycle-request.json",
                             "stateful-provider.json", "inference.json"):
            value = read_json(path)
            require(value.get("manifest_sha256") == digest, "measurement manifest binding mismatch: " + path.name)
    require({str(path.relative_to(root)) for path in root.rglob("*")
             if path.is_file() and path.name != "SHA256SUMS"} == seen, "checksum file coverage mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify exact campaign manifest and all artifact hashes.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.output.resolve(strict=True))
    print("PASS: manifest binding and complete SHA256SUMS coverage")
