#!/usr/bin/env python3
"""Export validated cycle JSONL records to lossless family CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.cycle_schema import CycleSchemaError, RecordType, parse_cycle_jsonl


def _header(prefix: str, name: str) -> str:
    """Encode a source key into a collision-free CSV namespace."""
    return prefix + name.encode("utf-8").hex()


def _cell(value: str) -> str:
    """Render a JSON value without losing null or nested structure."""
    parsed = json.loads(value)
    if parsed is None:
        return ""
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _write_family_csv(path: Path, source: Path, names: List[str], owned: List[Path]) -> None:
    """Write one staged family CSV from its canonical JSONL records."""
    temporary = Path(tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)[1])
    owned.append(temporary)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow([item for name in names for item in (_header("field:", name), _header("present:", name))])
            with source.open("r", encoding="utf-8") as records:
                for line in records:
                    decoded = json.loads(line)
                    writer.writerow([item for name in names for item in (
                        _cell(json.dumps(decoded[name], separators=(",", ":"))) if name in decoded else "",
                        str(name in decoded).lower(),
                    )])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(input_path: Path, output_dir: Path) -> None:
    """Validate the complete input and publish deterministic family CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: Dict[RecordType, Path] = {}
    names: Dict[RecordType, set[str]] = {}
    owned: List[Path] = []
    try:
        for record in parse_cycle_jsonl(input_path):
            family = record.record_type
            if family not in staged:
                staged[family] = Path(tempfile.mkstemp(prefix=f".{family.value}.", suffix=".jsonl.tmp", dir=output_dir)[1])
                owned.append(staged[family])
                names[family] = set()
            decoded = json.loads(record.canonical_json)
            names[family].update(decoded.keys())
            with staged[family].open("a", encoding="utf-8") as stream:
                stream.write(record.canonical_json + "\n")
        destinations = [output_dir / f"{family.value}.csv" for family in staged]
        collisions = [path for path in destinations if path.exists()]
        if collisions:
            raise OSError(f"destination already exists: {collisions[0]}")
        for family in RecordType:
            if family in staged:
                _write_family_csv(output_dir / f"{family.value}.csv", staged[family], sorted(names[family]), owned)
        for path in staged.values():
            path.unlink()
    except (CycleSchemaError, OSError, json.JSONDecodeError):
        for path in owned:
            if path.exists():
                path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Export cycle JSONL to per-family CSV")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        export(arguments.input, arguments.output)
    except (CycleSchemaError, OSError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
