#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
# How to run: python3 tests/parse-stripe-ws-profile.py PATH [--schema-only]

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeGuard


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

COMMON_REQUIRED_FIELDS = frozenset({
    "layer", "call_id", "stripe_count", "slot", "row_start", "row_count",
    "I", "J", "K", "original_tile_I", "call_tile_I", "tile_J", "tile_K", "submit_start",
    "submit_end", "submit_cycles", "slot_wait_start", "slot_wait_end", "slot_wait_cycles",
    "call_wall", "call_load", "call_exe", "call_store", "counter_width_bits", "units", "valid",
})
INTEGER_FIELDS = frozenset({
    "call_id", "stripe_count", "slot", "row_start", "row_count", "I", "J", "K",
    "original_tile_I", "call_tile_I", "tile_J", "tile_K", "submit_start", "submit_end",
    "submit_cycles", "slot_wait_start", "slot_wait_end", "slot_wait_cycles", "call_wall",
    "call_load", "call_exe", "call_store", "counter_width_bits",
})


class SchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StripeRecord:
    layer: str
    call_id: int
    stripe_idx: int
    stripe_count: int
    slot: int
    row_start: int
    row_count: int
    I: int
    J: int
    K: int
    original_tile_I: int
    call_tile_I: int
    tile_J: int
    tile_K: int
    submit_start: int
    submit_end: int
    submit_cycles: int
    slot_wait_start: int
    slot_wait_end: int
    slot_wait_cycles: int
    call_wall: int
    call_load: int
    call_exe: int
    call_store: int
    counter_width_bits: int
    units: str
    valid: bool


def is_json_value(value: object) -> TypeGuard[JsonValue]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def parse_json_record(line: str, line_number: int) -> dict[str, JsonValue]:
    raw: object = json.loads(line)
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise SchemaError(f"line {line_number}: record must be a JSON object")
    record: dict[str, JsonValue] = {}
    for key, value in raw.items():
        if not is_json_value(value):
            raise SchemaError(f"line {line_number}: invalid JSON value for {key}")
        record[key] = value
    return record


def required_string(record: dict[str, JsonValue], name: str, line_number: int) -> str:
    value = record.get(name)
    if not isinstance(value, str):
        raise SchemaError(f"line {line_number}: {name} must be a string")
    return value


def required_integer(record: dict[str, JsonValue], name: str, line_number: int) -> int:
    value = record.get(name)
    if type(value) is not int:
        raise SchemaError(f"line {line_number}: {name} must be an integer")
    return value


def required_boolean(record: dict[str, JsonValue], name: str, line_number: int) -> bool:
    value = record.get(name)
    if type(value) is not bool:
        raise SchemaError(f"line {line_number}: {name} must be a boolean")
    return value


def parse_record(record: dict[str, JsonValue], line_number: int) -> StripeRecord:
    is_v2 = "op" in record or "stripe_id" in record
    identity_fields = (
        frozenset({"schema", "version", "op", "stripe_id"})
        if is_v2
        else frozenset({"event", "stripe_idx"})
    )
    missing = (COMMON_REQUIRED_FIELDS | identity_fields) - record.keys()
    if missing:
        raise SchemaError(f"line {line_number}: missing fields: {', '.join(sorted(missing))}")
    if is_v2:
        if required_string(record, "schema", line_number) != "gemmini.cycle":
            raise SchemaError(f"line {line_number}: schema must be gemmini.cycle")
        if required_integer(record, "version", line_number) != 2:
            raise SchemaError(f"line {line_number}: version must be 2")
        if required_string(record, "op", line_number) != "gemmini.ws_stripe":
            raise SchemaError(f"line {line_number}: op must be gemmini.ws_stripe")
        stripe_idx = required_integer(record, "stripe_id", line_number)
    else:
        if required_string(record, "event", line_number) != "stripe_ws_profile":
            raise SchemaError(f"line {line_number}: event must be stripe_ws_profile")
        stripe_idx = required_integer(record, "stripe_idx", line_number)
    integer_values = {
        name: required_integer(record, name, line_number) for name in INTEGER_FIELDS
    }
    integer_values["stripe_idx"] = stripe_idx
    if any(value < 0 for value in integer_values.values()):
        raise SchemaError(f"line {line_number}: integer fields must be nonnegative")
    units = required_string(record, "units", line_number)
    if units != "cycles":
        raise SchemaError(f"line {line_number}: units must be cycles")
    layer = required_string(record, "layer", line_number)
    valid = required_boolean(record, "valid", line_number)
    call_wall = integer_values["call_wall"]
    if valid != (call_wall <= 2**32 - 1):
        raise SchemaError(f"line {line_number}: valid does not match call_wall range")
    if integer_values["counter_width_bits"] != 32:
        raise SchemaError(f"line {line_number}: counter_width_bits must be 32")
    if integer_values["stripe_count"] == 0 or integer_values["row_count"] == 0:
        raise SchemaError(f"line {line_number}: stripe_count and row_count must be positive")
    if integer_values["stripe_idx"] >= integer_values["stripe_count"]:
        raise SchemaError(f"line {line_number}: stripe_idx outside stripe_count")
    if integer_values["slot"] != integer_values["stripe_idx"] % 2:
        raise SchemaError(f"line {line_number}: slot must match stripe_idx modulo two")
    if integer_values["call_tile_I"] == 0 or integer_values["original_tile_I"] == 0:
        raise SchemaError(f"line {line_number}: tile sizes must be positive")
    if integer_values["row_start"] + integer_values["row_count"] > integer_values["I"]:
        raise SchemaError(f"line {line_number}: row range exceeds I")
    if integer_values["submit_end"] < integer_values["submit_start"]:
        raise SchemaError(f"line {line_number}: submit range is inverted")
    if integer_values["submit_cycles"] != integer_values["submit_end"] - integer_values["submit_start"]:
        raise SchemaError(f"line {line_number}: submit_cycles does not match submit range")
    if integer_values["slot_wait_end"] < integer_values["slot_wait_start"]:
        raise SchemaError(f"line {line_number}: slot wait range is inverted")
    if integer_values["slot_wait_cycles"] != integer_values["slot_wait_end"] - integer_values["slot_wait_start"]:
        raise SchemaError(f"line {line_number}: slot_wait_cycles does not match slot wait range")
    return StripeRecord(
        layer=layer,
        call_id=integer_values["call_id"],
        stripe_idx=integer_values["stripe_idx"],
        stripe_count=integer_values["stripe_count"],
        slot=integer_values["slot"],
        row_start=integer_values["row_start"],
        row_count=integer_values["row_count"],
        I=integer_values["I"],
        J=integer_values["J"],
        K=integer_values["K"],
        original_tile_I=integer_values["original_tile_I"],
        call_tile_I=integer_values["call_tile_I"],
        tile_J=integer_values["tile_J"],
        tile_K=integer_values["tile_K"],
        submit_start=integer_values["submit_start"],
        submit_end=integer_values["submit_end"],
        submit_cycles=integer_values["submit_cycles"],
        slot_wait_start=integer_values["slot_wait_start"],
        slot_wait_end=integer_values["slot_wait_end"],
        slot_wait_cycles=integer_values["slot_wait_cycles"],
        call_wall=call_wall,
        call_load=integer_values["call_load"],
        call_exe=integer_values["call_exe"],
        call_store=integer_values["call_store"],
        counter_width_bits=integer_values["counter_width_bits"],
        units=units,
        valid=valid,
    )


def load_records(path: Path) -> list[StripeRecord]:
    records: list[StripeRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise SchemaError(f"line {line_number}: empty JSONL record")
            records.append(parse_record(parse_json_record(line, line_number), line_number))
    if not records:
        raise SchemaError("input contains no records")
    return records


def split_groups(records: list[StripeRecord]) -> list[list[StripeRecord]]:
    groups: list[list[StripeRecord]] = []
    seen_call_ids: set[int] = set()
    for record in records:
        if not groups or groups[-1][0].call_id != record.call_id:
            if record.call_id in seen_call_ids:
                raise SchemaError("call_id group is not contiguous")
            seen_call_ids.add(record.call_id)
            groups.append([])
        groups[-1].append(record)
    for previous, current in zip(groups, groups[1:]):
        if current[0].call_id <= previous[0].call_id:
            raise SchemaError("call_id must increase between groups")
    return groups


def validate_group(group: list[StripeRecord]) -> None:
    first = group[0]
    if len(group) != first.stripe_count:
        raise SchemaError("group length does not match stripe_count")
    static_fields = (
        "layer", "stripe_count", "I", "J", "K", "original_tile_I", "tile_J", "tile_K",
        "units", "call_wall", "call_load", "call_exe", "call_store"
    )
    for index, record in enumerate(group):
        if record.stripe_idx != index:
            raise SchemaError("stripe_idx values must be ordered and contiguous")
        if any(getattr(record, field) != getattr(first, field) for field in static_fields):
            raise SchemaError("group metadata is inconsistent")


def validate_coverage(group: list[StripeRecord]) -> None:
    expected_row = 0
    for record in group:
        if record.row_start != expected_row:
            raise SchemaError("row ranges contain a gap or overlap")
        expected_row += record.row_count
    if expected_row != group[0].I:
        raise SchemaError("row ranges do not cover I")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stripe_ws_profile JSONL records")
    parser.add_argument("path", type=Path)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--expect-I", dest="expect_i", type=int)
    parser.add_argument("--expect-stripes", type=int)
    parser.add_argument("--expect-final-row-count", type=int)
    parser.add_argument("--expect-final-call-tile-I", dest="expect_final_call_tile_i", type=int)
    return parser.parse_args(argv)


def validate_expectations(groups: list[list[StripeRecord]], args: argparse.Namespace) -> None:
    for group in groups:
        first = group[0]
        if args.expect_i is not None and first.I != args.expect_i:
            raise SchemaError(f"expected I={args.expect_i}, got {first.I}")
        if args.expect_stripes is not None and len(group) != args.expect_stripes:
            raise SchemaError(f"expected {args.expect_stripes} stripes, got {len(group)}")
        if args.expect_final_row_count is not None and group[-1].row_count != args.expect_final_row_count:
            raise SchemaError("final row_count does not match expectation")
        if (
            args.expect_final_call_tile_i is not None
            and group[-1].call_tile_I != args.expect_final_call_tile_i
        ):
            raise SchemaError("final call_tile_I does not match expectation")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        records = load_records(args.path)
        groups = split_groups(records)
        for group in groups:
            validate_group(group)
            if not args.schema_only:
                validate_coverage(group)
        validate_expectations(groups, args)
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"validated {len(records)} records in {len(groups)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
