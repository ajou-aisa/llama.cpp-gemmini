#!/usr/bin/env python3
"""Strict JSON helpers for residual profiles; Python 3.9+ and standard library only."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Union

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, List["JsonValue"], Dict[str, "JsonValue"]]


class CycleSchemaError(Exception):
    """A cycle JSONL boundary failure tied to its physical input line."""

    __slots__ = ("line_number", "detail")

    line_number: int
    detail: str

    def __init__(self, line_number: int, detail: str) -> None:
        self.line_number = line_number
        self.detail = detail
        super().__init__(line_number, detail)

    def __str__(self) -> str:
        return f"line {self.line_number}: {self.detail}"


class _DuplicateKeyError(Exception):
    pass


class _InvalidConstantError(Exception):
    pass


def _pairs_to_mapping(pairs: List[Tuple[str, JsonValue]]) -> Dict[str, JsonValue]:
    result: Dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> JsonValue:
    raise _InvalidConstantError(value)


def _require(record: Mapping[str, JsonValue], name: str, line_number: int) -> JsonValue:
    if name not in record:
        raise CycleSchemaError(line_number, f"missing required field {name!r}")
    return record[name]


def _optional_string(record: Mapping[str, JsonValue], name: str, line_number: int) -> Optional[str]:
    value = _require(record, name, line_number)
    if value is None:
        return None
    if type(value) is not str:
        raise CycleSchemaError(line_number, f"field {name!r} must be a string or null")
    return value


def _optional_integer(record: Mapping[str, JsonValue], name: str, line_number: int) -> Optional[int]:
    value = _require(record, name, line_number)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise CycleSchemaError(line_number, f"field {name!r} must be a non-negative integer or null")
    return value


def parse_json_line(line: str, line_number: int) -> JsonValue:
    """Decode JSON while rejecting duplicate keys and non-JSON constants."""
    try:
        return json.loads(line, object_pairs_hook=_pairs_to_mapping, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise CycleSchemaError(line_number, f"malformed JSON at column {error.colno}") from None
    except _DuplicateKeyError as error:
        raise CycleSchemaError(line_number, f"duplicate key {error.args[0]!r}") from None
    except _InvalidConstantError as error:
        raise CycleSchemaError(line_number, f"invalid JSON constant {error.args[0]!r}") from None
