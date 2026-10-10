"""Read the original FP GGUF sequentially and decode stored HP1 weights."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import BinaryIO, Final

import numpy as np
from numpy.typing import NDArray

ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType, ReaderTensor

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class InputError(Exception):
    detail: str


@dataclass(frozen=True, slots=True)
class FloatTensor:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int


class FloatStream:
    """Accumulate a whole-file digest while consuming the GGUF once."""

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.position = 0
        self.digest = hashlib.sha256()

    def read(self, size: int) -> bytes:
        data = self.source.read(size)
        if len(data) != size:
            raise InputError(f"Truncated FP GGUF at byte {self.position}: wanted {size}, got {len(data)}")
        self.digest.update(data)
        self.position += size
        return data

    def integer(self, size: int) -> int:
        return int.from_bytes(self.read(size), "little")

    def string(self) -> str:
        return self.read(self.integer(8)).decode("utf-8")

    def skip_value(self, kind: int) -> None:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind in sizes:
            self.read(sizes[kind])
        elif kind == 8:
            self.read(self.integer(8))
        elif kind == 9:
            subtype, count = self.integer(4), self.integer(8)
            for _ in range(count):
                self.skip_value(subtype)
        else:
            raise InputError(f"Unknown GGUF metadata type {kind}")

    def header(self) -> list[FloatTensor]:
        if self.read(4) != b"GGUF" or self.integer(4) != 3:
            raise InputError("Expected little-endian GGUF version 3")
        tensors, fields = self.integer(8), self.integer(8)
        alignment = 32
        for _ in range(fields):
            name, kind = self.string(), self.integer(4)
            if name == "general.alignment":
                if kind != 4:
                    raise InputError("GGUF alignment must be UINT32")
                alignment = self.integer(4)
            else:
                self.skip_value(kind)
        entries: list[FloatTensor] = []
        for _ in range(tensors):
            name = self.string()
            shape = tuple(self.integer(8) for _ in range(self.integer(4)))
            kind, offset = self.integer(4), self.integer(8)
            if kind not in (0, 1):
                raise InputError(f"FP reference contains quantized tensor {name}: type {kind}")
            entries.append(FloatTensor(name, shape, np.dtype("<f4" if kind == 0 else "<f2"), offset))
        if alignment <= 0:
            raise InputError("Invalid GGUF alignment")
        self.read((-self.position) % alignment)
        return sorted(entries, key=lambda tensor: tensor.offset)

    def skip_to(self, offset: int) -> None:
        if offset < self.position:
            raise InputError("Overlapping or unsorted tensor offsets")
        while self.position < offset:
            self.read(min(offset - self.position, 1024 * 1024))

    def finish(self) -> str:
        while data := self.source.read(1024 * 1024):
            self.digest.update(data)
            self.position += len(data)
        return self.digest.hexdigest()


def hp1_values(raw: NDArray[np.uint8], bits: int) -> tuple[FloatArray, NDArray[np.int16]]:
    width = 24 if bits == 4 else 40
    code_bytes = 16 if bits == 4 else 32
    blocks = raw.reshape(-1, width)
    exponent = blocks[:, code_bytes:code_bytes + 2].copy().view("<i2").reshape(-1)
    channel = blocks[:, -4:].copy().view("<f4").reshape(-1)
    if np.any((exponent < 0) & (exponent != -32768)):
        raise InputError("HP1 contains an invalid negative exponent")
    scale = np.ldexp(channel, np.where(exponent == -32768, 0, exponent).astype(np.int32))
    scale[exponent == -32768] = 0
    if bits == 4:
        packed = blocks[:, :16]
        codes = np.concatenate((packed & 15, packed >> 4), axis=1).astype(np.int16) - 8
    else:
        codes = blocks[:, :32].copy().view(np.int8).astype(np.int16)
    return (codes.astype(np.float32) * scale[:, None]).reshape(-1), codes.reshape(-1)


def quant_values(tensor: ReaderTensor, start: int, end: int) -> tuple[FloatArray, NDArray[np.int16] | None]:
    raw = tensor.data.reshape(tensor.n_elements // int(tensor.shape[0]), -1)[start:end]
    if tensor.tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
        return raw.astype(np.float32).reshape(-1), None
    if tensor.tensor_type not in (GGMLQuantizationType.Q4_HP1, GGMLQuantizationType.Q8_HP1):
        raise InputError(f"Unexpected weight type: {tensor.name} {tensor.tensor_type.name}")
    return hp1_values(raw, 4 if tensor.tensor_type == GGMLQuantizationType.Q4_HP1 else 8)


def group_name(tensor: FloatTensor) -> str:
    if len(tensor.shape) == 2 and tensor.name.startswith("blk."):
        return f"B{int(tensor.name.split('.')[1]):02d}"
    if tensor.name == "token_embd.weight":
        return "E"
    return "auxiliary"
