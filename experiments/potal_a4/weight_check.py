from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

from .model import Weights
from .run import digest


def main() -> None:
    model, library, output = (Path(arg) for arg in sys.argv[1:])
    decoded = Weights(model)
    native = ctypes.CDLL(str(library)).dequantize_row_q4_hp1
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    native.restype = None
    compared = 0
    tensor_count = 0
    for tensor in GGUFReader(model).tensors:
        if tensor.tensor_type != GGMLQuantizationType.Q4_HP1:
            continue
        shape = tuple(int(v) for v in tensor.shape[::-1])
        values = decoded.values[tensor.name].reshape(-1, shape[-1])
        raw_rows = tensor.data.reshape(len(values), -1)
        for row in sorted({0, len(values) // 2, len(values) - 1}):
            raw = np.ascontiguousarray(raw_rows[row])
            actual = np.empty(shape[-1], dtype=np.float32)
            native(raw.ctypes.data, actual.ctypes.data, actual.size)
            np.testing.assert_array_equal(actual, values[row])
            compared += actual.size
        tensor_count += 1
    report = {"native_decoder": str(library), "library_sha256": digest(library),
              "model_sha256": digest(model), "compared_weights": compared,
              "tensors": tensor_count, "bitwise_equal": True}
    with output.open("x") as target:
        json.dump(report, target, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
