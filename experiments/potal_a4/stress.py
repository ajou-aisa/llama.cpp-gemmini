from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from .attention import p_quant
from .metrics import Totals
from .stream import stream_quant
from .types import MainRule, PPolicy, StreamPolicy


def main() -> None:
    output = Path(sys.argv[1])
    records = []
    for length in (128, 256, 512, 1024):
        probabilities = np.tril(np.ones((length, length), dtype=np.float32)) / np.arange(1, length + 1, dtype=np.float32)[:, None]
        for width in (32, 64):
            for rows in (0, 32, 8, 1):
                for bits in (4, 6, 8):
                    quantized = p_quant(probabilities, PPolicy(width, rows, bits))
                    measured = Totals().observe(probabilities, quantized).probability(quantized.values)
                    records.append({"case": "causal_uniform", "context": length, "head_width": width,
                                    "p_rows": rows, "p_bits": bits, "last_output": float(quantized.values[-1].sum()),
                                    "metrics": asdict(measured)})
    rng = np.random.default_rng(17)
    ordinary = rng.normal(size=(64, 256)).astype(np.float32)
    for amplitude in (1, 16, 256, 65536):
        x = ordinary.copy()
        x.ravel()[::997] *= amplitude
        for rule in (MainRule.SELECTIVE, MainRule.DIRECT):
            for upper in (0, 1, 2, 8):
                quantized = stream_quant(x, StreamPolicy(64, rule=rule, upper_limbs=upper))
                measured = Totals().observe(x, quantized)
                records.append({"case": "outlier_sweep", "amplitude": amplitude,
                                "rule": rule, "upper_limbs": upper, "metrics": asdict(measured)})
    with output.open("x") as target:
        json.dump(records, target, indent=2)
    print(f"Measured {len(records)} constrained synthetic cases: {output}")


if __name__ == "__main__":
    main()
