"""Compare nine native FP32 and CPU graph logits using only the standard library."""

from __future__ import annotations

from array import array
from math import isfinite, sqrt
from pathlib import Path
from typing import Final

OUTPUT: Final = Path("/media/alveo/LaCie/fp32-5c-d8/sw")
VOCAB: Final = 151936


def load(path: Path) -> array[float]:
    """Read one complete little-endian binary32 vocabulary row."""
    values: array[float] = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, VOCAB)
        if stream.read(1):
            raise RuntimeError(path)
    return values


def main() -> None:
    """Print finite-logit and token sanity evidence, not a bit-exact claim."""
    print("graph,logits,max_abs,rmse,pearson,top1_fp32,top1_cpu,bitwise_equal")
    for graph in range(9):
        a = load(OUTPUT / "target" / f"logits-{graph}.bin")
        b = load(OUTPUT / "cpu-target" / f"logits-{graph}.bin")
        if not all(isfinite(x) and isfinite(y) for x, y in zip(a, b, strict=True)):
            raise RuntimeError(graph)
        mean_a, mean_b = sum(a) / VOCAB, sum(b) / VOCAB
        errors = [x - y for x, y in zip(a, b, strict=True)]
        covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
        variance_a = sum((x - mean_a) ** 2 for x in a)
        variance_b = sum((y - mean_b) ** 2 for y in b)
        correlation = covariance / sqrt(variance_a * variance_b)
        top_a = max(range(VOCAB), key=a.__getitem__)
        top_b = max(range(VOCAB), key=b.__getitem__)
        print(
            f"{graph},{VOCAB},{max(map(abs, errors)):.17g},"
            f"{sqrt(sum(x * x for x in errors) / VOCAB):.17g},"
            f"{correlation:.17g},{top_a},{top_b},{int(a == b)}",
        )


if __name__ == "__main__":
    main()
