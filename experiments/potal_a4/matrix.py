from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys
from typing import assert_never

from .types import MODES, QuantError


class Phase(str, Enum):
    VALID = "valid"
    HOLDOUT = "holdout"
    CONTEXT = "context"
    ALL = "all"


def run_case(task: tuple[Path, Path, str, str, int, int]) -> str:
    root, output, mode, split, context, chunks = task
    target = output / f"{split}{context}-{mode}.json"
    if target.exists():
        raise QuantError(f"Refusing existing result {target}; use a fresh output directory")
    command = [sys.executable, "-m", "experiments.potal_a4.run", str(root / "models/gpt2.Q4_HP1.gguf"),
               str(output / f"{split}.i32"), str(context), str(chunks), mode, str(target)]
    with target.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                timeout=1800, env={**os.environ, "OPENBLAS_NUM_THREADS": "2"})
    if result.returncode:
        raise QuantError(f"Failed {mode}; inspect {target.with_suffix('.log')}")
    return target.name


def main() -> None:
    root, output = (Path(arg) for arg in sys.argv[1:3])
    selected = ("weights_only", "linear_selective", "linear_direct", "original_a4nks", "direct_stripe", "direct_p8", "direct_p6", "direct_limb1")
    phase = Phase(sys.argv[3]) if len(sys.argv) == 4 else Phase.VALID
    match phase:
        case Phase.VALID:
            tasks = [(root, output, mode.name, "valid", 256, 2) for mode in MODES]
        case Phase.HOLDOUT:
            tasks = [(root, output, name, "test", 512, 8) for name in selected]
        case Phase.CONTEXT:
            tasks = [(root, output, name, "test", context, 4) for context in (128, 1024)
                     for name in ("weights_only", "original_a4nks", "direct_p8")]
        case Phase.ALL:
            tasks = [(root, output, mode.name, "valid", 256, 2) for mode in MODES]
            tasks += [(root, output, name, "test", 512, 8) for name in selected]
            tasks += [(root, output, name, "test", context, 4) for context in (128, 1024)
                      for name in ("weights_only", "original_a4nks", "direct_p8")]
        case unreachable:
            assert_never(unreachable)
    with ThreadPoolExecutor(max_workers=2) as workers:
        for result in workers.map(run_case, tasks):
            print(result, flush=True)


if __name__ == "__main__":
    main()
