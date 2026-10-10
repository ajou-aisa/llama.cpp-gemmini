from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    binary, output = (Path(arg).resolve() for arg in sys.argv[1:])
    records = []
    for kind in ("narrow", "sparse", "dense", "wide", "carry"):
        for mode in ("clipped", "direct"):
            for repeat in range(3):
                command = [str(binary), mode, kind, "512", "4096"]
                result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
                record = json.loads(result.stdout)
                record["repeat"] = repeat
                records.append(record)
                print(json.dumps(record), flush=True)
    with output.open("x") as target:
        json.dump(records, target, indent=2)


if __name__ == "__main__":
    main()
