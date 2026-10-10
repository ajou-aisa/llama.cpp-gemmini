from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    source_root, output = (Path(s) for s in sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    tokenizer_path = source_root / "models/gpt2-base/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for split in ("valid", "test"):
        source = source_root / f"wikitext-2-raw/wiki.{split}.raw"
        tokens = np.asarray(tokenizer.encode(source.read_text(), add_special_tokens=False).ids, dtype="<i4")
        tokens[:8192].tofile(output / f"{split}.i32")
        record = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                  "total_tokens": len(tokens), "saved_tokens": min(len(tokens), 8192)}
        (output / f"{split}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
