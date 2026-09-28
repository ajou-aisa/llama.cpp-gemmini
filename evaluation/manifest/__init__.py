"""Validated evaluation identity and exact-file SHA256 binding."""

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scripts.eval.eval_common import Record, decode, integer, require, text


@dataclass(frozen=True, slots=True)
class Manifest:
    model: str
    precision: str
    dim: int
    sha256: str
    seed: int
    chunk_policy: str

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        payload = path.read_bytes()
        value = decode(payload.decode("utf-8"))
        required = {"model", "dataset", "tokenizer_sha256", "chunk_policy", "precision",
                    "dim", "BK", "seed", "git_sha"}
        require(set(value) == required, "manifest fields do not match evaluation manifest v1")
        require(value["dataset"] == "WikiText-2", "unsupported manifest dataset")
        require(integer(value, "BK", 1) == 32, "manifest BK must be 32")
        require(integer(value, "dim") in (16, 32, 64), "unsupported manifest DIM")
        require(value["precision"] in ("A4W4", "A8W8"), "unsupported manifest precision")
        for key, length in (("tokenizer_sha256", 64), ("git_sha", 40)):
            require(re.fullmatch(r"[0-9a-f]{" + str(length) + "}", text(value, key)) is not None,
                    "invalid manifest hash: " + key)
        require(integer(value, "seed") < 4294967295, "manifest seed exceeds explicit native seed domain")
        return cls(text(value, "model"), text(value, "precision"), integer(value, "dim"),
                   sha256(payload).hexdigest(), integer(value, "seed"), text(value, "chunk_policy"))

    def identity(self) -> Record:
        return {"model": self.model, "precision": self.precision, "dim": self.dim,
                "manifest_sha256": self.sha256}
