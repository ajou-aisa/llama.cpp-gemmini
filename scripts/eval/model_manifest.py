#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run:
#   python3 -B scripts/eval/model_manifest.py create --hf-dir models/gpt2/hf --hf-repo openai-community/gpt2 \
#     --hf-revision SHA --f16 models/gpt2/gpt2.fp16.gguf --convert-argv '...' --converter-python .venv/bin/python \
#     --quantizer build-arm64/bin/llama-quantize --quantized Q8_HP1=models/gpt2/gpt2.Q8_HP1.gguf --output model-manifest.json
#   python3 -B scripts/eval/model_manifest.py verify --manifest model-manifest.json --model models/gpt2/gpt2.Q8_HP1.gguf
"""Host-local model provenance: upstream checkpoint, conversion, quantization and GGUF identity of every artifact.

A manifest freezes the model identity that performance and metric runs must share. It records how the files were
made on this host; it never claims equality with a model produced elsewhere. `verify` re-hashes the model file and
returns its manifest entry, so a run cannot silently use different bytes under the same manifest.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from eval_common import Json, Record, read_json, record, require, sha256, text, write_json

REPO: Final = Path(__file__).resolve().parents[2]
SCHEMA: Final = "potal-model-manifest"
UPSTREAM_FILES: Final = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt", "generation_config.json")
GGUF_KEYS: Final = ("general.architecture", "general.name", "general.file_type", "general.quantization_version",
                    "tokenizer.ggml.model", "tokenizer.ggml.pre")


def git(*argv: str, repo: Path = REPO) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()


def gguf_metadata(path: Path) -> Record:
    sys.path.insert(0, str(REPO / "gguf-py"))
    gguf: Any = importlib.import_module("gguf")
    reader: Any = gguf.GGUFReader(path)
    fields: Any = reader.fields

    def value(key: str) -> Json:
        if key not in fields:
            return None
        field: Any = fields[key]
        part: Any = field.parts[field.data[0]]
        return str(bytes(part).decode()) if field.types[0].name == "STRING" else int(part[0])

    architecture = value("general.architecture")
    metadata: Record = {key: value(key) for key in GGUF_KEYS}
    for suffix in ("context_length", "embedding_length", "block_count", "attention.head_count"):
        metadata[f"{architecture}.{suffix}"] = value(f"{architecture}.{suffix}")
    tokens: Any = fields.get("tokenizer.ggml.tokens")
    metadata["tokenizer.vocab_size"] = len(tokens.data) if tokens is not None else None
    file_type = metadata["general.file_type"]
    metadata["quantization"] = (str(gguf.LlamaFileType(file_type).name)
                                if isinstance(file_type, int) and file_type in gguf.LlamaFileType._value2member_map_
                                else None)
    types: dict[str, int] = {}
    for tensor in reader.tensors:
        types[tensor.tensor_type.name] = types.get(tensor.tensor_type.name, 0) + 1
    metadata["tensor_types"] = {key: types[key] for key in sorted(types)}
    metadata["tensor_count"] = len(reader.tensors)
    return metadata


def artifact(path: Path) -> Record:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size,
            "gguf": gguf_metadata(resolved)}


def tokenizer_identity(metadata: Record, hf: Record) -> Record:
    """The tokenizer is the GGUF-embedded vocabulary; the upstream tokenizer files are its source."""
    return {"model": metadata.get("tokenizer.ggml.model"), "pre": metadata.get("tokenizer.ggml.pre"),
            "vocab_size": metadata.get("tokenizer.vocab_size"),
            "upstream_files": {name: hf.get(name) for name in ("tokenizer.json", "vocab.json", "merges.txt")}}


def python_packages(python: Path) -> Record:
    probe = ("import importlib.metadata as m, json\n"
             "names = ('torch', 'transformers', 'safetensors', 'numpy', 'sentencepiece', 'gguf')\n"
             "print(json.dumps({n: (m.version(n) if n in {d.metadata['Name'].lower() for d in m.distributions()} "
             "else None) for n in names}))")
    done = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=False)
    require(done.returncode == 0, "converter python probe failed: " + done.stderr.strip())
    version = subprocess.run([str(python), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    return {"python": version, "packages": json.loads(done.stdout)}


def quantizer_identity(binary: Path) -> Record:
    done = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    lines: list[Json] = [line for line in (done.stdout + done.stderr).splitlines() if line.startswith(("version:", "built with"))]
    build = binary.resolve().parents[1]
    cache = build / "CMakeCache.txt"
    options: Record = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            key, _, value = line.partition("=")
            name = key.split(":", 1)[0]
            if name.startswith(("GGML_", "LLAMA_", "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER:", "CMAKE_CXX_COMPILER:")):
                options[name] = value
    return {"path": str(binary.resolve()), "sha256": sha256(binary.resolve()), "version_output": lines,
            "build_dir": str(build), "cmake_cache_options": options}


def create(args: argparse.Namespace) -> None:
    hf_dir = args.hf_dir.resolve(strict=True)
    hf: Record = {name: sha256(hf_dir / name) for name in UPSTREAM_FILES if (hf_dir / name).is_file()}
    require("model.safetensors" in hf and "config.json" in hf, "upstream checkpoint files missing")
    f16 = artifact(args.f16)
    quantized: Record = {}
    for item in args.quantized:
        name, _, location = item.partition("=")
        require(bool(name) and bool(location), "--quantized expects TYPE=PATH")
        entry = artifact(Path(location))
        entry["quantize_argv"] = [str(args.quantizer), str(args.f16), location, name]
        quantized[name] = entry
    converter = REPO / "convert_hf_to_gguf.py"
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary", "HEAD", "--", "convert_hf_to_gguf.py",
                                    "gguf-py", "src", "ggml", "tools/quantize", "include"])
    reproduction: Record | None = None
    if args.reproduction is not None:
        reproduced: Record = {name: sha256(path) for name, path in
                      ((p.name, p) for p in sorted(args.reproduction.resolve(strict=True).glob("*.gguf")))}
        originals = {Path(text(record(value), "path")).name: text(record(value), "sha256")
                     for value in [f16, *quantized.values()]}
        matches: Record = {name: reproduced.get(name) == digest for name, digest in originals.items()}
        reproduction = {"directory": str(args.reproduction.resolve()), "sha256": reproduced,
                        "byte_identical": matches,
                        "status": "REPRODUCED" if all(value is True for value in matches.values()) else "DIFFERS"}
    manifest: Record = {
        "schema": SCHEMA, "version": 1, "scope": "HOST_LOCAL_MODEL_PROVENANCE",
        "host": {"uname": list(__import__("platform").uname())},
        "upstream": {"repository": args.hf_repo, "revision": args.hf_revision, "local_dir": str(hf_dir), "files": hf},
        "conversion": {"argv": list[Json](shlex.split(args.convert_argv)), "cwd": str(REPO),
                       "converter": {"path": str(converter), "sha256": sha256(converter)},
                       "source": {"repo": REPO.name, "head": git("rev-parse", "HEAD"),
                                  "branch": git("branch", "--show-current"),
                                  "conversion_paths_clean": not diff},
                       "environment": python_packages(args.converter_python)},
        "f16": f16,
        "quantizer": {**quantizer_identity(args.quantizer), "source_head": git("rev-parse", "HEAD")},
        "quantized": quantized,
        "tokenizer": tokenizer_identity(record(f16["gguf"]), hf),
        "reproduction": reproduction,
    }
    output = args.output.resolve()
    require(not output.exists(), "manifest output exists: " + str(output))
    write_json(output, manifest)
    print(json.dumps({"manifest": str(output), "sha256": sha256(output)}))


def model_entry(manifest_path: Path, model: Path) -> Record:
    """Manifest entry of exactly this model file; the model bytes must match the frozen identity."""
    manifest = read_json(manifest_path.resolve(strict=True))
    require(manifest.get("schema") == SCHEMA and manifest.get("version") == 1, "unsupported model manifest")
    digest = sha256(model.resolve(strict=True))
    entries: dict[str, Record] = {"F16": record(manifest.get("f16"))}
    entries.update({name: record(value) for name, value in record(manifest.get("quantized")).items()})
    matches = [(name, entry) for name, entry in entries.items() if entry.get("sha256") == digest]
    require(len(matches) == 1, "model is not a frozen artifact of the manifest: " + str(model))
    name, entry = matches[0]
    return {"manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path.resolve())},
            "artifact": name, "sha256": digest, "gguf": entry.get("gguf"),
            "upstream": {key: record(manifest.get("upstream")).get(key) for key in ("repository", "revision")},
            "tokenizer": manifest.get("tokenizer")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--hf-dir", type=Path, required=True)
    make.add_argument("--hf-repo", required=True)
    make.add_argument("--hf-revision", required=True)
    make.add_argument("--f16", type=Path, required=True)
    make.add_argument("--convert-argv", required=True, help="the exact conversion command line")
    make.add_argument("--converter-python", type=Path, required=True)
    make.add_argument("--quantizer", type=Path, required=True)
    make.add_argument("--quantized", action="append", default=[], help="TYPE=PATH, repeatable")
    make.add_argument("--reproduction", type=Path, help="directory of independently regenerated GGUF files")
    make.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            create(args)
        else:
            print(json.dumps(model_entry(args.manifest, args.model), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"model manifest failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
