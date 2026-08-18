#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
# ─── How to run ───
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run: uv run scripts/gemmini-rmd-bundle.py manifest --help
# 3. Or: chmod +x scripts/gemmini-rmd-bundle.py && ./scripts/gemmini-rmd-bundle.py --self-test
# ──────────────────
"""Create and verify immutable Gemmini RMD runtime bundles."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Final, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gemmini_rmd_bundle import (
    BundleError,
    Code,
    ManifestRequest,
    create,
    load,
    verify,
)

GUEST_MODEL: Final = "/root/workspace/llama.cpp/models/gpt2.Q8_0.gguf"

type Command = Literal["manifest", "verify"]


class Args(argparse.Namespace):
    """Mutable argparse boundary populated in place."""

    def __init__(self) -> None:
        super().__init__()
        self.self_test: bool = False
        self.command: Command | None = None
        self.bundle: Path = Path()
        self.binary: Path = Path()
        self.project_library: list[Path] = []
        self.profile: str = ""
        self.source_hash: str = ""
        self.working_state_hash: str = ""
        self.build_flag: list[str] = []
        self.binary_version: str = ""
        self.compiler: Path = Path()
        self.compiler_flag: list[str] = []
        self.dependency: list[Path] = []
        self.config: list[Path] = []
        self.model: Path = Path()
        self.guest_model_path: str = GUEST_MODEL
        self.system_input: list[Path] = []
        self.readelf: str = "readelf"
        self.expected_manifest: Path | None = None


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    _ = root.add_argument("--self-test", action="store_true")
    commands = root.add_subparsers(dest="command")
    make = commands.add_parser("manifest")
    _ = make.add_argument("--bundle", type=Path, required=True)
    _ = make.add_argument("--binary", type=Path, required=True)
    _ = make.add_argument("--project-library", action="append", type=Path, default=[])
    _ = make.add_argument("--profile", choices=("summary", "detail"), required=True)
    _ = make.add_argument("--source-hash", required=True)
    _ = make.add_argument("--working-state-hash", required=True)
    _ = make.add_argument("--build-flag", action="append", default=[])
    _ = make.add_argument("--binary-version", required=True)
    _ = make.add_argument("--compiler", type=Path, required=True)
    _ = make.add_argument("--compiler-flag", action="append", default=[])
    _ = make.add_argument("--dependency", action="append", type=Path, default=[])
    _ = make.add_argument("--config", action="append", type=Path, default=[])
    _ = make.add_argument("--model", type=Path, required=True)
    _ = make.add_argument("--guest-model-path", default=GUEST_MODEL)
    _ = make.add_argument("--system-input", action="append", type=Path, default=[])
    _ = make.add_argument("--readelf", default="readelf")
    check = commands.add_parser("verify")
    _ = check.add_argument("--bundle", type=Path, required=True)
    _ = check.add_argument("--readelf", default="readelf")
    _ = check.add_argument("--expected-manifest", type=Path)
    return root


def arguments() -> Args:
    args = Args()
    _ = parser().parse_args(namespace=args)
    return args


def request(args: Args) -> ManifestRequest:
    return ManifestRequest(
        args.bundle,
        args.binary,
        tuple(args.project_library),
        args.profile,
        args.source_hash,
        args.working_state_hash,
        tuple(args.build_flag),
        args.binary_version,
        args.compiler,
        tuple(args.compiler_flag),
        tuple(args.dependency),
        tuple(args.config),
        args.model,
        args.guest_model_path,
        tuple(args.system_input),
        args.readelf,
    )


def executable(path: Path, body: str) -> None:
    _ = path.write_text(body)
    path.chmod(0o755)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="gemmini-rmd-bundle-") as raw:
        root = Path(raw)
        source = root / "source"
        source.mkdir()
        binary = source / "llama-cli"
        library = source / "libggml-fixture.so"
        _ = binary.write_bytes(b"\x7fELFfixture-binary")
        _ = library.write_bytes(b"\x7fELFfixture-library")
        binary.chmod(0o755)
        library.chmod(0o755)
        header = root / "gemmini.h"
        _ = header.write_text("#define GEMMINI_FIXTURE 1\n")
        compiler = root / "compiler"
        executable(
            compiler,
            '#!/bin/sh\nif [ "$1" = --version ]; then echo fixture-cc-1; else printf "x: %s\\n" "$HEADER_PATH"; fi\n',
        )
        readelf = root / "readelf"
        executable(
            readelf,
            "#!/bin/sh\ncase \"$2\" in *llama-cli) echo ' (NEEDED) [libggml-fixture.so]'; echo ' (RUNPATH) [$ORIGIN]' ;; *) echo ' (NEEDED) [libc.so.6]' ;; esac\n",
        )
        model = root / "model.gguf"
        _ = model.write_bytes(b"real model fixture")
        config = root / "CMakeCache.txt"
        _ = config.write_text("CYCLE_DETAIL=0\n")
        os.environ["HEADER_PATH"] = str(header)
        bundle = root / "bundle"
        spec = ManifestRequest(
            bundle,
            binary,
            (library,),
            "summary",
            "source-v1",
            "a" * 64,
            ("-DCYCLE_DETAIL=0",),
            "llama-fixture-v1",
            compiler,
            ("-Ifixture",),
            (header,),
            (config,),
            model,
            GUEST_MODEL,
            (),
            str(readelf),
        )
        manifest_path = create(spec)
        _ = verify(bundle, str(readelf))
        manifest = load(manifest_path)
        bundled = bundle / "bin" / library.name
        bundled.chmod(0o755)
        _ = bundled.write_bytes(b"\x7fELFmutated")
        bundled.chmod(0o555)
        try:
            _ = verify(bundle, str(readelf))
        except BundleError as error:
            if error.code is not Code.HASH_MISMATCH:
                raise
        else:
            raise BundleError(Code.HASH_MISMATCH, "mutation was not detected")
        if (
            not manifest["gemmini_headers"]
            or manifest["guest_model_path"] != GUEST_MODEL
        ):
            raise BundleError(Code.PROVENANCE_MISMATCH, "self-test provenance missing")
    print("SELF_TEST_OK HASH_MISMATCH")


def main() -> int:
    args = arguments()
    try:
        if args.self_test:
            self_test()
            return Code.OK
        match args.command:
            case "manifest":
                print(create(request(args)))
            case "verify":
                expected = (
                    load(args.expected_manifest) if args.expected_manifest else None
                )
                print(
                    verify(args.bundle, args.readelf, expected)[
                        "canonical_bundle_sha256"
                    ]
                )
            case None:
                parser().print_help(sys.stderr)
                return Code.USAGE
    except BundleError as error:
        print(error, file=sys.stderr)
        return error.code
    return Code.OK


if __name__ == "__main__":
    raise SystemExit(main())
