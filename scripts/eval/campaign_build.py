"""Fresh, independently configured measurement builds and command receipts.

This is the single place that turns an evaluation configuration into CMake options, for the llama builds
(PoTal, FullCPU, cycle and metric kinds) and for the IM2P cycle-model library. Only platform_profile() may
differ between hosts (library suffix, binary format, install name/rpath, host ISA); semantic options
(precision, DIM, backend, quantization, cycle-model flags, source set) never depend on the platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from eval_common import (
    Json,
    Record,
    artifact_snapshot,
    compiled_info,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)

REPO: Final = Path(__file__).resolve().parents[2]
TARGETS: Final = ["llama-eval-workload", "test-evaluation-workload", "test-evaluation-trace",
                 "test-gemmini-evaluation-metrics", "test-cycle-sim-reader",
                 "test-gemmini-cycle-sim-log", "test-cycle-sim-coverage"]
PLATFORMS: Final = frozenset({("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "aarch64"), ("Linux", "x86_64")})
MACHINES: Final = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}
# Instrumentation and locations do not change numerics or cycle semantics; everything else is semantic.
NON_SEMANTIC: Final = frozenset({"LOG_CYCLE", "GGML_CPU_CYCLE_LOG", "CYCLE_DETAIL", "LOG_DEBUG",
                                 "IM2P_SIM_ROOT", "CMAKE_EXPORT_COMPILE_COMMANDS"})
CYCLE_MODEL_OPTIONS: Final = {"CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"}
SOURCE_REPOS: Final = ("llama.cpp-gemmini", "IM2P.sim", "RISC-V-DynDNN-gemmini-include")
INCLUDE_REPO: Final = REPO.parent / "RISC-V-DynDNN-gemmini-include"


def command(argv: list[str], directory: Path, name: str, timeout: int = 1800, cwd: Path = REPO) -> None:
    """Run one bounded command, retaining its exact argv, output and exit status."""
    with (directory / (name + ".log")).open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            write_json(directory / (name + ".json"), {"argv": list(argv), "exit_code": None,
                       "status": "TIMEOUT", "timeout_seconds": timeout})
            raise
    write_json(directory / (name + ".json"), {"argv": list(argv), "cwd": str(cwd),
               "exit_code": result.returncode})
    require(result.returncode == 0, "command failed: " + str(directory / (name + ".log")))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Platform:
    """Host-only build facts; nothing here may reach a semantic option."""
    system: str
    machine: str
    binary_format: str
    shared_library_suffix: str
    install_name: str
    cmake_options: tuple[tuple[str, str], ...] = ()

    def record(self) -> Record:
        value: Record = {key: item for key, item in asdict(self).items() if key != "cmake_options"}
        value["cmake_options"] = {key: item for key, item in self.cmake_options}
        return value


def platform_profile(system: str | None = None, machine: str | None = None) -> Platform:
    name = system or platform.system()
    raw = machine or platform.machine()
    arch = MACHINES.get(raw.lower(), raw)
    require((name, arch) in PLATFORMS, f"unsupported evaluation host {name}/{raw}")
    darwin = name == "Darwin"
    return Platform(name, arch, "Mach-O" if darwin else "ELF", ".dylib" if darwin else ".so",
                    "CMake default @rpath install name" if darwin else "CMake default RPATH")


def semantic_options(options: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in sorted(options.items()) if key not in NON_SEMANTIC}


def with_platform(options: dict[str, str], host: Platform) -> dict[str, str]:
    extra = dict(host.cmake_options)
    require(not set(extra) & set(options) and not semantic_options(extra),
            "platform options may not name semantic or existing options")
    return {**options, **extra}


@dataclass(frozen=True, slots=True)
class LlamaPlan:
    kind: str
    options: dict[str, str]
    targets: tuple[str, ...]
    extra_targets: tuple[str, ...]
    verify_filter: str
    generator: str | None

    @property
    def semantic_options_sha256(self) -> str:
        return digest(semantic_options(self.options))

    def record(self) -> Record:
        options: Record = {key: value for key, value in self.options.items()}
        return {"kind": self.kind, "options": options, "targets": list(self.targets),
                "extra_targets": list(self.extra_targets), "verify_filter": self.verify_filter,
                "generator": self.generator, "semantic_options_sha256": self.semantic_options_sha256}


def llama_plan(kind: str, precision: str, dim: int, im2p: Path, matmul_mode: str = "FULL",
               extra_targets: tuple[str, ...] = ()) -> LlamaPlan:
    """Options, targets and verification filter of one llama build kind; platform independent."""
    require(kind in ("cycle", "activation", "residual", "scu", "potal-host", "potal-host-nocpulog",
                     "fullcpu-host"),
            "invalid measurement kind")
    require(precision in ("a4w4", "a8w8") and dim in (16, 32, 64) and
            matmul_mode in ("FULL", "STRIPE_PIPELINE") and
            (kind in ("cycle", "potal-host", "potal-host-nocpulog") or matmul_mode == "FULL"),
            "invalid measurement build profile")
    bits = 4 if precision == "a4w4" else 8
    options = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "BUILD_SHARED_LIBS": "OFF", "LLAMA_BUILD_TESTS": "ON", "LLAMA_BUILD_TOOLS": "ON",
        "LLAMA_CURL": "OFF", "GGML_GEMMINI": "ON", "GGML_METAL": "OFF",
        "GGML_ACCELERATE": "OFF", "GGML_BLAS": "OFF", "GGML_CUDA": "OFF",
        "GGML_OPENMP": "OFF", "CYCLE_SIM": "1", "LOG_CYCLE": "0", "LOG_DEBUG": "0",
        "CYCLE_DETAIL": "0", "GGML_CPU_CYCLE_LOG": "OFF", "GGML_BACKEND_DL": "OFF",
        "GGML_GEMMINI_EXECUTION_BACKEND": "IM2P_SIM", "IM2P_SIM_IMPLEMENTATION": "GEMMINI_HP1",
        "IM2P_SIM_ROOT": str(im2p), "GGML_GEMMINI_OPTION": "WS",
        "GGML_GEMMINI_ACTIVATION_QUANT": "EXSIA", "GGML_GEMMINI_BLOCK_SIZE": "32",
        "GGML_GEMMINI_ACTIVATION_BITS": str(bits), "GGML_GEMMINI_WEIGHT_BITS": str(bits),
        "GGML_GEMMINI_DIM": str(dim), "GGML_GEMMINI_DEFAULT_MATMUL_MODE": matmul_mode,
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND": "WS", "GGML_GEMMINI_ENABLE_RMD": "ON",
        "GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE": "OFF",
        "GGML_GEMMINI_ACT_METRICS": str(int(kind == "activation")),
        "GGML_GEMMINI_ACT_QUANT_METRICS": "0",
        "GGML_GEMMINI_RESIDUAL_METRICS": str(int(kind == "residual")),
        "GGML_GEMMINI_SCALE_METRICS": str(int(kind == "scu")),
    }
    if kind.startswith("potal-host"):
        options.update({"LOG_CYCLE": "1",
                        "GGML_CPU_CYCLE_LOG": "OFF" if kind == "potal-host-nocpulog" else "ON"})
    targets = list(TARGETS)
    verify_filter = ("^(test-evaluation-(workload|trace|build-options|metric-framework)|"
                     "test-gemmini-(evaluation-metrics|cycle-sim-log)|"
                     "test-cycle-sim-(reader|coverage|build-contract))$")
    if kind == "fullcpu-host":
        for key in [name for name in options if name.startswith("IM2P_SIM")]:
            del options[key]
        options.update({"GGML_GEMMINI_OPTION": "CPU", "GGML_GEMMINI_EXECUTION_BACKEND": "HARDWARE",
                        "GGML_GEMMINI_DEFAULT_RMD_BACKEND": "CPU", "CYCLE_SIM": "0",
                        "LOG_CYCLE": "1", "GGML_CPU_CYCLE_LOG": "ON"})
        targets = ["llama-eval-workload", "test-evaluation-workload", "test-evaluation-trace"]
        verify_filter = "^test-evaluation-(workload|trace|build-options)$"
    return LlamaPlan(kind, options, (*targets, *extra_targets), extra_targets, verify_filter,
                     "Ninja" if kind.startswith(("potal-host", "fullcpu-host")) else None)


def llama_argv(plan: LlamaPlan, output: Path, jobs: int, host: Platform) -> list[list[str]]:
    options = with_platform(plan.options, host)
    generator = ["-G", plan.generator] if plan.generator else []
    return [["cmake", "-S", str(REPO), "-B", str(output), *generator,
             *(f"-D{key}={value}" for key, value in options.items())],
            ["cmake", "--build", str(output), "--parallel", str(jobs), "--target", *plan.targets],
            ["ctest", "--test-dir", str(output), "--output-on-failure", "-R", plan.verify_filter]]


def cycle_model_argv(output: Path, im2p: Path, jobs: int, host: Platform) -> list[list[str]]:
    options = with_platform(dict(CYCLE_MODEL_OPTIONS), host)
    return [["cmake", "-S", str(im2p / "sim/cycle"), "-B", str(output),
             *(f"-D{key}={value}" for key, value in options.items())],
            ["cmake", "--build", str(output), "--parallel", str(jobs)],
            ["ctest", "--test-dir", str(output), "--output-on-failure"]]


def toolchain() -> Record:
    def first_line(*argv: str) -> str | None:
        executable = shutil.which(argv[0])
        if executable is None:
            return None
        done = subprocess.run([executable, *argv[1:]], capture_output=True, text=True, timeout=30, check=False)
        lines = (done.stdout or done.stderr).strip().splitlines()
        return lines[0] if lines else None
    return {"uname": list(platform.uname()), "cc": first_line("cc", "--version"),
            "cxx": first_line("c++", "--version"), "cmake": first_line("cmake", "--version"),
            "ninja": first_line("ninja", "--version"), "make": first_line("make", "--version"),
            "python": platform.python_version()}


def repository_state(repo: Path) -> Record:
    def git(*argv: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"])
    return {"repo": repo.name, "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "tracked_clean": not diff, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def include_headers(repo: Path = INCLUDE_REPO) -> Record:
    """Every header on the CMake include path of the Gemmini include repo (top level, then include/), hashed.

    Git HEAD and the tracked diff miss untracked or ignored headers, so the header bytes themselves are part of
    the source identity and therefore of every llama build cache key."""
    rows: Record = {str(path.relative_to(repo)): sha256(path)
                    for directory in (repo, repo / "include") for path in sorted(directory.glob("*.h"))}
    require("gemmini.h" in rows and "gemmini_params.h" in rows, "Gemmini include repo lacks gemmini.h/gemmini_params.h")
    return rows


def source_state() -> Record:
    state: Record = {name: repository_state(REPO.parent / name) for name in SOURCE_REPOS}
    record(state["RISC-V-DynDNN-gemmini-include"])["headers"] = include_headers()
    return state


def include_closure(build_dir: Path, repo: Path = INCLUDE_REPO) -> Record:
    """Include-repo headers the compiler actually read, hashed; empty when none were read.

    Ninja keeps compiler dependencies in its deps log; Makefile builds keep the compiler depfiles (*.o.d)."""
    tokens: list[str] = []
    if (build_dir / "build.ninja").is_file():
        done = subprocess.run(["ninja", "-C", str(build_dir), "-t", "deps"], capture_output=True, text=True, check=False)
        require(done.returncode == 0, "ninja deps log unavailable: " + str(build_dir))
        tokens = [line.strip() for line in done.stdout.splitlines() if line.startswith("    ")]
    else:
        depfiles = sorted(build_dir.rglob("*.o.d"))
        require(bool(depfiles), "compiler dependency files missing: " + str(build_dir))
        for depfile in depfiles:
            tokens.extend(depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split())
    root = repo.resolve()
    used = {(build_dir / token).resolve() for token in tokens}
    return {str(path.relative_to(root)): sha256(path) for path in sorted(used) if path.is_relative_to(root)}


def build(kind: str, precision: str, dim: int, output: Path, im2p: Path, jobs: int,
          matmul_mode: str = "FULL", extra_targets: tuple[str, ...] = ()) -> Path:
    """Configure, compile and verify a new directory; never reuse a user's build cache."""
    plan = llama_plan(kind, precision, dim, im2p, matmul_mode, extra_targets)
    require(jobs > 0, "invalid measurement build profile")
    host = platform_profile()
    output.mkdir(parents=True, exist_ok=False)
    producer_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    source_diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
    with (output / "producer.diff").open("xb") as stream:
        stream.write(source_diff)
    for name, argv in zip(("configure", "build", "verify"), llama_argv(plan, output, jobs, host)):
        command(argv, output, name)
    runner = output / "bin/llama-eval-workload"
    info = compiled_info(runner)
    bits = 4 if precision == "a4w4" else 8
    expected = (int(kind == "activation"), int(kind == "residual"), int(kind == "scu"))
    require(tuple(info.get(key) for key in ("activation_metrics", "residual_metrics", "scale_metrics"))
            == expected, "compiled collectors are not independent")
    if kind == "fullcpu-host":
        require(info.get("cpu_only") is True and info.get("cycle_sim") == 0 and
                info.get("log_cycle") == 1 and info.get("ggml_cpu_cycle_log") == 1,
                "FullCPU host-timing build profile differs from request")
    else:
        require(info.get("dim") == dim and info.get("activation_bits") == bits and
                info.get("weight_bits") == bits and info.get("cycle_sim") == 1 and
                info.get("matmul_mode") == matmul_mode, "compiled build profile differs from request")
        require(not kind.startswith("potal-host") or (info.get("log_cycle") == 1 and
                info.get("ggml_cpu_cycle_log") == int(kind == "potal-host")),
                "PoTal host-timing build requires LOG_CYCLE=1 and the requested CPU cycle instrumentation")
    write_json(output / "build-info.json", info)
    write_json(output / "artifacts.json", artifact_snapshot(runner))
    extras: Record = {name: sha256(output / "bin" / name) for name in extra_targets}
    headers = include_closure(output)
    require("gemmini.h" in headers, "GGML_GEMMINI build did not read the include-repo gemmini.h")
    write_json(output / "build-receipt.json", {"kind": kind, "options": plan.record()["options"],
               "runner_sha256": sha256(runner), "verification": "PASS", "producer_git_sha": producer_head,
               "producer_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
               "semantic_options_sha256": plan.semantic_options_sha256, "platform": host.record(),
               "toolchain": toolchain(), "sources": source_state(), "extra_targets": extras,
               "include_repo_closure": headers})
    return runner


def source_closure(build_dir: Path, root: Path) -> Record:
    """Project files named by the compiler depfiles of the cycle-model objects, hashed; system headers counted."""
    project: set[Path] = set()
    external: set[Path] = set()
    system = 0
    include = INCLUDE_REPO.resolve()
    for depfile in sorted((build_dir / "CMakeFiles/im2p_cycle_objects.dir").rglob("*.o.d")):
        for token in depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split():
            path = (build_dir / token).resolve()
            if path.is_relative_to(root):
                project.add(path)
            elif path.is_relative_to(include):
                external.add(path)
            else:
                system += 1
    require(bool(project), "cycle-model dependency files missing")
    # The cycle-model cache key names only IM2P.sim; an include-repo header in the closure would make it stale.
    require(not external, "cycle model reads Gemmini include-repo headers: " + ", ".join(map(str, sorted(external))))
    files: Record = {str(path.relative_to(root)): sha256(path) for path in sorted(project)}
    return {"files": files, "closure_sha256": digest(files), "system_header_references": system,
            "include_repo_dependency": "NONE"}


def build_cycle_model(output: Path, im2p: Path, jobs: int) -> Record:
    """Configure, compile and ctest the cycle-model library with a source/toolchain/binary receipt."""
    host = platform_profile()
    output.mkdir(parents=True, exist_ok=False)
    for name, argv in zip(("configure", "build", "verify"), cycle_model_argv(output, im2p, jobs, host)):
        command(argv, output, name, cwd=im2p)
    library = output / ("libim2p_cycle_model" + host.shared_library_suffix)
    require(library.is_file(), "cycle shared library missing")
    units: list[dict[str, str]] = json.loads((output / "compile_commands.json").read_text())
    link = output / "CMakeFiles/im2p_cycle_model_shared.dir/link.txt"
    options: Record = {key: value for key, value in with_platform(dict(CYCLE_MODEL_OPTIONS), host).items()}
    commands: list[Json] = [{"file": unit["file"], "command": unit["command"]} for unit in units]
    receipt: Record = {"schema": "im2p-cycle-model-build-receipt", "version": 1,
        "options": options, "platform": host.record(), "toolchain": toolchain(),
        "source": repository_state(im2p), "source_closure": source_closure(output, im2p.resolve()),
        "compile_commands": commands,
        "link_command": link.read_text().strip() if link.is_file() else None,
        "library": {"path": str(library), "sha256": sha256(library)},
        "static_library": {"path": str(output / "libim2p_cycle_model.a"),
                           "sha256": sha256(output / "libim2p_cycle_model.a")},
        "verification": "PASS"}
    write_json(output / "build-receipt.json", receipt)
    return receipt


def cached(cache: Path, name: str, identity: Record, make: Callable[[Path], object]) -> tuple[Path, bool]:
    """Reuse a completed build only for an identical source/configuration/toolchain identity."""
    path = cache / f"{name}-{digest(identity)[:16]}"
    marker, pending = path / "cache-identity.json", cache / f"{path.name}.pending"
    if marker.is_file():
        require(read_json(marker) == identity, "build cache identity collision: " + str(path))
        return path, True
    if pending.is_file():
        shutil.rmtree(path, ignore_errors=True)
    require(not path.exists(), "unowned build cache entry: " + str(path))
    cache.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(identity, sort_keys=True) + "\n")
    make(path)
    write_json(marker, identity)
    pending.unlink()
    return path, False


def cached_llama_build(cache: Path, kind: str, precision: str, dim: int, im2p: Path, jobs: int,
                       matmul_mode: str = "FULL", extra_targets: tuple[str, ...] = ()) -> tuple[Path, bool]:
    plan = llama_plan(kind, precision, dim, im2p, matmul_mode, extra_targets)
    identity: Record = {"build": "llama", "plan": plan.record(), "platform": platform_profile().record(),
                        "toolchain": toolchain(), "sources": source_state()}
    path, hit = cached(cache, kind, identity,
                       lambda output: build(kind, precision, dim, output, im2p, jobs, matmul_mode, extra_targets))
    receipt = read_json(path / "build-receipt.json")
    extras = record(receipt.get("extra_targets"))
    require(sha256(path / "bin/llama-eval-workload") == receipt.get("runner_sha256") and
            all(sha256(path / "bin" / name) == value for name, value in extras.items()),
            "cached build artifact changed after its receipt")
    return path, hit


def cached_cycle_model(cache: Path, im2p: Path, jobs: int) -> tuple[Path, bool]:
    options: Record = {key: value for key, value in CYCLE_MODEL_OPTIONS.items()}
    identity: Record = {"build": "cycle-model", "options": options, "platform": platform_profile().record(),
                        "toolchain": toolchain(), "source": repository_state(im2p)}
    path, hit = cached(cache, "cycle-model", identity, lambda output: build_cycle_model(output, im2p, jobs))
    library = record(read_json(path / "build-receipt.json").get("library"))
    require(sha256(Path(text(library, "path"))) == library.get("sha256"), "cached cycle library changed after its receipt")
    return path, hit


def snapshot(output: Path) -> None:
    """Preserve requested pre-change git evidence without modifying any checkout."""
    output.mkdir(parents=True, exist_ok=False)
    rows: Record = {}
    for name in SOURCE_REPOS:
        repo = REPO.parent / name
        fields: Record = {}
        for label, args in (("status", ["status", "--short"]), ("diff", ["diff", "--binary"]),
                            ("branch", ["branch", "--show-current"]), ("head", ["rev-parse", "HEAD"])):
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                    text=True, check=True)
            with (output / (name + "-" + label + ".txt")).open("x", encoding="utf-8") as stream:
                stream.write(result.stdout)
            fields[label] = result.stdout.strip()
        rows[name] = fields
    write_json(output / "repositories.json", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("cycle", "activation", "residual", "scu", "potal-host",
                                           "potal-host-nocpulog", "fullcpu-host", "cycle-model"), required=True)
    parser.add_argument("--matmul-mode", choices=("FULL", "STRIPE_PIPELINE"), default="FULL")
    parser.add_argument("--precision", choices=("a4w4", "a8w8"))
    parser.add_argument("--dim", type=int, choices=(16, 32, 64))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--extra-target", action="append", default=[], help="additional CMake target, e.g. llama-perplexity")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and commands; build nothing")
    parser.add_argument("--system", help="dry-run platform override (Darwin or Linux)")
    parser.add_argument("--machine", help="dry-run architecture override (arm64, aarch64, x86_64)")
    args = parser.parse_args()
    require(args.dry_run or (args.system is None and args.machine is None), "platform overrides are dry-run only")
    output, im2p = args.output.resolve(), args.im2p.resolve()
    if args.kind == "cycle-model":
        if args.dry_run:
            host = platform_profile(args.system, args.machine)
            print(json.dumps({"platform": host.record(), "options": with_platform(dict(CYCLE_MODEL_OPTIONS), host),
                              "argv": cycle_model_argv(output, im2p, args.jobs, host)}, indent=2, sort_keys=True))
            return 0
        receipt = build_cycle_model(output, im2p, args.jobs)
        print(json.dumps({"library": receipt["library"]}))
        return 0
    require(args.precision is not None and args.dim is not None, "--precision and --dim are required")
    if args.dry_run:
        host = platform_profile(args.system, args.machine)
        plan = llama_plan(args.kind, args.precision, args.dim, im2p, args.matmul_mode, tuple(args.extra_target))
        print(json.dumps({"platform": host.record(), "plan": plan.record(), "argv": llama_argv(plan, output, args.jobs, host)},
                         indent=2, sort_keys=True))
        return 0
    runner = build(args.kind, args.precision, args.dim, output, im2p, args.jobs, args.matmul_mode,
                   tuple(args.extra_target))
    print(json.dumps({"runner": str(runner), "sha256": sha256(runner)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
