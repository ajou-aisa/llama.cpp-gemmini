from __future__ import annotations

from collections import Counter
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from typing import TypedDict

import numpy as np

from .types import I64, QuantError


class BatchWork(ct.Structure):
    _fields_ = [("a", ct.POINTER(ct.c_int8)), ("b", ct.POINTER(ct.c_int8)),
                ("rows", ct.POINTER(ct.c_int64)), ("shifts", ct.POINTER(ct.c_int64)),
                ("regions", ct.POINTER(ct.c_int64)), ("m", ct.c_int), ("k", ct.c_int),
                ("n", ct.c_int), ("region_count", ct.c_int)]


class CpuBatch:
    def __init__(self, lib: ct.CDLL, direct: bool = True) -> None:
        self.lib, self.direct = lib, direct
        lib.potal_replay_work_size.restype = ct.c_size_t
        if lib.potal_replay_work_size() != ct.sizeof(BatchWork):
            raise QuantError("Native batch descriptor size differs")
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        lib.potal_replay_batch.argtypes = [ct.POINTER(BatchWork), ct.c_int, ct.c_int, ct.c_int, ct.c_int, i64, u64]
        lib.potal_replay_batch.restype = ct.c_int
        lib.potal_replay_error.restype = ct.c_char_p
        self.work: list[BatchWork] = []
        self.buffers: list[tuple[np.ndarray, ...]] = []
        self.times: Counter[str] = Counter()
        self.max_extra_bytes = 0

    def add(self, a: I64, b: I64, rows: I64, shifts: I64,
            tiles: tuple[tuple[slice, slice], ...] | None) -> None:
        begin = perf_counter_ns()
        if (a.ndim != 2 or b.ndim != 2 or b.shape[0] != a.shape[1] or
                rows.shape != (len(a),) or shifts.shape != (b.shape[1],) or
                a.min(initial=0) < -8 or a.max(initial=0) > 7 or b.min(initial=0) < -8 or b.max(initial=0) > 7):
            raise QuantError("Batch requires signed nibble matrices and matching row/scale maps")
        aa, bb = np.ascontiguousarray(a, dtype=np.int8), np.ascontiguousarray(b, dtype=np.int8)
        rr, ss = np.ascontiguousarray(rows, dtype=np.int64), np.ascontiguousarray(shifts, dtype=np.int64)
        regions = np.asarray([(0, len(a), 0, b.shape[1])] if tiles is None else
                             [(r.start, r.stop, c.start, c.stop) for r, c in tiles], dtype=np.int64).reshape(-1, 4)
        buffers = (aa, bb, rr, ss, regions)
        pointers = [v.ctypes.data_as(ct.POINTER(ct.c_int8 if i < 2 else ct.c_int64)) for i, v in enumerate(buffers)]
        self.work.append(BatchWork(*pointers, len(a), a.shape[1], b.shape[1], len(regions)))
        self.buffers.append(buffers)
        self.times["prepare_ns"] += perf_counter_ns() - begin

    def flush(self, acc: I64) -> None:
        begin = perf_counter_ns()
        stats = np.zeros(8, np.uint64)
        work = (BatchWork * len(self.work))(*self.work)
        queued_bytes = sum(v.nbytes for buffers in self.buffers for v in buffers) + ct.sizeof(work)
        try:
            status = self.lib.potal_replay_batch(work, len(work), acc.shape[0], acc.shape[1], self.direct, acc, stats)
            if status:
                raise QuantError(self.lib.potal_replay_error().decode())
            for index, key in enumerate(("validate_allocate_ns", "dot_ns", "merge_ns", "scratch_bytes",
                                         "raw_dot_calls", "full_partial_bytes", "tile_partial_bytes", "work_items")):
                if key != "scratch_bytes":
                    self.times[key] += int(stats[index])
            self.times["ffi_calls"] += 1
            self.times["execute_ns"] += perf_counter_ns() - begin
            self.max_extra_bytes = max(self.max_extra_bytes, queued_bytes + int(stats[3]))
        finally:
            self.work.clear()
            self.buffers.clear()


class CpuDot:
    """Existing native integer dot, shared by model replay and component replay."""
    def __init__(self, lib, mode: int) -> None:
        import numpy as np
        self.np, self.lib, self.mode = np, lib, mode
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        lib.potal_replay_dot.argtypes = [i8, i8, i32, ct.c_int, ct.c_int, ct.c_int, ct.c_int, u64]
        lib.potal_replay_dot.restype = ct.c_int
        self.times = Counter()
        self.max_extra_bytes = 0

    def __call__(self, a, b, lane):
        np = self.np
        begin = perf_counter_ns()
        aa = np.ascontiguousarray(a, dtype=np.int8)
        bb = np.ascontiguousarray(b, dtype=np.int8)
        result = np.empty((len(aa), bb.shape[1]), dtype=np.int32)
        stats = np.zeros(4, np.uint64)
        mode = self.mode if lane >= 2 else 0
        status = self.lib.potal_replay_dot(aa, bb, result, len(aa), aa.shape[1], bb.shape[1], mode, stats)
        if status: raise RuntimeError("Raw dot rejected shape")
        result = result.astype(np.int64)
        key = "upper2plus" if lane >= 2 else "main_or_upper1"
        self.times[key + "_pack_ns"] += int(stats[0])
        self.times[key + "_dot_ns"] += int(stats[1])
        self.times[key + "_call_ns"] += perf_counter_ns()-begin
        self.times["ffi_calls"] += 1
        self.max_extra_bytes = max(self.max_extra_bytes, int(stats[3]))
        return result


def prepare_head(role, streamed, stationary, times):
    from .profile_gemm import checked_bound, unpack
    from .profile_ops import probability, query, stationary_codes
    if role == 0:
        operand, _ = query(streamed, len(stationary), times)
        weights = stationary_codes(stationary.T, 8, times, "k")
    else:
        operand = probability(streamed, stationary.shape[0], times)
        weights = stationary_codes(stationary.T, 4, times, "v")
    plan = unpack(operand)
    checked_bound(plan, weights, 0 if role == 0 else 8)
    return operand, plan, weights


class ComponentRun(TypedDict):
    mode: str
    repeat: int
    gemm_ns: int
    qk_fragments: dict[str, int]
    kernel_times_ns: dict[str, int]
    component_times_ns: dict[str, int]
    max_workspace_bytes: int


class ComponentTiming(TypedDict):
    capture: str
    sha256: str
    bitwise_integer_gemm: bool
    runs: list[ComponentRun]


def measure_components(lib: ct.CDLL, paths: list[Path], skip_qk_tiles: bool,
                       batch_native: bool = False) -> list[ComponentTiming]:
    from .profile_gemm import QKTileMask, gemm
    from .run import digest

    names = ("dense", "qk_mask_skip", "batch_full_partial", "batch_direct") if batch_native else (
        "dense", "sparse_multiply", "qk_mask_skip" if skip_qk_tiles else "sparse_unit_add")
    modes = (0, 1, 2, 3) if batch_native else ((0, 2) if skip_qk_tiles else (0, 1, 2))
    timings = []
    for path in paths:
        role = 0 if path.name.startswith("qk") else 1
        role_name, zp = ("qk", 0) if role == 0 else ("pv", 8)
        with np.load(path) as saved:
            _, plan, weights = prepare_head(role, saved["streamed"], saved["stationary"], {})
            reference = gemm(plan, weights, role_name, {}, zp)
            assert np.array_equal(reference.astype(np.float32), saved["result"])
            mask_values = saved["mask"].copy() if skip_qk_tiles and role == 0 else None
        live = ~np.isneginf(mask_values) if mask_values is not None else np.ones(reference.shape, bool)
        repeats: list[ComponentRun] = []
        reference_counts = None
        for repeat in range(6):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                use_batch = batch_native and mode >= 2
                executor = CpuBatch(lib, mode == 3) if use_batch else CpuDot(lib, 0 if skip_qk_tiles else mode)
                counts, stages = Counter(), {}
                started = perf_counter_ns()
                masked = mask_values is not None and mode != 0
                mask = QKTileMask.from_values(mask_values) if masked else None
                result = gemm(plan, weights, role_name, stages, zp, None if use_batch else executor,
                              mask, counts if role == 0 else None, executor if use_batch else None)
                elapsed = perf_counter_ns()-started
                assert np.array_equal(result[live], reference[live]), (path.name, names[mode])
                if mask is not None:
                    assert np.all(result[mask.blocked] == 0)
                    if reference_counts is None:
                        reference_counts = counts
                    assert counts == reference_counts, (path.name, names[mode], counts, reference_counts)
                if repeat:
                    repeats.append({"mode":names[mode], "repeat":repeat, "gemm_ns":elapsed,
                                    "qk_fragments":dict(counts), "component_times_ns":stages,
                                    "kernel_times_ns":dict(executor.times), "max_workspace_bytes":executor.max_extra_bytes})
        timings.append({"capture":path.name, "sha256":digest(path), "bitwise_integer_gemm":True, "runs":repeats})
        print("timed " + path.name, flush=True)
    return timings


def batch_components(args: list[str]) -> None:
    import platform

    from .run import digest
    from .verify import check_native_batch

    library, captures, output = (Path(value).resolve() for value in args)
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(captures.glob("qk-*.npz")) + sorted(captures.glob("pv-*.npz"))
    if not paths:
        raise QuantError("No QK/PV captures found")
    check_native_batch(str(library))
    timings = measure_components(ct.CDLL(str(library)), paths, True, True)
    report = {"scope":"CPU native call batching and direct wide accumulation; existing raw integer dot, quantization and mask plan",
              "library":str(library), "library_sha256":digest(library), "platform":platform.platform(),
              "numpy":np.__version__, "cpu_dot_threads":1, "component_timings":timings,
              "accuracy":"Exact against independent NumPy int64 GEMM and saved float32 outputs; valid QK entries, all PV entries",
              "timing":"One warmup then five rotated-order measurements per capture; includes mask classification, planning, packing, FFI, dot, merge and publication; excludes common quantization",
              "workspace":"Batch maximum includes queued buffers/descriptors and native private accumulator/scratch, not process peak or Python intermediates; legacy field counts sparse workspace only",
              "limitations":["CPU component time, not Gemmini/IM2P cycles or whole-model speed",
                             "Prepared Python reference operands remain int64; submitted native operands are signed-nibble int8",
                             "Int64 overflow is rejected; arbitrary exact wide accumulator remains separate work"],
              "source_sha256":{p.name:digest(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".cpp")}}
    with output.open("x") as target:
        json.dump(report, target, indent=2)
    print(json.dumps({"complete":str(output), "captures":len(timings)}), flush=True)


def attention_replay(args: list[str], skip_qk_tiles: bool = False, batch_native: bool = False) -> None:
    """Paired native-model accuracy replay, plus isolated CPU component timings.

    Extends the existing native runner instead of implementing another Llama
    forward. The native graph still executes its original floating dots; its
    end-to-end wall time therefore is diagnostic, not a backend speedup claim.
    """
    import hashlib
    import platform

    import numpy as np

    from .profile_gemm import QKTileMask, gemm
    from .run import digest, peak_rss_bytes

    if len(args) != 6:
        raise ValueError("attention LIBRARY MODEL TEXT CONTEXT CHUNKS NEW_OUTPUT_DIRECTORY")
    library, model, text_path = (Path(v).resolve() for v in args[:3])
    context, chunks = map(int, args[3:5])
    output = Path(args[5]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if context % 32 or not 32 <= context <= 1024 or chunks < 1:
        raise ValueError("Require context multiple of 32 in 32..1024 and positive chunks")
    lib = ct.CDLL(str(library))
    lib.potal_replay_abi_version.restype = ct.c_int
    if lib.potal_replay_abi_version() != 2:
        raise RuntimeError("Rebuild native replay: mask-aware ABI version 2 required")
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int64)
    hook_type = ct.CFUNCTYPE(ct.c_int, ct.c_int, ct.c_int, fp, fp, fp, ip, ip, ip, fp, ip, ct.c_void_p)
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.potal_replay_open.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, hook_type, ct.c_void_p]
    lib.potal_replay_open.restype = ct.c_void_p
    lib.potal_replay_close.argtypes = [ct.c_void_p]
    lib.potal_replay_error.restype = ct.c_char_p
    lib.potal_replay_tokenize.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int, i32, ct.c_int]
    lib.potal_replay_tokenize.restype = ct.c_int
    lib.potal_replay_eval.argtypes = [ct.c_void_p, i32, ct.c_int, f64]
    lib.potal_replay_eval.restype = ct.c_int
    lib.potal_replay_step.argtypes = [ct.c_void_p, i32, i32, i32, ct.c_int, ct.c_int]
    lib.potal_replay_step.restype = ct.c_int
    lib.potal_replay_vocabulary.argtypes = [ct.c_void_p]
    lib.potal_replay_vocabulary.restype = ct.c_int
    lib.potal_replay_logits.argtypes = [ct.c_void_p]
    lib.potal_replay_logits.restype = fp

    # Independent integer dots, including every signed digit, tails, empty rows
    # and non-unit coefficients. No reliance on equal aggregate PPL for parity.
    rng = np.random.default_rng(0)
    primitive_checks = 0
    for m, k, n in ((1, 1, 1), (3, 31, 7), (33, 32, 35), (32, 32, 512)):
        for fixture in range(3):
            a = rng.integers(-8, 8, (m, k), dtype=np.int8)
            if fixture == 1: a = np.clip(a, -1, 1)
            if fixture == 2: a[rng.random(a.shape) < .9] = 0
            b = rng.integers(-8, 8, (k, n), dtype=np.int8)
            expected = a.astype(np.int64) @ b.astype(np.int64)
            for mode in range(3):
                assert np.array_equal(CpuDot(lib, mode)(a, b, 2), expected)
                primitive_checks += 1

    mode_name = ("dense", "sparse_multiply", "batch_direct" if batch_native else ("qk_mask_skip" if skip_qk_tiles else "sparse_unit_add"))
    state = {"mode": 0, "chunk": 0, "error": None, "trace": None, "times": None, "dot": None}
    histograms = {"qk": {}, "pv": {}}
    sample_paths = []
    observations = {"qk": Counter(), "pv": Counter()}
    shape_records = set()

    def reset_replay(mode: int, chunk: int, qa: bool = False) -> None:
        state.update(mode=mode, chunk=chunk, qa=qa, error=None, trace=hashlib.sha256(),
                     probabilities=hashlib.sha256(), times={}, fragments=Counter(), mask_samples=[],
                     dot=CpuBatch(lib) if batch_native and mode == 2 else CpuDot(lib, 0 if skip_qk_tiles else mode))

    def array(pointer, dims):
        shape = tuple(int(dims[i]) for i in (3, 2, 1, 0))
        return np.ctypeslib.as_array(pointer, shape=(int(np.prod(shape)),)).reshape(shape)[0]

    @hook_type
    def hook(role, layer, raw_stationary, raw_streamed, raw_output, shape_w, shape_x, shape_y,
             raw_mask, shape_mask, _):
        try:
            w, x, y = array(raw_stationary, shape_w), array(raw_streamed, shape_x), array(raw_output, shape_y)
            assert len(x) % len(w) == 0
            mask = None
            if role == 0:
                started = perf_counter_ns()
                mask_values = array(raw_mask, shape_mask)[0, :y.shape[1]]
                mask = QKTileMask.from_values(mask_values)
                assert (~mask.blocked).any(axis=1).all(), "Fully masked real query row"
                state["times"]["qk_mask_classify"] = state["times"].get("qk_mask_classify", 0) + perf_counter_ns() - started
                if skip_qk_tiles:
                    state["trace"].update(mask_values.tobytes())
                if state["qa"] and layer == 0:
                    state["mask_samples"].append(mask_values.copy())
            role_name = "qk" if role == 0 else "pv"
            shape_records.add((role_name, tuple(x.shape), tuple(w.shape), tuple(y.shape)))
            for head in range(len(x)):
                wh = w[head // (len(x)//len(w))]
                operand, plan, weights = prepare_head(role, x[head], wh, state["times"])
                use_batch = batch_native and state["mode"] == 2
                result = gemm(plan, weights, role_name, state["times"], 0 if role == 0 else 8, None if use_batch else state["dot"],
                              mask if skip_qk_tiles and state["mode"] == 2 else None,
                              state["fragments"] if role == 0 else None, state["dot"] if use_batch else None)
                assert np.isfinite(result).all()
                y[head] = result.astype(np.float32)
                # Hash exact packets, stationary codes/scales, and published
                # outputs for every head in every layer and chunk.
                trace = state["trace"]
                for packet in operand.packets:
                    for value in (packet.main, packet.digits, packet.row_lane, packet.columns, packet.scales):
                        trace.update(value.tobytes())
                trace.update(weights.theta.tobytes())
                for codes, _ in weights.passes: trace.update(codes.tobytes())
                trace.update((y[head][~mask.blocked] if skip_qk_tiles and mask is not None else y[head]).tobytes())
                if role == 1:
                    state["probabilities"].update(x[head].tobytes())
                if state["mode"] == 0 and not state["qa"]:
                    observation = observations[role_name]
                    observation["heads"] += 1
                    observation["values"] += x[head].size
                    observation["main_fragments"] += sum(p.main_fragments for p in operand.packets)
                    observation["upper_fragments_one_pass"] += sum(p.upper_fragments for p in operand.packets)
                    for stripe in plan:
                        for lane, rows, plane in stripe.lanes:
                            hist = histograms[role_name].setdefault(str(lane), Counter())
                            values, counts = np.unique(plane, return_counts=True)
                            hist.update({str(int(v)): int(c) for v, c in zip(values, counts)})
                    if state["chunk"] == 0 and layer in (0, 5, 8, 11, 15) and head in (0, len(x)//2, len(x)-1):
                        path = output / f"{role_name}-l{layer}-h{head}.npz"
                        saved_mask = {"mask": mask_values} if role == 0 else {}
                        np.savez_compressed(path, streamed=x[head], stationary=wh, result=y[head], **saved_mask)
                        sample_paths.append(path)
            return 0
        except Exception as error:
            import traceback
            state["error"] = traceback.format_exc()
            return 1

    handle = lib.potal_replay_open(str(model).encode(), context, 4, 1, hook, None)
    if not handle: raise RuntimeError(lib.potal_replay_error().decode())
    records = []
    try:
        text = text_path.read_bytes()
        buffer = np.empty(len(text)+16, np.int32)
        count = lib.potal_replay_tokenize(handle, text, len(text), buffer, len(buffer))
        assert count >= context*chunks, count
        tokens = buffer[:context*chunks].copy()
        tokens.tofile(output / "tokens.i32")
        vocabulary = lib.potal_replay_vocabulary(handle)
        for chunk in range(chunks):
            pair = {}
            # Alternate order to expose warm-cache/order bias. Neither model
            # forward wall time nor download/build overlap is a speedup result.
            for mode in ((0, 2) if chunk % 2 == 0 else (2, 0)):
                reset_replay(mode, chunk)
                nll = np.empty(context//2-1, np.float64)
                started = perf_counter_ns()
                calls = lib.potal_replay_eval(handle, tokens[chunk*context:(chunk+1)*context], context, nll)
                if calls < 0: raise RuntimeError(state["error"] or lib.potal_replay_error().decode())
                assert np.isfinite(nll).all()
                logits = np.ctypeslib.as_array(lib.potal_replay_logits(handle), shape=(len(nll), vocabulary))
                record = {"chunk":chunk, "mode":mode_name[mode], "callbacks":calls,
                          "nll":nll.tolist(), "nll_sum":float(nll.sum()), "ppl":float(np.exp(nll.mean())),
                          "logits_sha256":hashlib.sha256(logits.tobytes()).hexdigest(),
                          "attention_trace_sha256":state["trace"].hexdigest(),
                          "softmax_sha256":state["probabilities"].hexdigest(),
                          "qk_fragments":dict(state["fragments"]),
                          "diagnostic_replay_wall_ns":perf_counter_ns()-started,
                          "component_times_ns":state["times"], "cpu_kernel_times_ns":dict(state["dot"].times),
                          "max_sparse_workspace_bytes":state["dot"].max_extra_bytes}
                pair[mode] = record
                records.append(record)
                with (output / "chunks.jsonl").open("a") as target: target.write(json.dumps(record)+"\n")
                print(json.dumps({k:record[k] for k in ("chunk","mode","ppl","diagnostic_replay_wall_ns")}),flush=True)
            for key in ("nll", "logits_sha256", "attention_trace_sha256", "softmax_sha256"):
                assert pair[0][key] == pair[2][key], (chunk, key)
    finally:
        lib.potal_replay_close(handle)

    mask_scenarios = []
    if skip_qk_tiles:
        capacity = max(context, 64)
        steps = (("tail-prefill", 0, 33, 0, True, False),
                 ("prefix", 0, 31, 0, True, False),
                 ("decode", 31, 1, 31, False, False),
                 ("cached-suffix", 32, 17, 32, False, False),
                 ("packed-sequences", 0, 34, 0, True, True))
        paired_steps = {}
        for mode in (0, 2):
            handle = lib.potal_replay_open(str(model).encode(), capacity, 4, 2, hook, None)
            if not handle:
                raise RuntimeError(lib.potal_replay_error().decode())
            key_positions = np.empty(0, np.int32)
            key_sequences = np.empty(0, np.int32)
            try:
                for name, first, count, position, clear, packed in steps:
                    reset_replay(mode, -1, True)
                    positions = np.tile(np.arange(17, dtype=np.int32), 2) if packed else np.arange(position, position + count, dtype=np.int32)
                    sequences = np.repeat(np.arange(2, dtype=np.int32), 17) if packed else np.zeros(count, np.int32)
                    key_positions = positions if clear else np.concatenate((key_positions, positions))
                    key_sequences = sequences if clear else np.concatenate((key_sequences, sequences))
                    step_tokens = np.resize(tokens, first + count)[first:first + count].copy()
                    calls = lib.potal_replay_step(handle, step_tokens, positions, sequences, count, clear)
                    if calls < 0:
                        raise RuntimeError(state["error"] or lib.potal_replay_error().decode())
                    actual = state["mask_samples"][0]
                    expected = ((key_positions[None, :] > positions[:, None]) |
                                (key_sequences[None, :] != sequences[:, None]))
                    assert np.array_equal(np.isneginf(actual[:, :len(key_positions)]), expected), name
                    assert np.isneginf(actual[:, len(key_positions):]).all(), name
                    logits = np.ctypeslib.as_array(lib.potal_replay_logits(handle), shape=(count, vocabulary))
                    record = {"scenario":name, "mode":mode_name[mode], "callbacks":calls,
                              "query_rows":count, "populated_keys":len(key_positions), "physical_keys":actual.shape[1],
                              "actual_mask_matches_positions_and_sequences":True,
                              "trace":state["trace"].hexdigest(), "softmax":state["probabilities"].hexdigest(),
                              "logits":hashlib.sha256(logits.tobytes()).hexdigest(), "fragments":dict(state["fragments"])}
                    if mode == 0:
                        paired_steps[name] = record
                    else:
                        for key in ("trace", "softmax", "logits"):
                            assert record[key] == paired_steps[name][key], (name, key)
                    mask_scenarios.append(record)
                    print(json.dumps(record), flush=True)
            finally:
                lib.potal_replay_close(handle)
        (output / "mask-scenarios.json").write_text(json.dumps(mask_scenarios, indent=2) + "\n")

    timings = measure_components(lib, sample_paths, skip_qk_tiles, batch_native)

    summary = {
        "scope":"Native model with experimental QK/PV output replacement; exact PoTal attention, CPU component timing only",
        "limitations":["Original native floating dots still execute; replay wall time is not integrated backend latency",
                       "CPU timings do not predict Gemmini cycles, transfers, overlap, or full-model speedup",
                       "Native existing linear path/geometry is retained; this does not certify original-spec linear integration"],
        "candidate":("Batch existing QK/PV raw K32 dots per stripe and accumulate valid regions directly in int64; QK fully masked tile skip"
                     if batch_native else "QK only: omit all-negative-infinity DIM32 tiles using actual graph masks and original residual rows"
                     if skip_qk_tiles else "Only upper lanes >=2: skip zero, add/subtract +/-1, multiply every other digit exactly"),
        "model":str(model),"model_sha256":digest(model),"text":str(text_path),"text_sha256":digest(text_path),
        "library":str(library),"library_sha256":digest(library),"tokens_sha256":digest(output/"tokens.i32"),
        "context":context,"chunks":chunks,"scored_tokens":chunks*(context//2-1),"primitive_checks":primitive_checks,
        "attention_geometry":{"banks":4,"bank_rows":4096,"acc_rows":512,"dim":32},
        "bitwise_packet_scale_attention_logits_and_token_nll":True,
        "qk_comparison":"valid entries only" if skip_qk_tiles else "all entries",
        "bitwise_softmax":True, "mask_scenarios":mask_scenarios,
        "fragment_accounting":"Executed per-lane replay dots, DIM32 padded; separate from pooled specification upper fragments",
        "ppl":{mode_name[mode]:float(np.exp(sum(r["nll_sum"] for r in records if r["mode"]==mode_name[mode])/(chunks*(context//2-1)))) for mode in (0,2)},
        "observations":observations,"digit_histograms":histograms,"shapes":sorted(shape_records),
        "platform":platform.platform(),"numpy":np.__version__,"native_threads":4,"cpu_dot_threads":1,
        "peak_rss_bytes":peak_rss_bytes(),"component_timings":timings,
        "source_sha256":{p.name:digest(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py",".cpp")}}
    (output/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"complete":str(output/"report.json"),"ppl":summary["ppl"]}),flush=True)


class CudaDot:
    """cuBLAS int8->int32 dots; original host integer SCU remains the owner."""
    def __init__(self) -> None:
        import numpy as np
        self.np = np
        self.rt = ct.CDLL("libcudart.so")
        self.blas = ct.CDLL("libcublas.so")
        self.rt.cudaGetErrorString.argtypes = [ct.c_int]
        self.rt.cudaGetErrorString.restype = ct.c_char_p
        self.rt.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        self.rt.cudaFree.argtypes = [ct.c_void_p]
        self.rt.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
        self.blas.cublasCreate_v2.argtypes = [ct.POINTER(ct.c_void_p)]
        self.blas.cublasDestroy_v2.argtypes = [ct.c_void_p]
        self.blas.cublasGetVersion_v2.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.blas.cublasGemmEx.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
            ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int,
            ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        self.handle = ct.c_void_p()
        self.check_blas(self.blas.cublasCreate_v2(ct.byref(self.handle)))
        self.alpha, self.beta = ct.c_int32(1), ct.c_int32(0)
        self.pool = {}
        self.retained_bytes = 0
        self.calls = 0
        self.transfer_bytes = 0
        self.versions = {}
        for name in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
            fn = getattr(self.rt, name)
            fn.argtypes = [ct.POINTER(ct.c_int)]
            value = ct.c_int()
            self.check(fn(ct.byref(value)))
            self.versions[name] = value.value
        value = ct.c_int()
        self.check_blas(self.blas.cublasGetVersion_v2(self.handle, ct.byref(value)))
        self.versions["cublasGetVersion"] = value.value

    def check(self, status):
        if status:
            raise RuntimeError(self.rt.cudaGetErrorString(status).decode())

    @staticmethod
    def check_blas(status):
        if status:
            raise RuntimeError(f"cuBLAS status {status}")

    def __call__(self, a, b, lane):
        np = self.np
        m, k = a.shape
        n = b.shape[1]
        if not (m > 0 and n > 0 and 0 < k <= 32 and b.shape[0] == k):
            raise ValueError("CUDA raw dot requires positive dimensions and K <= 32")
        # Keep DIM32 fragment padding, including tails and compacted K blocks.
        mpad, kp, npad = (m+31)//32*32, 32, (n+31)//32*32
        shape = mpad, kp, npad
        if shape not in self.pool:
            aa = np.zeros((mpad, kp), np.int8)
            bb = np.zeros((npad, kp), np.int8)
            out = np.empty((mpad, npad), np.int32)
            pointers = []
            try:
                for array in (aa, bb, out):
                    p = ct.c_void_p()
                    self.check(self.rt.cudaMalloc(ct.byref(p), array.nbytes))
                    pointers.append(p)
            except Exception:
                for p in pointers: self.rt.cudaFree(p)
                raise
            self.retained_bytes += aa.nbytes + bb.nbytes + out.nbytes
            self.pool[shape] = aa, bb, out, pointers
        aa, bb, out, (da, db, dc) = self.pool[shape]
        aa[:m, :k] = a
        bb[:n, :k] = b.T
        # A smaller matrix can reuse a larger rounded shape. Clear old padding.
        aa[:, k:] = 0
        aa[m:, :] = 0
        bb[n:, :] = 0
        bb[:, k:] = 0
        self.check(self.rt.cudaMemcpy(da, aa.ctypes.data, aa.nbytes, 1))
        self.check(self.rt.cudaMemcpy(db, bb.ctypes.data, bb.nbytes, 1))
        # Column-major TN layout for cuBLAS IMMA. CUDA_R_8I=3,
        # CUDA_R_32I=10, CUBLAS_COMPUTE_32I=72; no floating multiply.
        self.check_blas(self.blas.cublasGemmEx(self.handle, 1, 0, npad, mpad, kp,
            ct.byref(self.alpha), db, 3, kp, da, 3, kp,
            ct.byref(self.beta), dc, 10, npad, 72, -1))
        # Blocking D2H makes all device work part of the measured call.
        self.check(self.rt.cudaMemcpy(out.ctypes.data, dc, out.nbytes, 2))
        self.calls += 1
        self.transfer_bytes += aa.nbytes + bb.nbytes + out.nbytes
        return out[:m, :n].astype(np.int64)

    def close(self):
        for _, _, _, pointers in self.pool.values():
            for pointer in pointers:
                self.check(self.rt.cudaFree(pointer))
        self.pool.clear()
        self.check_blas(self.blas.cublasDestroy_v2(self.handle))


def cuda_replay(args: list[str]) -> None:
    """Four-way timings on saved native-model heads, with independent parity."""
    import os
    import platform
    import numpy as np
    from .profile_gemm import gemm
    from .run import digest

    if len(args) != 3:
        raise ValueError("cuda LIBRARY CAPTURE_DIRECTORY NEW_REPORT_JSON")
    library, capture_root, output = (Path(v).resolve() for v in args)
    paths = sorted(capture_root.glob("*-l*-h*.npz"))
    if output.exists() or not paths:
        raise ValueError("Require saved captures and a new report path")
    lib = ct.CDLL(str(library))
    gpu = CudaDot()
    names = ("dense", "sparse_multiply", "sparse_unit_add", "cuda_dense")
    dots = [CpuDot(lib, mode) for mode in range(3)] + [gpu]
    records, primitive_checks = [], 0
    try:
        rng = np.random.default_rng(5)
        # Last two shapes exercise cache reuse with changing row/K/column tails.
        for m, k, n in ((1,1,1), (3,31,7), (33,32,35), (64,32,512), (128,32,64), (3,29,5), (3,31,7)):
            for fixture in range(3):
                a = rng.integers(-8, 8, (m,k), dtype=np.int8)
                if fixture == 1: a = np.clip(a, -1, 1)
                if fixture == 2: a[rng.random(a.shape) < .9] = 0
                b = rng.integers(-8, 8, (k,n), dtype=np.int8)
                expected = a.astype(np.int64) @ b.astype(np.int64)
                for dot in dots:
                    assert np.array_equal(dot(a, b, 2), expected)
                    primitive_checks += 1
        print(json.dumps({"integer_primitive_checks":primitive_checks}), flush=True)
        for path in paths:
            with np.load(path) as saved:
                role = 0 if path.name.startswith("qk") else 1
                preparation = {}
                _, plan, weights = prepare_head(role, saved["streamed"], saved["stationary"], preparation)
                role_name, zp = ("qk", 0) if role == 0 else ("pv", 8)
                reference = gemm(plan, weights, role_name, {}, zp)
                assert np.array_equal(reference.astype(np.float32), saved["result"])
            runs = []
            # Warm all modes once, then rotate order across five measurements.
            for repeat in range(6):
                order = list(range(4))
                order = order[repeat % 4:] + order[:repeat % 4]
                for mode in order:
                    dot = dots[mode]
                    if mode < 3: dot.times.clear()
                    before_calls, before_bytes = gpu.calls, gpu.transfer_bytes
                    times = {}
                    start = perf_counter_ns()
                    result = gemm(plan, weights, role_name, times, zp, dot)
                    elapsed = perf_counter_ns()-start
                    assert np.array_equal(result, reference), (path.name, names[mode])
                    if repeat:
                        run = {"mode":names[mode], "repeat":repeat, "gemm_ns":elapsed,
                               "component_times_ns":times}
                        if mode < 3:
                            run.update(kernel_times_ns=dict(dot.times), max_sparse_workspace_bytes=dot.max_extra_bytes)
                        else:
                            run.update(cuda_dot_calls=gpu.calls-before_calls, copy_bytes=gpu.transfer_bytes-before_bytes)
                        runs.append(run)
            records.append({"capture":path.name, "sha256":digest(path), "bitwise_integer_gemm":True,
                            "published_float32_equal":True, "preparation_times_ns":preparation, "runs":runs})
            print("timed " + path.name, flush=True)
        report = {
            "scope":"CPU/CUDA component replay of saved original-policy Llama QK/PV heads; original host SCU unchanged",
            "candidate":"CPU upper lanes >=2: skip zero, specialize +/-1; CUDA uses dense int8 dots for every lane",
            "cuda_api":"cublasGemmEx/CUBLAS_COMPUTE_32I", "cuda_versions":gpu.versions,
            "fragment_padding":32, "primitive_checks":primitive_checks,
            "integer_and_published_bitwise":True, "component_timings":records,
            "cuda_dot_calls":gpu.calls, "host_device_copy_bytes":gpu.transfer_bytes,
            "retained_device_workspace_bytes":gpu.retained_bytes, "retained_host_workspace_bytes":gpu.retained_bytes,
            "workspace_excludes":"cuBLAS/runtime internal allocations, returned arrays, and common host SCU buffers",
            "limitations":["Includes padding, dtype conversion, copies, synchronization, and original host SCU/zero point",
                           "Quantization is common preparation excluded from GEMM timing",
                           "One warmup per mode/capture; five rotated-order measurements with retained GPU buffers",
                           "This is neither full-model CUDA PPL nor Gemmini latency; native linear path is unchanged"],
            "library":str(library), "library_sha256":digest(library),
            "source_sha256":{p.name:digest(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".cpp")},
            "environment":{key:os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "platform":platform.platform(), "numpy":np.__version__}
        with output.open("x") as target: json.dump(report, target, indent=2)
        print(json.dumps({"complete":str(output), "captures":len(records)}), flush=True)
    finally:
        gpu.close()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "batch-components":
        batch_components(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "qk-batch":
        attention_replay(sys.argv[2:], skip_qk_tiles=True, batch_native=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "qk-mask":
        attention_replay(sys.argv[2:], skip_qk_tiles=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "attention":
        attention_replay(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "cuda":
        cuda_replay(sys.argv[2:])
        return
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
