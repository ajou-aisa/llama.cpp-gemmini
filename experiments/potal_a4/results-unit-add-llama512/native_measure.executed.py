from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def attention_replay(args: list[str]) -> None:
    """Paired native-model accuracy replay, plus isolated CPU component timings.

    Extends the existing native runner instead of implementing another Llama
    forward. The native graph still executes its original floating dots; its
    end-to-end wall time therefore is diagnostic, not a backend speedup claim.
    """
    import ctypes as ct
    from collections import Counter
    import hashlib
    import platform
    from time import perf_counter_ns

    import numpy as np

    from .profile_gemm import checked_bound, gemm, unpack
    from .profile_ops import probability, query, stationary_codes
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
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int64)
    hook_type = ct.CFUNCTYPE(ct.c_int, ct.c_int, ct.c_int, fp, fp, fp, ip, ip, ip, ct.c_void_p)
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.potal_replay_dot.argtypes = [i8, i8, i32, ct.c_int, ct.c_int, ct.c_int, ct.c_int, u64]
    lib.potal_replay_dot.restype = ct.c_int
    lib.potal_replay_open.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, hook_type, ct.c_void_p]
    lib.potal_replay_open.restype = ct.c_void_p
    lib.potal_replay_close.argtypes = [ct.c_void_p]
    lib.potal_replay_error.restype = ct.c_char_p
    lib.potal_replay_tokenize.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int, i32, ct.c_int]
    lib.potal_replay_tokenize.restype = ct.c_int
    lib.potal_replay_eval.argtypes = [ct.c_void_p, i32, ct.c_int, f64]
    lib.potal_replay_eval.restype = ct.c_int
    lib.potal_replay_vocabulary.argtypes = [ct.c_void_p]
    lib.potal_replay_vocabulary.restype = ct.c_int
    lib.potal_replay_logits.argtypes = [ct.c_void_p]
    lib.potal_replay_logits.restype = fp

    class Dot:
        def __init__(self, mode: int) -> None:
            self.mode = mode
            self.times = Counter()
            self.max_extra_bytes = 0

        def __call__(self, a, b, lane):
            begin = perf_counter_ns()
            aa = np.ascontiguousarray(a, dtype=np.int8)
            bb = np.ascontiguousarray(b, dtype=np.int8)
            result = np.empty((len(aa), bb.shape[1]), dtype=np.int32)
            stats = np.zeros(4, np.uint64)
            mode = self.mode if lane >= 2 else 0
            status = lib.potal_replay_dot(aa, bb, result, len(aa), aa.shape[1], bb.shape[1], mode, stats)
            if status: raise RuntimeError("Raw dot rejected shape")
            result = result.astype(np.int64)
            key = "upper2plus" if lane >= 2 else "main_or_upper1"
            self.times[key + "_pack_ns"] += int(stats[0])
            self.times[key + "_dot_ns"] += int(stats[1])
            self.times[key + "_call_ns"] += perf_counter_ns()-begin
            self.max_extra_bytes = max(self.max_extra_bytes, int(stats[3]))
            return result

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
                assert np.array_equal(Dot(mode)(a, b, 2), expected)
                primitive_checks += 1

    def prepare_head(role, streamed, stationary, times):
        if role == 0:
            operand, _ = query(streamed, len(stationary), times)
            weights = stationary_codes(stationary.T, 8, times, "k")
        else:
            operand = probability(streamed, stationary.shape[0], times)
            weights = stationary_codes(stationary.T, 4, times, "v")
        plan = unpack(operand)
        checked_bound(plan, weights, 0 if role == 0 else 8)
        return operand, plan, weights

    mode_name = ("dense", "sparse_multiply", "sparse_unit_add")
    state = {"mode": 0, "chunk": 0, "error": None, "trace": None, "times": None, "dot": None}
    histograms = {"qk": {}, "pv": {}}
    sample_paths = []
    observations = {"qk": Counter(), "pv": Counter()}
    shape_records = set()

    def array(pointer, dims):
        shape = tuple(int(dims[i]) for i in (3, 2, 1, 0))
        return np.ctypeslib.as_array(pointer, shape=(int(np.prod(shape)),)).reshape(shape)[0]

    @hook_type
    def hook(role, layer, raw_stationary, raw_streamed, raw_output, shape_w, shape_x, shape_y, _):
        try:
            w, x, y = array(raw_stationary, shape_w), array(raw_streamed, shape_x), array(raw_output, shape_y)
            assert len(x) % len(w) == 0
            role_name = "qk" if role == 0 else "pv"
            shape_records.add((role_name, tuple(x.shape), tuple(w.shape), tuple(y.shape)))
            for head in range(len(x)):
                wh = w[head // (len(x)//len(w))]
                operand, plan, weights = prepare_head(role, x[head], wh, state["times"])
                result = gemm(plan, weights, role_name, state["times"], 0 if role == 0 else 8, state["dot"])
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
                trace.update(y[head].tobytes())
                if state["mode"] == 0:
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
                        np.savez_compressed(path, streamed=x[head], stationary=wh, result=y[head])
                        sample_paths.append(path)
            return 0
        except Exception as error:
            import traceback
            state["error"] = traceback.format_exc()
            return 1

    handle = lib.potal_replay_open(str(model).encode(), context, 4, hook, None)
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
                state.update(mode=mode, chunk=chunk, error=None, trace=hashlib.sha256(), times={}, dot=Dot(mode))
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
                          "diagnostic_replay_wall_ns":perf_counter_ns()-started,
                          "component_times_ns":state["times"], "cpu_kernel_times_ns":dict(state["dot"].times),
                          "max_sparse_workspace_bytes":state["dot"].max_extra_bytes}
                pair[mode] = record
                records.append(record)
                with (output / "chunks.jsonl").open("a") as target: target.write(json.dumps(record)+"\n")
                print(json.dumps({k:record[k] for k in ("chunk","mode","ppl","diagnostic_replay_wall_ns")}),flush=True)
            for key in ("nll", "logits_sha256", "attention_trace_sha256"):
                assert pair[0][key] == pair[2][key], (chunk, key)
    finally:
        lib.potal_replay_close(handle)

    # Time fixed real-model inputs after end-to-end numerical replay. All three
    # modes preserve every digit; sparse_multiply isolates zero skipping from
    # the additional +/-1 specialization. Quantization is common preparation.
    timings = []
    for path in sample_paths:
        saved = np.load(path)
        role = 0 if path.name.startswith("qk") else 1
        _, plan, weights = prepare_head(role, saved["streamed"], saved["stationary"], {})
        reference = gemm(plan, weights, "qk" if role == 0 else "pv", {}, 0 if role == 0 else 8)
        assert np.array_equal(reference.astype(np.float32), saved["result"])
        repeats = []
        for repeat in range(6):
            for mode in ((0, 1, 2) if repeat % 2 == 0 else (2, 1, 0)):
                dot = Dot(mode)
                started = perf_counter_ns()
                result = gemm(plan, weights, "qk" if role == 0 else "pv", {}, 0 if role == 0 else 8, dot)
                elapsed = perf_counter_ns()-started
                assert np.array_equal(result, reference)
                if repeat:
                    repeats.append({"mode":mode_name[mode], "repeat":repeat, "gemm_ns":elapsed,
                                    "kernel_times_ns":dict(dot.times), "max_sparse_workspace_bytes":dot.max_extra_bytes})
        timings.append({"capture":path.name, "sha256":digest(path), "bitwise_integer_gemm":True, "runs":repeats})
        print("timed "+path.name,flush=True)

    summary = {
        "scope":"Native model with experimental QK/PV output replacement; exact PoTal attention, CPU component timing only",
        "limitations":["Original native floating dots still execute; replay wall time is not integrated backend latency",
                       "CPU timings do not predict Gemmini cycles, transfers, overlap, or full-model speedup",
                       "Native existing linear path/geometry is retained; this does not certify original-spec linear integration"],
        "candidate":"Only upper lanes >=2: skip zero, add/subtract +/-1, multiply every other digit exactly",
        "model":str(model),"model_sha256":digest(model),"text":str(text_path),"text_sha256":digest(text_path),
        "library":str(library),"library_sha256":digest(library),"tokens_sha256":digest(output/"tokens.i32"),
        "context":context,"chunks":chunks,"scored_tokens":chunks*(context//2-1),"primitive_checks":primitive_checks,
        "attention_geometry":{"banks":4,"bank_rows":4096,"acc_rows":512,"dim":32},
        "bitwise_packet_scale_attention_logits_and_token_nll":True,
        "ppl":{mode_name[mode]:float(np.exp(sum(r["nll_sum"] for r in records if r["mode"]==mode_name[mode])/(chunks*(context//2-1)))) for mode in (0,2)},
        "observations":observations,"digit_histograms":histograms,"shapes":sorted(shape_records),
        "platform":platform.platform(),"numpy":np.__version__,"native_threads":4,"cpu_dot_threads":1,
        "peak_rss_bytes":peak_rss_bytes(),"component_timings":timings,
        "source_sha256":{p.name:digest(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py",".cpp")}}
    (output/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"complete":str(output/"report.json"),"ppl":summary["ppl"]}),flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "attention":
        attention_replay(sys.argv[2:])
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
