# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2"]
# ///
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analyze_lm_head import digest, logit_metrics
from weight_io import ROOT, GGUFReader, hp1_values, quant_values

HERE = Path(__file__).resolve().parent
LIBRARY = ROOT / "build-metal-llama-full/quality-potal4-d16/bin/libggml-base.dylib"
FloatArray = NDArray[np.float64]


def rotate(values: FloatArray, signs: FloatArray) -> FloatArray:
    result = np.array(values * signs, dtype=np.float64, copy=True)
    blocks = result.reshape(-1, 256)
    stride = 1
    while stride < 256:
        pairs = blocks.reshape(-1, 2 * stride)
        left, right = pairs[:, :stride].copy(), pairs[:, stride:].copy()
        pairs[:, :stride], pairs[:, stride:] = left + right, left - right
        stride *= 2
    return result / 16


def self_check() -> None:
    rng = np.random.default_rng(42)
    h, w, error = rng.normal(size=(5, 768)), rng.normal(size=(11, 768)), rng.normal(size=(11, 768))
    signs = rng.choice([-1.0, 1.0], 768)
    projection = np.linalg.qr(rng.normal(size=(768, 3)))[0]
    mean = rng.normal(size=768)
    q = w + error
    np.testing.assert_allclose(rotate(h, signs) @ rotate(w, signs).T, h @ w.T, atol=1e-12)
    np.testing.assert_allclose(rotate(h, signs) @ rotate(error, signs).T, h @ error.T, atol=1e-12)
    np.testing.assert_allclose((h - mean) @ q.T + mean @ w.T - h @ w.T, (h - mean) @ error.T, atol=1e-12)
    projected = h @ projection
    corrected = h @ q.T - projected @ (error @ projection).T
    np.testing.assert_allclose(corrected - h @ w.T, (h - projected @ projection.T) @ error.T, atol=1e-12)


def layer_traces() -> dict:
    report = {}
    model_paths = {
        "gpt2-fp16": ROOT / "models/gpt2.fp16.gguf",
        "gpt2-q4": ROOT / "output/experiment/potal-ppl-20261001/gpt2.Q4_0.head-original-F16.gguf",
        "llama-q4": ROOT / "output/experiment/potal-ppl-20261001/llama3.2-1B.Q4_0.head-original-F16.gguf",
    }
    for name in ("gpt2-fp16", "gpt2-q4", "llama-q4"):
        directory = HERE / "data" / f"{name}-layer-trace"
        nodes: dict[int, dict] = {}
        with (directory / "channels.csv").open() as source:
            for row in csv.DictReader(source):
                node = nodes.setdefault(int(row["ordinal"]), dict(name=row["node"], op=row["op"], rows=int(row["rows"]),
                                                                  mean=[], mean_square=[], max_abs=[]))
                assert int(row["channel"]) == len(node["mean"])
                assert int(row["rows"]) == 512
                for key in ("mean", "mean_square", "max_abs"):
                    node[key].append(float(row[key]))
        top = [496, 430, 36] if name.startswith("gpt2") else [1564, 1645, 1021]
        named = {node["name"]: node for node in nodes.values()}
        for node in nodes.values():
            energy = np.array(node["mean_square"])
            node["top3_energy_percent"] = float(100 * energy[top].sum() / energy.sum())
            node["top3_rms"] = np.sqrt(energy[top]).tolist()
            node["top3_mean"] = np.array(node["mean"])[top].tolist()
            node["all_channel_rms"] = float(np.sqrt(energy.mean()))
        final = np.fromfile(directory / "head-input.f32", dtype=np.float32).reshape(512, -1).astype(np.float64)
        np.testing.assert_allclose(final.mean(axis=0), named["result_norm"]["mean"], atol=1e-12)
        np.testing.assert_allclose(np.square(final).mean(axis=0), named["result_norm"]["mean_square"], atol=1e-12)
        branches = []
        previous = named["inpL"] if name.startswith("gpt2") else named["inp_embd"]
        for block in range(12 if name.startswith("gpt2") else 16):
            ffn_input, ffn_output, output = (named[f"{label}-{block}"] for label in ("ffn_inp", "ffn_out", "l_out"))
            np.testing.assert_allclose(np.array(ffn_input["mean"]) + ffn_output["mean"], output["mean"], atol=2e-5)
            for channel in top:
                branches.append(dict(block=block, channel=channel,
                                     attention_mean_increment=ffn_input["mean"][channel] - previous["mean"][channel],
                                     ffn_mean_increment=ffn_output["mean"][channel],
                                     output_mean=output["mean"][channel], output_rms=output["mean_square"][channel] ** 0.5))
            previous = output
        compact = [{key: value for key, value in node.items() if key not in ("mean", "mean_square", "max_abs")}
                   for node in nodes.values()]
        report[name] = dict(channels=top, nodes=compact, branches=branches,
                            model_path=str(model_paths[name]), model_sha256=digest(model_paths[name]),
                            input_sha256=digest(directory / "head-input.f32"),
                            statistics_sha256=digest(directory / "channels.csv"),
                            tokens_sha256=digest(directory / "tokens.i32"))
    assert report["gpt2-fp16"]["tokens_sha256"] == report["gpt2-q4"]["tokens_sha256"]
    return report


def normalization(h: FloatArray, reader: GGUFReader, top: NDArray[np.int64]) -> dict:
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gamma = tensors["output_norm.weight"].data.astype(np.float64).reshape(-1)
    beta_tensor = tensors.get("output_norm.bias")
    beta = np.zeros_like(gamma) if beta_tensor is None else beta_tensor.data.astype(np.float64).reshape(-1)
    assert np.all(gamma != 0)
    before = (h - beta) / gamma
    def summary(values: FloatArray) -> dict:
        energy = np.square(values).mean(axis=0)
        mean = values.mean(axis=0)
        return dict(top3_energy_percent=float(100 * energy[top].sum() / energy.sum()),
                    mean_vector_energy_percent=float(100 * np.square(mean).sum() / energy.sum()),
                    top3_rms=np.sqrt(energy[top]).tolist(), top3_mean=mean[top].tolist(),
                    top3_std=values.std(axis=0)[top].tolist(),
                    row_mean_range=[float(values.mean(axis=1).min()), float(values.mean(axis=1).max())],
                    row_mean_square_range=[float(np.square(values).mean(axis=1).min()), float(np.square(values).mean(axis=1).max())])
    if beta_tensor is not None:
        np.testing.assert_allclose(before.mean(axis=1), 0, atol=6e-8)
    normalized_energy = np.square(before).mean(axis=1)
    assert np.all(normalized_energy > 0) and np.all(normalized_energy <= 1 + 2e-6)
    return dict(gamma_median=float(np.median(gamma)), gamma_max=float(gamma.max()),
                top3_gamma=gamma[top].tolist(), top3_beta=beta[top].tolist(),
                before_affine=summary(before), after_affine=summary(h))


def native_quantize(library: ctypes.CDLL, values: FloatArray, bits: int) -> tuple[NDArray[np.uint8], FloatArray]:
    values_f32 = np.ascontiguousarray(values, dtype=np.float32)
    packed = np.empty((values.shape[0], values.shape[1] // 32 * (24 if bits == 4 else 40)), dtype=np.uint8)
    function = getattr(library, f"quantize_q{bits}_hp1")
    assert function(values_f32, packed, *values.shape, None) == packed.nbytes
    restored, _ = hp1_values(packed, bits)
    return packed, restored.reshape(values.shape).astype(np.float64)


def replay(model: str, library: ctypes.CDLL) -> dict:
    previous = json.loads((HERE / "data/lm-head-diagnostics.json").read_text())[model]
    paths = {key: Path(value) for key, value in previous["paths"].items()}
    for key, path in paths.items():
        assert digest(path) == previous["model_sha256"][key]
    readers = {key: GGUFReader(str(path)) for key, path in paths.items()}
    heads = {key: next(t for t in reader.tensors if t.name == "token_embd.weight") for key, reader in readers.items()}
    vocab, width = previous["head_shape"]
    capture = Path(previous["capture"]["native_run"]) / "head-input.f32"
    assert digest(capture) == previous["capture"]["head_input_sha256"]
    all_h = np.fromfile(capture, dtype=np.float32).reshape(8192, width).astype(np.float64)
    calibration = all_h[:4096]
    test_rows = (np.arange(16, 32)[:, None] * 256 + np.array([0, 64, 128, 192])).reshape(-1)
    inputs = all_h[test_rows]
    top = np.argsort(np.square(calibration).mean(axis=0))[-3:][::-1]
    mean = calibration.mean(axis=0)
    gram = calibration.T @ calibration / len(calibration)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    basis = np.ascontiguousarray(eigenvectors[:, -3:][:, ::-1])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(gram @ basis, basis * eigenvalues[-3:][::-1], atol=1e-9)
    projected = inputs @ basis
    report = dict(paths=previous["paths"], model_sha256=previous["model_sha256"],
                  head_input_sha256=previous["capture"]["head_input_sha256"],
                  calibration_rows=[0, 4096], test_rows=test_rows.tolist(), top3_channels=top.tolist(),
                  principal_energy_percent=(100 * eigenvalues[-3:][::-1] / eigenvalues.sum()).tolist(),
                  normalization=normalization(all_h, readers["FP16"], top),
                  precisions={})
    del all_h, gram, eigenvectors
    reference_logits = np.empty((len(inputs), vocab), dtype=np.float64)
    drop_logits = np.empty_like(reference_logits)
    fp_columns = np.empty((vocab, 3), dtype=np.float64)
    column_max = np.zeros(width)
    head_hash = hashlib.sha256()
    for start in range(0, vocab, 256):
        end = min(vocab, start + 256)
        raw = heads["FP16"].data.reshape(vocab, width)[start:end]
        head_hash.update(raw.tobytes())
        weights = raw.astype(np.float64)
        fp_columns[start:end] = weights[:, top]
        reference_logits[:, start:end] = inputs @ weights.T
        drop_logits[:, start:end] = reference_logits[:, start:end] - inputs[:, top] @ weights[:, top].T
        column_max = np.maximum(column_max, np.abs(weights).max(axis=0))
    assert head_hash.hexdigest() == previous["reference_head_sha256"]
    report["drop3_from_fp16_without_correction"] = logit_metrics(reference_logits, drop_logits)
    report["fp16_top3_weight_columns"] = dict(mean=fp_columns.mean(axis=0).tolist(),
                                              std=fp_columns.std(axis=0).tolist(),
                                              correlation=np.corrcoef(fp_columns.T).tolist())
    del drop_logits
    scale = np.sqrt(np.abs(calibration).max(axis=0) / column_max)
    assert np.isfinite(scale).all() and np.all(scale > 0)
    scaled_inputs = inputs / scale
    signs = np.random.default_rng(42).choice([-1.0, 1.0], width)
    rotated_inputs = rotate(inputs, signs)
    report["rescale_alpha"] = 0.5
    report["rotation"] = "seed 42 signs then block-diagonal normalized Hadamard-256; head only"
    report["transform_unquantized_max_abs_error"] = {"rescale": 0.0, "rotation": 0.0}
    for bits in (4, 8):
        names = ["current_hp1", "exact_top3_columns", "exact_mean_bias", "exact_principal1", "exact_principal3"]
        if model == "gpt2":
            names.extend(["rescale_then_requantize", "rotate_then_requantize"])
        logits = {name: np.empty_like(reference_logits) for name in names}
        quantized_columns = np.empty_like(fp_columns)
        squared_error, top_error, column_error = 0.0, 0.0, np.zeros(width)
        for start in range(0, vocab, 256):
            end = min(vocab, start + 256)
            weights = heads["FP16"].data.reshape(vocab, width)[start:end].astype(np.float64)
            quantized, _ = quant_values(heads[f"INT{bits}"], start, end)
            quantized = quantized.reshape(-1, width).astype(np.float64)
            quantized_columns[start:end] = quantized[:, top]
            error = quantized - weights
            column_error += np.square(error).sum(axis=0)
            base = inputs @ quantized.T
            logits["current_hp1"][:, start:end] = base
            correction = inputs[:, top] @ error[:, top].T
            logits["exact_top3_columns"][:, start:end] = base - correction
            logits["exact_mean_bias"][:, start:end] = base - mean @ error.T
            projection_error = error @ basis
            logits["exact_principal1"][:, start:end] = base - projected[:, :1] @ projection_error[:, :1].T
            logits["exact_principal3"][:, start:end] = base - projected @ projection_error.T
            squared_error += float(np.square(error).sum())
            top_error += float(np.square(error[:, top]).sum())
            if model == "gpt2" or start == 0:
                packed, restored = native_quantize(library, weights, bits)
                stored = heads[f"INT{bits}"].data.reshape(vocab, -1)[start:end]
                assert np.array_equal(packed, stored), (model, bits, start, "HP1 producer mismatch")
                np.testing.assert_array_equal(restored, quantized)
            if model == "gpt2":
                _, scaled_weights = native_quantize(library, weights * scale, bits)
                logits["rescale_then_requantize"][:, start:end] = scaled_inputs @ scaled_weights.T
                rotated_weights = rotate(weights, signs)
                _, rotated_quantized = native_quantize(library, rotated_weights, bits)
                logits["rotate_then_requantize"][:, start:end] = rotated_inputs @ rotated_quantized.T
                for name, candidate in (("rescale", scaled_inputs @ (weights * scale).T),
                                        ("rotation", rotated_inputs @ rotated_weights.T)):
                    discrepancy = float(np.max(np.abs(candidate - reference_logits[:, start:end])))
                    report["transform_unquantized_max_abs_error"][name] = max(report["transform_unquantized_max_abs_error"][name], discrepancy)
                    np.testing.assert_allclose(candidate, reference_logits[:, start:end], atol=1e-10, rtol=1e-10)
            if start % 16384 == 0:
                print(f"{model} INT{bits}: {end}/{vocab} vocabulary rows", flush=True)
        report["precisions"][str(bits)] = dict(metrics={name: logit_metrics(reference_logits, value) for name, value in logits.items()},
                                               top3_weight_error_energy_percent=100 * top_error / squared_error,
                                               top3_weight_column_std=quantized_columns.std(axis=0).tolist(),
                                               top3_weight_error_std=(quantized_columns - fp_columns).std(axis=0).tolist(),
                                               weight_error_rms_per_channel=np.sqrt(column_error / vocab).tolist(),
                                               native_producer_byte_check_rows=vocab if model == "gpt2" else 256)
        print(model, bits, json.dumps(report["precisions"][str(bits)]["metrics"]), flush=True)
        del logits
    return report


def plots(report: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output = HERE / "figures"
    with PdfPages(output / "activation-amplification.pdf") as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
        colors = ("#c24e30", "#2876a4", "#3b916b")
        for index, name in enumerate(("gpt2-fp16", "gpt2-q4")):
            trace = report["traces"][name]
            nodes = [n for n in trace["nodes"] if n["name"] == "inpL" or n["name"].startswith("l_out-")]
            for ch, color, column in zip(trace["channels"], colors, range(3)):
                axes[0, index].plot(range(-1, 12), [n["top3_rms"][column] for n in nodes], "o-", color=color, label=f"feature {ch}")
            axes[0, index].set(title=f"{name}: residual-stream channel RMS", xlabel="Block index (0-based); -1 = input", ylabel="RMS over 512 tokens", yscale="log")
            axes[0, index].set_xticks(range(-1, 12))
            axes[0, index].legend()
        for index, name in enumerate(("gpt2-fp16", "llama-q4")):
            trace = report["traces"][name]
            end = [n for n in trace["nodes"] if n["name"] in ("norm", "result_norm")]
            axes[1, 0].plot([0, 1], [n["top3_energy_percent"] for n in end], "o-", label=name)
        axes[1, 0].set(xticks=[0, 1], xticklabels=["Before learned\ngamma/beta", "After learned\ngamma/beta"], ylabel="Fixed three features / total energy (%)", ylim=(0, 101), title="Final normalization: same 512-token traces")
        axes[1, 0].legend()
        trace = report["traces"]["gpt2-fp16"]
        branches = [b for b in trace["branches"] if b["channel"] == 496]
        axes[1, 1].bar(np.arange(12) - .18, [b["attention_mean_increment"] for b in branches], .36, label="Attention branch")
        axes[1, 1].bar(np.arange(12) + .18, [b["ffn_mean_increment"] for b in branches], .36, label="FFN branch")
        axes[1, 1].set(title="GPT-2 FP16: feature 496 mean additions", xlabel="Block index (0-based)", ylabel="Mean signed increment", xticks=range(12))
        axes[1, 1].legend()
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle("Large activations already exist before quantization\nGPT-2 FP16 and Q4-body controls; Llama Q4-body reference", fontsize=17)
        fig.savefig(output / "activation-layer-trace.png", dpi=300)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
        display = {"current_hp1": "Current HP1", "exact_top3_columns": "3 exact columns", "exact_mean_bias": "Center + exact bias",
                   "exact_principal1": "1 exact direction", "exact_principal3": "3 exact directions",
                   "rescale_then_requantize": "Scale + re-quantize", "rotate_then_requantize": "Rotate + re-quantize",
                   "vocab_center_then_requantize": "Center head weights + re-quantize"}
        for ax, model in zip(axes, ("gpt2", "llama")):
            rows = report["replay"][model]["precisions"]["4"]["metrics"]
            bars = ax.barh(list(display[name] for name in rows), [row["centered_rmse"] for row in rows.values()], color="#2876a4")
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
            ax.invert_yaxis()
            ax.set(title=f"{model}: current HP1 INT4 head", xlabel="Centered logit RMSE vs FP16 head")
            ax.margins(x=.2)
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("Head-only counterfactuals on held-out captured inputs\n64 rows / full vocabulary; calibration = earlier 16 chunks; FP64 dot; not full PPL", fontsize=16)
        fig.savefig(output / "activation-head-corrections.png", dpi=300)
        pdf.savefig(fig)
        plt.close(fig)


def vocab_centering(report: dict, library: ctypes.CDLL) -> None:
    for model, entry in report["replay"].items():
        previous = json.loads((HERE / "data/lm-head-diagnostics.json").read_text())[model]
        vocab, width = previous["head_shape"]
        all_h = np.fromfile(Path(previous["capture"]["native_run"]) / "head-input.f32", dtype=np.float32).reshape(8192, width)
        inputs = all_h[entry["test_rows"]].astype(np.float64)
        reader = GGUFReader(previous["paths"]["FP16"])
        head = next(t for t in reader.tensors if t.name == "token_embd.weight")
        column_mean = np.zeros(width)
        for start in range(0, vocab, 256):
            column_mean += head.data.reshape(vocab, width)[start:start+256].astype(np.float64).sum(axis=0)
        column_mean /= vocab
        reference = np.empty((len(inputs), vocab), dtype=np.float64)
        logits = {bits: np.empty_like(reference) for bits in (4, 8)}
        max_identity_error = 0.0
        for start in range(0, vocab, 256):
            end = min(vocab, start + 256)
            weights = head.data.reshape(vocab, width)[start:end].astype(np.float64)
            reference[:, start:end] = inputs @ weights.T
            centered = weights - column_mean
            unquantized = inputs @ centered.T + (inputs @ column_mean)[:, None]
            max_identity_error = max(max_identity_error, float(np.max(np.abs(unquantized - reference[:, start:end]))))
            np.testing.assert_allclose(unquantized, reference[:, start:end], atol=1e-10, rtol=1e-10)
            for bits in (4, 8):
                _, quantized = native_quantize(library, centered, bits)
                logits[bits][:, start:end] = inputs @ quantized.T
        entry["vocab_center_unquantized_identity_max_abs_error"] = max_identity_error
        for bits, output in logits.items():
            metrics = logit_metrics(reference, output)
            entry["precisions"][str(bits)]["metrics"]["vocab_center_then_requantize"] = metrics
            print(model, bits, "vocabulary centering", json.dumps(metrics), flush=True)
    (HERE / "data/activation-amplification.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    self_check()
    library = ctypes.CDLL(str(LIBRARY))
    for bits in (4, 8):
        function = getattr(library, f"quantize_q{bits}_hp1")
        function.argtypes = [np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
                             np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    report = dict(quantizer_library=str(LIBRARY), quantizer_library_sha256=digest(LIBRARY),
                  trace_library=str(ROOT / "output/experiment/q4-rounding-20260930/build-cpu/bin/libllama.dylib"),
                  trace_library_sha256=digest(ROOT / "output/experiment/q4-rounding-20260930/build-cpu/bin/libllama.dylib"),
                  trace_source_commit="b2f6b7630da21d17f0ca34e08cfd1e2d762a47b5",
                  trace_text_sha256=digest(ROOT / "wikitext-2-raw/wiki.test.raw"),
                  trace_settings=dict(ctx=512, batch=512, ubatch=512, gpu_layers=0, threads=1, kv_dtype="F16", flash_attention=False, logits_for_all_tokens=True),
                  scope="Diagnostic CPU traces and fixed-input head replay; no ExSIA/SCU/RMD/Metal execution or full PPL",
                  traces=layer_traces(), replay={})
    for model in ("gpt2", "llama"):
        report["replay"][model] = replay(model, library)
        (HERE / "data/activation-amplification.json").write_text(json.dumps(report, indent=2) + "\n")
    vocab_centering(report, library)
    plots(report)
    with (HERE / "data/activation-head-corrections.csv").open("w") as output:
        fields = ["model", "bits", "method", "centered_rmse", "centered_relative_l2_percent", "top1_agreement_percent", "mean_kl_from_fp16"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for model, replay_data in report["replay"].items():
            for bits, precision in replay_data["precisions"].items():
                for method, metrics in precision["metrics"].items():
                    writer.writerow(dict(model=model, bits=bits, method=method, **metrics))
    print("Saved activation-amplification.json, corrections CSV, two-page PDF and PNGs", flush=True)


if __name__ == "__main__":
    main()
