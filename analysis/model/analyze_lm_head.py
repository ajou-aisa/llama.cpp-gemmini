# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2"]
# ///
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
from typing import TypedDict

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from weight_io import ROOT, GGUFReader, InputError, quant_values
from weight_stats import Moments

HERE = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "output/experiment"
LABELS = ("FP16", "INT4", "INT8")
COLORS = ("#2864a5", "#bf5434", "#278879")


class LogitMetrics(TypedDict):
    centered_rmse: float
    centered_relative_l2_percent: float
    top1_agreement_percent: float
    mean_kl_from_fp16: float


def logit_metrics(reference: NDArray[np.float64], candidate: NDArray[np.float64]) -> LogitMetrics:
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise InputError("Invalid replay logits")
    ref = reference - reference.mean(axis=1, keepdims=True)
    delta = candidate - reference
    delta -= delta.mean(axis=1, keepdims=True)
    def log_probability(values: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = values - values.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    lp, lq = log_probability(reference), log_probability(candidate)
    return LogitMetrics(
        centered_rmse=float(np.sqrt(np.mean(delta * delta))),
        centered_relative_l2_percent=float(100 * np.linalg.norm(delta) / np.linalg.norm(ref)),
        top1_agreement_percent=float(100 * np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))),
        mean_kl_from_fp16=float(np.mean(np.sum(np.exp(lp) * (lp - lq), axis=1))),
    )


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture(path: Path, k: int, vocab: int) -> tuple[NDArray[np.float32], float]:
    manifest = dict(line.split("=", 1) for line in (path / "manifest.txt").read_text().splitlines())
    assert manifest["exit_status"] == "0"
    assert manifest["git_sha"] == "b2f6b7630da21d17f0ca34e08cfd1e2d762a47b5"
    with (path / "capture-shapes.csv").open() as source:
        shapes = list(csv.DictReader(source))
    assert [int(row["chunk"]) for row in shapes] == list(range(32))
    assert all((int(row["k"]), int(row["rows"]), int(row["vocab"])) == (k, 256, vocab) for row in shapes)
    values = np.fromfile(path / "head-input.f32", dtype=np.float32).reshape(8192, k)
    assert np.isfinite(values).all()
    match = re.search(r"Final estimate: PPL = ([0-9.]+)", (path / "raw/stderr.txt").read_text())
    assert match is not None
    return values, float(match[1])


def analyze(model: str, base: ctypes.CDLL, cpu: ctypes.CDLL) -> dict:
    data = json.loads((HERE / "data" / f"{model}.json").read_text())
    old = json.loads((EXPERIMENT / "potal-model-compare-20260930/comparison.json").read_text())[model]
    native_path, exact_path = Path(old["native_run"]), Path(old["exact_F32_run"])
    k, vocab = old["K"], old["vocab"]
    h, native_ppl = capture(native_path, k, vocab)
    exact_h, exact_ppl = capture(exact_path, k, vocab)
    assert np.array_equal(h.view(np.uint32), exact_h.view(np.uint32))
    assert digest(native_path / "token-embeddings.f32") == digest(exact_path / "token-embeddings.f32")
    del exact_h
    packed = np.empty(h.size // 256 * 292, dtype=np.uint8)
    rounded = np.empty_like(h)
    cpu.quantize_row_q8_K(h, packed, h.size)
    base.dequantize_row_q8_K(packed, rounded, h.size)
    energy = np.square(h, dtype=np.float64).sum(axis=0)
    top = np.argsort(energy)[-3:][::-1]
    energy_percent = energy / energy.sum() * 100
    zeroed = 100 * np.count_nonzero((h != 0) & (rounded == 0)) / np.count_nonzero(h)
    assert np.isclose(energy_percent[top].sum(), old["top3_activation_energy_percent"], atol=1e-10, rtol=0)
    assert np.isclose(zeroed, old["q8_K_zeroed_nonzero_percent"], atol=1e-10, rtol=0)
    capture_info = dict(native_run=str(native_path), exact_run=str(exact_path),
                        head_input_sha256=digest(native_path / "head-input.f32"),
                        rows=8192, scored_rows=8160, native_ppl=native_ppl, exact_head_ppl=exact_ppl,
                        head_inputs_and_embeddings_bit_identical=True,
                        energy_percent_by_channel=energy_percent.tolist(), top3_channels=top.tolist(),
                        top3_energy_percent=float(energy_percent[top].sum()),
                        q8_K_zeroed_nonzero_percent=float(zeroed))
    if model == "gpt2":
        injected = EXPERIMENT / "q6-head-20260930/experiment/f32-q8-injected-20260929-174801"
        injected_h, injected_ppl = capture(injected, k, vocab)
        assert np.array_equal(rounded.view(np.uint32), injected_h.view(np.uint32))
        assert digest(native_path / "head-input.f32") == digest(injected / "head-input-original.f32")
        assert digest(native_path / "token-embeddings.f32") == digest(injected / "token-embeddings.f32")
        assert injected_ppl == native_ppl
        capture_info.update(injected_run=str(injected), injected_ppl=injected_ppl,
                            injected_inputs_equal_native_q8_K=True)
        del injected_h
    del packed, rounded
    rows = (np.arange(32)[:, None] * 256 + np.array([0, 64, 128, 192])).reshape(-1)
    inputs = h[rows].astype(np.float64)
    del h
    reference_path = (ROOT / "models/gpt2.fp16.gguf" if model == "gpt2" else
                      EXPERIMENT / "potal-ppl-20261001/llama3.2-1B.Q4_0.head-original-F16.gguf")
    paths = {"FP16": reference_path, "INT4": ROOT / data["input"]["int4"],
             "INT8": ROOT / data["input"]["int8"]}
    hashes = {label: digest(path) for label, path in paths.items()}
    for label, key in (("INT4", "int4"), ("INT8", "int8")):
        assert hashes[label] == data["sha256"][key]
    if model == "gpt2":
        assert hashes["FP16"] == data["sha256"]["reference"]
    readers = {label: GGUFReader(str(path)) for label, path in paths.items()}
    heads = {label: next(t for t in reader.tensors if t.name == "token_embd.weight")
             for label, reader in readers.items()}
    for label, reader in readers.items():
        assert not any(t.name == "output.weight" for t in reader.tensors)
        assert tuple(heads[label].shape) == (k, vocab)
    assert heads["FP16"].tensor_type.name == "F16"
    logits = {label: np.empty((len(rows), vocab), dtype=np.float64) for label in LABELS}
    top_error = {label: np.empty((len(rows), vocab), dtype=np.float64) for label in LABELS[1:]}
    moments = {label: Moments() for label in LABELS}
    head_digest = hashlib.sha256()
    chunk_rows = max(1, 262144 // k)
    for start in range(0, vocab, chunk_rows):
        end = min(vocab, start + chunk_rows)
        raw = heads["FP16"].data.reshape(vocab, k)[start:end]
        head_digest.update(raw.tobytes())
        reference = raw.astype(np.float32).reshape(-1)
        for label in LABELS:
            values, codes = (reference, None) if label == "FP16" else quant_values(heads[label], start, end)
            moments[label].add(values, reference, codes)
            weights = values.reshape(-1, k).astype(np.float64)
            logits[label][:, start:end] = inputs @ weights.T
            if label != "FP16":
                difference = weights[:, top] - reference.reshape(-1, k)[:, top]
                top_error[label][:, start:end] = inputs[:, top] @ difference.T
        if start // chunk_rows % 128 == 0:
            print(f"{model}: head vocabulary {end}/{vocab}", flush=True)
    for label, stats in moments.items():
        expected = data["groups"][label]["E"]
        assert np.array_equal(stats.fine_histogram, expected["fine_histogram"])
        for key, value in stats.summary().items():
            if value is None or expected[key] is None:
                assert value == expected[key], (model, label, key)
            else:
                assert np.isclose(value, expected[key], rtol=1e-11, atol=1e-14), (model, label, key)
    replay = {}
    for label in LABELS[1:]:
        stats = logit_metrics(logits["FP16"], logits[label])
        error = top_error[label]
        error -= error.mean(axis=1, keepdims=True)
        replay[label] = dict(stats, top3_channel_only_centered_error_rmse=float(np.sqrt(np.mean(error * error))))
    for reader in readers.values():
        reader.data._mmap.madvise(mmap.MADV_DONTNEED)
    print(f"{model}: {json.dumps(replay)}", flush=True)
    return dict(title=data["title"], head_shape=[vocab, k], paths={key: str(p) for key, p in paths.items()},
                model_sha256=hashes, reference_head_sha256=head_digest.hexdigest(),
                current_head_statistics_match_prior_analysis=True, capture=capture_info,
                replay_rows=rows.tolist(), replay=replay,
                fp16_centered_logit_rms=float(np.std(logits["FP16"], axis=1).dot(np.std(logits["FP16"], axis=1)) / len(rows)) ** 0.5)


def plots(report: dict) -> None:
    output = HERE / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(output / "lm-head-comparison.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.85, hspace=0.5, wspace=0.23)
        fig.suptitle("LM head weights: FP16 vs current PoTal HP1", fontsize=23, weight="bold", y=0.97)
        fig.text(0.075, 0.91, "Shared token embedding / output matrix. All weight values; 1/32-log2 magnitude bins; no smoothing.", fontsize=12)
        for row, model in enumerate(("gpt2", "llama")):
            data = json.loads((HERE / "data" / f"{model}.json").read_text())
            edges = np.exp2(data["fine_histogram_log2_edges"])
            for col, (label, color) in enumerate(zip(LABELS, COLORS)):
                ax = axes[row, col]
                stats = data["groups"][label]["E"]
                percent = np.array(stats["fine_histogram"]) / stats["count"] * 100
                ax.stairs(percent, edges, fill=True, color=color, alpha=0.75, linewidth=0.3)
                ax.set(xscale="log", yscale="symlog", xlim=(2**-16, 2), ylim=(0, 35))
                ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
                ax.set_yticks([0, 0.01, 0.1, 1, 10, 30], ["0", "0.01", "0.1", "1", "10", "30"])
                ax.set_title(f"{data['title']} / {label}", fontsize=14, color=color, weight="bold")
                ax.set_xlabel("Restored |weight| (log scale)")
                if col == 0:
                    ax.set_ylabel("Weights per bin (%) / symlog")
                ax.grid(axis="y", alpha=0.2)
                relative = "reference" if label == "FP16" else f"relative L2 error {stats['relative_l2_error']*100:.3f}%"
                ax.text(0.02, 0.95, f"RMS {stats['rms']:.5f} | max {stats['max_abs']:.4f}\n"
                        f"zero {stats['zero_fraction']*100:.2f}% | {relative}",
                        transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88))
        fig.text(0.075, 0.045, "Zeros are reported separately. Shared axes across all six panels. Full histograms are retained in data/*-magnitude-fine.csv.\n"
                 "Similar relative weight error does not imply similar logit error: the input activations and their direction also matter.", fontsize=11, linespacing=1.6)
        fig.savefig(output / "lm-head-fp16-hp1-int4-int8.png", dpi=400)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.subplots_adjust(left=0.065, right=0.97, bottom=0.30, top=0.77, wspace=0.35)
        fig.suptitle("Why GPT-2's LM head is sensitive: three separate measurements", fontsize=21, weight="bold", y=0.95)
        for model, color in (("gpt2", COLORS[1]), ("llama", COLORS[0])):
            row = report[model]
            values = np.sort(row["capture"]["energy_percent_by_channel"])[::-1]
            axes[0].plot(np.arange(1, len(values)+1), np.cumsum(values), color=color,
                         label=f"{row['title']}: top 3 = {values[:3].sum():.2f}%")
        axes[0].set(xscale="log", xlabel="Channels, ranked by energy", ylabel="Cumulative input energy (%)", ylim=(0, 101),
                    title="Historical CPU head inputs\n8,192 captured rows / model")
        axes[0].legend(fontsize=9, loc="lower right")
        axes[0].grid(alpha=0.2)
        for index, model in enumerate(("gpt2", "llama")):
            for offset, label, color in ((-0.18, "INT4", COLORS[1]), (0.18, "INT8", COLORS[2])):
                value = report[model]["replay"][label]["centered_rmse"]
                axes[1].bar(index + offset, value, width=0.34, color=color, label=label if index == 0 else None)
                axes[1].text(index + offset, value * 1.12, f"{value:.3f}", ha="center", fontsize=10)
        axes[1].set(yscale="log", ylabel="Centered logit RMSE (log scale)",
                    title="Current HP1 weights, fixed historical H\n128 rows, entire vocabulary, FP64 dot")
        axes[1].set_xticks([0, 1], ["GPT-2", "Llama-3.2-1B"])
        axes[1].legend(fontsize=9)
        axes[1].margins(y=0.35)
        for index, model in enumerate(("gpt2", "llama")):
            for offset, key, color, title in ((-0.18, "native_ppl", "#85929e", "Native Q8_K input"),
                                            (0.18, "exact_head_ppl", "#278879", "Same W, FP input")):
                value = report[model]["capture"][key]
                axes[2].bar(index + offset, value, width=0.34, color=color, label=title if index == 0 else None)
                axes[2].text(index + offset, value + 1, f"{value:.2f}", ha="center", fontsize=10)
        axes[2].set(ylabel="PPL (first 32 chunks only)", ylim=(0, 66),
                    title="Historical CPU control: same Q6_K W\nInput quantization changes; H is fixed")
        axes[2].set_xticks([0, 1], ["GPT-2", "Llama-3.2-1B"])
        axes[2].legend(fontsize=9)
        fig.text(0.065, 0.10, "H was captured from Q4_0-body / Q6_K-head CPU runs in September 2026. It is not a current PoTal activation trace.\n"
                 "Middle: weight-only sensitivity replay; no ExSIA, RMD, SCU or PPL evaluation. Right: 8,160 scored tokens, not full-dataset PPL.\n"
                 "GPT-2 control: restoring native Q8_K-rounded H in the FP head restores PPL 52.2543. Current PoTal causality remains untested.",
                 fontsize=10, linespacing=1.6)
        fig.savefig(output / "lm-head-sensitivity.png", dpi=400)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    toy = np.array([[1., 2., 4.], [-2., 3., 0.]])
    invariant = logit_metrics(toy, toy + np.array([[10.], [-5.]]))
    assert invariant["centered_rmse"] == 0 and abs(invariant["mean_kl_from_fp16"]) < 1e-14
    changed = logit_metrics(toy, toy[:, ::-1])
    assert changed["centered_rmse"] > 0 and changed["mean_kl_from_fp16"] > 0
    library = EXPERIMENT / "q4-rounding-20260930/build-cpu/bin"
    base, cpu = (ctypes.CDLL(str(library / name)) for name in ("libggml-base.dylib", "libggml-cpu.dylib"))
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    bytes_array = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    cpu.quantize_row_q8_K.argtypes = [floats, bytes_array, ctypes.c_int64]
    cpu.quantize_row_q8_K.restype = None
    base.dequantize_row_q8_K.argtypes = [bytes_array, floats, ctypes.c_int64]
    base.dequantize_row_q8_K.restype = None
    report = {model: analyze(model, base, cpu) for model in ("gpt2", "llama")}
    report["scope"] = "Current HP1 weight statistics; historical CPU Q4_0-body head captures and controls; current HP1 weight-only FP64 replay on those fixed inputs. Not current PoTal PPL or backend arithmetic."
    report["sampling"] = "4 scored head rows per chunk: offsets 0,64,128,192; all 32 historical chunks. Full vocabulary. Fixed deterministic sample; different tokenizers cover different text extents."
    report["reference"] = "Llama FP16 head reused from an archived original-F16 control model. All head statistics and fine histograms, including HP1 errors against it, rechecked against the whole-source-SHA-verified analysis. Archived model and head SHA are recorded separately."
    report["metrics"] = "Logit errors are centered per row across vocabulary; this removes softmax-invariant constant offsets. Top1 is agreement with the FP16-head replay, not ground-truth accuracy. KL uses that replay's full-vocabulary probability distribution. Top3-only error is a separate projection; its energy is not additive with remaining channels."
    (HERE / "data/lm-head-diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plots(report)
    print(HERE / "data/lm-head-diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
