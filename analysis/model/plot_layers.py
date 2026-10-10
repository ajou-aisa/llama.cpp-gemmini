# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2"]
# ///
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    labels = ("FP16", "INT4", "INT8")
    colors = ("#2864a5", "#bf5434", "#278879")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    for model in ("gpt2", "llama"):
        data = json.loads((root / "data" / f"{model}.json").read_text())
        groups = data["groups"]
        names = ["E"] + sorted(name for name in groups["FP16"] if name.startswith("B"))
        edges = np.array(data["histogram_log2_edges"])
        centers = (edges[1:] + edges[:-1]) / 2
        visible = (centers >= -16) & (centers <= 6)
        y_limit = max(max(groups[label][name]["histogram"]) / groups[label][name]["count"] * 100
                      for label in labels for name in names) * 1.12
        output = root / "figures" / "layers" / model
        output.mkdir(parents=True, exist_ok=True)
        with PdfPages(output.parent / f"{model}-layers-fp16-hp1.pdf") as pdf:
            for name in names:
                title = "Embedding / tied LM head" if name == "E" else f"Transformer block {int(name[1:]):02d}"
                tensors = [tensor for tensor in data["tensors"]
                           if tensor["precision"] == "FP16" and tensor["group"] == name]
                fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), sharex=True, sharey=True)
                fig.subplots_adjust(left=0.06, right=0.975, bottom=0.34, top=0.73, wspace=0.16)
                fig.text(0.06, 0.94, f"{data['title']}  /  {title}", fontsize=21,
                         weight="bold", color="#132d42")
                fig.text(0.06, 0.887,
                         f"{len(tensors)} matrix tensor(s) | {groups['FP16'][name]['count']:,} weights | FP16 vs PoTal HP1",
                         fontsize=12, color="#395569")
                fig.text(0.06, 0.837,
                         "Observed histogram points joined by straight lines. Identical axes across all layers and precisions of this model.",
                         fontsize=10)
                for ax, label, color in zip(axes, labels, colors):
                    stats = groups[label][name]
                    histogram = np.array(stats["histogram"])
                    assert int(histogram.sum()) + round(stats["zero_fraction"] * stats["count"]) == stats["count"]
                    values = histogram[visible] / stats["count"] * 100
                    ax.plot(centers[visible], values, color=color, linewidth=1.1,
                            marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.8)
                    ax.set(xlim=(-16, 6), ylim=(0, y_limit), xlabel="Weight magnitude |W| (log2 scale)")
                    ax.set_xticks([-16, -12, -8, -4, 0, 6],
                                  [r"$2^{-16}$", r"$2^{-12}$", r"$2^{-8}$", r"$2^{-4}$", "1", "64"])
                    ax.grid(alpha=0.18)
                    ax.set_title(label if label == "FP16" else f"{label} (Q{4 if label == 'INT4' else 8}_HP1)",
                                 fontsize=15, color=color, weight="bold", pad=13)
                    hidden = int(histogram[~visible].sum()) / stats["count"] * 100
                    summary = (f"RMS  {stats['rms']:.6g}\nMax |W|  {stats['max_abs']:.6g}\n"
                               f"Zero weights  {stats['zero_fraction']*100:.4f}%\n")
                    summary += "FP16 reference" if label == "FP16" else (
                        f"Relative L2 error  {stats['relative_l2_error']*100:.3f}%\nSQNR  {stats['sqnr_db']:.3f} dB")
                    summary += f"\nNonzero mass outside x-axis  {hidden:.4f}%"
                    fig.text(ax.get_position().x0, 0.26, summary, fontsize=10,
                             linespacing=1.45, color="#213e52", va="top")
                axes[0].set_ylabel("Weights per bin (%)")
                components = ", ".join(tensor["tensor"].removesuffix(".weight").split(".")[-1] for tensor in tensors)
                fig.text(0.06, 0.058, f"Included matrices: {components}.", fontsize=9, color="#526371")
                fig.text(0.06, 0.023,
                         "Each point: 100 x bin count / total weights. Bin width: 0.25 log2 units. Zero is reported separately. Norm/bias excluded.",
                         fontsize=9, color="#526371")
                figure_name = "embedding" if name == "E" else f"block-{int(name[1:]):02d}"
                fig.savefig(output / f"{figure_name}.png", dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"{model}: {len(names)} layer pages -> {output.parent / f'{model}-layers-fp16-hp1.pdf'}")


if __name__ == "__main__":
    main()
