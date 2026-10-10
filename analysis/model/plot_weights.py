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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import numpy as np


def model_page(path: Path) -> Figure:
    data = json.loads(path.read_text())
    labels = ("FP16", "INT4", "INT8")
    colors = ("#2864a5", "#bf5434", "#278879")
    groups = data["groups"]
    names = ["E"] + sorted(name for name in groups["FP16"] if name.startswith("B"))
    edges = np.array(data["fine_histogram_log2_edges"])
    centers = (edges[1:] + edges[:-1]) / 2
    visible = (centers >= -16) & (centers <= 6)
    x, y = np.meshgrid(centers[visible], np.arange(len(names)))
    distributions = {label: np.array([groups[label][name]["fine_histogram"] for name in names]) /
                     np.array([groups[label][name]["count"] for name in names])[:, None] * 100
                     for label in labels}
    peak = max(float(values[:, visible].max()) for values in distributions.values())
    z_limit = np.log10(1 + peak / 0.01) * 1.08
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(21, 12.7), facecolor="white")
    upper_grid = fig.add_gridspec(1, 4, left=0.02, right=0.965, bottom=0.51, top=0.835, wspace=0.02)
    lower_grid = fig.add_gridspec(1, 3, left=0.065, right=0.95, bottom=0.24, top=0.422, wspace=0.27)
    meta = data["metadata"]
    family = meta["general.architecture"]
    architecture = (f"{meta[family + '.block_count']} blocks | width {meta[family + '.embedding_length']} | "
                    f"FFN {meta[family + '.feed_forward_length']} | ")
    architecture += "MHA 12 heads | LayerNorm + GELU | learned positions" if family == "gpt2" else "GQA 32 Q / 8 KV heads | RMSNorm + SwiGLU | RoPE"
    fig.text(0.055, 0.955, f"{data['title']}  /  FP16 vs PoTal HP1", fontsize=23, weight="bold", color="#132d42")
    fig.text(0.055, 0.914, architecture, fontsize=12, color="#395569")
    fig.text(0.055, 0.875, "First two surfaces: original FP16 height, colored by the observed fraction mapped to zero. INT4 / INT8: restored weight distributions.", fontsize=11)
    magnitude_max = max(groups[label][name]["max_abs"] for label in labels for name in names) * 1.4
    magnitude_min = min(groups[label][name]["rms"] for label in labels for name in names) / 2
    zero_max = max(1.0, max(groups[label][name]["zero_fraction"] * 100 for label in labels for name in names) * 1.1)
    ticks = sorted(set([0, len(names)-1] + list(range(1, len(names), 2))))
    tick_labels = ["E" if index == 0 else str(index - 1) for index in ticks]
    block_ticks = [0, 1, 5, 9, len(names)-1]
    block_labels = ["E" if index == 0 else str(index - 1) for index in block_ticks]
    zero_cmap = LinearSegmentedColormap.from_list("zero_fraction", ["#adc5d8", "#ffe0a0", "#cf3927"])
    zero_norm = Normalize(0, 100)
    reference_counts = np.array([groups["FP16"][name]["fine_histogram"] for name in names])[:, visible]
    panels = (("FP16: INT4 zero map", "FP16", "INT4"), ("FP16: INT8 zero map", "FP16", "INT8"),
              ("INT4 (Q4_HP1)", "INT4", None), ("INT8 (Q8_HP1)", "INT8", None))
    for column, (title, label, zero_target) in enumerate(panels):
        ax = fig.add_subplot(upper_grid[0, column], projection="3d")
        z = np.log10(1 + distributions[label][:, visible] / 0.01)
        if zero_target is not None:
            zero_counts = np.array([groups[zero_target][name]["zeroed_reference_histogram"] for name in names])[:, visible]
            assert np.all((zero_counts >= 0) & (zero_counts <= reference_counts))
            percent = np.divide(100.0 * zero_counts, reference_counts,
                                out=np.zeros_like(z), where=reference_counts != 0)
            facecolors = zero_cmap(zero_norm(percent))
            facecolors[reference_counts == 0, 3] = 0
            ax.plot_surface(x, y, z, facecolors=facecolors, shade=False,
                            rstride=1, cstride=1, linewidth=0, antialiased=True)
        else:
            ax.plot_surface(x, y, z, cmap="viridis", norm=Normalize(0, z_limit),
                            rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.view_init(elev=27, azim=-62)
        ax.set_box_aspect((1.5, 1.2, 0.85))
        ax.set(xlim=(-16, 6), ylim=(0, len(names)-1), zlim=(0, z_limit),
               xlabel="Weight magnitude |W|", ylabel="Block", zlabel="Weights/bin (%)\nlog1p scale")
        ax.set_xticks([-16, -12, -8, -4, 0, 6], ["1.53e-5", "2.44e-4", "0.00391", "0.0625", "1", "64"])
        frequency_ticks = np.array([0, 0.01, 0.1, 1, 10, 30])
        ax.set_zticks(np.log10(1 + frequency_ticks / 0.01), [f"{value:g}" for value in frequency_ticks])
        ax.set_yticks(block_ticks, block_labels)
        ax.tick_params(labelsize=8, pad=0)
        ax.xaxis.labelpad = 7
        ax.yaxis.labelpad = 4
        ax.zaxis.labelpad = 7
        ax.set_title(title, fontsize=14, color=colors[labels.index(label)], weight="bold", pad=13)
    colorbar = fig.colorbar(ScalarMappable(norm=zero_norm, cmap=zero_cmap),
                           cax=fig.add_axes((0.135, 0.483, 0.245, 0.011)), orientation="horizontal")
    colorbar.set_ticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("FP16 nonzero weights becoming 0 within each magnitude bin", fontsize=9, labelpad=3)
    fig.text(0.58, 0.471, "Surface height uses the same frequency scale in all four panels.\n"
             "Original FP16 zeros are excluded from the zero maps.", fontsize=10, color="#395569", linespacing=1.5)
    for column, (label, color) in enumerate(zip(labels, colors)):
        lower = fig.add_subplot(lower_grid[0, column])
        lower.set_title(f"{label}: magnitude and zero fraction", fontsize=12, color=color, weight="bold")
        positions = np.arange(len(names))
        for key, title, line_color, style in (("rms", "RMS", "#2864a5", "-"),
                                              ("p99_abs_approx", "P99 |W| (approx.)", "#278879", "--"),
                                              ("max_abs", "Max |W|", "#bf5434", "-")):
            lower.plot(positions, [groups[label][name][key] for name in names],
                       label=title, color=line_color, linestyle=style, marker="o", markersize=3, linewidth=1.7)
        lower.set_yscale("log")
        lower.set_ylim(magnitude_min, magnitude_max)
        lower.set_xticks(ticks, tick_labels)
        lower.set_xlabel("Block index (E = tied embedding / LM head)", fontsize=9)
        lower.set_ylabel("Restored weight magnitude", fontsize=9)
        lower.grid(axis="y", alpha=0.2)
        right = lower.twinx()
        right.plot(positions, [groups[label][name]["zero_fraction"] * 100 for name in names],
                   color="#717a85", linestyle=":", linewidth=1.5, label="Zero weights (%)")
        right.set_ylim(0, zero_max)
        right.set_ylabel("Zero weights (%)", color="#717a85", fontsize=9)
        right.tick_params(axis="y", labelsize=8, colors="#717a85")
        if column == 0:
            handles, texts = lower.get_legend_handles_labels()
            other, other_texts = right.get_legend_handles_labels()
            lower.legend(handles + other, texts + other_texts, fontsize=8, loc="upper left", framealpha=0.9)
        stats = data["linear_summary"][label]
        hidden = sum(int(np.array(groups[label][name]["fine_histogram"])[~visible].sum()) for name in names) / stats["count"] * 100
        footer = f"Matrix weights: {stats['count']:,}  |  zero {stats['zero_fraction']*100:.2f}%\n"
        footer += f"RMS {stats['rms']:.5g}  |  max |W| {stats['max_abs']:.5g}\n"
        footer += "FP16 reference" if label == "FP16" else f"Relative L2 error {stats['relative_l2_error']*100:.2f}%  |  SQNR {stats['sqnr_db']:.2f} dB"
        if label != "FP16":
            footer += f"\nFP16 nonzeros mapped to 0: {stats['zeroed_nonzero_fraction']*100:.4f}% of weights"
        footer += f"\nNonzero mass outside top x-axis: {hidden:.4f}%"
        fig.text(0.07 + column * 0.31, 0.094, footer, fontsize=9, linespacing=1.55, color="#213e52")
    fig.text(0.055, 0.028,
             "Zero maps compare the same weight positions: FP16 != 0 and decoded HP1 == 0, binned by the original FP16 magnitude. E = tied embedding / LM head.\n"
             "Surface bin width: 1/32 log2. Frequency height = log10(1 + percent/0.01); ticks show actual percent. P99 uses coarse bins. No smoothing or bin downsampling.",
             fontsize=9, color="#526371", linespacing=1.5)
    return fig


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "figures"
    output.mkdir(exist_ok=True)
    with PdfPages(output / "gpt2-llama-weight-comparison.pdf") as pdf:
        for model in ("gpt2", "llama"):
            figure = model_page(root / "data" / f"{model}.json")
            figure.savefig(output / f"{model}-fp16-hp1-int4-int8.png", dpi=400)
            pdf.savefig(figure)
            plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
