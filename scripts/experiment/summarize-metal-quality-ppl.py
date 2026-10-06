#!/usr/bin/env python3
import csv
import math
from pathlib import Path
import sys


def summarize(directory: Path) -> None:
    with (directory / "results.tsv").open(newline="") as source:
        records = list(csv.DictReader(source, delimiter="\t"))
    values: dict[tuple[str, str, int, int], str] = {}
    for row in records:
        method, model = row["method"], row["model"]
        bits, dim = int(row["bits"]), int(row["dim"])
        if method not in ("RTN-W", "RTN-WA", "PoTal") or model not in ("gpt2", "llama"):
            raise ValueError("Unexpected result profile")
        if bits not in (4, 8) or dim not in (16, 32, 64) or int(row["exit"]) != 0:
            raise ValueError("Invalid result profile/status")
        if not math.isfinite(float(row["ppl"])) or float(row["ppl"]) <= 0:
            raise ValueError("Invalid PPL")
        key = method, model, bits, dim if method == "PoTal" else 0
        if key in values:
            raise ValueError(f"Duplicate result: {key}")
        values[key] = f'{float(row["ppl"]):.2f}'

    def cells(method: str, dim: int = 0) -> list[str]:
        return [values.get((method, model, bits, dim), "TBD")
                for model in ("gpt2", "llama") for bits in (4, 8)]

    markdown = [
        "# WikiText-2 PPL",
        "",
        f"New measurements completed: {len(values)}/20. FP16 values are user-provided references.",
        "",
        "| Method | A/W | DIM | GPT-2 n=4 | GPT-2 n=8 | Llama n=4 | Llama n=8 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        "| FP16 (reference) | 16/16 | -- | 27.19 | 27.19 | 10.19 | 10.19 |",
    ]
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{WikiText-2 perplexity (PPL$\downarrow$).}",
        r"RTN-W/WA use $B_K{=}32$ round-to-nearest weight/weight-and-activation quantization without PoT alignment or residual compensation.}",
        r"\label{tab:quality_ppl}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multirow{2}{*}{A/W} & \multirow{2}{*}{DIM}",
        r"& \multicolumn{2}{c}{GPT-2} & \multicolumn{2}{c}{Llama-3.2-1B} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"&&& $n{=}4$ & $n{=}8$ & $n{=}4$ & $n{=}8$ \\",
        r"\midrule",
        r"FP16 & 16/16 & -- & \multicolumn{2}{c}{27.19} & \multicolumn{2}{c}{10.19} \\",
    ]
    for method, aw in (("RTN-W", "16/n"), ("RTN-WA", "n/n")):
        row = cells(method)
        markdown.append(f"| {method} | {aw} | -- | " + " | ".join(row) + " |")
        latex.append(f"{method} & $" + aw + "$ & -- & " + " & ".join(row) + r" \\")
    latex.append(r"\midrule")
    for dim in (16, 32, 64):
        row = cells("PoTal", dim)
        markdown.append(f"| PoTal | n/n | {dim} | " + " | ".join(row) + " |")
        prefix = r"\multirow{3}{*}{\textbf{PoTal}} & \multirow{3}{*}{$n/n$}" if dim == 16 else "&"
        latex.append(prefix + f" & {dim} & " + " & ".join(row) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (directory / "quality_ppl.md").write_text("\n".join(markdown) + "\n")
    (directory / "quality_ppl.tex").write_text("\n".join(latex) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize-metal-quality-ppl.py RESULT_DIRECTORY")
    summarize(Path(sys.argv[1]))
