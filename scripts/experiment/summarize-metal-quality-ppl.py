#!/usr/bin/env python3
import csv
import math
from pathlib import Path
import sys


def summarize(directory: Path) -> None:
    with (directory / "results.tsv").open(newline="") as source:
        records = list(csv.DictReader(source, delimiter="\t"))
    values: dict[tuple[str, str, int, int], str] = {}
    full = True
    for row in records:
        method, model = row["method"], row["model"]
        bits, dim = int(row["bits"]), int(row["dim"])
        if method not in ("RTN-W", "RTN-WA1", "RTN-WA2", "PoTal") or model not in ("gpt2", "llama"):
            raise ValueError("Unexpected result profile")
        if bits not in (4, 8) or dim not in (16, 32, 64) or int(row["exit"]) != 0:
            raise ValueError("Invalid result profile/status")
        if not math.isfinite(float(row["ppl"])) or float(row["ppl"]) <= 0:
            raise ValueError("Invalid PPL")
        expected_chunks = 559 if model == "gpt2" else 564
        full = full and int(row["chunks"]) == expected_chunks and int(row["scored_tokens"]) == expected_chunks * 255
        key = method, model, bits, dim if method in ("PoTal", "RTN-WA2") else 0
        if key in values:
            raise ValueError(f"Duplicate result: {key}")
        values[key] = f'{float(row["ppl"]):.2f}'

    def cells(method: str, dim: int = 0) -> list[str]:
        return [values.get((method, model, bits, dim), "TBD")
                for model in ("gpt2", "llama") for bits in (4, 8)]

    markdown = [
        "# WikiText-2 PPL",
        "",
        (f"Validated full-corpus results: {len(values)}/32. FP16 values are user-provided references." if full
         else "Partial-corpus diagnostics only; these values are not full WikiText-2 PPL."),
        "",
        "| Method | A/W | DIM | GPT-2 n=4 | GPT-2 n=8 | Llama n=4 | Llama n=8 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if full:
        markdown.append("| FP16 (reference) | 16/16 | -- | 27.19 | 27.19 | 10.19 | 10.19 |")
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{WikiText-2 perplexity (PPL$\downarrow$).}" if full else r"\caption{\textbf{Partial-corpus diagnostic PPL; not full WikiText-2.}",
        r"RTN-W/WA1 share baseline weights. WA2/PoTal share HP1 weights including the LM head; WA2 disables outlier selection and residual compensation.}",
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
    ]
    if full:
        latex.append(r"FP16 & 16/16 & -- & \multicolumn{2}{c}{27.19} & \multicolumn{2}{c}{10.19} \\")
    for method, aw in (("RTN-W", "16/n"), ("RTN-WA1", "n/n")):
        row = cells(method)
        markdown.append(f"| {method} | {aw} | -- | " + " | ".join(row) + " |")
        latex.append(f"{method} & $" + aw + "$ & -- & " + " & ".join(row) + r" \\")
    latex.append(r"\midrule")
    for method in ("RTN-WA2", "PoTal"):
        for dim in (16, 32, 64):
            row = cells(method, dim)
            markdown.append(f"| {method} | n/n | {dim} | " + " | ".join(row) + " |")
            latex.append(method + f" & $n/n$ & {dim} & " + " & ".join(row) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (directory / "quality_ppl.md").write_text("\n".join(markdown) + "\n")
    (directory / "quality_ppl.tex").write_text("\n".join(latex) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize-metal-quality-ppl.py RESULT_DIRECTORY")
    summarize(Path(sys.argv[1]))
