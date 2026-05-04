#!/usr/bin/env python3
"""
Sweep all (q_code, delta_code) candidate combinations for a given model,
apply each via set_ethos_config.py, run llama-perplexity, and record PPL.

q_code  : all 2^N combinations of {7, 8} across N layers (32 for 5-layer models)
delta_code: fully determined by q per layer using DELTA_MAP

Each run appends to ethos_ppl_<model>.csv. Re-running automatically resumes
from where it left off by scanning the existing CSV for completed combinations.

Usage:
  python ethos_profile.py gpt2
  python ethos_profile.py llama
"""
import argparse, csv, json, re, subprocess, sys
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SET_CONFIG = SCRIPT_DIR / "set_ethos_config.py"
BIN        = SCRIPT_DIR / "build-arm64/bin/llama-perplexity"
WIKI       = SCRIPT_DIR / "wikitext-2-raw/wiki.test.raw"

MODEL_FILES = {
    "gpt2":  "gpt2.F16.gguf",
    "llama": "Llama-3.2-1B.f16.gguf",
    "qwen2": "qwen2-0.5B.f16.gguf",
}

PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
ETA_RE = re.compile(r"ETA\s+([\d.]+)\s+minutes")

CONFIG_PATH = SCRIPT_DIR / "ggml/src/ggml-gemmini/ethos_config.json"

# Per-layer delta lookup: DELTA_MAP[model][layer_name][q_value] = delta_value
DELTA_MAP = {
    "gpt2": {
        "attn_norm":   {7: 3, 8: 4},
        "kqv_out":     {7: 2, 8: 3},
        "ffn_norm":    {7: 2, 8: 3},
        "ffn_gelu":    {7: 0, 8: 1},
        "result_norm": {7: 0, 8: 1},
    },
    "llama": {
        "attn_norm":    {7: 4, 8: 4},
        "kqv_out":      {7: 3, 8: 4},
        "ffn_norm":     {7: 4, 8: 4},
        "ffn_gate_par": {7: 2, 8: 2},
        "result_norm":  {7: 0, 8: 1},
    },
    "qwen2": {
        "attn_norm":    {7: 3, 8: 4},
        "kqv_out":      {7: 2, 8: 3},
        "ffn_norm":     {7: 3, 8: 4},
        "ffn_gate_par": {7: 1, 8: 2},
        "result_norm":  {7: 0, 8: 0},
    },
}


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_combinations(layer_names: list[str], model: str):
    """Yield all (q_code, delta_code) pairs: q in {7,8}^N, delta from DELTA_MAP."""
    delta_map = DELTA_MAP.get(model)
    if delta_map is None:
        sys.exit(f"No DELTA_MAP entry for model '{model}'")
    missing = [l for l in layer_names if l not in delta_map]
    if missing:
        sys.exit(f"Layer(s) not in DELTA_MAP['{model}']: {missing}")
    for q_vals in product([7, 8], repeat=len(layer_names)):
        q_code     = "".join(str(q) for q in q_vals)
        delta_code = "".join(str(delta_map[layer][q])
                             for layer, q in zip(layer_names, q_vals))
        yield q_code, delta_code


# ---------------------------------------------------------------------------
# Perplexity runner
# ---------------------------------------------------------------------------

def run_perplexity(model_file: str) -> tuple[float | None, float | None]:
    cmd = [str(BIN), "-m", str(SCRIPT_DIR / "models" / model_file), "-f", str(WIKI)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    eta_printed = False
    for line in proc.stdout:
        lines.append(line)
        if not eta_printed:
            m_eta = ETA_RE.search(line)
            if m_eta:
                print(f"  ETA {m_eta.group(1)} minutes", flush=True)
                eta_printed = True
    proc.wait()
    output = "".join(lines)
    m = PPL_RE.search(output)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep ethos config combinations and collect PPL results"
    )
    parser.add_argument("model", help="Architecture name (e.g. gpt2, llama)")
    args = parser.parse_args()

    model_file = MODEL_FILES.get(args.model)
    if not model_file:
        sys.exit(f"Unknown model '{args.model}'. Available: {list(MODEL_FILES)}")

    csv_path   = SCRIPT_DIR / f"ethos_ppl_{args.model}.csv"
    fieldnames = ["model", "q_code", "delta_code", "ppl", "ppl_sigma"]

    # Auto-resume: scan existing CSV for already-completed combinations
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["q_code"], row["delta_code"]))
        if done:
            print(f"Auto-resume: {len(done)} combinations already recorded, skipping")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    out = open(csv_path, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    config = json.loads(CONFIG_PATH.read_text())
    # Layer order from config is canonical; DELTA_MAP is used as a lookup only.
    layer_names = list(config["architectures"][args.model]["layers"].keys())
    combinations = list(generate_combinations(layer_names, args.model))
    print(f"Total combinations: {len(combinations)}")

    n_run = n_resumed = 0

    try:
        for q, delta in combinations:
            if (q, delta) in done:
                n_resumed += 1
                continue

            # Apply config
            subprocess.run(
                [sys.executable, str(SET_CONFIG), args.model, q, delta],
                check=True, capture_output=True,
            )

            print(f"[{n_run}] q={q}  d={delta}", flush=True)

            ppl, sigma = run_perplexity(model_file)
            if ppl is None:
                print("  WARNING: PPL parse failed — skipping")
                continue

            print(f"  PPL={ppl}  +/-{sigma}")
            writer.writerow({
                "model":      args.model,
                "q_code":     q,
                "delta_code": delta,
                "ppl":        ppl,
                "ppl_sigma":  sigma,
            })
            out.flush()
            n_run += 1

    finally:
        out.close()

    print(f"\nDone. {n_run} run, {n_resumed} skipped (already done).")
    print(f"Results → {csv_path}")


if __name__ == "__main__":
    main()
