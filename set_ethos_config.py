#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "ggml/src/ggml-gemmini/ethos_config.json"

def main():
    parser = argparse.ArgumentParser(
        description="Apply q/delta codes to ethos_config.json for a given architecture"
    )
    parser.add_argument("model", help="Architecture name (e.g. gpt2, llama, qwen2)")
    parser.add_argument("q_code", help="Digit string, one per layer in config order")
    parser.add_argument("delta_code", help="Digit string, one per layer in config order")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())

    arch = config["architectures"].get(args.model)
    if arch is None:
        sys.exit(f"Unknown model '{args.model}'. Available: {list(config['architectures'])}")

    layers = arch["layers"]
    layer_names = list(layers.keys())
    n = len(layer_names)

    # Validate code lengths
    for code_name, code in [("q_code", args.q_code), ("delta_code", args.delta_code)]:
        if len(code) != n:
            sys.exit(f"{code_name} length {len(code)} != number of layers {n} ({layer_names})")
        if not code.isdigit():
            sys.exit(f"{code_name} must be digits only, got '{code}'")

    # Apply codes
    for i, name in enumerate(layer_names):
        layers[name]["q"] = int(args.q_code[i])
        layers[name]["delta"] = int(args.delta_code[i])

    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Updated {args.model} layers in {CONFIG_PATH}:")
    for i, name in enumerate(layer_names):
        print(f"  {name}: q={args.q_code[i]}, delta={args.delta_code[i]}")

if __name__ == "__main__":
    main()
