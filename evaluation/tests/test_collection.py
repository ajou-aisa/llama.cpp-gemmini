"""Collector adapter checks; synthetic runner boundary, never a numerical measurement."""

import sys
from pathlib import Path
from unittest.mock import patch

from evaluation.manifest import Manifest
from scripts.eval.eval_common import Json, Record, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import metric_run


def collection(root: Path) -> None:
    # Given an explicit manifest seed and exact chunk policy plus small dummy collector inputs.
    directory = root / "collector-adapter"
    directory.mkdir()
    for name in ("runner", "model.gguf", "wiki.test.raw"):
        (directory / name).write_text("synthetic adapter input\n", encoding="utf-8")
    raw_manifest = read_json(root / "evaluation_manifest.json")
    raw_manifest["chunk_policy"] = "METRIC_PREFILL_256:split=test:max_chunks=1:context=256:tail=drop:output=second_half"
    manifest_path = directory / "evaluation_manifest.json"
    write_json(manifest_path, raw_manifest)
    manifest = Manifest.load(manifest_path)
    info: Record = {"activation_metrics": 1, "residual_metrics": 0, "scale_metrics": 0,
                    "cycle_sim": 1, "backend": "IM2P_SIM", "hp1": True, "gemmini": 1,
                    "gemmini_option": "WS", "activation_bits": 8, "weight_bits": 8,
                    "dim": 64, "activation_mode": "EXSIA", "block_size": 32}
    args = metric_run.parser_for("activation").parse_args([
        "--runner", str(directory / "runner"), "--model", str(directory / "model.gguf"),
        "--dataset", str(directory / "wiki.test.raw"), "--split", "test", "--max-chunks", "1",
        "--evaluation-manifest", str(manifest_path), "--output", str(directory / "output")])

    def runner(command: list[str], output: Path, timeout: int) -> None:
        assert timeout > 0
        seed = int(command[command.index("--seed") + 1]) if "--seed" in command else 0
        binding = command[command.index("--manifest-sha256") + 1] if "--manifest-sha256" in command else ""
        assert binding == manifest.sha256
        (output / "activation-quant-metrics.jsonl").write_text("adapter-only\n", encoding="utf-8")
        (output / "native").mkdir()
        tokens: list[Json] = [1] * 256
        chunk: Record = {"chunk_id": 0, "token_offset": 0, "input_tokens": tokens, "complete": True}
        native: Record = {"workload": "METRIC_PREFILL_256", "complete": True,
                          "output_mask": "second_half", "seed": seed, "tokens": 256,
                          "complete_chunks": 1, "selected_chunks": 1, "dropped_tail_tokens": 0,
                          "context_tokens": 256, "batch": 256, "ubatch": 256, "threads": 1,
                          "threads_batch": 1, "add_special": True, "parse_special": False,
                          "trailing_lf_removed": False, "add_bos": False, "bos_token": 0,
                          "bos_policy": "native", "kv_policy": "clear-per-chunk",
                          "chunks": [chunk]}
        write_json(output / "native/workload.json", native)

    # When using the real collection adapter with only the external binary boundary substituted.
    with patch("metric_run.compiled_info", return_value=info), patch("metric_run.clean_environment"), \
            patch("metric_run.run", side_effect=runner):
        _, binding = metric_run.collect_metric(args, "activation")
    # Then the requested seed survives actual command construction and returned workload verification.
    assert binding["manifest_sha256"] == manifest.sha256
    assert read_json(directory / "output/native/workload.json")["seed"] == 1234
    assert (directory / "output/evaluation_manifest.json").read_bytes() == manifest_path.read_bytes()
