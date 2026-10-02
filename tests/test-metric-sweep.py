# /// script
# requires-python = ">=3.11"
# dependencies = ["pytest"]
# ///
# How to run: python3 -B -m pytest -q tests/test-metric-sweep.py
"""`metric all`: matrix, shared builds, workload anchor, identity checks, summary/CSV/tables, failures and resume, on
synthetic campaign.py run directories (nothing is built or collected)."""
from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/eval"
sys.path.insert(0, str(EVAL))

import metric_run
import metric_sweep
import metric_table
import run_measurement as wrapper
from campaign_inputs import checksums
from eval_common import EvaluationError, read_json, sha256, write_json

REDUCER = sha256(ROOT / "evaluation/weight_alignment/__init__.py")
REDUCERS = {kind: sha256(ROOT / "evaluation" / module / "__init__.py")
            for kind, module in (("activation", "activation"), ("residual", "residual"), ("scu", "weight_alignment"))}
COLLECTOR = {name: sha256(ROOT / name) for name in ("scripts/eval/campaign.py", "scripts/eval/metric_run.py",
                                                    "scripts/eval/campaign_metrics.py", "evaluation/reducer/__init__.py")}

STAMP = "20260101T000000Z"
MODELS, PRECISIONS, DIMS = ("gpt2", "llama3.2-1B"), ("a4w4", "a8w8"), (16, 32, 64)
DISPLAY = {"gpt2": "GPT-2 124M", "llama3.2-1B": "Llama-3.2-1B"}
SINK = {"activation": "activation_metrics", "residual": "residual_metrics", "scu": "scale_metrics"}
CSV_HEADER = ["model", "precision", "dim",
              "activation_recall", "activation_jaccard", "activation_requant_ratio", "activation_residual_fraction",
              "scu_dense_avg_delta_w", "scu_dense_max_delta_w", "scu_dense_update_fraction",
              "scu_residual_avg_delta_w", "scu_residual_max_delta_w", "scu_residual_update_fraction",
              "scu_overall_avg_delta_w", "scu_overall_max_delta_w", "scu_overall_update_fraction",
              "residual_retained_row_factor", "residual_retained_k_factor", "residual_logical_ratio",
              "residual_padded_ratio"]
Key = tuple[str, str, int, str]  # (model, precision, DIM, metric kind)


def values(kind: str, model: str, precision: str, dim: int) -> dict[str, object]:
    """Distinct, exactly representable metric values per configuration and field."""
    base = MODELS.index(model) * 0.25 + PRECISIONS.index(precision) * 0.0625 + DIMS.index(dim) * 0.015625

    def fraction(index: int) -> float:
        return 0.5 + base + index / 1024

    if kind == "activation":
        ratios = ("recall", "jaccard", "requant_ratio", "residual_fraction", "fp_fraction")
        counts = ("fp_selected", "potal_selected", "intersection", "union", "valid_positions", "residual_nnz",
                  "eligible_logical_blocks", "unique_actual_requantized_blocks", "p3_requantization_events")
        return {**{name: fraction(index) for index, name in enumerate(ratios)},
                **{name: 1000 + index for index, name in enumerate(counts)}}
    if kind == "scu":
        def population(index: int) -> dict[str, object]:
            return {"avg_delta_w": 1 + index + base, "max_delta_w": 3 + index, "scu_update_fraction": fraction(index),
                    "delta_w_sum": 10 + index, "alignment_count": 20 + index, "updated_partial_sum_count": 30 + index,
                    "total_partial_sum_count": 40 + index}
        overall = population(2)
        return {**overall, "dense": population(0), "residual": population(1), "overall": overall}
    ratios = ("retained_row_factor", "retained_k_factor", "logical_ratio", "padded_ratio")
    counts = ("logical_main_macs", "logical_residual_macs", "physical_main_macs", "physical_residual_macs",
              "zero_limb_pruned_count", "original_radix_rows", "retained_rows", "original_k", "compact_k")
    return {**{name: fraction(index) for index, name in enumerate(ratios)},
            **{name: 2000 + index for index, name in enumerate(counts)}}


def write_run(output: Path, kind: str, options: dict[str, str], chunk_ids: list[int]) -> None:
    """A finished campaign.py run directory with every file the sweep and campaign_verify read."""
    model, precision, dim = options["--model"], options["--precision"], int(options["--dim"])
    bits = 4 if precision == "a4w4" else 8
    (output / "collection").mkdir(parents=True)
    manifest = output / "evaluation_manifest.json"
    if "--evaluation-manifest" in options:  # campaign.py copies the shared manifest bytes
        manifest.write_bytes(Path(options["--evaluation-manifest"]).read_bytes())
    else:
        write_json(manifest, {"model": DISPLAY[model], "dataset": "WikiText-2", "tokenizer_sha256": "t-" + model,
                              "precision": precision.upper(), "dim": dim, "BK": 32, "seed": int(options["--seed"]),
                              "git_sha": "g" * 40, "chunk_policy": f"max_chunks={options['--max-chunks']}"})
    digest = sha256(manifest)
    identity = f"{model}:" + ",".join(map(str, chunk_ids))  # tokens depend on the tokenizer, never on the DIM
    write_json(output / "collection/workload-binding.json", {
        "model_sha256": f"m-{model}-{precision}", "dataset_sha256": "d", "split": "test", "manifest_sha256": digest,
        "native_workload_identity": identity, "requested_max_chunks": int(options["--max-chunks"]),
        "workload": {"chunks": [{"chunk_id": chunk} for chunk in chunk_ids]}})
    prepared = Path(options["--prepared-build"]).resolve()
    runner = prepared / "bin/llama-eval-workload"
    sinks = {name: int(name == SINK[kind]) for name in SINK.values()}
    write_json(output / "manifest.json", {
        "kind": kind, "manifest_sha256": digest, "model_sha256": f"m-{model}-{precision}", "dataset_sha256": "d",
        "runner": str(runner), "build_hash": sha256(runner), "build_receipt_sha256": sha256(prepared / "build-receipt.json"),
        "build_info": {"activation_bits": bits, "weight_bits": bits, "dim": dim, "block_size": 32, **sinks},
        "model_manifest": None, "native_workload_identity": identity, "measurement_domain": "metric",
        **({"scu_collection_mode": options["--scu-mode"], "scu_reducer_sha256": REDUCER} if kind == "scu" else {})})
    write_json(output / "summary.json", {"manifest_sha256": digest, **values(kind, model, precision, dim)})
    checksums(output)


def write_combined_run(output: Path, options: dict[str, str], chunk_ids: list[int]) -> None:
    """A finished `campaign.py metrics-all` run: one collection, one manifest, one summary per metric kind."""
    model, precision, dim = options["--model"], options["--precision"], int(options["--dim"])
    bits = 4 if precision == "a4w4" else 8
    (output / "collection").mkdir(parents=True)
    manifest = output / "evaluation_manifest.json"
    write_json(manifest, {"model": DISPLAY[model], "dataset": "WikiText-2", "tokenizer_sha256": "t-" + model,
                          "precision": precision.upper(), "dim": dim, "BK": 32, "seed": int(options["--seed"]),
                          "git_sha": "g" * 40, "chunk_policy": f"max_chunks={options['--max-chunks']}"})
    digest = sha256(manifest)
    identity = f"{model}:" + ",".join(map(str, chunk_ids))
    write_json(output / "collection/workload-binding.json", {
        "model_sha256": f"m-{model}-{precision}", "dataset_sha256": "d", "split": "test", "manifest_sha256": digest,
        "native_workload_identity": identity, "requested_max_chunks": int(options["--max-chunks"]),
        "workload": {"chunks": [{"chunk_id": chunk} for chunk in chunk_ids]}})
    workers = int(options["--workers"])
    write_json(output / "collection/shards.json", {"manifest_sha256": digest, "shards": [
        {"first_chunk": index, "chunks": 1, "wall_seconds": 1.0} for index in range(workers)]})
    for kind in ("activation", "residual", "scu"):
        (output / kind).mkdir()
        write_json(output / kind / "summary.json", {"manifest_sha256": digest, **values(kind, model, precision, dim)})
    prepared = Path(options["--prepared-build"]).resolve()
    runner = prepared / "bin/llama-eval-workload"
    write_json(output / "manifest.json", {
        "kind": "metrics-all", "manifest_sha256": digest, "model_sha256": f"m-{model}-{precision}",
        "dataset_sha256": "d", "runner": str(runner), "build_hash": sha256(runner),
        "build_receipt_sha256": sha256(prepared / "build-receipt.json"),
        "build_info": {"activation_bits": bits, "weight_bits": bits, "dim": dim, "block_size": 32,
                       **dict.fromkeys(SINK.values(), 1)},
        "model_manifest": None, "native_workload_identity": identity, "measurement_domain": "metric",
        "kinds": ["activation", "residual", "scu"], "scu_collection_mode": options["--scu-mode"],
        "scu_reducer_sha256": REDUCER, "reducer_sha256": REDUCERS, "collector_sha256": COLLECTOR,
        "threads": int(options["--threads"]), "threads_batch": int(options["--threads-batch"]), "workers": workers,
        "metric_execution": metric_run.metric_execution(options["--terminal-lm-head"]),
        "timing": {"native_collection": 1.0}})
    checksums(output)


class Fakes:
    """The build cache and the campaign.py processes the sweep starts, replaced by synthetic directories."""

    def __init__(self) -> None:
        self.builds: list[tuple[str, str, int]] = []
        self.calls: list[list[str]] = []
        self.fail: set[Key] = set()
        self.chunks: dict[Key, list[int]] = {}
        self.sources = ""  # a later source state (commit, merge) resolves every profile to another build

    def build(self, cache: Path, kind: str, precision: str, dim: int, im2p: Path, jobs: int) -> tuple[Path, bool]:
        path = cache / f"{kind}-{precision}-d{dim}{self.sources}"
        if path.exists():
            return path, True
        self.builds.append((kind, precision, dim))
        (path / "bin").mkdir(parents=True)
        (path / "bin/llama-eval-workload").write_text(f"runner {kind} {precision} {dim}")
        write_json(path / "build-receipt.json", {"kind": kind, "verification": "PASS",
                   "runner_sha256": sha256(path / "bin/llama-eval-workload"),
                   "semantic_options_sha256": f"semantic-{kind}-{precision}-{dim}"})
        return path, False

    def process(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == [sys.executable, "-B", str(EVAL / "campaign.py")]
        argv = command[3:]
        self.calls.append(argv)
        kind, options = argv[0], dict(zip(argv[1::2], argv[2::2]))
        key = (options["--model"], options["--precision"], int(options["--dim"]), kind)
        output = Path(options["--output"])
        if key in self.fail:  # a campaign that stopped after creating its run directory
            (output / "collection").mkdir(parents=True)
            return subprocess.CompletedProcess(command, 1, "", "campaign failed: native collection failed\n")
        if kind == "metrics-all":
            write_combined_run(output, options, self.chunks.get(key, [0, 3]))
        else:
            write_run(output, kind, options, self.chunks.get(key, [0, 3]))
        return subprocess.CompletedProcess(command, 0, f"{output}\n", "")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    value = Fakes()
    monkeypatch.setattr(metric_sweep, "cached_llama_build", value.build)
    monkeypatch.setattr(metric_sweep.subprocess, "run", value.process)
    return value


def sweep(tmp_path: Path, *argv: str, collection: str = "separate") -> int:
    """`metric all` on the fakes; the separate collection (one run per metric kind) unless asked otherwise."""
    return metric_sweep.main([*argv, "--im2p", str(tmp_path), "--build-cache", str(tmp_path / "cache"),
                              "--collection", collection])


def table(output: str, title: str) -> list[list[str]]:
    """Body rows of one box table of the stdout."""
    block = output.split(title + "\n", 1)[1].split("\n\n", 1)[0]
    rows = [[cell.strip() for cell in line.strip("│").split("│")] for line in block.splitlines() if line.startswith("│")]
    return rows[1:]


# Entry point

def test_run_measurement_dispatches_metric_all_to_the_sweep() -> None:
    assert wrapper.delegate(["metric", "all"], STAMP) == (
        EVAL / "metric_sweep.py", ["--output", f"runs/metrics/sweep-{STAMP}"])
    assert wrapper.delegate(["metric", "all", "--models", "gpt2", "--dims", "32"], STAMP)[1] == [
        "--models", "gpt2", "--dims", "32", "--output", f"runs/metrics/sweep-{STAMP}"]
    for extra in (["--resume", "runs/metrics/sweep-x"], ["--dry-run"], ["--output", "out"]):
        assert wrapper.delegate(["metric", "all", *extra], STAMP)[1] == extra
    # The individual metric paths are unchanged.
    assert wrapper.delegate(["metric", "scu", "--model", "gpt2", "--output", "o"], STAMP) == (
        EVAL / "campaign.py", ["scu", "--model", "gpt2", "--output", "o"])


def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    assert sweep(tmp_path, "--dry-run") == 0
    output = capsys.readouterr().out
    for line in ("12 configurations", "36 metric runs", "18 build configurations", "models = gpt2, llama3.2-1B",
                 "precisions = a4w4, a8w8", "DIMs = 16, 32, 64", "max_chunks = 0 (all complete chunks)",
                 "seed = 1234", "SCU mode = aggregate"):
        assert line + "\n" in output
    commands = [shlex.split(line) for line in output.splitlines() if line.startswith("  ") and "--output" in line]
    assert len(commands) == 36 and [command[0] for command in commands[:3]] == ["activation", "residual", "scu"]
    assert commands[1][commands[1].index("--evaluation-manifest") + 1] == "OUTPUT/gpt2/a4w4/d16/activation/evaluation_manifest.json"
    assert len([line for line in output.splitlines() if "semantic_options_sha256=" in line]) == 18
    assert fakes.builds == [] and fakes.calls == [] and not (tmp_path / "cache").exists()


# Matrix, build reuse, summary, CSV and tables

def test_full_matrix_shares_builds_and_aggregates_every_configuration(
        tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "sweep"
    assert sweep(tmp_path, "--output", str(root)) == 0
    output = capsys.readouterr().out
    # Given the default matrix: 2 x 2 x 3 configurations, 3 metric runs each, one build per kind x precision x DIM.
    configs = [(model, precision, dim) for model in MODELS for precision in PRECISIONS for dim in DIMS]
    assert len(fakes.calls) == 36 and [(call[2], call[4], int(call[6]), call[0]) for call in fakes.calls] == [
        (*config, kind) for config in configs for kind in ("activation", "residual", "scu")]
    assert len(fakes.builds) == len(set(fakes.builds)) == 18
    summary = read_json(root / "metric-summary.json")
    rows = summary["configurations"]
    assert summary["status"] == "PASS" and summary["failures"] == [] and len(rows) == 12
    assert summary["counts"] == {"configurations": 12, "metric_runs": 36, "builds": 18, "native_collections": 36,
                                 "passed_configurations": 12}
    assert [(row["model"], row["precision"], row["dim"]) for row in rows] == configs
    # Both models use the same prepared build of a metric/precision/DIM; every build is recorded per configuration.
    used: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        for kind, build in row["builds"].items():
            assert set(build) == {"name", "path", "build_receipt_sha256", "runner_sha256", "semantic_options_sha256"}
            used.setdefault((kind, row["precision"], row["dim"]), set()).add(json.dumps(build, sort_keys=True))
    assert len(used) == 18 and all(len(builds) == 1 for builds in used.values())
    assert sorted(path.name for path in (root / "builds").iterdir()) == sorted(
        f"{kind}-{precision}-d{dim}" for kind, precision, dim in used)
    # JSON keeps every copied field, including the SCU populations and count diagnostics.
    first = rows[0]
    assert first["activation"] == {name: value for name, value in values("activation", "gpt2", "a4w4", 16).items()}
    assert first["scu"] == {"collection_mode": "aggregate",
                            **{name: values("scu", "gpt2", "a4w4", 16)[name] for name in ("dense", "residual", "overall")}}
    assert first["residual"] == values("residual", "gpt2", "a4w4", 16)
    assert first["runs"] == {kind: f"gpt2/a4w4/d16/{kind}" for kind in ("activation", "residual", "scu")}
    # CSV: one row per configuration, the paper columns mapped exactly.
    with (root / "metric-summary.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == CSV_HEADER
        table_rows = list(reader)
    assert len(table_rows) == 12
    for row, (model, precision, dim) in zip(table_rows, configs):
        activation, scu, residual = (values(kind, model, precision, dim) for kind in ("activation", "scu", "residual"))
        expected = [model, precision, dim, *(activation[name] for name in ("recall", "jaccard", "requant_ratio",
                    "residual_fraction")),
                    *(scu[population][name] for population in ("dense", "residual", "overall")
                      for name in ("avg_delta_w", "max_delta_w", "scu_update_fraction")),
                    *(residual[name] for name in ("retained_row_factor", "retained_k_factor", "logical_ratio",
                                                  "padded_ratio"))]
        assert [row[name] for name in CSV_HEADER] == [str(value) for value in expected]
    # stdout: one row per configuration in each paper table, ordered by model, precision, DIM.
    for title in ("Activation Adaptation", "SCU Weight-Scale Alignment — Dense", "Residual Overhead"):
        body = table(output, title)
        assert [(cells[0], cells[1], int(cells[2])) for cells in body] == [
            (DISPLAY[model], precision.upper(), dim) for model, precision, dim in configs], title
    assert table(output, "Activation Adaptation")[0][3:] == ["50.00%", "50.10%", "50.20%", "50.29%"]
    assert table(output, "SCU Weight-Scale Alignment — Dense")[0][3:] == ["1.00", "3", "50.00%"]
    assert "SCU Weight-Scale Alignment — Residual" not in output and "status: PASS" in output


def test_scu_breakdown_all_and_plain_tables(tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    assert sweep(tmp_path, "--output", str(tmp_path / "s"), "--models", "gpt2", "--precisions", "a8w8",
                 "--dims", "32", "--scu-breakdown", "all", "--plain") == 0
    output = capsys.readouterr().out
    for title in ("SCU Weight-Scale Alignment - Dense", "SCU Weight-Scale Alignment - Residual",
                  "SCU Weight-Scale Alignment - Overall"):
        assert title + "\nModel       Prec  DIM  Avg dW  Max dW  Update Frac\n" in output
    assert "│" not in output and "Δ" not in output


def test_table_rendering_is_fixed() -> None:
    entry = [("GPT-2 124M", "A8W8", 32, {"recall": 0.982, "jaccard": 0.947, "requant_ratio": 0.031,
                                          "residual_fraction": None})]
    assert metric_table.metric_table("activation", entry) == """\
Activation Adaptation
┌────────────┬──────┬─────┬────────┬─────────┬──────────┬───────────────┐
│ Model      │ Prec │ DIM │ Recall │ Jaccard │ Requant. │ Residual Frac │
├────────────┼──────┼─────┼────────┼─────────┼──────────┼───────────────┤
│ GPT-2 124M │ A8W8 │  32 │ 98.20% │  94.70% │    3.10% │             - │
└────────────┴──────┴─────┴────────┴─────────┴──────────┴───────────────┘"""
    assert metric_table.metric_table("activation", entry, plain=True) == """\
Activation Adaptation
Model       Prec  DIM  Recall  Jaccard  Requant.  Residual Frac
GPT-2 124M  A8W8   32  98.20%   94.70%     3.10%              -"""


# Workload identity

def test_residual_and_scu_reuse_the_activation_workload(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a4w4") == 0
    for call in fakes.calls:
        options = dict(zip(call[1::2], call[2::2]))
        anchor = root / f"gpt2/a4w4/d{options['--dim']}/activation"
        if call[0] == "activation":
            assert "--evaluation-manifest" not in options and "--workload-manifest" not in options
        else:
            assert options["--evaluation-manifest"] == f"{anchor}/evaluation_manifest.json"
            assert options["--workload-manifest"] == f"{anchor}/collection/workload-binding.json"
        assert options["--prepared-build"] == str(root / f"builds/{call[0]}-a4w4-d{options['--dim']}")


def test_a_metric_run_with_other_chunks_fails_the_configuration(
        tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.chunks[("gpt2", "a8w8", 32, "residual")] = [7]
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32") == 1
    summary = read_json(root / "metric-summary.json")
    assert summary["status"] == "FAILED" and summary["configurations"] == []
    [failure] = summary["failures"]
    assert failure["stage"] == "identity" and "chunk_ids, native_workload_identity" in failure["error"]
    with (root / "metric-summary.csv").open() as stream:
        assert stream.read().splitlines() == [",".join(CSV_HEADER)]  # never a partial or zero row
    assert "Failed configurations" in capsys.readouterr().out


def test_dims_share_one_chunk_population(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8") == 0
    summary = read_json(root / "metric-summary.json")
    rows = summary["configurations"]
    # Given DIM 16/32/64 of one model and precision: same chunks and workload, different DIM and builds.
    assert summary["dim_identity"] == [{"model": "gpt2", "precision": "a8w8", "dims": [16, 32, 64], "status": "SAME",
                                        "native_workload_identity": "gpt2:0,3", "chunk_ids": [0, 3]}]
    shared = [row["shared_identity"] for row in rows]
    assert [row["dim"] for row in shared] == [16, 32, 64] and all(row["chunk_ids"] == [0, 3] for row in shared)
    assert all({key: value for key, value in row.items() if key != "dim"} ==
               {key: value for key, value in shared[0].items() if key != "dim"} for row in shared)
    for kind in ("activation", "residual", "scu"):
        assert len({row["builds"][kind]["semantic_options_sha256"] for row in rows}) == 3
    # When one DIM selects other chunks (consistently for its three metrics), that configuration fails.
    for kind in ("activation", "residual", "scu"):
        fakes.chunks[("gpt2", "a8w8", 64, kind)] = [5]
    other = tmp_path / "other"
    assert sweep(tmp_path, "--output", str(other), "--models", "gpt2", "--precisions", "a8w8") == 1
    [failure] = read_json(other / "metric-summary.json")["failures"]
    assert failure["dim"] == 64 and "DIM 64 workload differs from DIM 16 in: native_workload_identity, chunk_ids" in failure["error"]


# Failures and resume

def test_fail_fast_and_keep_going(tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.fail.add(("gpt2", "a8w8", 32, "residual"))
    assert sweep(tmp_path, "--output", str(tmp_path / "fast"), "--models", "gpt2", "--precisions", "a8w8") == 1
    assert [call[0] for call in fakes.calls] == ["activation", "residual", "scu", "activation", "residual"]
    fast = read_json(tmp_path / "fast/metric-summary.json")
    assert [row["dim"] for row in fast["configurations"]] == [16] and len(fast["failures"]) == 1
    capsys.readouterr()
    fakes.calls.clear()
    assert sweep(tmp_path, "--output", str(tmp_path / "all"), "--models", "gpt2", "--precisions", "a8w8",
                 "--keep-going") == 1
    assert len(fakes.calls) == 9  # SCU of DIM 32 still runs on its activation anchor
    summary = read_json(tmp_path / "all/metric-summary.json")
    assert summary["status"] == "FAILED" and [row["dim"] for row in summary["configurations"]] == [16, 64]
    assert [(row["dim"], row["stage"]) for row in summary["failures"]] == [(32, "residual")]
    output = capsys.readouterr().out
    assert len(table(output, "Activation Adaptation")) == 2
    assert table(output, "Failed configurations") == [
        ["gpt2", "A8W8", "32", "residual", "campaign.py residual failed: campaign failed: native collection failed"]]


def test_resume_reuses_complete_runs_and_retries_incomplete_ones(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    fakes.fail.add(("gpt2", "a8w8", 32, "scu"))
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "16,32") == 1
    assert read_json(root / "metric-summary.json")["status"] == "FAILED"
    partial = root / "gpt2/a8w8/d32/scu"
    assert partial.is_dir() and not (partial / "SHA256SUMS").exists()
    # The recorded matrix and options are the sweep's; --resume refuses to change them.
    with pytest.raises(SystemExit):
        metric_sweep.parser().parse_args(["--resume", str(root), "--output", "x"])
    assert metric_sweep.main(["--resume", str(root), "--dims", "64"]) == 1
    fakes.fail.clear()
    fakes.calls.clear()
    before = {path: path.stat().st_mtime_ns for path in partial.rglob("*")}
    assert metric_sweep.main(["--resume", str(root)]) == 0
    # Only the incomplete metric runs again, in a new directory; the incomplete one is left as it was.
    assert [(call[0], call[call.index("--output") + 1]) for call in fakes.calls] == [("scu", f"{partial}.retry-1")]
    assert {path: path.stat().st_mtime_ns for path in partial.rglob("*")} == before
    summary = read_json(root / "metric-summary.json")
    assert summary["status"] == "PASS" and len(summary["configurations"]) == 2
    assert summary["configurations"][1]["runs"]["scu"] == "gpt2/a8w8/d32/scu.retry-1"
    assert read_json(root / "metric-summary.attempt-1.json")["status"] == "FAILED"
    assert (root / "gpt2/a8w8/d32/scu.campaign.log").is_file() and (root / "metric-summary.attempt-1.csv").is_file()
    # A finished sweep is not resumed again.
    assert metric_sweep.main(["--resume", str(root)]) == 1


def test_resume_rejects_a_corrupt_complete_run(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    fakes.fail.add(("gpt2", "a8w8", 32, "activation"))
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "16,32") == 1
    fakes.fail.clear()
    summary = root / "gpt2/a8w8/d16/residual/summary.json"
    summary.write_text(summary.read_text().replace("2000", "2001"))
    fakes.calls.clear()
    assert metric_sweep.main(["--resume", str(root)]) == 1
    # The corrupt run is neither reused nor silently replaced; the sweep stops there.
    assert fakes.calls == [] and not (root / "gpt2/a8w8/d16/residual.retry-1").exists()
    [failure] = read_json(root / "metric-summary.json")["failures"]
    assert (failure["dim"], failure["stage"]) == (16, "residual") and "artifact hash changed: summary.json" in failure["error"]


def test_resume_keeps_its_builds_after_the_sources_change(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    fakes.fail.add(("gpt2", "a8w8", 32, "metrics-all"))
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "16,32",
                 collection="combined") == 1
    started = {link.name: link.resolve() for link in (root / "builds").iterdir()}
    # A commit or merge after the start: the build cache now resolves these profiles to other builds.
    fakes.sources = "-merged"
    fakes.fail.clear()
    fakes.builds.clear()
    fakes.calls.clear()
    runner = started["metrics-all-a8w8-d32"] / "bin/llama-eval-workload"
    original = runner.read_text()
    runner.write_text("changed after its receipt")
    assert metric_sweep.main(["--resume", str(root)]) == 1  # a recorded build that changed is refused, not replaced
    [failure] = read_json(root / "metric-summary.json")["failures"]
    assert failure["stage"] == "build metrics-all" and "no longer holds the verified build" in failure["error"]
    runner.write_text(original)
    assert metric_sweep.main(["--resume", str(root)]) == 0
    assert fakes.builds == [] and {link.name: link.resolve() for link in (root / "builds").iterdir()} == started
    [call] = fakes.calls  # only the unfinished configuration runs, on the build the sweep started with
    assert Path(call[call.index("--prepared-build") + 1]).resolve() == started["metrics-all-a8w8-d32"]


def test_reused_scu_run_without_the_split_is_refused(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32") == 0
    run = root / "gpt2/a8w8/d32"
    runs = {kind: run / kind for kind in ("activation", "residual", "scu")}
    old = read_json(runs["scu"] / "summary.json")
    del old["dense"]
    (runs["scu"] / "summary.json").unlink()
    write_json(runs["scu"] / "summary.json", old)
    with pytest.raises(EvaluationError, match="no dense/residual/overall split"):
        metric_sweep.configuration(root, metric_sweep.recorded(root), ("gpt2", "a8w8", 32), runs,
                                   {(kind, "a8w8", 32): root / f"builds/{kind}-a8w8-d32" for kind in runs})


# Individual metric command: stdout contract kept, table on stderr

def test_individual_campaign_keeps_stdout_and_prints_its_table_on_stderr(tmp_path: Path) -> None:
    prepared, _ = Fakes().build(tmp_path / "cache", "activation", "a8w8", 32, tmp_path, 1)
    run = tmp_path / "run"
    write_run(run, "activation", {"--model": "gpt2", "--precision": "a8w8", "--dim": "32", "--seed": "1234",
                                  "--max-chunks": "1", "--prepared-build": str(prepared)}, [0])
    code = f'''
import sys
from pathlib import Path
sys.path.insert(0, {str(EVAL)!r})
import campaign
campaign.run_campaign = lambda args: Path({str(run)!r})
sys.argv = ["campaign.py", "activation", "--model", "gpt2", "--precision", "a8w8", "--dim", "32"]
raise SystemExit(campaign.main())
'''
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"{run}\n"
    title, header, row = done.stderr.splitlines()
    assert (title, header) == ("Activation Adaptation", "Model       Prec  DIM  Recall  Jaccard  Requant.  Residual Frac")
    assert row.startswith("GPT-2 124M  A8W8   32  57.81%")


# SCU collection mode

def test_metric_all_collects_scu_in_aggregate_mode_and_guards_detailed_full_corpus(
        tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32") == 0
    modes = {call[0]: dict(zip(call[1::2], call[2::2])).get("--scu-mode") for call in fakes.calls}
    assert modes == {"activation": None, "residual": None, "scu": "aggregate"}
    assert read_json(root / "manifest.json")["scu_mode"] == "aggregate"
    assert read_json(root / "metric-summary.json")["configurations"][0]["scu"]["collection_mode"] == "aggregate"
    with (root / "metric-summary.csv").open() as stream:  # the paper columns do not depend on the mode
        assert next(csv.reader(stream)) == CSV_HEADER
    # Detailed SCU of the full corpus is refused before anything is built; bounded or acknowledged runs are allowed.
    fakes.builds.clear()
    assert sweep(tmp_path, "--output", str(tmp_path / "full"), "--scu-mode", "detailed") == 1
    assert not (tmp_path / "full").exists() and fakes.builds == []
    assert sweep(tmp_path, "--output", str(tmp_path / "one"), "--models", "gpt2", "--precisions", "a8w8",
                 "--dims", "32", "--scu-mode", "detailed", "--max-chunks", "1") == 0
    assert read_json(tmp_path / "one/metric-summary.json")["configurations"][0]["scu"]["collection_mode"] == "detailed"
    assert sweep(tmp_path, "--dry-run", "--scu-mode", "detailed", "--allow-large-raw-scu") == 0


def test_resume_never_reuses_an_scu_run_of_another_mode_or_reducer(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    fakes.fail.add(("gpt2", "a8w8", 32, "scu"))
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "16,32") == 1
    assert metric_sweep.main(["--resume", str(root), "--scu-mode", "detailed"]) == 1  # the mode is recorded
    # A complete DIM 16 SCU run that claims detailed collection is not an aggregate result.
    binding = root / "gpt2/a8w8/d16/scu/manifest.json"
    value = read_json(binding)
    value["scu_collection_mode"] = "detailed"
    binding.unlink()
    write_json(binding, value)
    runs = {kind: root / f"gpt2/a8w8/d16/{kind}" for kind in ("activation", "residual", "scu")}
    builds = {(kind, "a8w8", 16): root / f"builds/{kind}-a8w8-d16" for kind in runs}
    with pytest.raises(EvaluationError, match="collected in detailed mode"):
        metric_sweep.configuration(root, metric_sweep.recorded(root), ("gpt2", "a8w8", 16), runs, builds)
    value.update(scu_collection_mode="aggregate", scu_reducer_sha256="0" * 64)
    binding.unlink()
    write_json(binding, value)
    with pytest.raises(EvaluationError, match="another SCU reducer source"):
        metric_sweep.configuration(root, metric_sweep.recorded(root), ("gpt2", "a8w8", 16), runs, builds)


# Combined collection: one forward per configuration (metrics-all build), exact shard merge

def test_combined_collection_is_one_forward_per_configuration(
        tmp_path: Path, fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "combined"
    assert sweep(tmp_path, "--output", str(root), "--threads", "4", "--workers", "2", collection="combined") == 0
    configs = [(model, precision, dim) for model in MODELS for precision in PRECISIONS for dim in DIMS]
    # 12 native collections, each producing all three metrics; 6 combined builds shared by both models.
    assert [(call[2], call[4], int(call[6]), call[0]) for call in fakes.calls] == [
        (*config, "metrics-all") for config in configs]
    options = [dict(zip(call[1::2], call[2::2])) for call in fakes.calls]
    assert {(o["--workers"], o["--threads"], o["--threads-batch"], o["--scu-mode"], o["--terminal-lm-head"])
            for o in options} == {("2", "4", "4", "aggregate", "metrics-only")}
    assert len(fakes.builds) == len(set(fakes.builds)) == 6 and {kind for kind, _, _ in fakes.builds} == {"metrics-all"}
    summary = read_json(root / "metric-summary.json")
    assert summary["status"] == "PASS" and summary["counts"] == {
        "configurations": 12, "metric_runs": 36, "builds": 6, "native_collections": 12, "passed_configurations": 12}
    assert summary["collection"] == {"mode": "combined", "threads": 4, "threads_batch": 4, "workers": 2,
                                     "terminal_lm_head": "metrics-only"}
    first = summary["configurations"][0]
    assert first["runs"] == {kind: f"gpt2/a4w4/d16/metrics-all/{kind}" for kind in ("activation", "residual", "scu")}
    assert first["builds"]["activation"] == first["builds"]["residual"] == first["builds"]["scu"]
    assert first["builds"]["scu"]["name"] == "metrics-all-a4w4-d16"
    assert (first["collection"]["mode"], first["collection"]["threads"], first["collection"]["workers"]) == (
        "combined", 4, 2) and len(first["collection"]["shards"]) == 2
    combined_tables = capsys.readouterr().out
    # The same metric values, CSV bytes and tables as the separate collection.
    separate = tmp_path / "separate"
    assert sweep(tmp_path, "--output", str(separate)) == 0
    assert (root / "metric-summary.csv").read_bytes() == (separate / "metric-summary.csv").read_bytes()
    other = read_json(separate / "metric-summary.json")["configurations"]
    assert [{kind: row[kind] for kind in ("activation", "residual", "scu")} for row in summary["configurations"]] == [
        {kind: row[kind] for kind in ("activation", "residual", "scu")} for row in other]
    separate_tables = capsys.readouterr().out
    for title in ("Activation Adaptation", "SCU Weight-Scale Alignment — Dense", "Residual Overhead"):
        assert table(combined_tables, title) == table(separate_tables, title)
    assert sweep(tmp_path, "--dry-run", "--threads", "4", "--workers", "2", collection="combined") == 0
    plan = capsys.readouterr().out
    for line in ("36 metric runs from 12 combined collections", "6 build configurations",
                 "collection = combined (threads = 4, threads_batch = 4, workers = 2, terminal lm_head = metrics-only)"):
        assert line + "\n" in plan
    assert len([line for line in plan.splitlines() if line.startswith("  metrics-all --model")]) == 12


def test_combined_collection_guards(tmp_path: Path, fakes: Fakes) -> None:
    # Detailed SCU only exists in the separate collection; thread settings only in the combined one.
    assert sweep(tmp_path, "--output", str(tmp_path / "detailed"), "--scu-mode", "detailed", "--max-chunks", "1",
                 collection="combined") == 1
    assert sweep(tmp_path, "--output", str(tmp_path / "threads"), "--threads", "4") == 1
    assert sweep(tmp_path, "--output", str(tmp_path / "workers"), "--workers", "0", collection="combined") == 1
    assert not any((tmp_path / name).exists() for name in ("detailed", "threads", "workers")) and fakes.calls == []


def test_resume_reuses_a_combined_run_only_with_its_threads_and_sources(tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32",
                 "--threads", "2", collection="combined") == 0
    recorded = metric_sweep.recorded(root)
    assert (recorded.collection, recorded.threads, recorded.threads_batch) == ("combined", 2, 2)
    assert metric_sweep.main(["--resume", str(root), "--threads", "4"]) == 1  # recorded, not overridable
    run = root / "gpt2/a8w8/d32/metrics-all"
    runs = dict.fromkeys(("activation", "residual", "scu"), run)
    builds = {("metrics-all", "a8w8", 32): root / "builds/metrics-all-a8w8-d32"}
    assert metric_sweep.configuration(root, recorded, ("gpt2", "a8w8", 32), runs, builds)["collection"]["threads"] == 2
    binding = read_json(run / "manifest.json")
    for change, reason in (({"threads": 8}, "other thread settings"),
                           ({"collector_sha256": {**COLLECTOR, "scripts/eval/metric_run.py": "0" * 64}},
                            "other collector/reducer sources")):
        (run / "manifest.json").unlink()
        write_json(run / "manifest.json", {**binding, **change})
        with pytest.raises(EvaluationError, match=reason):
            metric_sweep.configuration(root, recorded, ("gpt2", "a8w8", 32), runs, builds)


def test_terminal_lm_head_mode_is_recorded_and_part_of_the_resume_identity(tmp_path: Path, fakes: Fakes) -> None:
    # The combined collection runs the terminal lm_head metrics-only by default; full is explicit and recorded.
    root = tmp_path / "s"
    assert sweep(tmp_path, "--output", str(root), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32",
                 collection="combined") == 0
    assert [dict(zip(call[1::2], call[2::2]))["--terminal-lm-head"] for call in fakes.calls] == ["metrics-only"]
    recorded = metric_sweep.recorded(root)
    assert recorded.terminal_lm_head == "metrics-only"
    assert read_json(root / "metric-summary.json")["configurations"][0]["collection"]["metric_execution"] == {
        "terminal_lm_head": "metrics-only", "lm_head_observation": "full", "lm_head_numerical_gemm": "elided",
        "logits_materialized": False, "reason": "terminal output not consumed by the metric-only prefill"}
    full = tmp_path / "full"
    assert sweep(tmp_path, "--output", str(full), "--models", "gpt2", "--precisions", "a8w8", "--dims", "32",
                 "--terminal-lm-head", "full", collection="combined") == 0
    assert read_json(full / "metric-summary.json")["collection"]["terminal_lm_head"] == "full"
    # The mode defines results: never changed on resume, never mixed between runs, only combined.
    assert metric_sweep.main(["--resume", str(root), "--terminal-lm-head", "full"]) == 1
    run = root / "gpt2/a8w8/d32/metrics-all"
    runs: dict[str, Path] = dict.fromkeys(("activation", "residual", "scu"), run)
    builds = {("metrics-all", "a8w8", 32): root / "builds/metrics-all-a8w8-d32"}
    binding = read_json(run / "manifest.json")
    for execution in (metric_run.metric_execution("full"), None):  # a run collected before terminal modes was full
        (run / "manifest.json").unlink()
        write_json(run / "manifest.json", {**binding, "metric_execution": execution})
        with pytest.raises(EvaluationError, match="terminal lm_head in another mode"):
            metric_sweep.configuration(root, recorded, ("gpt2", "a8w8", 32), runs, builds)
    assert sweep(tmp_path, "--output", str(tmp_path / "separate"), "--terminal-lm-head", "metrics-only") == 1
    manifest = read_json(root / "manifest.json")
    del manifest["terminal_lm_head"]
    (root / "manifest.json").unlink()
    write_json(root / "manifest.json", manifest)
    assert metric_sweep.recorded(root).terminal_lm_head == "full"


def test_shard_plan_and_workload_merge() -> None:
    import metric_run
    # Contiguous shards cover every selected chunk exactly once; sizes differ by at most one.
    assert metric_run.shard_plan(1118, 4) == [(0, 280), (280, 280), (560, 279), (839, 279)]
    for selected, workers in ((1, 4), (32, 1), (32, 3), (1129, 6), (7, 7)):
        shards = metric_run.shard_plan(selected, workers)
        chunks = [chunk for first, count in shards for chunk in range(first, first + count)]
        assert chunks == list(range(selected)) and len(shards) == min(selected, workers)
        assert max(count for _, count in shards) - min(count for _, count in shards) <= 1
    with pytest.raises(EvaluationError):
        metric_run.shard_plan(0, 2)
    # Shard workloads join into one workload only when every run field agrees.
    def shard(first: int, count: int, **change: object) -> dict[str, object]:
        return {"workload": "METRIC_PREFILL_256", "threads": 4, "tokens": 9000, "first_chunk": first,
                "selected_chunks": count, "chunks": [{"chunk_id": chunk} for chunk in range(first, first + count)],
                **change}
    merged = metric_run.merge_workloads([shard(0, 2), shard(2, 1)])
    assert (merged["first_chunk"], merged["selected_chunks"], [row["chunk_id"] for row in merged["chunks"]]) == (
        0, 3, [0, 1, 2])
    with pytest.raises(EvaluationError, match="different workloads"):
        metric_run.merge_workloads([shard(0, 2), shard(2, 1, threads=8)])


FAKE_RUNNER = r'''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
if argv == ["--build-info"]:
    print(json.dumps({"schema": "potal-evaluation-build", "version": 1, "activation_metrics": 1, "residual_metrics": 1,
                      "scale_metrics": 1, "cycle_sim": 1, "backend": "IM2P_SIM", "hp1": True, "gemmini": 1,
                      "gemmini_option": "WS", "activation_bits": 8, "weight_bits": 8, "dim": 64,
                      "activation_mode": "EXSIA", "block_size": 32, "rmd_enabled": 1, "rmd_backend": "WS"}))
    sys.exit(0)
plan = "--plan-only" in argv
values = [value for value in argv if value != "--plan-only"]
options = dict(zip(values[::2], values[1::2]))
total = int(os.environ["FAKE_CHUNKS"])
if plan:
    selected = total if options["--max-chunks"] == "0" else min(total, int(options["--max-chunks"]))
    print(json.dumps({"schema": "potal-native-chunk-plan", "version": 1, "tokens": total * 256, "complete_chunks": total,
                      "selected_chunks": selected, "first_chunk": 0, "dropped_tail_tokens": 0, "context_tokens": 256}))
    sys.exit(0)
first, count = int(options["--first-chunk"]), int(options["--max-chunks"])
if str(first) in os.environ.get("FAKE_FAIL", "").split(","):
    sys.exit(3)
common = {"version": 1, "run_id": options["--run-id"], "workload_id": "METRIC_PREFILL_256",
          "manifest_sha256": options["--manifest-sha256"], "precision": "A8W8", "dim": 64}
def stream(path, schema, run, rows, end):
    lines = [{**common, "schema": schema, "kind": "RUN", **run}]
    lines += [{**common, "schema": schema, **row} for row in rows]
    lines.append({**common, "schema": schema, "kind": "RUN_END", "success": True, **end})
    Path(path).write_text("".join(json.dumps({**row, "sequence": index}) + "\n" for index, row in enumerate(lines)))
chunks = list(range(first, first + count))
counts = []
for local, chunk in enumerate(chunks):
    fp, selected, both = 1 + chunk % 3, 2 + chunk % 2, 1
    counts.append({"kind": "COUNTS", "chunk_id": chunk, "invocation_id": local, "layer": "lm_head", "m": 1, "k": 64,
                   "valid_positions": 64, "finite_positions": 64, "nonfinite_positions": 0, "reference_complete": True,
                   "reference_invalid_reason": None, "fp_selected": fp, "potal_selected": selected,
                   "intersection": both, "union": fp + selected - both, "residual_nnz": chunk % 2,
                   "eligible_logical_blocks": 2, "unique_actual_requantized_blocks": 1, "p3_requantization_events": 2})
stream(options["--activation-output"], "im2p-activation-quant-metrics",
       {"definition_status": "CONFIRMED_BY_USER", "reference_revision": "signed-row-original-bk32-population-2sigma-v1"},
       counts, {"invocation_count": count, "observation_count": count, "reference_complete": True})
residual = []
for local, chunk in enumerate(chunks):
    at = {"chunk_id": chunk, "invocation_id": local, "layer": "lm_head", "stripe_id": 0}
    residual += [{**at, "kind": "MAIN_STRIPE", "row_begin": 0, "row_count": 2, "m": 2, "n": 16, "k": 64,
                  "physical_fragments": 2},
                 {**at, "kind": "RADIX_STRIPE", "radix_limb_count": 2, "original_rows": 4, "original_radix_rows": 4,
                  "main_original_rows": 2, "original_k": 64},
                 {**at, "kind": "COMPACT_WORK", "m": 3, "n": 16, "k": 4, "original_k": 64, "source_row_begin": 0,
                  "source_row_count": 2, "tile_i_count": 1, "tile_j_count": 1, "tile_k_count": 1, "radix_limb_count": 2,
                  "original_rows": 4, "retained_rows": 3, "zero_limb_pruned_count": 1, "retained_k": 4, "compact_k": 4,
                  "physical_fragments": 2,
                  "runs": [{"original_block_id": 0, "original_k_mask": 3, "compact_k_begin": 0, "compact_k_count": 2},
                           {"original_block_id": 1, "original_k_mask": 5, "compact_k_begin": 2, "compact_k_count": 2}],
                  "row_map": [{"original_lane_id": 0, "source_row": 0}, {"original_lane_id": 0, "source_row": 1},
                              {"original_lane_id": 1, "source_row": 0}]}]
stream(options["--residual-output"], "im2p-residual-path-metrics", {}, residual,
       {"invocation_count": count, "observation_count": len(residual)})
scale = []
for chunk in chunks:
    for work, offset in (("DENSE", chunk % 4), ("RESIDUAL", 1 + chunk % 2)):
        scale.append({"kind": "AGGREGATE", "chunk_id": chunk, "layer": "lm_head", "work_type": work,
                      "delta_w_sum": 2 * offset, "max_delta_w": offset, "alignment_count": 2,
                      "updated_partial_sum_count": 8 if offset else 0, "total_partial_sum_count": 8,
                      "zero_weight_count": 0})
stream(options["--scale-output"], "im2p-scale-alignment-aggregate",
       {"scale_domain": "hp1_block_pot_to_channel_anchor", "collection_mode": "aggregate"}, scale,
       {"invocation_count": count, "observation_count": len(scale), "alignment_count": 4 * count,
        "scale_invocation_count": count})
native = Path(options["--output-dir"])
native.mkdir()
elided = options["--terminal-lm-head"] == "metrics-only"
logits = {"logits_materialized": False, "lm_head_numerical_elisions": 1} if elided else {}
(native / "workload.json").write_text(json.dumps({
    "terminal_lm_head": options["--terminal-lm-head"], "lm_head_numerical_execution": not elided,
    "logits_materialized": not elided,
    "workload": "METRIC_PREFILL_256", "complete": True, "output_mask": "second_half", "seed": int(options["--seed"]),
    "context_tokens": 256, "tokens": total * 256, "complete_chunks": total, "selected_chunks": count,
    "first_chunk": first, "dropped_tail_tokens": 0, "batch": 256, "ubatch": 256, "threads": int(options["--threads"]),
    "threads_batch": int(options["--threads-batch"]), "add_special": True, "parse_special": False,
    "trailing_lf_removed": True, "add_bos": False, "bos_token": 1, "bos_policy": "replace_chunk_first",
    "kv_policy": "clear_per_chunk",
    "chunks": [{"chunk_id": chunk, "token_offset": chunk * 256, "input_tokens": [chunk], "complete": True, **logits}
               for chunk in chunks]}))
'''


def test_combined_collection_shards_merge_exactly_and_resume_finished_shards(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import metric_run
    from campaign_metrics import outputs

    from evaluation.manifest import Manifest
    runner = tmp_path / "bin/llama-eval-workload"
    runner.parent.mkdir()
    runner.write_text(f"#!{sys.executable}\n" + FAKE_RUNNER)
    runner.chmod(0o755)
    model, dataset = tmp_path / "model.gguf", tmp_path / "wiki.test.raw"
    model.write_bytes(b"model")
    dataset.write_bytes(b"dataset")
    manifest_path = tmp_path / "evaluation_manifest.json"
    write_json(manifest_path, {"model": "GPT-2 124M", "dataset": "WikiText-2", "tokenizer_sha256": "1" * 64,
                               "chunk_policy": metric_run.chunk_policy(0), "precision": "A8W8", "dim": 64, "BK": 32,
                               "seed": 1234, "git_sha": "2" * 40})
    manifest = Manifest.load(manifest_path)
    monkeypatch.setenv("FAKE_CHUNKS", "7")

    def collect(name: str, workers: int, reuse: Path | None = None,
                terminal: str = "full") -> tuple[Path, dict[str, object]]:
        root = tmp_path / name
        raws, binding = metric_run.collect_combined(runner, model, dataset, manifest_path, root / "collection", 0,
                                                    workers, 2, 2, 60, reuse, terminal)
        chunks = set(range(7))
        for kind in ("activation", "residual", "scu"):
            (root / kind).mkdir()
            outputs(kind, root / "collection/shards.json", manifest, root / kind, {"lm_head"}, chunks,
                    shards=raws[kind])
        return root, binding

    def results(root: Path) -> dict[str, object]:
        names = {"activation": "activation_metrics.json", "residual": "residual_metrics.json",
                 "scu": "scale_alignment_metrics.json"}
        return {kind: {key: value for key, value in read_json(root / kind / name).items() if key != "input_sha256"}
                for kind, name in names.items()}

    single, binding = collect("one", 1)
    assert binding["workers"] == 1 and binding["workload"]["selected_chunks"] == 7
    for workers in (3, 7, 9):  # 9 workers still make 7 one-chunk shards
        sharded, other = collect(f"w{workers}", workers)
        assert results(sharded) == results(single), workers
        assert other["native_workload_identity"] == binding["native_workload_identity"]
        assert len(read_json(sharded / "collection/shards.json")["shards"]) == min(workers, 7)
    # A failed shard fails the collection; a retry links the finished shards and runs only the failed one.
    monkeypatch.setenv("FAKE_FAIL", "0")
    with pytest.raises(EvaluationError, match="native combined collection failed"):
        collect("failed", 3)
    monkeypatch.setenv("FAKE_FAIL", "")
    resumed, binding = collect("resumed", 3, tmp_path / "failed/collection")
    assert sorted(binding["reused_shards"]) == ["shard-001", "shard-002"] and results(resumed) == results(single)
    assert (resumed / "collection/shard-001/activation-quant-metrics.jsonl").samefile(
        tmp_path / "failed/collection/shard-001/activation-quant-metrics.jsonl")
    # Another shard plan, or a damaged stream, is never reused.
    assert collect("replanned", 2, tmp_path / "failed/collection")[1]["reused_shards"] == {}
    damaged = tmp_path / "failed/collection/shard-002/residual-path-metrics.jsonl"
    damaged.write_text("\n".join(damaged.read_text().splitlines()[:-1]) + "\n")
    _, binding = collect("damaged", 3, tmp_path / "failed/collection")
    assert sorted(binding["reused_shards"]) == ["shard-001"]
    # The terminal lm_head mode is part of every shard command: a metrics-only collection gives the same metrics,
    # records its elided lm_head, and never reuses a shard of the full mode.
    elided, binding = collect("metrics-only", 3, tmp_path / "failed/collection", "metrics-only")
    execution, workload = binding["metric_execution"], binding["workload"]
    assert isinstance(execution, dict) and isinstance(workload, dict)
    assert results(elided) == results(single) and binding["reused_shards"] == {}
    assert execution["lm_head_numerical_gemm"] == "elided"
    assert all(row["logits_materialized"] is False for row in workload["chunks"])
    with pytest.raises(EvaluationError, match="invalid terminal lm_head mode"):
        collect("unknown", 3, None, "none")
