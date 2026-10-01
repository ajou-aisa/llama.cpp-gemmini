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

import metric_sweep
import metric_table
import run_measurement as wrapper
from campaign_inputs import checksums
from eval_common import EvaluationError, read_json, sha256, write_json

REDUCER = sha256(ROOT / "evaluation/weight_alignment/__init__.py")

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


class Fakes:
    """The build cache and the campaign.py processes the sweep starts, replaced by synthetic directories."""

    def __init__(self) -> None:
        self.builds: list[tuple[str, str, int]] = []
        self.calls: list[list[str]] = []
        self.fail: set[Key] = set()
        self.chunks: dict[Key, list[int]] = {}

    def build(self, cache: Path, kind: str, precision: str, dim: int, im2p: Path, jobs: int) -> tuple[Path, bool]:
        path = cache / f"{kind}-{precision}-d{dim}"
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
        write_run(output, kind, options, self.chunks.get(key, [0, 3]))
        return subprocess.CompletedProcess(command, 0, f"{output}\n", "")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    value = Fakes()
    monkeypatch.setattr(metric_sweep, "cached_llama_build", value.build)
    monkeypatch.setattr(metric_sweep.subprocess, "run", value.process)
    return value


def sweep(tmp_path: Path, *argv: str) -> int:
    return metric_sweep.main([*argv, "--im2p", str(tmp_path), "--build-cache", str(tmp_path / "cache")])


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
    assert summary["counts"] == {"configurations": 12, "metric_runs": 36, "builds": 18, "passed_configurations": 12}
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
