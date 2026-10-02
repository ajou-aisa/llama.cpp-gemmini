# /// script
# requires-python = ">=3.11"
# dependencies = ["pytest"]
# ///
# How to run: python3 -B -m pytest -q tests/test-measurement-domains.py
"""Measurement interface: one canonical entry point, exactly three domains (performance, timeline, metric), isolated
builds and outputs."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/eval"
sys.path.insert(0, str(EVAL))

import measurement_domains as domains
import run_measurement as wrapper
from campaign_build import METRIC_SINKS, llama_plan
from eval_common import EvaluationError, sha256, write_json

STAMP = "20260101T000000Z"
SINKS = ("GGML_GEMMINI_ACT_METRICS", "GGML_GEMMINI_RESIDUAL_METRICS", "GGML_GEMMINI_SCALE_METRICS")


def isolated(code: str, *argv: str) -> None:
    """Run a check in its own interpreter: the metric engine imports llama's `evaluation` package, whose
    `scripts.eval.*` imports cannot share a process with IM2P's own top-level `scripts` package."""
    done = subprocess.run([sys.executable, "-B", "-c", f"import sys; sys.path.insert(0, {str(EVAL)!r})\n" + code, *argv],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


# Canonical entry point

@pytest.mark.parametrize("argv, script, arguments", [
    (["performance", "--model", "models/gpt2/gpt2.Q8_HP1.gguf", "--precision", "a8w8", "--dim", "32"],
     "run_cycle_evaluation.py",
     ["--run", "performance", "--model", "models/gpt2/gpt2.Q8_HP1.gguf", "--precision", "a8w8", "--dim", "32",
      "--output", f"runs/performance/{STAMP}-gpt2.Q8_HP1-a8w8-d32-hp1"]),
    (["performance", "--smoke", "--model=m.gguf", "--timeline", "compact"], "run_cycle_evaluation.py",
     ["--run", "performance", "--smoke", "--model=m.gguf", "--timeline", "compact",
      "--output", f"runs/performance/{STAMP}-m-a8w8-d32-hp1-smoke"]),
    (["timeline", "--from-run", "runs/performance/x"], "run_cycle_evaluation.py",
     ["--run", "timeline", "--from-run", "runs/performance/x", "--output", f"runs/timeline/{STAMP}-x"]),
    (["metric", "activation", "--model", "gpt2", "--precision", "a4w4", "--dim", "16"], "campaign.py",
     ["activation", "--model", "gpt2", "--precision", "a4w4", "--dim", "16",
      "--output", f"runs/metrics/activation/{STAMP}-gpt2-a4w4-d16"]),
    (["metric", "residual", "--model", "gpt2", "--output", "out"], "campaign.py",
     ["residual", "--model", "gpt2", "--output", "out"]),
    (["metric", "scu", "--model", "llama3.2-1B", "--output=out"], "campaign.py",
     ["scu", "--model", "llama3.2-1B", "--output=out"]),
])
def test_each_domain_dispatches_to_its_one_delegate(argv: list[str], script: str, arguments: list[str]) -> None:
    assert wrapper.delegate(argv, STAMP) == (EVAL / script, arguments)
    assert (EVAL / script).is_file()


@pytest.mark.parametrize("argv, reason", [
    ([], "choose a domain"), (["perf"], "choose a domain"), (["quality", "ppl"], "choose a domain"),
    (["metric"], "metric requires"), (["metric", "cycle"], "metric requires"), (["metric", "ppl"], "metric requires"),
    (["performance", "--run", "timeline"], "selects the mode"),
    (["timeline", "--run=performance", "--from-run", "x"], "selects the mode"),
    (["timeline"], "requires --from-run"),
])
def test_wrapper_rejects_mixed_or_unknown_domains(argv: list[str], reason: str) -> None:
    with pytest.raises(wrapper.UsageError, match=reason):
        wrapper.delegate(argv, STAMP)


def test_help_and_dry_run_get_no_default_output() -> None:
    for extra in (["--help"], ["-h"], ["--dry-run"]):
        _, arguments = wrapper.delegate(["performance", *extra], STAMP)
        assert "--output" not in arguments


def test_wrapper_holds_no_measurement_logic() -> None:
    # The wrapper may import the domain definitions only; the engines stay separate processes.
    tree = ast.parse((EVAL / "run_measurement.py").read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"__future__", "json", "os", "sys", "datetime", "pathlib", "measurement_domains"}


def test_main_reports_usage_errors_without_running_anything(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "execv", lambda *_: (_ for _ in ()).throw(AssertionError("must not exec")))
    assert wrapper.main(["quality", "ppl"]) == 2
    assert "choose a domain: performance, timeline, metric" in capsys.readouterr().err
    assert wrapper.main(["describe", "--json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["schema"] == "potal-measurement-domains" and set(described["measurements"]) == set(domains.MEASUREMENTS)
    assert described["domains"] == ["performance", "timeline", "metric"]
    assert described["metric_kinds"] == ["activation", "residual", "scu"]


def engine_modules(root: str) -> tuple[set[str], set[str]]:
    """Static import closure of a scripts/eval module: (scripts/eval modules, top-level project packages)."""
    local = {path.stem for path in EVAL.glob("*.py")}
    seen: set[str] = set()
    packages: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((EVAL / f"{name}.py").read_text())
        names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module} | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        pending.extend(item for item in names if item in local)
        packages.update(item.split(".")[0] for item in names if item.split(".")[0] in ("evaluation", "sim"))
    return seen - {root}, packages


def test_metric_and_performance_engines_share_only_build_and_identity_modules() -> None:
    performance, performance_packages = engine_modules("run_cycle_evaluation")
    metrics, metric_packages = engine_modules("campaign")
    # Given the two engines, the only shared code is the build authority, the model identity and the helpers.
    assert performance & metrics == {"campaign_build", "eval_common", "model_manifest"}
    # The metric engine has no replay, scheduler, timeline or stage cache, and never imports the IM2P cycle package.
    assert not metrics & {"run_cycle_evaluation", "schedule_engine", "e2e_timeline", "stage_cache", "offline_pipeline",
                          "end_to_end", "target_admission"} and metric_packages == {"evaluation"}
    # The performance/timeline engine has no metric collector or reducer.
    assert not performance & {"campaign", "metric_run", "campaign_metrics", "campaign_stream", "metric_reducers"}
    assert performance_packages == {"sim"}


# Terminology and dependency matrix

def test_the_interface_has_exactly_three_domains() -> None:
    assert domains.DOMAINS == ("performance", "timeline", "metric")
    assert domains.METRIC_KINDS == ("activation", "residual", "scu")
    by_domain: dict[str, set[str]] = {}
    for name, row in domains.MEASUREMENTS.items():
        by_domain.setdefault(row.domain, set()).add(name)
    assert by_domain == {"performance": {"performance"}, "timeline": {"timeline"},
                         "metric": {"activation", "residual", "scu"}}
    assert {name: title for name, (title, _) in domains.TERMINOLOGY.items()} == {
        "performance": "Performance", "timeline": "Timeline", "metric": "Metrics"}
    # Every command of the wrapper is one of the three domains and every delegate mode is an interface mode.
    assert {row.command[0] for row in domains.MEASUREMENTS.values()} == set(domains.DOMAINS)
    assert {row.delegate for row in domains.MEASUREMENTS.values()} == {
        ("run_cycle_evaluation.py", "--run", "performance"), ("run_cycle_evaluation.py", "--run", "timeline"),
        ("campaign.py", "activation"), ("campaign.py", "residual"), ("campaign.py", "scu")}
    # Nothing else is defined here; perplexity is named once, as being outside the interface.
    described = domains.description()
    outside = described.pop("outside_interface")
    assert outside == ("Perplexity/model-quality evaluation is intentionally outside this measurement interface and "
                       "is run with llama-perplexity.")
    assert not re.search(r"\b(quality|ppl|perplexity)\b", json.dumps(described).lower())


def test_dependency_matrix_separates_the_domains() -> None:
    matrix = domains.dependency_matrix()
    assert list(matrix) == ["performance", "timeline", "activation", "residual", "scu"]
    assert all(list(row) == [key for key, _ in domains.MATRIX_COLUMNS] and len(row) == 9 for row in matrix.values())
    assert matrix["performance"]["scheduler"] == "yes" and matrix["performance"]["metric_instrumentation"] == "no"
    # Timeline export and metrics need neither the cycle model nor PoTal/FullCPU/PMU.
    for name in ("timeline", "activation", "residual", "scu"):
        assert [matrix[name][key] for key in ("cycle_model", "potal", "fullcpu", "pmu")] == ["no"] * 4
    assert all(matrix[name]["metric_instrumentation"] == "yes" and matrix[name]["scheduler"] == "no" and
               matrix[name]["timeline"] == "no" for name in domains.METRIC_KINDS)
    assert matrix["timeline"]["timeline"] == "yes" and matrix["timeline"]["wikitext"] == "no"


# Build isolation (campaign_build is the only option source)

def test_each_metric_build_enables_only_its_own_sink() -> None:
    builds = {row["measurement"] + ":" + str(row["build_kind"]): row for row in domains.build_mapping()}
    expected = {"activation:activation": ("1", "0", "0"), "residual:residual": ("0", "1", "0"), "scu:scu": ("0", "0", "1"),
                "performance:potal-host": ("0", "0", "0"), "performance:fullcpu-host": ("0", "0", "0")}
    for name, flags in expected.items():
        shown = builds[name]["flags"]
        assert isinstance(shown, dict) and tuple(shown[key] for key in SINKS) == flags, name
    assert builds["timeline:None"]["build_kind"] is None
    assert {row["binary"] for row in builds.values() if row["build_kind"] not in (None, "cycle-model")} == {
        "llama-eval-workload"}
    # The mapping is read from llama_plan, not restated.
    plan = llama_plan("scu", "a8w8", 32, ROOT.parent / "IM2P.sim")
    assert builds["scu:scu"]["semantic_options_sha256"] == plan.semantic_options_sha256
    # Instrumentation builds are distinct build identities, also from the performance build.
    digests = [builds[name]["semantic_options_sha256"] for name in
               ("activation:activation", "residual:residual", "scu:scu", "performance:potal-host")]
    assert len(set(digests)) == 4


def test_campaign_build_is_the_only_build_option_source() -> None:
    # Given the sink table of the build authority, every llama build kind sets exactly these three options from it.
    assert {kind: option for kind, (option, _) in METRIC_SINKS.items()} == dict(zip(domains.METRIC_KINDS, SINKS))
    for kind in ("cycle", "activation", "residual", "scu", "potal-host", "potal-host-nocpulog", "fullcpu-host"):
        options = llama_plan(kind, "a8w8", 32, ROOT.parent / "IM2P.sim").options
        assert {name: options[name] for name in SINKS} == {
            option: str(int(kind == metric)) for metric, (option, _) in METRIC_SINKS.items()}, kind
    # No other script of scripts/eval runs CMake or names a sink option.
    for path in sorted(EVAL.glob("*.py")) + sorted(EVAL.glob("*.sh")):
        if path.name in ("campaign_build.py", "cycle_trace_capture.py"):
            continue  # the authority; the certification capture only reads the options back from a receipt
        text = path.read_text()
        assert not re.search(r"""["']cmake["']""", text) and not any(name in text for name in SINKS), path.name
    assert not re.search(r"""["']cmake["']""", (EVAL / "cycle_trace_capture.py").read_text())
    # The audit table names real files and the authority first.
    audited = [name for row in domains.BUILD_SCRIPTS for name in row[0].split(", ")]
    assert audited[0] == "scripts/eval/campaign_build.py" and all((ROOT / name).is_file() for name in audited)
    assert all(len(row) == len(domains.BUILD_SCRIPT_FIELDS) for row in domains.BUILD_SCRIPTS)


def test_a_metric_run_rejects_a_build_with_another_sink() -> None:
    isolated('''
from eval_common import EvaluationError
from metric_run import validate_metric_recipe

def compiled(**changes):
    info = {"activation_metrics": 0, "residual_metrics": 0, "scale_metrics": 0, "cycle_sim": 1, "backend": "IM2P_SIM",
            "hp1": True, "gemmini": 1, "gemmini_option": "WS", "activation_bits": 8, "weight_bits": 8, "dim": 32,
            "activation_mode": "EXSIA", "block_size": 32, "rmd_enabled": 1, "rmd_backend": "WS"}
    info.update(changes)
    return info

def rejected(info, recipe):
    try:
        validate_metric_recipe(info, recipe)
    except EvaluationError:
        return True
    return False

flags = {"activation": "activation_metrics", "residual": "residual_metrics", "scale": "scale_metrics"}
for recipe, flag in flags.items():
    validate_metric_recipe(compiled(**{flag: 1}), recipe)                 # its own single sink
    assert rejected(compiled(), recipe)                                   # a performance build (no sink)
    for other in flags.values():
        if other != flag:
            assert rejected(compiled(**{flag: 1, other: 1}), recipe)      # a second sink is enabled
            assert rejected(compiled(**{other: 1}), recipe)               # another metric's build
''')


# Output contract and shared identity

def test_metric_runs_share_canonical_output_names(tmp_path: Path) -> None:
    isolated('''
from pathlib import Path
import campaign
root = Path(sys.argv[1])
for kind, raw_name in (("activation", "activation-quant-metrics.jsonl"), ("residual", "residual-path-metrics.jsonl"),
                       ("scu", "scale-alignment-metrics.jsonl.gz")):
    output = root / kind
    (output / "collection").mkdir(parents=True)
    summary, layers = campaign.OUTPUT_FILENAMES[kind]
    for path in (output / summary, output / layers, output / "collection/request.json", output / "collection" / raw_name):
        path.write_text(path.name)
    campaign.canonical_aliases(output, kind, output / "collection" / raw_name)
    raw_alias = "raw.jsonl.gz" if raw_name.endswith(".gz") else "raw.jsonl"
    for alias, target in (("summary.json", summary), ("layers.json", layers),
                          ("request.json", "collection/request.json"), (raw_alias, "collection/" + raw_name)):
        assert (output / alias).samefile(output / target), (kind, alias)
''', str(tmp_path))
    for kind in domains.METRIC_KINDS:
        declared = domains.MEASUREMENTS[kind].outputs
        assert any(item.endswith("(= summary.json)") for item in declared)
        assert any(item.endswith("(= layers.json)") for item in declared)
        assert "request.json" in declared and ("raw.jsonl.gz" if kind == "scu" else "raw.jsonl") in declared


def test_one_model_manifest_identity_is_shared_while_builds_stay_separate(tmp_path: Path) -> None:
    import run_cycle_evaluation as runner
    from model_manifest import SCHEMA, model_entry
    model, manifest = tmp_path / "gpt2.Q8_HP1.gguf", tmp_path / "model-manifest.json"
    model.write_bytes(b"frozen model bytes")
    write_json(manifest, {"schema": SCHEMA, "version": 1, "f16": {"sha256": "0" * 64},
                          "quantized": {"Q8_HP1": {"sha256": sha256(model), "path": str(model), "gguf": {"quantization": "Q8_HP1"}}},
                          "upstream": {"repository": "openai-community/gpt2", "revision": "r"}, "tokenizer": {"sha256": "t"}})
    # The performance runner resolves the model through model_manifest.model_entry ...
    assert runner.model_entry is model_entry
    entry = model_entry(manifest, model)
    assert entry["artifact"] == "Q8_HP1" and entry["sha256"] == sha256(model)
    # ... and so does the metric campaign, which accepts the same manifest.
    isolated('''
import json
from pathlib import Path
import campaign
import model_manifest
assert campaign.model_entry is model_manifest.model_entry
manifest, model = Path(sys.argv[1]), Path(sys.argv[2])
args = campaign.parser().parse_args(["activation", "--model", "gpt2", "--precision", "a8w8", "--dim", "32",
                                     "--model-manifest", str(manifest)])
assert args.model_manifest == manifest
assert campaign.model_entry(args.model_manifest, model) == json.loads(sys.argv[3])
''', str(manifest), str(model), json.dumps(entry))
    # A model outside the manifest is rejected in every domain.
    other = tmp_path / "other.gguf"
    other.write_bytes(b"other")
    with pytest.raises(EvaluationError, match="not a frozen artifact"):
        model_entry(manifest, other)


def identity_runs(root: Path, dim: int = 32, manifest: bool = True) -> list[Path]:
    """One finished run directory per domain: performance, timeline export, and two metric kinds."""
    entry = {"artifact": "Q8_HP1", "manifest": {"sha256": "manifest"},
             "gguf": {"general.architecture": "gpt2", "quantization": "MOSTLY_Q8_HP1"}} if manifest else None
    sinks = {"activation_metrics": 0, "residual_metrics": 0, "scale_metrics": 0}
    performance, export = root / "performance", root / "timeline"
    for path in (performance / "build", export):
        path.mkdir(parents=True)
    write_json(performance / "performance.json", {"workload": {
        "profile": f"a8w8-d{dim}-hp1", "dataset": {"sha256": "data"},
        "model": {"sha256": "model", "architecture": "gpt2", "quantization": "MOSTLY_Q8_HP1", "manifest": entry}}})
    write_json(performance / "build/cycle-model.json", {"receipt": {"library": {"sha256": "library"}}})
    write_json(performance / "build/potal.json", {
        "receipt": {"kind": "potal-host", "semantic_options_sha256": "potal", "runner_sha256": "potal-binary"},
        "build_info": {"block_size": 32, **sinks}})
    write_json(export / "export.json", {"source_run": str(performance)})
    runs = [performance, export]
    for kind, key in (("activation", "activation_metrics"), ("scu", "scale_metrics")):
        metric = root / kind
        (metric / "build").mkdir(parents=True)
        write_json(metric / "manifest.json", {
            "kind": kind, "model_sha256": "model", "dataset_sha256": "data", "model_manifest": entry,
            "build_hash": f"{kind}-binary", "build_receipt_sha256": f"{kind}-receipt",
            "runner": str(metric / "build/bin/llama-eval-workload"),
            "build_info": {"activation_bits": 8, "weight_bits": 8, "dim": dim, "block_size": 32, **sinks, key: 1}})
        write_json(metric / "build/build-receipt.json", {"kind": kind, "semantic_options_sha256": kind,
                                                         "runner_sha256": f"{kind}-binary"})
        runs.append(metric)
    return runs


def test_domains_share_one_identity_and_record_their_own_builds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import measurement_identity as identity
    runs = identity_runs(tmp_path / "same")
    rows = [identity.run_identity(run) for run in runs]
    # Given one model and configuration measured in the three domains, the shared identity is the same record.
    assert [(row["domain"], row["measurement"]) for row in rows] == [
        ("performance", "performance"), ("timeline", "timeline"), ("metric", "activation"), ("metric", "scu")]
    assert {row["domain"] for row in rows} == set(domains.DOMAINS)
    assert all(row["shared"] == rows[0]["shared"] for row in rows) and tuple(sorted(rows[0]["shared"])) == tuple(sorted(identity.SHARED))
    assert identity.compare(rows) == {"same_model_and_configuration": True, "differing": {}, "unrecorded": {}}
    # And: every run records its own build identity; none is required to equal another.
    builds = [rows[0]["builds"]["potal"], rows[2]["builds"]["activation"], rows[3]["builds"]["scu"]]
    assert len({build["semantic_options_sha256"] for build in builds}) == 3
    assert len({build["binary_sha256"] for build in builds}) == 3 and rows[1]["builds"] == {}
    assert [build["metric_sinks"] for build in builds] == [
        {"activation": 0, "residual": 0, "scu": 0}, {"activation": 1, "residual": 0, "scu": 0},
        {"activation": 0, "residual": 0, "scu": 1}]
    assert identity.main([str(run) for run in runs]) == 0
    assert json.loads(capsys.readouterr().out)["comparison"]["same_model_and_configuration"] is True
    # When a run of another configuration is compared, the differing field is named and the exit status is 1.
    other = identity_runs(tmp_path / "other", dim=16)[2]
    assert identity.main([str(runs[0]), str(other)]) == 1
    assert list(json.loads(capsys.readouterr().out)["comparison"]["differing"]) == ["dim"]
    # A run without a model manifest leaves the manifest fields unrecorded; that is reported, not a mismatch.
    bare = identity_runs(tmp_path / "bare", manifest=False)[2]
    comparison = identity.compare([rows[0], identity.run_identity(bare)])
    assert comparison["same_model_and_configuration"] is True
    assert set(comparison["unrecorded"]) == {"model_manifest_sha256", "model_artifact", "architecture", "quantization"}
    # Anything that is not a run of the three domains is refused.
    assert identity.main([str(tmp_path)]) == 2
    assert "not a finished performance, timeline or metric run" in capsys.readouterr().err


def test_identity_helper_is_dispatched_without_measuring(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(os, "execv", lambda _, argv: calls.append(argv))
    wrapper.main(["identity", "runs/a", "runs/b"])
    assert calls == [[sys.executable, "-B", str(EVAL / "measurement_identity.py"), "runs/a", "runs/b"]]
    # The reader imports neither engine.
    assert engine_modules("measurement_identity") == ({"campaign_build", "eval_common"}, set())


# Documentation stays generated from the definitions

def test_readme_tables_are_the_generated_ones() -> None:
    text = (EVAL / "README.md").read_text()
    for section in domains.SECTIONS:
        begin, end = f"<!-- BEGIN GENERATED {section} -->\n", f"<!-- END GENERATED {section} -->"
        assert text[text.index(begin) + len(begin):text.index(end)] == domains.markdown(section), section
    for row in domains.MEASUREMENTS.values():
        assert row.example in text


def test_every_script_in_scripts_eval_is_classified() -> None:
    files = {path.name for path in EVAL.iterdir() if path.suffix in (".py", ".sh", ".c")}
    classified = [script for script, _, _, _ in domains.SCRIPTS]
    assert len(classified) == len(set(classified)) and set(classified) == files
    assert {status for _, status, _, _ in (*domains.SCRIPTS, *domains.OUTSIDE)} <= set(domains.STATUSES)
    canonical = {script for script, status, _, _ in domains.SCRIPTS if status == "CANONICAL"}
    assert canonical == {"run_measurement.py", "run_cycle_evaluation.py", "campaign.py"}
