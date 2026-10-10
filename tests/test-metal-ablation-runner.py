from pathlib import Path
import os
import hashlib
import json
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/experiment/run-metal-ablation.py"


def test_dry_run_has_all_paired_models() -> None:
    process = subprocess.run([sys.executable, str(RUNNER), "--dry-run"], text=True, capture_output=True, check=True)
    lines = [line.split("\t") for line in process.stdout.splitlines() if "\t" in line]
    models = {line[0]: line[1] for line in lines}
    assert len(models) == 32
    for family in ("gpt2", "llama"):
        for bits in (4, 8):
            assert models[f"{family}-rtnw{bits}"] == models[f"{family}-rtnwa1-n{bits}"]
            for dim in (16, 32, 64):
                assert models[f"{family}-rtnwa2-n{bits}-d{dim}"] == models[f"{family}-potal{bits}-d{dim}"]
    assert sum(line[2] == "selection=1" and line[3] == "RC=1" for line in lines) == 12


def test_predecessor_requires_terminal_case_records(tmp_path: Path) -> None:
    predecessor = runpy.run_path(str(RUNNER))["predecessor_finished"]
    (tmp_path / "queue.txt").write_text("first\nsecond\n")
    (tmp_path / "runner.pid").write_text(str(os.getpid()))
    (tmp_path / "status.tsv").write_text("case\tstate\texit\nfirst\tcomplete\t0\nsecond\trunning\t\n")
    assert predecessor(tmp_path) is False
    with (tmp_path / "status.tsv").open("a") as stream:
        stream.write("second\tcomplete\t0\n")
    assert predecessor(tmp_path) is True


def test_interruption_is_not_completion(tmp_path: Path) -> None:
    predecessor = runpy.run_path(str(RUNNER))["predecessor_finished"]
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    (tmp_path / "queue.txt").write_text("first\n")
    (tmp_path / "runner.pid").write_text(str(child.pid))
    (tmp_path / "status.tsv").write_text("case\tstate\texit\nfirst\trunning\t\n")
    with pytest.raises(SystemExit, match="interrupted"):
        predecessor(tmp_path)


@pytest.mark.parametrize("mismatch", [None, "partial", "model", "cpu_head", "missing_gpu", "selection", "short_command"])
def test_result_reuse_requires_full_matching_gpu_run(tmp_path: Path, mismatch: str | None) -> None:
    module = runpy.run_path(str(RUNNER))
    reuse = module["result_row"]
    reuse.__globals__["ROOT"] = tmp_path
    case = next(case for case in module["cases"]() if case.name == "gpt2-rtnwa1-n4")
    model_dir = tmp_path / "scripts/experiment"
    model_dir.mkdir(parents=True)
    (model_dir / "default-ppl-models.sha256").write_text(f"{'a' * 64} models/default/{case.model}\n")
    corpus = tmp_path / "wikitext-2-raw"
    corpus.mkdir()
    (corpus / "wiki.test.raw").write_bytes(b"fixture corpus")
    result = tmp_path / "result"
    result.mkdir()
    (result / "result.tsv").write_text("RTN-WA1\tgpt2\t4\t16\t559\t142545\t50.0\t100\t0\n")
    (result / "exit-status.txt").write_text("0\n")
    (result / "command.txt").write_text("runner --chunks -1 --ctx-size 512 --batch-size 512 --ubatch-size 512 --cache-type-k f16 --cache-type-v f16 --seed 42")
    digest = hashlib.sha256(b"fixture corpus").hexdigest()
    (result / "inputs-binaries.sha256").write_text(f"{'a' * 64} /models/{case.model}\n{digest} /data/wiki.test.raw\n")
    proof = dict(schema="metal-cpu-exact-ppl", graph="cpu_equivalent_metal", version=3, complete=True,
                 bits=4, dim=16, compute="INT", activation="BLOCK", scored_tokens=142545,
                 observed_matmuls=49*559, verified_matmuls=49*559, cpu_q6_head_matmuls=0,
                 gpu_q6_head_matmuls=559, float_gpu_calls=0, integer_gpu_launches=559,
                 attention_gpu_calls=24*559, observed_attention_matmuls=24*559, verified_attention_matmuls=24*559,
                 residual_gpu_launches=0, activation_fp16=False, outlier_selection=False,
                 residual_enabled=False, q6_head_activation_bits=4)
    if mismatch == "partial":
        proof["scored_tokens"] = 255
    elif mismatch == "model":
        (result / "inputs-binaries.sha256").write_text(f"{'b' * 64} /models/{case.model}\n{digest} /data/wiki.test.raw\n")
    elif mismatch == "cpu_head":
        proof["cpu_q6_head_matmuls"] = 559
    elif mismatch == "missing_gpu":
        proof["integer_gpu_launches"] = 0
    elif mismatch == "selection":
        proof["outlier_selection"] = True
    elif mismatch == "short_command":
        (result / "command.txt").write_text("runner --chunks 1")
    (result / "metal-cpu-exact-proof.json").write_text(json.dumps(proof))
    assert (reuse(result, case) is not None) == (mismatch is None)


@pytest.mark.parametrize("chunks,tokens,full", [(1, 255, False), (559, 142545, True)])
def test_diagnostic_results_are_never_labelled_full(tmp_path: Path, chunks: int, tokens: int, full: bool) -> None:
    summarize = runpy.run_path(str(ROOT / "scripts/experiment/summarize-metal-quality-ppl.py"))["summarize"]
    (tmp_path / "results.tsv").write_text("method\tmodel\tbits\tdim\tchunks\tscored_tokens\tppl\tprocess_seconds\texit\n"
                                        f"RTN-W\tgpt2\t4\t16\t{chunks}\t{tokens}\t50\t100\t0\n")
    summarize(tmp_path)
    text = (tmp_path / "quality_ppl.md").read_text()
    assert ("Validated full-corpus" in text) == full
    assert ("27.19" in text) == full


def test_queue_continues_after_failed_build_and_runs_full_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = runpy.run_path(str(RUNNER))
    run = module["main"]
    matrix = tuple(case for case in module["cases"]() if case.name in ("gpt2-rtnw4", "gpt2-rtnw8"))
    calls: list[list[str]] = []
    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1 if "gpt2-rtnw4" in command else 0)
    def row(directory: Path, case: object) -> str | None:
        if directory.name == "gpt2-rtnw8":
            return "RTN-W\tgpt2\t8\t16\t559\t142545\t27.0\t100\t0\n"
        return None
    run.__globals__.update(ROOT=ROOT, cases=lambda: matrix, result_row=row,
                           subprocess=SimpleNamespace(run=execute, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--output", str(tmp_path)])
    assert run() == 1
    assert (tmp_path / "state.txt").read_text() == "failed\n"
    status = (tmp_path / "status.tsv").read_text()
    assert "gpt2-rtnw4\tfailed" in status and "gpt2-rtnw8\tcomplete" in status
    actual = next(command for command in calls if "--chunks" in command)
    assert actual[actual.index("--chunks")+1] == "-1"


def test_waiting_cli_keeps_queue_idle_and_rejects_duplicate(tmp_path: Path) -> None:
    predecessor = tmp_path / "before"
    predecessor.mkdir()
    (predecessor / "queue.txt").write_text("first\n")
    (predecessor / "runner.pid").write_text(str(os.getpid()))
    (predecessor / "status.tsv").write_text("case\tstate\nfirst\trunning\n")
    output = tmp_path / "after"
    command = [sys.executable, str(RUNNER), "--after", str(predecessor), "--output", str(output)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (output / "state.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.poll() is None
        assert (output / "state.txt").read_text() == "waiting\n"
        assert len((output / "queue.txt").read_text().splitlines()) == 32
        assert not (output / "status.tsv").exists()
        duplicate = subprocess.run(command, capture_output=True, text=True, check=False)
        assert duplicate.returncode != 0 and "Queue already active" in duplicate.stderr
        assert int((output / "runner.pid").read_text()) == process.pid
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_followup_dry_run_excludes_all_predecessor_cases(tmp_path: Path) -> None:
    excluded = {f"llama-rtnw{bits}" for bits in (4, 8)} | {
        f"llama-potal{bits}-d{dim}" for bits in (4, 8) for dim in (16, 32, 64)}
    (tmp_path / "queue.txt").write_text("\n".join(sorted(excluded)) + "\n")
    result = subprocess.run([sys.executable, str(RUNNER), "--dry-run", "--after", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    queued = {line.split("\t")[0] for line in result.stdout.splitlines() if "\t" in line}
    assert len(queued) == 24 and not queued.intersection(excluded)


@pytest.mark.parametrize("available", [True, False])
def test_excluded_cases_are_never_rerun_even_if_their_results_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    module = runpy.run_path(str(RUNNER))
    run = module["main"]
    matrix = tuple(case for case in module["cases"]() if case.name in ("gpt2-rtnw4", "gpt2-rtnw8"))
    before = tmp_path / "before"
    before.mkdir()
    (before / "queue.txt").write_text("gpt2-rtnw4\n")
    terminal = "complete" if available else "failed"
    (before / "status.tsv").write_text(f"case\tstate\ngpt2-rtnw4\t{terminal}\n")
    output = tmp_path / "after"
    calls: list[list[str]] = []
    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)
    def row(directory: Path, case: object) -> str | None:
        if directory.name == "gpt2-rtnw4" and available:
            return "RTN-W\tgpt2\t4\t16\t559\t142545\t48.0\t100\t0\n"
        if directory.name == "gpt2-rtnw8" and directory.parent.name.startswith("attempt-"):
            return "RTN-W\tgpt2\t8\t16\t559\t142545\t27.0\t100\t0\n"
        return None
    run.__globals__.update(cases=lambda: matrix, result_row=row,
                           subprocess=SimpleNamespace(run=execute, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--after", str(before), "--output", str(output)])
    assert run() == int(not available)
    assert (output / "queue.txt").read_text() == "gpt2-rtnw8\n"
    assert (output / "excluded-cases.txt").read_text() == "gpt2-rtnw4\n"
    assert all("gpt2-rtnw4" not in command for command in calls)
    assert sum("--chunks" in command for command in calls) == 1
    status = (output / "status.tsv").read_text()
    expected = "reused" if available else "excluded-unavailable"
    assert f"gpt2-rtnw4\t{expected}" in status and "gpt2-rtnw8\tcomplete" in status
