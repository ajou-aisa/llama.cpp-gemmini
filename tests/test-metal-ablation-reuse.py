from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/experiment/run-metal-ablation.py"


def archived_rtnw(tmp_path: Path, bits: int) -> Path:
    directory = tmp_path / f"gpt2-rtnw{bits}"
    directory.mkdir()
    model = f"gpt2.Q{bits}_0.gguf"
    manifest = {Path(name).name: digest for digest, name in
                (line.split() for line in (ROOT / "scripts/experiment/default-ppl-models.sha256").read_text().splitlines())}
    corpus = hashlib.sha256((ROOT / "wikitext-2-raw/wiki.test.raw").read_bytes()).hexdigest()
    (directory / "inputs-binaries.sha256").write_text(f"{manifest[model]} /models/{model}\n{corpus} /data/wiki.test.raw\n")
    (directory / "result.tsv").write_text(f"RTN-W\tgpt2\t{bits}\t16\t559\t142545\t31.0\t100\t0\n")
    (directory / "command.txt").write_text("runner --chunks -1 --ctx-size 512 --batch-size 512 --ubatch-size 512 --cache-type-k f16 --cache-type-v f16 --seed 42")
    (directory / "exit-status.txt").write_text("0\n")
    proof = dict(schema="metal-cpu-exact-ppl", graph="original_cpu", version=1, complete=True,
                 compute="FLOAT", activation="BLOCK", bits=bits, dim=16, scored_tokens=142545,
                 observed_matmuls=49*559, verified_matmuls=49*559, float_gpu_calls=49*559,
                 integer_gpu_launches=0, cpu_q6_head_matmuls=0, gpu_q6_head_matmuls=559 if bits == 4 else 0,
                 activation_fp16=True)
    (directory / "metal-cpu-exact-proof.json").write_text(json.dumps(proof))
    return directory


@pytest.mark.parametrize("mismatch", [None, "partial", "fp32", "model", "cpu_head"])
def test_explicit_cpu_attention_reuse_keeps_quality_checks(tmp_path: Path, mismatch: str | None) -> None:
    directory = archived_rtnw(tmp_path, 4)
    module = runpy.run_path(str(RUNNER))
    case = next(case for case in module["cases"]() if case.name == directory.name)
    proof_path = directory / "metal-cpu-exact-proof.json"
    proof = json.loads(proof_path.read_text())
    if mismatch == "partial":
        proof["scored_tokens"] = 255
    if mismatch == "fp32":
        proof["activation_fp16"] = False
    if mismatch == "model":
        (directory / "inputs-binaries.sha256").write_text("")
    if mismatch == "cpu_head":
        proof["cpu_q6_head_matmuls"] = 559
    proof_path.write_text(json.dumps(proof))
    assert module["result_row"](directory, case) is None
    row = module["result_row"](directory, case, allow_cpu_attention=True)
    assert (row is not None) == (mismatch is None)


def test_waiting_queue_excludes_archived_rtnw_without_touching_predecessor(tmp_path: Path) -> None:
    sources = [archived_rtnw(tmp_path, bits) for bits in (4, 8)]
    before = tmp_path / "before"
    before.mkdir()
    names = [f"llama-rtnw{bits}" for bits in (4, 8)] + [
        f"llama-potal{bits}-d{dim}" for bits in (4, 8) for dim in (16, 32, 64)]
    (before / "queue.txt").write_text("\n".join(names) + "\n")
    (before / "runner.pid").write_text(str(os.getpid()))
    status = "case\tstate\nllama-potal8-d16\trunning\n"
    (before / "status.tsv").write_text(status)
    output = tmp_path / "after"
    output.mkdir()
    (output / "reuse.tsv").write_text("case\tsource\n" + "".join(f"{p.name}\t{p}\n" for p in sources))
    process = subprocess.Popen([sys.executable, str(RUNNER), "--after", str(before), "--output", str(output)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (output / "state.txt").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.poll() is None
        queued = set((output / "queue.txt").read_text().splitlines())
        assert len(queued) == 22
        assert not queued.intersection(names + [p.name for p in sources])
        assert len((output / "excluded-cases.txt").read_text().splitlines()) == 10
        assert (output / "state.txt").read_text() == "waiting\n"
        assert (before / "status.tsv").read_text() == status
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_archived_results_reach_table_without_rerunning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [archived_rtnw(tmp_path, bits) for bits in (4, 8)]
    (tmp_path / "reuse.tsv").write_text("case\tsource\n" + "".join(f"{p.name}\t{p}\n" for p in sources))
    module = runpy.run_path(str(RUNNER))
    matrix = tuple(case for case in module["cases"]() if case.name in {p.name for p in sources})
    run = module["main"]
    run.__globals__["cases"] = lambda: matrix
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--output", str(tmp_path)])
    assert run() == 0
    assert (tmp_path / "queue.txt").read_text() == ""
    assert len((tmp_path / "results.tsv").read_text().splitlines()) == 3
    status = (tmp_path / "status.tsv").read_text()
    for source in sources:
        assert f"{source.name}\treused\t{source}" in status
        assert json.loads((source / "metal-cpu-exact-proof.json").read_text())["graph"] == "original_cpu"
    assert not list(tmp_path.glob("*-build.log"))
