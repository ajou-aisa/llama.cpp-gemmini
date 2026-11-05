#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars", "pydantic", "matplotlib"]
# ///
# How to run: .\scripts\experiment\start-ties-away.ps1; monitor with watch-ties-away.ps1
"""Remeasure native RTN baselines and ties-away MXINT/PoTal with separate proofs."""
import fcntl
from functools import lru_cache
from importlib import import_module
import json
from pathlib import Path
import re
import sys
import time
from typing import Final, TypedDict

runtime = import_module("sensitivity-runtime")
queue = runtime.queue
ROOT: Final = runtime.ROOT
OUTPUT: Final = Path("/home/mojoi/potal-results/ties-away-full-24")
CONFIG: Final = Path(__file__).with_name("ties-away-ppl-jobs.json")
original_inputs, original_settings, original_verify = queue.inputs, queue.settings, queue.verify
original_run_case = queue.run_case


class RTNResult(TypedDict):
    chunks: int
    scored_tokens: int
    nll: float
    ppl: float
    cuda_proof: dict[str,str | int | bool]


class MXResult(RTNResult):
    hp1_proof: dict[str,str | int | bool]


type Result = runtime.Result | MXResult | RTNResult


def prepare() -> None:
    """Create four native RTN-WA, four MXINT-WA and sixteen PoTal measurements."""
    old = queue.read(ROOT/"scripts/experiment/cuda-ppl-jobs.rtx3070.json")["jobs"]
    jobs = []
    for job in old:
        if job["kind"] == "rtn" and job["bits"] in (4,8):
            jobs.append(dict(job,binary="/home/mojoi/build/potal-ties-away-rtn/bin/llama-perplexity"))
    for job in old:
        if job["kind"] == "rtn-mx":
            jobs.append(dict(job,binary="/home/mojoi/build/potal-ties-away-rtn/bin/llama-perplexity"))
    potal = sorted((j for j in old if j["kind"] == "potal"),
                   key=lambda j:(j["bits"],-j["anchor"],j["family"]))
    for job in potal:
        for scope in ("none","both"):
            jobs.append(dict(job,id=job["id"]+("-linear" if scope == "none" else "-both"),
                attention_scope=scope,runtime_anchor=job["anchor"],policy="",profile="",
                binary=f'/home/mojoi/build/potal-ties-away-a{job["bits"]}/bin/llama-perplexity'))
    for job in jobs:
        if job["kind"] == "rtn": continue
        job["model"] = job["model"].replace("_RNE.","_TIES_AWAY.")
        job["rounding"] = "ties_away"
    queue.require(len(jobs) == 24,"Incomplete RTN/ties-away experiment matrix")
    queue.write(CONFIG,dict(dataset="wikitext-2-raw/wiki.test.raw",rounding="native_rtn_and_ties_away_mxint_potal",jobs=jobs))
    print(f"Prepared {len(jobs)} affected full-corpus experiments: {CONFIG}")


def inputs(job: runtime.Job, dataset: Path) -> dict[str,str]:
    """Pin the actual model report and driver as well as the executable and data."""
    if job["kind"] == "potal": return dict(runtime.inputs(job,dataset),**{str(Path(__file__)):runtime.digest(Path(__file__))})
    identity = original_inputs(job,dataset)
    identity[str(Path(__file__))] = runtime.digest(Path(__file__))
    if job["kind"] == "rtn": return identity
    sidecar = ROOT/(job["model"]+".mxint.json")
    report = queue.read(sidecar)
    queue.require(report.get("rounding") == "nearest" and report.get("tie_break") == "away_from_zero" and
                  report.get("sha256") == runtime.digest(ROOT/job["model"]),"Expected verified ties-away weights")
    identity.update({str(p):runtime.digest(p) for p in (sidecar,Path(__file__))})
    return identity


def settings(job: runtime.Job, dataset: Path, phase: str, identity: dict[str,str]) -> runtime.gpu.Manifest:
    """Use only the new policy; QK and PV are enabled together or both use the common baseline."""
    return runtime.settings(job,dataset,phase,identity) if job["kind"] == "potal" else original_settings(job,dataset,phase,identity)


def verify(log: Path, job: runtime.Job, chunks: int) -> Result:
    """Reject the old BLOCK_RTN activation proof even if a stale result is copied here."""
    if job["kind"] == "potal": return runtime.verify(log,job,chunks)
    if job["kind"] == "rtn": return original_verify(log,job,chunks)
    result = original_verify(log,dict(job,kind="rtn"),chunks)
    text = log.read_text(errors="replace")
    markers = re.findall(r"MXINT_WA_PROOF (\{[^\r\n]*\})",text)
    queue.require(len(markers) == 1,"Missing true MXINT activation proof")
    proof = json.loads(markers[0])
    dense = (7*(24 if job["family"] == "qwen2-0.5B" else 16)+1)*chunks
    queue.require(proof == dict(activation="MXINT_POT",rounding="ties_away",block_size=32,
        outlier_selection=False,stripe_folding=False,residual=False,
        mxint4_matmuls=dense if job["bits"] == 4 else 0,mxint8_matmuls=dense if job["bits"] == 8 else 0),
        "Wrong MXINT-WA rounding/scaling/coverage")
    return dict(**result,hp1_proof=proof)


@lru_cache(maxsize=40)
def verified_for_watch(log: str, job: str, chunks: int, signature: tuple[tuple[int,int],...]) -> Result:
    """Final logs are revalidated when their files change, then reused between refreshes."""
    return verify(Path(log),json.loads(job),chunks)


def watch(once: bool) -> None:
    """Show the new matrix while keeping the old comparison available in its original monitor."""
    monitor = import_module("watch-cuda-ppl")
    past = import_module("watch-attention-ppl")
    def cached(log: Path, job: runtime.Job, chunks: int) -> Result:
        signature = tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in
                          (log,log.with_name("result.json"),log.with_name("manifest.json"),log.with_name("exit-status.txt")))
        return verified_for_watch(str(log),json.dumps(job,sort_keys=True),chunks,signature)
    while True:
        if not once: print("\033[2J\033[H",end="")
        jobs = queue.read(CONFIG)["jobs"]
        state = monitor.read(OUTPUT/"status.json")
        print("REMEASURE: native RTN + ties-away MXINT/PoTal | * = partial PPL")
        print("Queue:",state.get("state","waiting"),state.get("job",""),state.get("phase",""))
        cells: dict[str,str] = {}; completed = 0; detail = "Preparing verified measurements."
        for job in jobs:
            full = OUTPUT/job["id"]/"full"
            result = monitor.read(full/"result.json")
            cell = "WAIT"
            if result:
                try:
                    checked = cached(full/"ppl.log",job,job["full_chunks"])
                    queue.require(all(result.get(k) == v for k,v in checked.items()) and
                                  (full/"exit-status.txt").read_text().strip() == "0" and
                                  monitor.read(full/"manifest.json").get("job") == job,"Result changed")
                    cell = f'{result["ppl"]:.6f}'; completed += 1
                except (OSError,ValueError,KeyError,RuntimeError): cell = "INVALID"
            elif state.get("job") == job["id"]:
                phase = state.get("phase","smoke")
                execution = monitor.read(OUTPUT/job["id"]/phase/"execution.json")
                running = bool(execution and queue.alive(execution))
                cell = "SMOKE" if phase == "smoke" and running else "RUN" if running else "STOPPED"
                path = OUTPUT/job["id"]/phase/"ppl.log"
                text = ""
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(max(0,path.stat().st_size-65536)); text = stream.read().decode(errors="replace")
                progress = re.findall(r"^\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$",text,re.M)
                count = int(progress[-1][0])//512+1 if progress else 0
                if progress and phase == "full": cell = progress[-1][1]+"*"
                detail = f'Current: {count}/{1 if phase == "smoke" else job["full_chunks"]} chunks'
                if execution: detail += " | elapsed "+monitor.duration(monitor.active_seconds(state,execution))
            cells[job["id"]] = cell
        print(detail)
        print(f'{"Method":<30}'+"".join(f'{name:>13}' for name in ("Qwen n=4","Qwen n=8","Llama n=4","Llama n=8")))
        groups = [("RTN-WA (Q4_0/Q8_0)","rtn",0,"none"),("MXINT-WA","rtn-mx",0,"none"),*(
            (f"PoTal TOP{anchor} {scope}","potal",anchor,scope)
            for scope in ("linear","QK+PV") for anchor in (2,3))]
        for label,kind,anchor,scope in groups:
            values = []
            for family in ("qwen2-0.5B","llama3.2-1B"):
                for bits in (4,8):
                    job = next(j for j in jobs if j["kind"] == kind and j["family"] == family and j["bits"] == bits and
                               (kind != "potal" or (j["anchor"] == anchor and
                                j["attention_scope"] == ("none" if scope == "linear" else "both"))))
                    values.append(cells[job["id"]])
            print(f'{label:<30}'+"".join(f'{value:>13}' for value in values))
        print(f"Completed remeasurement: {completed}/{len(jobs)} full runs. Historical results below use the previous policy.")
        print("\n".join(past.historical_lines()))
        previous = past.attention_rows()
        for anchor in (2,3):
            values = [next(r.ppl for r in previous if r.name == f'{model}-top{anchor}-n{bits}-both')
                      for model in ("qwen","llama") for bits in (4,8)]
            print(f'{f"PoTal TOP{anchor} QK+PV (old)":<30}'+"".join(f'{value:>13}' for value in values))
        sensitivity = monitor.read(runtime.OUTPUT/"sensitivity-status.json")
        print("Ctrl+C closes monitor only | Sensitivity:",sensitivity.get("state","queued after 24 full runs"))
        if once: return
        time.sleep(5)


def main() -> None:
    """Run serially on the shared GPU lock, inherited by the actual inference child."""
    queue.inputs,queue.settings,queue.verify,queue.digest = inputs,settings,verify,runtime.digest
    if "--run" not in sys.argv:
        queue.main(); return
    with Path("/home/mojoi/potal-results/rtx3070-full-24/.run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def locked_run(job: runtime.Job, dataset: Path, output: Path, phase: str,
                       identity: dict[str,str], lock_fd: int) -> None:
            original_run_case(job,dataset,output,phase,identity,lock.fileno())
        queue.run_case = locked_run
        queue.main()


if __name__ == "__main__":
    if "--prepare" in sys.argv: prepare()
    elif "--watch" in sys.argv or "--status" in sys.argv:
        try: watch("--status" in sys.argv)
        except KeyboardInterrupt: print("\nMonitor closed.")
    else: main()
