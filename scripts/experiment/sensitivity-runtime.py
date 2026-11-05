#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic"]
# ///
# How to run: imported by sensitivity-experiment.py
"""Reuse the verified CUDA queue with runtime anchor and sensitivity proofs."""
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
import json
import os
from pathlib import Path
import re
from typing import Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field

gpu = import_module("cuda-attention-ppl")
queue = import_module("cuda-ppl-queue")
analysis = import_module("sensitivity-analysis")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/home/mojoi/potal-results/anchor-sensitivity-ties-away-v1")
EXPORT = ROOT.parent / "experiment-tools/anchor-sensitivity-ties-away-20261102"
original_digest = queue.digest


class Job(gpu.Job):
    runtime_anchor: int
    policy: str
    profile: str


class Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_name: Literal["potal-sensitivity-v1"] = Field(alias="schema")
    key: str
    node: str
    bits: Literal[4, 8]
    selected_anchor: Literal[2, 3]
    logical_macs: int = Field(gt=0)
    main_fragments: int = Field(gt=0)
    residual_fragments: int = Field(ge=0)
    saturations: Literal[0]
    call: int = Field(ge=0)


class Result(gpu.Result):
    fragment_macs_per_chunk: float
    nll_per_chunk: list[float]
    policy_proof: dict[str, str | int | bool]


@lru_cache(maxsize=128)
def signature_digest(path: str, size: int, modified: int) -> str:
    """Rehash a file whenever its size or nanosecond modification time changes."""
    return original_digest(Path(path))


def digest(path: Path) -> str:
    """Avoid repeatedly reading the same multi-gigabyte model within this process."""
    stat = path.stat()
    return signature_digest(str(path),stat.st_size,stat.st_mtime_ns)


def inputs(job: Job, dataset: Path) -> dict[str, str]:
    """Pin isolated binaries, policies, the dataset and every analysis/runner module."""
    identity = gpu.inputs(dict(job, anchor=2),dataset)
    sidecar = ROOT / (job["model"]+".mxint.json")
    report = queue.read(sidecar)
    queue.require(report.get("rounding") == "nearest" and report.get("tie_break") == "away_from_zero" and
                  digest(ROOT/job["model"]) == report.get("sha256"),"Expected verified ties-away weights")
    paths = [Path(__file__), ROOT / "scripts/experiment/sensitivity-analysis.py",
             ROOT / "scripts/experiment/sensitivity-experiment.py",
             ROOT / "scripts/experiment/sensitivity-report.py",sidecar]
    if job["policy"]: paths.append(Path(job["policy"]))
    identity.update({str(path):digest(path) for path in paths})
    return identity


def settings(job: Job, dataset: Path, phase: str, identity: Mapping[str, str]) -> gpu.Manifest:
    """Keep attention jointly enabled and change only the declared folding anchors."""
    manifest = gpu.settings(job,dataset,phase,identity)
    manifest["command"][manifest["command"].index("--chunks")+1] = str(1 if phase == "smoke" else job["full_chunks"])
    manifest["environment"]["GGML_GEMMINI_ANCHOR_DEFAULT"] = str(job["runtime_anchor"])
    if job["policy"]: manifest["environment"]["GGML_GEMMINI_ANCHOR_POLICY"] = job["policy"]
    if job["profile"]: manifest["environment"]["GGML_GEMMINI_SENSITIVITY_LOG"] = job["profile"]
    else: manifest["environment"]["GGML_GEMMINI_ANCHOR_TRACE"] = "1"
    return manifest


def policy_entries(path: str) -> dict[str, int]:
    """Load the same exact-key TSV policy consumed by the backend."""
    if not path: return {}
    entries = [line.split("\t") for line in Path(path).read_text().splitlines()
               if line and not line.startswith("#")]
    queue.require(all(len(row) == 2 and row[1] in ("2","3") for row in entries),"Malformed anchor policy")
    policy = {key:int(anchor) for key,anchor in entries}
    queue.require(len(policy) == len(entries),"Duplicate anchor policy keys")
    return policy


def verify(log: Path, job: Job, chunks: int) -> Result:
    """Require GPU/SCU coverage, the actual policy, complete traces and finite NLL."""
    expected = policy_entries(job["policy"])
    checked = gpu.verify(log,dict(job, anchor=0 if expected else job["runtime_anchor"]),chunks)
    text = log.read_text(errors="replace")
    markers = re.findall(r"POTAL_ANCHOR_POLICY_PROOF (\{[^\r\n]*\})",text)
    queue.require(len(markers) == 1,"Missing policy completion proof")
    proof = json.loads(markers[0])
    queue.require(proof == dict(schema="potal-anchor-policy-v1",complete=True,
                  default_anchor=job["runtime_anchor"],policy_keys=len(expected),used_policy_keys=len(expected),
                  profiling=bool(job["profile"]),rounding="ties_away"),"Policy proof differs from requested settings")
    layers = {"qwen2-0.5B":24,"llama3.2-1B":16}[job["family"]]
    if job["profile"]:
        rows = analysis.aggregate(Path(job["profile"]),chunks,layers)
        cost = sum(r[f'macs_top{job["runtime_anchor"]}'] for r in rows)
    else:
        nodes = [Node.model_validate_json(row) for row in re.findall(r"POTAL_ANCHOR_NODE (\{[^\r\n]*\})",text)]
        operations = (7+(2 if job["attention_scope"] == "both" else 0))*layers+1
        queue.require(len(nodes) == operations*chunks,"Incomplete actual operation trace")
        calls: dict[str,list[int]] = defaultdict(list)
        for node in nodes:
            queue.require(node.selected_anchor == expected.get(node.key,job["runtime_anchor"]),"Anchor ignored")
            calls[node.key].append(node.call)
        queue.require(len(calls) == operations and all(sorted(c) == list(range(chunks)) for c in calls.values()),
                      "Missing/duplicate actual operation trace")
        cost = 32*sum(n.main_fragments+n.residual_fragments for n in nodes)/chunks
    per_chunk = [float(row) for row in re.findall(r"PPL_CHUNK [^\r\n]*?\bnll=([0-9.eE+-]+)",text)]
    queue.require(len(per_chunk) == chunks,"Missing paired NLL observations")
    return dict(**checked,fragment_macs_per_chunk=cost,nll_per_chunk=per_chunk,policy_proof=proof)


def run(base: gpu.Job, name: str, dataset: Path, chunks: int, lock_fd: int,
        anchor: int = 2, policy: Path | None = None, profile: bool = False) -> Result:
    """Resume verified cases; retain interrupted data rather than overwriting it."""
    group = f'{base["family"]}-n{base["bits"]}'
    directory = OUTPUT / group
    trace = directory / name / "full/profile.jsonl"
    job: Job = dict(base,id=name,anchor=anchor,full_chunks=chunks,runtime_anchor=anchor,
                    policy=str(policy) if policy else "", profile=str(trace) if profile else "")
    directory.mkdir(parents=True,exist_ok=True)
    status = dict(state="running",profile=group,case=name,pid=os.getpid(),
                  dataset=dataset.name,chunks=chunks,log=str(directory/name/"full/ppl.log"))
    queue.write(OUTPUT / "sensitivity-status.json",status)
    queue.write(EXPORT / "status.json",status)
    queue.inputs,queue.settings,queue.verify,queue.digest = inputs,settings,verify,digest
    identity = inputs(job,dataset)
    queue.run_case(job,dataset,directory,"full",identity,lock_fd)
    return queue.read(directory/name/"full/result.json")
