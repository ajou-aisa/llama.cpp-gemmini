from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from campaign_build import command
from eval_common import (
    Record,
    integer,
    read_json,
    record,
    records,
    require,
    sha256,
    text,
    write_json,
)

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])

from sim.cycle.execution_sequence_provider import StatefulSequenceProvider
from sim.cycle.stateful_sequence_evidence import EvidenceContext

COUNTERS: Final = ("fragment_count", "loop_count", "planner_loop_count", "load_request_count",
                  "load_response_count", "store_request_count", "store_response_count",
                  "scale_request_count", "scale_response_count", "event_count")


class CampaignProvider(StatefulSequenceProvider):
    def counters(self) -> Record:
        value = self._session.counters()
        return {key: int(getattr(value, key)) for key in COUNTERS}


def context_from(certificate: Path, root: Path) -> EvidenceContext:
    doc = read_json(certificate)
    parents = record(doc.get("parents"))
    evidence = (Path(text(record(doc.get("evidence_input")), "path"))
                if doc.get("schema") == "stateful-full374-replay-v1" else None)
    return EvidenceContext(root, Path(text(record(doc.get("library")), "path")),
        Path(text(record(doc.get("shared_library")), "path")),
        Path(text(record(parents.get("base")), "path")),
        Path(text(record(parents.get("run_aware")), "path")),
        Path(text(record(parents.get("service")), "path")), tag6_evidence_input=evidence)


def native_build(im2p: Path, output: Path, jobs: int) -> Path:
    output.mkdir()
    command(["cmake", "-S", str(im2p / "sim/cycle"), "-B", str(output),
             "-DCMAKE_BUILD_TYPE=Release"], output, "configure")
    command(["cmake", "--build", str(output), "--parallel", str(jobs)], output, "build")
    command(["ctest", "--test-dir", str(output), "--output-on-failure"], output, "verify")
    candidates = [output / name for name in ("libim2p_cycle_model.dylib", "libim2p_cycle_model.so")]
    found = [path for path in candidates if path.is_file()]
    require(len(found) == 1, "cycle shared library missing")
    return found[0]


def collect(args: argparse.Namespace, runner: Path, model: Path, dataset: Path,
            manifest: Record, output: Path, binding: Record) -> None:
    info = record(binding["build_info"])
    require(all(info.get(key) == 0 for key in ("activation_metrics", "residual_metrics", "scale_metrics")),
            "cycle build must disable all metric collectors")
    new_library = native_build(args.im2p.resolve(), output / "cycle-library-build", args.jobs)
    certificate = args.certificate.resolve(strict=True)
    context = context_from(certificate, args.evidence_root.resolve(strict=True))
    library = (args.library or new_library).resolve(strict=True)
    source_mode = "FRESH_NATIVE_PREFILL"
    if args.trace_source is None:
        require(args.max_chunks == 1, "fresh cycle admission takes one exact trace; run chunks independently")
        command([str(runner), "--model", str(model), "--file", str(dataset),
                 "--output-dir", str(output / "native"), "--cycle-trace", "--max-chunks", "1",
                 "--seed", str(args.seed), "--manifest-sha256", text(manifest, "manifest_sha256")], output, "inference", args.timeout)
        trace = output / "native/chunk-0/npu-cycle-trace.jsonl"
    else:
        source_mode = "REPLAY_EXISTING_SOURCE"
        trace = args.trace_source.resolve(strict=True)
        require(args.source_provenance is not None, "existing trace requires --source-provenance")
        provenance_path = args.source_provenance.resolve(strict=True)
        provenance = read_json(provenance_path)
        inputs = record(provenance.get("inputs"))
        source_trace = record(record(provenance.get("artifacts")).get("npu_trace"))
        require(inputs.get("model_sha256") == sha256(model) and inputs.get("dataset_sha256") == sha256(dataset),
                "trace provenance does not bind requested model/dataset")
        require(source_trace.get("sha256") == sha256(trace) and
                (provenance_path.parent / text(source_trace, "path")).resolve() == trace,
                "trace provenance hash/path mismatch")
        require(record(provenance.get("run")).get("process_exit_code") == 0,
                "trace source inference did not complete")
        run = record(provenance["run"])
        receipt_path = provenance_path.parent / text(run, "command_receipt")
        workload_path = provenance_path.parent / "native/workload.json"
        require(sha256(receipt_path) == run.get("command_receipt_sha256") and
                sha256(workload_path) == run.get("workload_sha256"), "source command/workload receipt changed")
        workload = read_json(workload_path)
        require(args.max_chunks == 1 and workload.get("seed") == args.seed and
                run.get("native_chunk") == 0 and run.get("native_input_tokens") == 256 and
                run.get("actual_sampler_calls") == 1 and run.get("decode_calls") == 0,
                "source replay requires its exact one-chunk 256+1 zero-decode recipe and seed")
        source_build = record(record(provenance["build"])["compiled_info"])
        require(all(source_build.get(key) == info.get(key) for key in ("activation_bits", "weight_bits", "dim")),
                "source trace profile differs from campaign build")
        binding["source_provenance"] = {"path": str(provenance_path), "sha256": sha256(provenance_path)}
    identity: Record = {**manifest, "trace_sha256": sha256(trace),
        "source_mode": source_mode, "cycle_library_sha256": sha256(library), "unit": "cycles",
        "cycle_count_is_not_latency_ms": True}
    rows: dict[int, Record] = {}
    phases: dict[int, Record] = {}
    for row in records(trace):
        if row.get("kind") == "NPU_WORK":
            rows[integer(row, "work_id")] = {key: row[key] for key in
                ("work_id", "layer", "provenance", "phase_id", "operation_id")}
        if row.get("kind") == "PHASE":
            phases[integer(row, "phase_id")] = row
    per_layer: dict[str, Record] = {}
    per_phase: dict[int, Record] = {}
    totals: Record = {key: 0 for key in (*COUNTERS, "dense_cycles", "residual_cycles", "service_cycles",
        "scu_drain_tail_cycles", "resource_drain_tail_cycles", "resource_ready_cycles", "submission_count", "logical_work_count")}
    with CampaignProvider(library, trace, certificate, context) as provider:
        require(provider.admission.production_admitted and
                len(provider.requests) == len(rows), "complete production trace admission required")
        require(provider.admission.profile == text(manifest, "precision").lower() + f"-d{integer(manifest, 'dim')}-hp1",
                "stateful provider profile differs from evaluation manifest")
        require(tuple(sorted(rows)) == tuple(item[0] for item in provider.admission.work_bindings),
                "trace metadata order differs from admitted work order")
        with (output / "per-work-cycle.jsonl").open("x", encoding="utf-8") as stream:
            for request, (work_id, metadata) in zip(provider.requests, sorted(rows.items()), strict=True):
                before = provider.counters()
                window = provider.execute(request, provider.native_cursor)
                after = provider.counters()
                service = window.result_ready_cycle - window.accepted_cycle
                counters = {key: integer(after, key) - integer(before, key) for key in COUNTERS}
                values = {**counters, "submission_count": counters["loop_count"], "logical_work_count": 1, "service_cycles": service,
                    "dense_cycles": service if metadata["provenance"] == "dense_main" else 0,
                    "residual_cycles": service if metadata["provenance"] == "residual" else 0,
                    "scu_drain_tail_cycles": window.final_scale_release_cycle - window.result_ready_cycle,
                    "resource_drain_tail_cycles": window.resource_ready_cycle - window.final_scale_release_cycle,
                    "resource_ready_cycles": window.resource_ready_cycle - window.offered_cycle}
                phase_id = integer(metadata, "phase_id")
                layer = text(metadata, "layer")
                groups = (totals, per_layer.setdefault(layer, {key: 0 for key in values}),
                          per_phase.setdefault(phase_id, {key: 0 for key in values}))
                for group in groups:
                    for key, value in values.items():
                        group[key] = integer(group, key) + value
                stream.write(json.dumps({**identity, **metadata, **values, "work_id": work_id,
                    "offered_cycle": window.offered_cycle, "accepted_cycle": window.accepted_cycle,
                    "result_ready_cycle": window.result_ready_cycle,
                    "final_scale_release_cycle": window.final_scale_release_cycle,
                    "resource_ready_cycle": window.resource_ready_cycle}) + "\n")
                stream.flush()
            provider.verify_complete()
        report: Record = {**identity, "production_admitted": True, "complete": True,
            "work_count": provider.invocations, "final_resource_cursor": provider.native_cursor,
            "profile": provider.admission.profile, "certificate_sha256": sha256(certificate),
            "state_domain_revision": provider.admission.scoped.state_domain_revision,
            "fresh_cycle_library": {"path": str(new_library), "sha256": sha256(new_library)},
            "runtime_cycle_library": {"path": str(library), "sha256": sha256(library)},
            "runtime_build_is_fresh": library == new_library,
            "fresh_build_runtime_byte_equivalence": sha256(new_library) == sha256(library),
            "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN"}
    require(totals["resource_ready_cycles"] == report["final_resource_cursor"], "resource cycle conservation failed")
    require(integer(totals, "dense_cycles") + integer(totals, "residual_cycles") == totals["service_cycles"],
            "dense/residual cycle accounting does not conserve service cycles")
    write_json(output / "provider-report.json", report)
    write_json(output / "cycle_metrics.json", {**identity, **totals, "scu_cycles": None,
        "scu_cycles_status": "NOT_EXPOSED_BY_STATEFUL_PROVIDER",
        "scu_drain_tail_definition": "final_scale_release minus result_ready; NOT total SCU active cycles",
        "traffic_unit": "native_request_response_transactions", "prefill_token_attribution": "BATCH_ONLY"})
    write_json(output / "per-layer-cycle.json", {**identity,
        "layers": [{**identity, "layer": layer, **values} for layer, values in sorted(per_layer.items())]})
    write_json(output / "per-token-cycle.json", {**identity, "attribution": "PHASE_BATCH_ONLY_FOR_PREFILL",
        "phases": [{**identity, "phase_id": phase, "phase_kind": phases[phase]["phase_kind"],
                    "decode_index": phases[phase]["decode_index"], "input_tokens": phases[phase]["input_tokens"],
                    **values} for phase, values in sorted(per_phase.items())],
        "per_prefill_token_cycles": None})
    binding.update({"trace_source": str(trace), "source_mode": source_mode,
                    "cycle_library_sha256": sha256(library), "provider_report_sha256": sha256(output / "provider-report.json")})


if __name__ == "__main__":
    request = read_json(Path(sys.argv[2]))
    options = record(request["options"])
    arguments = argparse.Namespace(**options)
    for key in ("im2p", "trace_source", "source_provenance", "certificate", "evidence_root", "library"):
        value = options.get(key)
        setattr(arguments, key, Path(value) if isinstance(value, str) else None)
    destination = Path(text(request, "output"))
    result_binding = record(request["binding"])
    collect(arguments, Path(text(request, "runner")), Path(text(request, "model")), Path(text(request, "dataset")),
            record(request["manifest"]), destination, result_binding)
    write_json(destination / "cycle-binding.json", result_binding)
