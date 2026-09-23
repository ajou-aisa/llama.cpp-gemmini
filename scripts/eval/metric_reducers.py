from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

from eval_common import Record, integer, ratio, record, records, require, sha256, text

ACT_COUNTS: Final = ("valid_positions", "fp_selected", "potal_selected", "intersection",
    "union", "residual_nnz", "eligible_logical_blocks", "unique_actual_requantized_blocks",
    "finite_positions", "nonfinite_positions", "p3_requantization_events")
FP_COUNTS: Final = {"fp_selected", "intersection", "union"}


def metric_rows(path: Path, schema: str) -> Iterator[Record]:
    iterator = records(path)
    first = next(iterator, {})
    require(first.get("kind") == "RUN" and first.get("schema") == schema and first.get("version") == 1,
            "wrong metric stream schema/version or missing RUN")
    run_id, workload_id = text(first, "run_id"), text(first, "workload_id")
    sequence = integer(first, "sequence")
    invocations: set[int] = set()
    ended = False
    reference_complete = True
    for row in iterator:
        require(not ended, "metric record follows RUN_END")
        require(row.get("schema") == schema and row.get("version") == 1 and
                row.get("run_id") == run_id and row.get("workload_id") == workload_id,
                "metric stream identity changed")
        current = integer(row, "sequence")
        require(current > sequence, "non-increasing metric sequence")
        sequence = current
        if row.get("kind") == "RUN_END":
            require(row.get("success") is True, "metric run failed")
            require(integer(row, "invocation_count") == len(invocations), "metric invocation coverage mismatch")
            if first.get("definition_status") == "CONFIRMED_BY_USER":
                require(row.get("reference_complete") is reference_complete, "ACT final reference coverage mismatch")
            ended = True
        else:
            require(row.get("kind") != "RUN", "duplicate metric RUN")
            invocations.add(integer(row, "invocation_id"))
            if first.get("definition_status") == "CONFIRMED_BY_USER":
                require(type(row.get("reference_complete")) is bool, "missing ACT reference validity")
                reference_complete = reference_complete and row["reference_complete"] is True
            yield row
    require(ended, "missing successful metric RUN_END")


def invocation(row: Record) -> tuple[int, int]:
    return integer(row, "chunk_id"), integer(row, "invocation_id")


def activation_summary(path: Path) -> Record:
    totals = {key: 0 for key in ACT_COUNTS}
    seen: set[tuple[int, int]] = set()
    header = next(records(path), {})
    status, revision = text(header, "definition_status"), text(header, "reference_revision")
    require((status, revision) in {("MISSING_REFERENCE_DEFINITION", "UNRESOLVED"),
            ("PROPOSED_NOT_CONFIRMED", "proposed-invocation-finite-population-v1"),
            ("CONFIRMED_BY_USER", "signed-row-original-bk32-population-2sigma-v1")},
            "unbound/unsupported activation reference authority")
    reference_complete = True
    for row in metric_rows(path, "im2p-activation-quant-metrics"):
        require(row.get("kind") == "COUNTS", "unexpected ACT record")
        key = invocation(row)
        require(key not in seen, "duplicate ACT chunk/invocation")
        seen.add(key)
        values = {name: integer(row, name) for name in ACT_COUNTS if name not in FP_COUNTS}
        m, k = integer(row, "m", 1), integer(row, "k", 1)
        require(values["valid_positions"] == m * k and
                values["finite_positions"] + values["nonfinite_positions"] == m * k,
                "ACT valid/original coordinate population mismatch")
        require(values["eligible_logical_blocks"] == m * ((k + 31) // 32),
                "ACT logical block denominator mismatch")
        require(values["potal_selected"] <= values["valid_positions"] and
                values["residual_nnz"] <= values["valid_positions"], "ACT selected count exceeds population")
        require(values["unique_actual_requantized_blocks"] <= values["eligible_logical_blocks"],
                "requantized block union exceeds eligible logical blocks")
        require(values["p3_requantization_events"] >= values["unique_actual_requantized_blocks"],
                "requantization event count below unique block union")
        if all(row.get(name) is None for name in FP_COUNTS):
            reference_complete = False
            if status == "CONFIRMED_BY_USER":
                require(row.get("reference_complete") is False and values["nonfinite_positions"] > 0 and
                        row.get("reference_invalid_reason") == "NONFINITE_INPUT_UNSPECIFIED",
                        "invalid ACT reference lacks explicit nonfinite reason")
        else:
            if status == "CONFIRMED_BY_USER":
                require(row.get("reference_complete") is True and values["nonfinite_positions"] == 0 and
                        row.get("reference_invalid_reason") is None, "ACT invalid reference cannot publish F counts")
            values.update({name: integer(row, name) for name in FP_COUNTS})
            require(values["fp_selected"] <= values["valid_positions"], "FP selection exceeds population")
            require(values["intersection"] <= min(values["fp_selected"], values["potal_selected"]),
                    "invalid ACT intersection")
            require(values["union"] == values["fp_selected"] + values["potal_selected"] - values["intersection"],
                    "invalid ACT union")
        for name, value in values.items():
            totals[name] += value
    require(bool(seen), "empty ACT observation stream")
    require(status == "CONFIRMED_BY_USER" or reference_complete == (status == "PROPOSED_NOT_CONFIRMED"),
            "reference counts/status mismatch")
    confirmed = status == "CONFIRMED_BY_USER" and reference_complete
    raw: Record = {name: value if reference_complete or name not in FP_COUNTS else None
                   for name, value in totals.items()}
    fractions: Record = {
        "recall": ratio(totals["intersection"], totals["fp_selected"]) if confirmed else None,
        "jaccard": ratio(totals["intersection"], totals["union"]) if confirmed else None,
        "fp_fraction": ratio(totals["fp_selected"], totals["valid_positions"]) if confirmed else None,
        "residual_fraction": ratio(totals["residual_nnz"], totals["valid_positions"]),
        "requant_fraction": ratio(totals["unique_actual_requantized_blocks"], totals["eligible_logical_blocks"]),
    }
    return {"schema": "potal-activation-quant-summary", "version": 1, "counts": raw,
            "ratios": fractions, "aggregation": "integer-sum-before-division",
            "definition_status": status, "definition_revision": revision,
            "reference_metric_publication": "READY" if confirmed else "NOT_READY",
            "reference_complete": reference_complete,
            "candidate_ratios": {"recall": ratio(totals["intersection"], totals["fp_selected"]),
                "jaccard": ratio(totals["intersection"], totals["union"]),
                "fp_fraction": ratio(totals["fp_selected"], totals["valid_positions"])} if reference_complete and not confirmed else None,
            "candidate_ratio_scope": "DIAGNOSTIC_PROPOSED_DEFINITION_ONLY",
            "undefined_ratio_policy": "null", "invocations": len(seen),
            "chunks": len({chunk for chunk, _ in seen}), "input_sha256": sha256(path)}


def compact_runs(row: Record) -> int:
    original_k, compact_k = integer(row, "original_k", 1), integer(row, "k", 1)
    raw = row.get("runs")
    require(isinstance(raw, list) and bool(raw), "missing compact runs")
    if not isinstance(raw, list):
        return 0
    cursor, previous = 0, -1
    for item in raw:
        run = record(item)
        block, mask = integer(run, "original_block_id"), integer(run, "original_k_mask", 1)
        count = integer(run, "compact_k_count", 1)
        require(block > previous and mask < 1 << 32 and mask.bit_count() == count,
                "invalid compact original block/mask/count")
        require(integer(run, "compact_k_begin") == cursor, "compact run gap/overlap")
        require(block * 32 + mask.bit_length() <= original_k, "compact run exceeds original K")
        cursor += count
        previous = block
    require(cursor == compact_k, "compact run coverage mismatch")
    return len(raw)


def residual_summary(path: Path, accept_proposed_weighting: bool = False) -> Record:
    mains: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    work_parents: set[tuple[int, int, int]] = set()
    dense_macs, expanded_macs, compact_macs, run_count = 0, 0, 0, 0
    for row in metric_rows(path, "im2p-residual-path-metrics"):
        key = (*invocation(row), integer(row, "stripe_id"))
        m, n, k = integer(row, "m", 1), integer(row, "n", 1), integer(row, "k", 1)
        kind = text(row, "kind")
        require(kind in {"MAIN_STRIPE", "COMPACT_WORK"}, "unexpected RES record")
        if kind == "MAIN_STRIPE":
            require(key not in mains, "duplicate MAIN_STRIPE")
            mains[key] = (m, n, k)
            dense_macs += m * n * k
        else:
            require(key in mains, "COMPACT_WORK lacks original MAIN_STRIPE")
            require(key not in work_parents, "more than one logical compact request per compatible invocation")
            require(n == mains[key][1] and integer(row, "original_k", 1) == mains[key][2],
                    "compact work parent N/original K mismatch")
            for name in ("tile_i_count", "tile_j_count", "tile_k_count"):
                integer(row, name, 1)
            rows = row.get("row_map")
            require(isinstance(rows, list) and len(rows) == m, "compact row map coverage mismatch")
            mappings = [record(item) for item in rows] if isinstance(rows, list) else []
            coordinates = {(integer(item, "original_lane_id"), integer(item, "source_row")) for item in mappings}
            require(len(coordinates) == m and all(source_row < mains[key][0] for _, source_row in coordinates),
                    "duplicate or out-of-range compact radix row")
            work_parents.add(key)
            run_count += compact_runs(row)
            expanded_macs += m * n * integer(row, "original_k", 1)
            compact_macs += m * n * k
    require(bool(mains), "empty MAIN_STRIPE coverage")
    return {"schema": "potal-residual-path-summary", "version": 1,
            "definition_revision": "proposed-v3-DTR-v1",
            "definition_status": "EXPLICIT_PROPOSED_WEIGHTING_ACCEPTED" if accept_proposed_weighting else "PROPOSED_NOT_RESEARCH_AUTHORITY",
            "raw": {"D": dense_macs, "T": expanded_macs, "R": compact_macs},
            "ratios": {"row_factor": ratio(expanded_macs, dense_macs),
                       "retained_K_factor": ratio(compact_macs, expanded_macs),
                       "residual_main_MAC": ratio(compact_macs, dense_macs)},
            "ratio_scope": "declared proposed algebra; not implicitly adopted research definition",
            "main_stripes": len(mains), "compact_works": len(work_parents), "runs": run_count,
            "residual_free_main_stripes": len(mains) - len(work_parents),
            "input_sha256": sha256(path), "undefined_ratio_policy": "null"}
