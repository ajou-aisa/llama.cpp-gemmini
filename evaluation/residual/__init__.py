"""Logical compaction and physical run-padding metrics from producer observations."""

from pathlib import Path

from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows, summary
from scripts.eval.eval_common import (
    EvaluationError,
    Record,
    integer,
    ratio,
    record,
    require,
)


def _fragments(row: Record, dim: int) -> int:
    raw = row.get("runs")
    require(isinstance(raw, list) and bool(raw), "missing compact runs")
    if not isinstance(raw, list):
        raise EvaluationError("missing compact runs")
    cursor, previous, fragments = 0, -1, 0
    original_k, compact_k = integer(row, "original_k", 1), integer(row, "k", 1)
    for item in raw:
        run = record(item)
        block, mask = integer(run, "original_block_id"), integer(run, "original_k_mask", 1)
        count = integer(run, "compact_k_count", 1)
        require(block > previous and mask < 1 << 32 and mask.bit_count() == count,
                "invalid compact original block/mask/count")
        require(integer(run, "compact_k_begin") == cursor, "compact run gap/overlap")
        require(block * 32 + mask.bit_length() <= original_k, "compact run exceeds original K")
        cursor += count
        fragments += (count + dim - 1) // dim
        previous = block
    require(cursor == compact_k, "compact run coverage mismatch")
    return fragments


def reduce(path: Path, manifest: Manifest) -> Record:
    rows = metric_rows(path, "im2p-residual-path-metrics", manifest)
    next(rows)
    mains: dict[tuple[int, int], Record] = {}
    radix_rows: dict[tuple[int, int], Record] = {}
    works: set[tuple[int, int]] = set()
    ranges: dict[int, list[tuple[int, int]]] = {}
    dense, compact, padded_dense, padded_compact = 0, 0, 0, 0
    original_rows, retained_rows, original_k, retained_k = 0, 0, 0, 0
    dim = manifest.dim
    for row in rows:
        key = (integer(row, "invocation_id"), integer(row, "stripe_id"))
        kind = row.get("kind")
        require(kind in ("MAIN_STRIPE", "RADIX_STRIPE", "COMPACT_WORK"), "unexpected RES record")
        if kind == "RADIX_STRIPE":
            require(key in mains and key not in radix_rows, "missing/duplicate radix parent")
            parent = mains[key]
            radix = integer(row, "radix_limb_count")
            main_m, main_k = integer(parent, "m", 1), integer(parent, "k", 1)
            require(integer(row, "main_original_rows", 1) == main_m and
                    integer(row, "original_k", 1) == main_k and
                    integer(row, "original_rows") == integer(row, "original_radix_rows") == main_m * radix,
                    "original radix row population mismatch")
            original_rows += main_m * radix
            radix_rows[key] = row
            continue
        m, n, k = (integer(row, field, 1) for field in ("m", "n", "k"))
        if kind == "MAIN_STRIPE":
            require(key not in mains, "duplicate MAIN_STRIPE")
            require(integer(row, "row_count", 1) == m, "main stripe row count mismatch")
            mains[key] = row
            ranges.setdefault(key[0], []).append((integer(row, "row_begin"), m))
            original_k += k
            dense += m * n * k
            k_fragments = sum((min(32, k - begin) + dim - 1) // dim for begin in range(0, k, 32))
            fragments = ((m + dim - 1) // dim) * ((n + dim - 1) // dim) * k_fragments
            require(integer(row, "physical_fragments", 1) == fragments,
                    "main physical fragments disagree with original BK32 boundaries")
            padded_dense += fragments * dim ** 3
        else:
            require(key in mains and key in radix_rows and key not in works, "missing/duplicate compact parent")
            parent = mains[key]
            parent_m = integer(parent, "m", 1)
            parent_k = integer(parent, "k", 1)
            radix = integer(radix_rows[key], "radix_limb_count", 1)
            require(n == integer(parent, "n", 1) and integer(row, "original_k", 1) == parent_k,
                    "compact parent N/original K mismatch")
            require(integer(row, "source_row_count", 1) == parent_m and
                    integer(row, "source_row_begin") == integer(parent, "row_begin"),
                    "compact source row span mismatch")
            require(integer(row, "radix_limb_count", 1) == radix and
                    integer(row, "original_rows", 1) == parent_m * radix and
                    integer(row, "retained_rows", 1) == m and
                    integer(row, "zero_limb_pruned_count") == parent_m * radix - m,
                    "compact radix pruning population mismatch")
            require(integer(row, "retained_k", 1) == integer(row, "compact_k", 1) == k,
                    "compact K population mismatch")
            mappings = row.get("row_map")
            require(isinstance(mappings, list) and len(mappings) == m, "compact row map coverage mismatch")
            coordinates: set[tuple[int, int]] = {
                (integer(record(item), "original_lane_id"), integer(record(item), "source_row"))
                for item in mappings} if isinstance(mappings, list) else set()
            require(len(coordinates) == m and all(lane < radix and source < parent_m
                    for lane, source in coordinates), "invalid compact radix row map")
            for field in ("tile_i_count", "tile_j_count", "tile_k_count"):
                integer(row, field, 1)
            fragments = ((m + dim - 1) // dim) * ((n + dim - 1) // dim) * _fragments(row, dim)
            require(integer(row, "physical_fragments", 1) == fragments,
                    "compact physical fragments disagree with original run boundaries")
            works.add(key)
            retained_rows += m
            retained_k += k
            compact += m * n * k
            padded_compact += fragments * dim ** 3
    require(bool(mains), "empty RES main observation stream")
    require(mains.keys() == radix_rows.keys(), "incomplete radix stripe coverage")
    require(works == {key for key, row in radix_rows.items() if integer(row, "radix_limb_count") > 0},
            "incomplete compact execution coverage")
    for spans in ranges.values():
        cursor = 0
        for begin, count in sorted(spans):
            require(begin == cursor, "main stripe overlap/gap")
            cursor += count
    return {**summary(manifest, path, "residual"), "logical_ratio": ratio(compact, dense),
            "padded_ratio": ratio(padded_compact, padded_dense),
            "retained_row_factor": ratio(retained_rows, original_rows),
            "retained_k_factor": ratio(retained_k, original_k),
            "logical_main_macs": dense, "logical_residual_macs": compact,
            "physical_main_macs": padded_dense, "physical_residual_macs": padded_compact,
            "original_radix_rows": original_rows, "retained_rows": retained_rows,
            "zero_limb_pruned_count": original_rows - retained_rows,
            "original_k": original_k, "compact_k": retained_k,
            "main_stripes": len(mains), "compact_works": len(works),
            "row_factor_denominator": "pre_pruning_radix_expanded_rows",
            "k_factor_denominator": "sum_original_K_per_main_stripe"}
