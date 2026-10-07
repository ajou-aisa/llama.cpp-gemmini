#!/usr/bin/env python3
# How to run: python3 -B scripts/eval/test_paired_capacity.py
"""Self-test for paired_capacity.py: tile-selector goldens and the capacity arithmetic of one known stripe."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_capacity import (acc_fits, contract, coverage, linear_fit, select_tiles,  # noqa: E402
                             stripe_metrics)

IM2P = Path(__file__).resolve().parents[3] / "IM2P.sim"
LARGE = 1 << 20
GOLDEN = {"a8w8-d16-hp1": (5, 5, 51), "a4w4-d16-hp1": (5, 5, 102), "a8w8-d32-hp1": (2, 2, 32),
          "a4w4-d32-hp1": (2, 2, 64), "a8w8-d64-hp1": (1, 1, 16), "a4w4-d64-hp1": (1, 1, 32)}


def main() -> int:
    hardware = {profile: contract(IM2P, profile) for profile in GOLDEN}
    for profile, expected in GOLDEN.items():
        hw = hardware[profile]
        tiles = select_tiles(LARGE, LARGE, LARGE, hw["dim"], hw["bank_count"], hw["bank_rows"], hw["accumulator_rows"])
        assert tiles == expected, (profile, tiles)
    hw = hardware["a8w8-d16-hp1"]
    # The fill loop widens J past sqrt(ACC/2/DIM) when one K32 block leaves scratchpad room.
    assert select_tiles(256, 2304, 32, 16, hw["bank_count"], hw["bank_rows"], hw["accumulator_rows"]) == (5, 6, 2)
    # gpt2 a8w8-d16 qkv_proj stripe 0: every row retained twice, 762 of 768 K active.
    runs = [{"original_block_id": block, "original_k_mask": (1 << 32) - 1 if block != 2 else 0xFFFFDFEF}
            for block in range(24)]
    main = {"m": 80, "n": 2304, "k": 768, "physical_fragments": 34560}
    work = {"m": 160, "n": 2304, "k": 762, "original_k": 768, "runs": runs, "physical_fragments": 69120}
    row = stripe_metrics(main, work, 256, hw)
    assert (row["tile_I"], row["tile_J"], row["I_M"], row["J"], row["I_R"]) == (5, 5, 5, 5, 10), row
    assert [row[f"groups_free_{factor}x"] for factor in (2, 3, 4)] == [7, 14, 20], row
    assert row["work_ids"] == 75 and row["blocks_active"] == 24 and row["f_run"] == 1.0
    assert row["max_chunks"] == 2 and row["ra_rows_max"] == 10 * 2 * 16
    assert row["sp_free"] == 4 * 4096 // 2 - (5 + 5) * 2 * 16
    assert [acc_fits(row, factor, 512, 16) for factor in (1, 2, 3, 4)] == [False, False, True, True]
    # A residual-free stripe and the coverage weighting by residual fragments.
    empty = stripe_metrics(main, {}, 256, hw)
    assert empty["I_R"] == 0 and empty["R_eq"] == 80.0 and empty["residual_fragments"] == 0
    small = dict(row, I_R=2, residual_fragments=10)
    result = coverage([row, small, empty], lambda candidate: acc_fits(candidate, 2, 512, 16))
    assert result == {"stripes": 0.5, "fragments": 10 / (69120 + 10)}, result
    assert linear_fit([(1, 10), (2, 12), (3, 14)]) == (8.0, 2.0)
    print("test_paired_capacity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
