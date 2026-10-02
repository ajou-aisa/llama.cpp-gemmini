#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B tests/test-evaluation-v3.py
from __future__ import annotations

from pathlib import Path
import importlib
import json
import subprocess
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from eval_common import Record, record


def write_stream(path, rows):
    for index, row in enumerate(rows):
        row.update(schema=rows[0]["schema"], version=1, sequence=index,
                   run_id="run", workload_id="work")
    rows[-1]["invocation_count"] = len({row["invocation_id"] for row in rows if "invocation_id" in row})
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


class EntrypointTests(unittest.TestCase):
    def test_independent_entrypoints_have_real_help(self) -> None:
        # Given three independently requested workflows.
        for name in ("activation_quant_metrics", "residual_path_metrics", "end_to_end"):
            with self.subTest(entrypoint=name):
                path = ROOT / "scripts/eval" / (name + ".py")
                # When invoking each public command.
                self.assertTrue(path.is_file(), f"missing independent entrypoint: {path}")
                result = subprocess.run([sys.executable, "-B", str(path), "--help"],
                                        capture_output=True, text=True, timeout=10)
                # Then the actual CLI describes executable options.
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)


class ReducerTests(unittest.TestCase):
    def reducer(self, name: str):
        path = ROOT / "scripts/eval/metric_reducers.py"
        self.assertTrue(path.is_file(), "metric reducers are not implemented")
        return getattr(importlib.import_module("metric_reducers"), name)

    def test_activation_microaverage_keeps_zero_denominators_undefined(self) -> None:
        reduce = self.reducer("activation_summary")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.jsonl"
            rows = [dict(kind="RUN", schema="im2p-activation-quant-metrics", version=1,
                         definition_status="PROPOSED_NOT_CONFIRMED",reference_revision="proposed-invocation-finite-population-v1"),
                    dict(kind="COUNTS", chunk_id=0, invocation_id=1, layer="linear", m=1,k=32,
                         valid_positions=32, fp_selected=0, potal_selected=2, intersection=0,
                         union=2,residual_nnz=1,eligible_logical_blocks=1,
                         unique_actual_requantized_blocks=1,nonfinite_positions=0,
                         finite_positions=32,p3_requantization_events=1,
                         definition_status="CONFIRMED",definition_revision="test-source-bound"),
                    dict(kind="RUN_END",success=True)]
            write_stream(path, rows)
            summary = reduce(path)
            self.assertIsNone(summary["candidate_ratios"]["recall"])
            self.assertEqual(summary["candidate_ratios"]["jaccard"], 0)

    def test_residual_main_denominator_includes_no_residual(self) -> None:
        reduce = self.reducer("residual_summary")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shapes.jsonl"
            rows = [dict(kind="RUN",schema="im2p-residual-path-metrics",version=1,
                         run_id="run",workload_id="work"),
                    dict(kind="MAIN_STRIPE",chunk_id=0,invocation_id=1,stripe_id=0,m=2,n=3,k=32),
                    dict(kind="MAIN_STRIPE",chunk_id=0,invocation_id=2,stripe_id=0,m=4,n=5,k=64),
                    dict(kind="RUN_END",success=True)]
            write_stream(path, rows)
            summary = reduce(path, False)
            self.assertEqual(summary["raw"], {"D":1472,"T":0,"R":0})
            self.assertEqual(summary["ratios"]["residual_main_MAC"], 0)
            self.assertIsNone(summary["ratios"]["retained_K_factor"])


class ApplicationTests(unittest.TestCase):
    def api(self):
        path = ROOT / "scripts/eval/application_results.py"
        self.assertTrue(path.is_file(), "application endpoint reducer missing")
        return importlib.import_module("application_results")

    def test_tpot_uses_last_minus_first_over_127(self) -> None:
        api = self.api()
        endpoints = dict(schema="potal-application-endpoints",version=1,chunk_id=0,
                         workload="E2E_GENERATION_256_128", t0_ns=100,
                         sample_accept_ns=[200+i*3 for i in range(128)],
                         generated_tokens=list(range(128)),samples=128,decode_calls=127,
                         complete=True,warmup=0,timing_source="steady_clock",timing_unit="ns",
                         excludes_terminal_io=True)
        result = api.application_result(endpoints)
        self.assertEqual(result["ttft_ns"], 100)
        self.assertEqual(result["tpot_ns"], {"numerator":381,"denominator":127})

    def test_incomplete_generation_is_rejected(self) -> None:
        api = self.api()
        with self.assertRaises(ValueError):
            api.application_result(dict(schema="potal-application-endpoints",version=1,
                                        samples=127,complete=False))

    def test_median_of_run_means_does_not_pool_intervals(self) -> None:
        api = self.api()
        rows = [dict(repetition=i,chunk_id=i,measurement_id=str(i),ttft_ns=i*10,
                     tpot_ns=dict(numerator=i*127,denominator=127),host_id="host",role="cuda",
                     comparison_contract="same",application_sha256=str(i),
                     measurement_kind="NATIVE_APPLICATION") for i in range(10)]
        result = api.aggregate_results(rows)
        self.assertEqual(result["ttft_ns"], {"numerator":45,"denominator":1})
        self.assertEqual(result["tpot_ns"], {"numerator":9,"denominator":2})


class BoundaryTests(unittest.TestCase):
    def test_confirmed_marker_cannot_invent_reference_authority(self) -> None:
        from metric_reducers import activation_summary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unbound.jsonl"
            write_stream(path,[dict(kind="RUN",schema="im2p-activation-quant-metrics",
                                   definition_status="CONFIRMED",reference_revision="manual-marker"),
                               dict(kind="RUN_END",success=True)])
            with self.assertRaisesRegex(ValueError,"reference authority"):
                activation_summary(path)

    def test_unknown_sampler_override_is_not_ignored(self) -> None:
        from e2e_run import settings_contract
        with self.assertRaises(ValueError):
            settings_contract(dict(schema="potal-evaluation-settings",version=1,seed=1,
                                   temperature=0,threads=1,threads_batch=1,batch_size=256,
                                   ubatch_size=256,sampler_policy="common_default_chain",
                                   split="test",chunk_id=0,top_k=1))

    def test_token_trajectory_mismatch_rejects_cost_pair(self) -> None:
        from application_results import validate_pair
        full: Record = dict(host_id="same",model_sha256="model",dataset_sha256="data",
                    input_tokens_sha256="prompt",generated_tokens_sha256="tokens-A",
                    cpu_kernel_contract_sha256="kernel",comparison_contract="same")
        potal: Record = dict(full, generated_tokens_sha256="tokens-B")
        with self.assertRaises(ValueError):
            validate_pair(full,potal)

    def test_repeated_application_hash_is_not_ten_measurements(self) -> None:
        from application_results import aggregate_results
        rows: list[Record] = [dict(repetition=i,chunk_id=i,measurement_id=str(i),application_sha256="same-sample") for i in range(10)]
        with self.assertRaises(ValueError):
            aggregate_results(rows)

    def test_stats_enabled_build_is_rejected_by_e2e(self) -> None:
        from eval_common import validate_recipe
        for activation, residual in ((1,0),(0,1),(1,1),(2,0),(True,0)):
            with self.subTest(activation=activation,residual=residual), self.assertRaises(ValueError):
                validate_recipe(dict(activation_metrics=activation,residual_metrics=residual,cycle_sim=0), "e2e")

    def test_residual_cpu_fallback_is_not_residual_free_coverage(self) -> None:
        from eval_common import validate_recipe
        with self.assertRaises(ValueError):
            validate_recipe(dict(activation_metrics=0,residual_metrics=1,cycle_sim=1,
                                 activation_mode="EXSIA",block_size=32,hp1=True,backend="IM2P_SIM",
                                 gemmini=1,gemmini_option="WS",activation_bits=8,weight_bits=8,dim=16,
                                 rmd_enabled=1,rmd_backend="CPU"), "residual")

    def test_actual_offloaded_profile_required_for_metrics(self) -> None:
        from eval_common import validate_recipe
        base = dict(activation_metrics=1,residual_metrics=0,cycle_sim=1,
                    activation_mode="EXSIA",block_size=32,gemmini=1,gemmini_option="WS",
                    backend="IM2P_SIM",hp1=True)
        for bits in (4,8):
            for dim in (16,32,64):
                validate_recipe(dict(base,activation_bits=bits,weight_bits=bits,dim=dim),"activation")
        for changes in (dict(gemmini=0),dict(gemmini_option="CPU"),dict(dim=128),dict(weight_bits=4),
                        dict(backend="FPGA_UART"),dict(cycle_sim=0)):
            wrong: Record = dict(base,activation_bits=8,weight_bits=8,dim=16)
            wrong.update(changes)
            with self.assertRaises(ValueError):
                validate_recipe(wrong,"activation")

    def test_proposed_activation_definition_blocks_reference_publication(self) -> None:
        from metric_reducers import activation_summary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.jsonl"
            rows = [dict(kind="RUN",schema="im2p-activation-quant-metrics",
                         definition_status="PROPOSED_NOT_CONFIRMED",reference_revision="proposed-invocation-finite-population-v1"),
                    dict(kind="COUNTS",chunk_id=0,invocation_id=0,m=1,k=32,valid_positions=32,
                         finite_positions=32,nonfinite_positions=0,fp_selected=2,potal_selected=3,
                         intersection=1,union=4,residual_nnz=5,eligible_logical_blocks=1,
                         unique_actual_requantized_blocks=1,p3_requantization_events=3),
                    dict(kind="RUN_END",success=True)]
            write_stream(path,rows)
            summary = activation_summary(path)
            self.assertEqual(record(summary["counts"])["residual_nnz"],5)
            self.assertIsNone(record(summary["ratios"])["recall"])
            self.assertEqual(summary["reference_metric_publication"],"NOT_READY")

    def test_duplicate_main_stripe_rejected(self) -> None:
        from metric_reducers import residual_summary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shapes.jsonl"
            row = dict(kind="MAIN_STRIPE",chunk_id=0,invocation_id=0,stripe_id=0,m=2,n=3,k=32)
            rows = [dict(kind="RUN",schema="im2p-residual-path-metrics"),dict(row),dict(row),
                    dict(kind="RUN_END",success=True)]
            write_stream(path,rows)
            with self.assertRaises(ValueError):
                residual_summary(path)

    def test_expanded_rows_and_skipped_runs_preserve_rectangle_mac_counts(self) -> None:
        from metric_reducers import residual_summary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shapes.jsonl"
            rows = [dict(kind="RUN",schema="im2p-residual-path-metrics"),
                    dict(kind="MAIN_STRIPE",chunk_id=0,invocation_id=0,stripe_id=0,m=1,n=5,k=128),
                    dict(kind="COMPACT_WORK",chunk_id=0,invocation_id=0,stripe_id=0,m=2,n=5,k=22,
                         original_k=128,tile_i_count=1,tile_j_count=1,tile_k_count=1,
                         row_map=[dict(original_lane_id=0,source_row=0),dict(original_lane_id=3,source_row=0)],
                         runs=[dict(original_block_id=0,original_k_mask=4095,compact_k_begin=0,compact_k_count=12),
                               dict(original_block_id=3,original_k_mask=1023,compact_k_begin=12,compact_k_count=10)]),
                    dict(kind="RUN_END",success=True)]
            write_stream(path,rows)
            summary = residual_summary(path,True)
            self.assertEqual(summary["raw"],dict(D=640,T=1280,R=220))
            ratios = record(summary["ratios"])
            row_factor, retained, total = ratios["row_factor"], ratios["retained_K_factor"], ratios["residual_main_MAC"]
            assert isinstance(row_factor,float) and isinstance(retained,float) and isinstance(total,float)
            self.assertEqual(row_factor,2)
            self.assertAlmostEqual(row_factor * retained,total)

    def test_cuda_partial_placement_rejected(self) -> None:
        from e2e_run import cuda_placement
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"native.log"
            path.write_text("load_tensors: offloaded 8/12 layers to GPU\n")
            with self.assertRaises(ValueError):
                cuda_placement(path)

    def test_fresh_output_is_preserved(self) -> None:
        from eval_common import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            path.write_text("USER_DATA")
            with self.assertRaises(ValueError):
                write_json(path,{"replaced":True})
            self.assertEqual(path.read_text(),"USER_DATA")


if __name__ == "__main__":
    unittest.main()
