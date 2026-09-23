#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B tests/test-evaluation-v3-pairing.py
from pathlib import Path
import importlib
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts/eval"))
from eval_common import Json, Record


def approved_settings() -> Record:
    return dict(schema="potal-evaluation-settings",version=2,seed=1234,temperature=0,
                threads=1,threads_batch=1,batch_size=256,ubatch_size=256,
                sampler_policy="user-confirmed-greedy-v3",split="test",chunk_ids=list(range(10)),
                top_k=0,top_p=1,min_p=0,repeat_penalty=1,repeat_last_n=0,grammar=None,
                eos_stopping=False,warmup=0)


class ApprovedPolicyTests(unittest.TestCase):
    def test_approved_recipe_and_distinct_ten_chunk_mapping(self):
        from e2e_run import settings_contract
        settings_contract(approved_settings())
        repeated_chunks: list[Json] = [0] * 10
        changes: tuple[Record, ...] = (dict(seed=1),dict(top_k=40),dict(top_p=0.95),dict(min_p=0.05),
                                      dict(eos_stopping=True),dict(chunk_ids=repeated_chunks),dict(warmup=1))
        for change in changes:
            wrong = approved_settings()
            wrong.update(change)
            with self.assertRaises(ValueError):
                settings_contract(wrong)

    def test_forced_cpu_is_not_128_actual_samples(self):
        module = ROOT / "scripts/eval/paired_inputs.py"
        self.assertTrue(module.is_file(),"forced CPU trajectory adapter missing")
        forced_result = importlib.import_module("paired_inputs").forced_result
        row = dict(schema="potal-application-endpoints",version=1,complete=True,
                   execution_kind="FORCED_CPU_COST_ONLY",trajectory_source="POTAL",cost_only=True,
                   samples=0,actual_samples=0,decode_calls=127,generated_tokens=list(range(128)),
                   sample_accept_ns=[],source_role="full_cpu")
        result = forced_result(row)
        self.assertEqual(result["actual_samples"],0)
        self.assertNotIn("ttft_ns",result)
        with self.assertRaises(ValueError):
            forced_result(dict(row,actual_samples=128))


if __name__ == "__main__":
    unittest.main()
