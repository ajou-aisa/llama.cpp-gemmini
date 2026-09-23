#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from application_results import application_result, cuda_placement, load_measurement
from eval_common import Record, sha256


def layer_count_fixture(root: Path, repetition: int) -> Path:
    destination = root / f'repetition-{repetition:02d}'
    (destination / 'native').mkdir(parents=True)
    endpoints: Record = dict(schema='potal-application-endpoints', version=1, chunk_id=repetition,
        workload='E2E_GENERATION_256_128', source_role='cuda', t0_ns=100,
        sample_accept_ns=[200 + repetition + index*3 for index in range(128)],
        generated_tokens=list(range(128)), samples=128, decode_calls=127,
        complete=True, warmup=0, timing_source='steady_clock', timing_unit='ns', excludes_terminal_io=True)
    endpoint_path = destination / 'native/application.jsonl'
    endpoint_path.write_text(json.dumps(endpoints) + '\n')
    process_log = destination / 'process.log'
    process_log.write_text('load_tensors: offloaded 12/12 layers to GPU\n')
    result = dict(schema='potal-e2e-run', version=1, role='cuda', measurement_kind='NATIVE_APPLICATION',
        host_id='fixture-host', repetition=repetition, chunk_id=repetition,
        measurement_id=f'fixture-{repetition}', comparison_contract='fixture',
        application_sha256=sha256(endpoint_path), actual_placement=cuda_placement(process_log),
        **application_result(endpoints))
    path = destination / 'result.json'
    path.write_text(json.dumps(result))
    return path


class CudaPlacementTests(unittest.TestCase):
    def test_generic_gpu_layer_count_is_not_complete_cuda_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'load.log'
            path.write_text('load_tensors: offloaded 12/12 layers to GPU\n')
            proof = cuda_placement(path)
            self.assertEqual(proof['coverage'], 'LAYER_COUNT_ONLY')
            self.assertFalse(proof['placement_complete'])
            self.assertIsNone(proof['verified_backend'])

    def test_layer_count_only_measurement_cannot_publish_verified_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'request.json').write_text(json.dumps(dict(
                build_info=dict(activation_metrics=0, residual_metrics=0, cycle_sim=0,
                                cuda=1, log_cycle=0, ggml_cpu_cycle_log=0),
                host=dict(host_id='fixture-host'))))
            paths = [layer_count_fixture(root, index) for index in range(10)]
            with self.assertRaisesRegex(ValueError, 'LAYER_COUNT_ONLY'):
                load_measurement(paths[0])
            command = [sys.executable, '-B', str(ROOT / 'scripts/eval/end_to_end.py'),
                       '--output', str(root / 'campaign'), 'aggregate']
            for path in paths:
                command.extend(('--result', str(path)))
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('LAYER_COUNT_ONLY', result.stderr)
            self.assertFalse((root / 'campaign').exists())


if __name__ == '__main__':
    unittest.main()
