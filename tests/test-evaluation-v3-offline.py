#!/usr/bin/env python3
from __future__ import annotations

from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
IM2P = ROOT.parent / 'IM2P.sim'
sys.path.insert(0, str(ROOT / 'scripts/eval'))
sys.path.insert(0, str(IM2P))
from end_to_end import main
from sim.tests.cycle.test_execution_cli import fixture_files


class StreamingPipelineTests(unittest.TestCase):
    def test_official_sqlite_adapter_and_scheduler_receive_diagnostic_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture_files(root)
            dataset = root / 'dataset.jsonl.gz'
            with gzip.open(dataset, 'wb') as stream:
                stream.write((root / 'dataset.jsonl').read_bytes())
            application = root / 'application.jsonl'
            application.write_text(json.dumps(dict(schema='potal-application-cpu', version=1,
                source_role='potal_collection', stage='sample_accept', sample_index=0,
                phase='prefill', decode_index=None, thread_id=1, thread_cpu_valid=True, thread_cpu_ns=2)) + '\n')
            declaration = json.loads((root / 'lifecycle.json').read_text())
            declaration['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            declaration['application'] = dict(sha256=hashlib.sha256(application.read_bytes()).hexdigest(),
                sampler_policy='SINGLE_CALLING_THREAD', resource='cpu:0', metric_policy='THREAD_CPU_NS_GANG',
                expected_samples=1, steps=[dict(sample_index=0, logits_ready=['operation:b'], next_decode_entries=[])])
            (root / 'lifecycle.json').write_text(json.dumps(declaration))
            arguments = ['--output', str(root / 'output'), '--im2p', str(IM2P),
                         '--lifecycle', str(root / 'lifecycle.json'), '--application', str(application),
                         '--streaming-ir', '--diagnostic-phase-table', str(root / 'phase.json'),
                         '--diagnostic-frequency-hz', '1000000000']
            for name in ('full-cpu-log', 'full-cpu-graph', 'full-cpu-provenance', 'potal-log',
                         'potal-graph', 'potal-provenance', 'npu-trace', 'library',
                         'cycle-certificate', 'run-aware-certificate'):
                arguments.extend(('--' + name, str(root / 'summary.json')))
            actual_run = subprocess.run

            def fixture_upstream(command, **options):
                module = command[3]
                if module == 'sim.cycle.npu_trace':
                    shutil.copyfile(root / 'npu.jsonl', command[command.index('--output') + 1])
                    return subprocess.CompletedProcess(command, 0)
                if module == 'sim.cycle.reconstruct':
                    shutil.copyfile(dataset, command[command.index('--output') + 1])
                    shutil.copyfile(root / 'summary.json', command[command.index('--summary') + 1])
                    return subprocess.CompletedProcess(command, 0)
                return actual_run(command, **options)

            with patch('offline_pipeline.subprocess.run', side_effect=fixture_upstream), \
                 patch.object(sys, 'argv', ['end_to_end.py', *arguments[:2], 'reconstruct', *arguments[2:]]):
                self.assertEqual(main(), 0)
            with closing(sqlite3.connect(root / 'output/schedule.sqlite')) as database:
                value = database.execute('SELECT body FROM results WHERE identity=?', ('application:sample:0',)).fetchone()
                self.assertEqual(json.loads(value[0])['result_ready_ns'], dict(numerator=15, denominator=1))
                manifest = json.loads(database.execute("SELECT body FROM metadata WHERE key='manifest'").fetchone()[0])
                self.assertFalse(manifest['paper_latency_ready'])
            receipt = json.loads((root / 'output/synthetic-schedule-command.json').read_text())
            self.assertIn('--synthetic', receipt['argv'])
            self.assertEqual(receipt['argv'][-1], str(root / 'output/schedule.sqlite'))
            result = json.loads((root / 'output/result.json').read_text())
            self.assertEqual(result['schedule'], 'SYNTHETIC_ONLY')
            self.assertFalse(result['E2E_RECONSTRUCTION_READY'])
            self.assertIsNone(result['TTFT'])
            self.assertIsNone(result['TPOT'])
            self.assertFalse((root / 'output/reconstructed-result.json').exists())


if __name__ == '__main__':
    unittest.main()
