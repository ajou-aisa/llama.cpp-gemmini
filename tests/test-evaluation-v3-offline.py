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
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
IM2P = ROOT.parent / 'IM2P.sim'
sys.path.insert(0, str(ROOT / 'scripts/eval'))
sys.path.insert(0, str(IM2P))
from end_to_end import main
from sim.tests.cycle.test_execution_cli import fixture_files


class StreamingPipelineTests(unittest.TestCase):
    def _arguments(self, root: Path, *mode: str) -> list[str]:
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
        arguments = ['--output', str(root / 'output'), 'reconstruct', '--im2p', str(IM2P),
                     '--lifecycle', str(root / 'lifecycle.json'), '--application', str(application),
                     *mode, '--diagnostic-phase-table', str(root / 'phase.json'),
                     '--diagnostic-frequency-hz', '1000000000']
        for name in ('full-cpu-log', 'full-cpu-graph', 'full-cpu-provenance', 'potal-log',
                     'potal-graph', 'potal-provenance', 'npu-trace', 'library',
                     'cycle-certificate', 'run-aware-certificate'):
            arguments.extend(('--' + name, str(root / 'summary.json')))
        return arguments

    def _invoke(self, root: Path, *mode: str, certified: bool = False) -> int:
        arguments = self._arguments(root, *mode)
        if certified:
            diagnostic = arguments.index('--diagnostic-phase-table')
            del arguments[diagnostic:diagnostic + 4]
            clock = root / 'clock.json'
            clock.write_text(json.dumps({'selected_frequency_hz': 1000000000}))
            for name, value in (('service-certificate', root / 'summary.json'),
                                ('clock-selection', clock), ('profile', 'fixture'),
                                ('potal-result', root / 'summary.json'), ('timing', root / 'summary.json'),
                                ('initial-scratchpad-half', 0), ('initial-accumulator-half', 0)):
                arguments.extend(('--' + name, str(value)))
        dataset = root / 'dataset.jsonl.gz'
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
            if certified and module == 'sim.cycle.execution_cli' and command[4] == 'schedule':
                Path(command[command.index('--output') + 1]).write_bytes(b'fixture schedule')
                return subprocess.CompletedProcess(command, 0)
            return actual_run(command, **options)

        with patch('offline_pipeline.subprocess.run', side_effect=fixture_upstream), \
             patch.object(sys, 'argv', ['end_to_end.py', *arguments]):
            return main()

    def test_official_sqlite_adapter_and_scheduler_receive_diagnostic_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(self._invoke(root, '--streaming-ir'), 0)
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

    def test_default_reconstruction_uses_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(self._invoke(root), 0)
            result = json.loads((root / 'output/result.json').read_text())
            self.assertEqual(result['execution_ir_format'], 'SQLITE')
            self.assertTrue((root / 'output/execution.sqlite').is_file())
            self.assertTrue((root / 'output/schedule.sqlite').is_file())

    def test_explicit_json_mode_is_for_small_debug_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(self._invoke(root, '--json-ir'), 0)
            result = json.loads((root / 'output/result.json').read_text())
            self.assertEqual(result['execution_ir_format'], 'JSON')
            self.assertTrue((root / 'output/execution-bundle.json').is_file())

    def test_large_json_input_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = self._arguments(root, '--json-ir')
            large = root / 'large.jsonl'
            with large.open('wb') as stream:
                stream.truncate(64 * 1024 * 1024 + 1)
            arguments[arguments.index('--full-cpu-log') + 1] = str(large)
            with patch('offline_pipeline.subprocess.run') as run, \
                 patch.object(sys, 'argv', ['end_to_end.py', *arguments]):
                self.assertEqual(main(), 1)
            run.assert_not_called()
            self.assertEqual(json.loads((root / 'output/failure.json').read_text())['boundary'], 'preflight')
            self.assertFalse((root / 'output/result.json').exists())

    def test_insufficient_disk_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=0)), \
                 patch('offline_pipeline.subprocess.run') as run, \
                 patch.object(sys, 'argv', ['end_to_end.py', *self._arguments(root)]):
                self.assertEqual(main(), 1)
            run.assert_not_called()
            self.assertEqual(json.loads((root / 'output/failure.json').read_text())['boundary'], 'preflight')
            self.assertFalse((root / 'output/result.json').exists())

    def test_timeout_records_failed_boundary_without_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.subprocess.run', side_effect=subprocess.TimeoutExpired(['replay'], 1)), \
                 patch.object(sys, 'argv', ['end_to_end.py', *self._arguments(root)]):
                self.assertEqual(main(), 1)
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(failure['boundary'], 'replay')
            self.assertEqual(failure['status'], 'TIMEOUT')
            self.assertFalse((root / 'output/result.json').exists())

    def test_failed_stage_records_exit_code_without_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.subprocess.run', return_value=subprocess.CompletedProcess(['replay'], 7)), \
                 patch.object(sys, 'argv', ['end_to_end.py', *self._arguments(root)]):
                self.assertEqual(main(), 1)
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(failure['boundary'], 'replay')
            self.assertEqual(failure['exit_code'], 7)
            self.assertFalse((root / 'output/result.json').exists())

    def test_existing_output_directory_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'output'
            output.mkdir()
            (output / 'marker.bin').write_bytes(b'keep this tree')
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(self._invoke(root), 1)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
            self.assertFalse((output / 'failure.json').exists())

    def test_certified_input_io_failure_has_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.load_potal_collection', side_effect=OSError('collection read failed')):
                self.assertEqual(self._invoke(root, certified=True), 1)
            self.assertFalse((root / 'output/result.json').exists())
            self.assertFalse((root / 'output/reconstructed-result.json').exists())
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(failure['boundary'], 'certified-reconstruction')

    def test_certified_endpoint_sqlite_failure_has_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.load_potal_collection', return_value={}), \
                 patch('offline_pipeline.prefill_dispatches', return_value=[0]), \
                 patch('offline_pipeline.scheduled_application_result', side_effect=sqlite3.OperationalError('read failed')):
                self.assertEqual(self._invoke(root, certified=True), 1)
            self.assertFalse((root / 'output/result.json').exists())
            self.assertFalse((root / 'output/reconstructed-result.json').exists())
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(failure['boundary'], 'certified-reconstruction')

    def test_certified_publication_io_failure_has_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.load_potal_collection', return_value={}), \
                 patch('offline_pipeline.prefill_dispatches', return_value=[0]), \
                 patch('offline_pipeline.scheduled_application_result', return_value={}), \
                 patch('offline_pipeline.reconstructed_row', return_value={'fixture': True}), \
                 patch('offline_pipeline.load_measurement', side_effect=OSError('candidate read failed')):
                self.assertEqual(self._invoke(root, certified=True), 1)
            self.assertFalse((root / 'output/result.json').exists())
            self.assertFalse((root / 'output/reconstructed-result.json').exists())
            failure = json.loads((root / 'output/failure.json').read_text())
            self.assertEqual(failure['boundary'], 'certified-publication')

    def test_semantic_certification_rejection_keeps_rejection_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('offline_pipeline.load_potal_collection', side_effect=ValueError('invalid source binding')):
                self.assertEqual(self._invoke(root, certified=True), 1)
            result = json.loads((root / 'output/result.json').read_text())
            self.assertEqual(result['schedule'], 'NOT_READY_CERTIFICATION_REJECTED')
            self.assertFalse(result['E2E_RECONSTRUCTION_READY'])
            self.assertFalse((root / 'output/reconstructed-result.json').exists())


if __name__ == '__main__':
    unittest.main()
