#!/usr/bin/env python3
"""CPU-only integration checks using actual queue and monitor CLI subprocesses."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
QUEUE = HERE / 'cuda-ppl-queue.py'
MONITOR = HERE / 'watch-cuda-ppl.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--proof-log', type=Path, help='Captured Qwen TOP2 A8 two-chunk log regression')
    args = parser.parse_args()
    args.receipts.mkdir(parents=True, exist_ok=True)
    receipts = []
    if args.proof_log:
        spec = importlib.util.spec_from_file_location('queue_under_test', QUEUE)
        queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(queue)
        job = dict(family='qwen2-0.5B', kind='potal', bits=8)
        result = queue.verify(args.proof_log, job, 2)
        assert result['nll'] == 1265.5980171565611 and result['scored_tokens'] == 510
        original = args.proof_log.read_text()
        with tempfile.TemporaryDirectory(prefix='potal-proof-qa-') as temp:
            log = Path(temp) / 'mutated.log'
            log.write_bytes(original.replace('cpu_q6_head_matmuls\n', 'cpu_q6_head_matmuls\r\n\r\n').encode())
            assert queue.verify(log, job, 2) == result
            variants = {
                'missing': original.replace('CUDA_CPU_EXACT_PROOF ', 'REMOVED_PROOF '),
                'duplicate': original + '\nCUDA_CPU_EXACT_PROOF {}\n',
                'truncated_duplicate': original + '\nCUDA_CPU_EXACT_PROOF {',
                'malformed_duplicate': original + '\nCUDA_CPU_EXACT_PROOF invalid\n',
                'malformed_single': original.replace('\"complete\":true', '\"complete\":invalid'),
                'wrong_count': original.replace('\"verified_matmuls\":338', '\"verified_matmuls\":337'),
                'injected_text': original.replace('cpu_q6_head_matmuls\n', 'cpu_q6_head_matmuls\nLOG_INTERLEAVING\n'),
            }
            for label, text in variants.items():
                assert text != original, label
                log.write_text(text)
                try:
                    queue.verify(log, job, 2)
                except (RuntimeError, ValueError):
                    pass
                else:
                    raise AssertionError('Accepted ' + label + ' proof')
        (args.receipts / 'captured-proof-result.json').write_text(json.dumps(result, indent=2) + '\n')
        receipts.append('PASS captured split-key proof exact NLL; missing/duplicate/truncated/malformed/wrong-count/text-injection rejected')
    machine = json.loads((HERE / 'cuda-ppl-jobs.rtx3070.json').read_text())
    kinds = [j['kind'] for j in machine['jobs']]
    assert kinds == ['fp16'] * 2 + ['rtn'] * 8 + ['rtn-mx'] * 4 + ['potal'] * 8 + ['nvfp4'] * 2
    assert [j['anchor'] for j in machine['jobs'] if j['kind'] == 'potal'] == [2]*4 + [3]*4
    assert [(j['weight_bits'], j['bits']) for j in machine['jobs'][2:10]] == [(4,16)]*2 + [(8,16)]*2 + [(4,4)]*2 + [(8,8)]*2
    assert len(set(j['id'] for j in machine['jobs'])) == 24
    receipts.append('PASS exact 24 unique job order, TOP2/TOP3 and RTN-W/WA bit widths')
    with tempfile.TemporaryDirectory(prefix='potal-queue-qa-') as temp:
        base = Path(temp)
        binary = base / 'build/bin/llama-perplexity'
        binary.parent.mkdir(parents=True)
        binary.write_text("""#!/usr/bin/env python3
import json, math, os, sys, time
from pathlib import Path
if Path(__file__).with_name('slow').exists(): time.sleep(30)
print('calculating perplexity over 1 chunks', flush=True)
print('PPL_CHUNK index=0 scored_tokens=255 nll=255', flush=True)
print('PPL_TOTALS chunks=1 total_scored_tokens=255 total_nll=255 corpus_ppl='+repr(math.e), flush=True)
a=int(os.environ['GGML_CUDA_RTN_A_BITS'])
print('CUDA_RTN_PROOF '+json.dumps(dict(complete=True, activation_bits=a, attention_bits=16, block_size=32, compensation=False, scored_tokens=255, observed_matmuls=217, verified_matmuls=217, dense_gpu_calls=169, attention_gpu_calls=48)), flush=True)
""")
        binary.chmod(0o755)
        (base / 'build/CMakeCache.txt').write_text('GGML_GEMMINI_ENABLE_A4NKS_ATTENTION:BOOL=OFF\nGGML_CUDA:BOOL=OFF\nGGML_GEMMINI_CUDA_RTN:BOOL=ON\nGGML_GEMMINI_ENABLE_RMD:BOOL=OFF\n')
        model, dataset = base / 'model.gguf', base / 'dataset.txt'
        model.write_text('fixture model')
        dataset.write_text('fixture corpus')
        config = base / 'jobs.json'
        job = dict(id='fixture', family='qwen2-0.5B', kind='rtn', bits=16, weight_bits=4,
                   full_chunks=1, binary=str(binary), model=str(model))
        config.write_text(json.dumps(dict(dataset=str(dataset), jobs=[job])))
        def command(output, *extra):
            return [sys.executable, str(QUEUE), '--config', str(config), '--output', str(output), *extra]
        def run(output, *extra, ok=True):
            result = subprocess.run(command(output, *extra), capture_output=True, text=True)
            assert (result.returncode == 0) == ok, result.stdout + result.stderr
            return result
        def wait_for(path):
            for _ in range(200):
                if path.exists(): return json.loads(path.read_text())
                time.sleep(.025)
            raise AssertionError('Timed out waiting for ' + str(path))
        output = base / 'completed'
        run(output, '--run')
        result = json.loads((output / 'fixture/full/result.json').read_text())
        assert result['activation_bits'] == 16 and result['weight_bits'] == 4
        assert 'Verified completed' in run(output, '--run').stdout
        model.write_text('changed fixture model')
        assert 'settings changed' in run(output, '--run', ok=False).stderr
        model.write_text('fixture model')
        run(output, '--run')
        receipts.append('PASS real smoke/full subprocess, RTN-W A16/W4 results, verified reuse, stale input rejection')
        snapshot = subprocess.run([sys.executable, str(MONITOR), '--output', str(output), '--config', str(config), '--json'], capture_output=True, text=True, check=True)
        assert json.loads(snapshot.stdout)['status']['state'] == 'complete'
        (args.receipts / 'monitor-complete.json').write_text(snapshot.stdout)
        full_log = output / 'fixture/full/ppl.log'
        original_log = full_log.read_text()
        full_log.write_text(original_log.replace('"complete": true', '"complete": false'))
        invalid = subprocess.run([sys.executable, str(MONITOR), '--output', str(output), '--config', str(config), '--json'], capture_output=True, text=True, check=True)
        assert 'INVALID' in json.loads(invalid.stdout)['display']
        full_log.write_text(original_log)
        receipts.append('PASS monitor refuses DONE when full GPU coverage proof is incomplete')
        config.write_text(config.read_text() + '\n')
        assert 'configuration changed' in run(output, '--run', ok=False).stderr
        receipts.append('PASS changed configuration hash rejected before reuse')
        config.write_text(json.dumps(dict(dataset=str(dataset), jobs=[job])))
        binary.with_name('slow').touch()
        active = base / 'active'
        process = subprocess.Popen(command(active, '--run'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        record = wait_for(active / 'fixture/smoke/execution.json')
        assert 'BlockingIOError' in run(active, '--run', ok=False).stderr
        live_log = active / 'fixture/smoke/ppl.log'
        live_log.write_text('PPL_CHUNK invalid\nPPL_CHUNK index=0 scored_tokens=255 nll=')
        snapshot = subprocess.run([sys.executable, str(MONITOR), '--output', str(active), '--config', str(config), '--json'], capture_output=True, text=True, check=True)
        assert 'SMOKE' in json.loads(snapshot.stdout)['display']
        (args.receipts / 'monitor-active.json').write_text(snapshot.stdout)
        live_log.write_text('')
        receipts.append('PASS live monitor tolerates malformed complete line and partial appended chunk')
        process.terminate()
        process.wait(timeout=20)
        assert not Path('/proc', str(record['pid'])).exists(), 'TERM left inference alive'
        binary.with_name('slow').unlink()
        run(active, '--run')
        assert list((active / 'fixture').glob('smoke.interrupted-*'))
        receipts.append('PASS live PID lock, monitor live subprocess, TERM child cleanup, interrupted archive/restart')
        binary.with_name('slow').touch()
        orphan = base / 'orphan'
        process = subprocess.Popen(command(orphan, '--run'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        record = wait_for(orphan / 'fixture/smoke/execution.json')
        process.kill()
        process.wait()
        refused = run(orphan, '--run', ok=False).stderr
        assert 'BlockingIOError' in refused or 'Previous inference is still alive' in refused
        os.kill(record['pid'], signal.SIGTERM)
        for _ in range(200):
            stat = Path('/proc', str(record['pid']), 'stat')
            if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
            time.sleep(.025)
        else:
            raise AssertionError('Orphan fixture remains alive')
        binary.with_name('slow').unlink()
        receipts.append('PASS SIGKILL orphan inference blocks queue restart')
        launched = base / 'systemd'
        launch = subprocess.run(['bash', str(HERE / 'start-cuda-ppl.sh'), '--config', str(config), '--output', str(launched)], capture_output=True, text=True, timeout=30)
        assert launch.returncode == 0, launch.stdout + launch.stderr
        for _ in range(400):
            status = launched / 'status.json'
            if status.exists() and json.loads(status.read_text()).get('state') == 'complete':
                break
            time.sleep(.025)
        else:
            raise AssertionError('systemd fixture did not complete: ' + (launched / 'queue.log').read_text())
        (args.receipts / 'launcher-systemd.txt').write_text(launch.stdout + launch.stderr + (launched / 'queue.log').read_text())
        receipts.append('PASS actual systemd launcher smoke/full with persistent log')
        windows_output = base / 'windows-anchor'
        windows_script = subprocess.check_output(['wslpath', '-w', str(HERE / 'start-cuda-ppl.ps1')], text=True).strip()
        anchor = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windows_script, '-Config', str(config), '-Output', str(windows_output)], capture_output=True, text=True, errors='replace', timeout=30)
        assert anchor.returncode == 0, anchor.stdout + anchor.stderr
        for _ in range(600):
            status = windows_output / 'status.json'
            if status.exists() and json.loads(status.read_text()).get('state') == 'complete':
                break
            time.sleep(.025)
        else:
            raise AssertionError('Windows hidden WSL anchor did not complete')
        (args.receipts / 'launcher-windows-anchor.txt').write_text(anchor.stdout + anchor.stderr + (windows_output / 'queue.log').read_text())
        receipts.append('PASS actual Windows hidden WSL anchor starts systemd queue after launcher exits')
        config.write_text(json.dumps(dict(dataset=str(dataset), jobs=[dict(job, bits=3)])))
        assert 'Unsupported job' in run(base / 'bad', '--list', ok=False).stderr
        config.write_text(json.dumps(dict(dataset=str(dataset), jobs=[dict(job, id='../../escape') ])))
        assert 'Invalid/duplicate' in run(base / 'bad', '--list', ok=False).stderr
        receipts.append('PASS malformed activation bits and path-like job IDs rejected')
    (args.receipts / 'summary.txt').write_text('\n'.join(receipts) + '\nTemporary fixture directory cleaned. No GPU inference was used.\n')
    print('\n'.join(receipts))


if __name__ == '__main__':
    main()
