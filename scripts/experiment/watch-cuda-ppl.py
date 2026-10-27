#!/usr/bin/env python3
"""Read-only CUDA PPL queue/build monitor. Ctrl+C only closes this display."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cuda_ppl_queue', Path(__file__).with_name('cuda-ppl-queue.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def show(output, config):
    plan = read(output / 'plan.json') or read(config)
    status = read(output / 'status.json')
    print('Queue:', status.get('state', 'not started'), status.get('job', ''), status.get('phase', ''))
    if status.get('error'):
        print('ERROR:', status['error'])
    for build in sorted({(ROOT / j['binary']).resolve().parent.parent for j in plan.get('jobs', [])}):
        try:
            log = (build / 'build.log').read_text(errors='replace')
            steps = re.findall(r'^\[(\d+)/(\d+)\] (.*)$', log, re.M)
            verification = (build / 'verification.log').read_text(errors='replace') if (build / 'verification.log').exists() else ''
            label = 'verified' if '100% tests passed' in verification else f'{steps[-1][0]}/{steps[-1][1]}' if steps else 'configuring'
            if 'FAILED:' in log or 'ninja: build stopped' in log:
                label = 'FAILED'
        except OSError:
            label = 'not built'
        print(f'Build {build.name}: {label}')
    print(f'\n{"Experiment":<27} {"A/W":<5} {"Status":<11} {"Chunks":>9} {"PPL":>11} {"Elapsed":>9} {"ETA~":>9}')
    done = 0
    for job in plan.get('jobs', []):
        result = read(output / job['id'] / 'full/result.json')
        state, count, ppl, elapsed, eta = 'WAIT', '-', '-', '--:--:--', '--:--:--'
        if result:
            done += 1
            state, count, ppl = 'DONE', f'{result["chunks"]}/{job["full_chunks"]}', f'{result["ppl"]:.6f}'
            elapsed = duration(result['seconds'])
        elif status.get('smoke_only') and (output / job['id'] / 'smoke/result.json').exists():
            smoke = read(output / job['id'] / 'smoke/result.json')
            state, count, ppl = 'SMOKE DONE', '1/1', f'{smoke["ppl"]:.6f}'
            elapsed = duration(smoke['seconds'])
        elif status.get('job') == job['id'] and status.get('state') == 'failed':
            state = 'FAILED'
        elif status.get('job') == job['id'] and status.get('state') == 'running':
            phase = status['phase']
            directory = output / job['id'] / phase
            execution = read(directory / 'execution.json')
            state = ('SMOKE' if phase == 'smoke' else 'RUNNING') if execution and queue.alive(execution) else 'STOPPED'
            try:
                text = (directory / 'ppl.log').read_text(errors='replace')
            except OSError:
                text = ''
            rows = re.findall(r'^\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$', text, re.M)
            seconds = time.time() - status['started_at']
            elapsed = duration(seconds)
            if rows:
                completed = int(rows[-1][0]) // 512 + 1
                total = 1 if phase == 'smoke' else job['full_chunks']
                count, ppl = f'{completed}/{total}', rows[-1][1] + '*'
                if state != 'STOPPED':
                    eta = duration(seconds / completed * max(0, total - completed))
        print(f'{job["id"]:<27} {job["bits"]}/{job["bits"]:<3} {state:<11} {count:>9} {ppl:>11} {elapsed:>9} {eta:>9}')
    print(f'\nCompleted: {done}/{len(plan.get("jobs", []))} full runs. * = partial cumulative PPL.')
    print('Ctrl+C closes this monitor only. No inference is started by this program.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/cuda-ppl')
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/experiment/cuda-ppl-jobs.example.json')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            if not args.once:
                print('\033[2J\033[H', end='')
            show(args.output.resolve(), args.config)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
