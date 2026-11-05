#!/usr/bin/env python3
"""Run the stock llama.cpp NVFP4 kernels and PPL policy on both base models."""
import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
BIN = Path('/home/mojoi/src/llama.cpp/build-cuda/bin')
UPSTREAM = BIN.parents[1]


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    folder: str
    stem: str
    chunks: int


class NativeResult(TypedDict):
    chunks: int
    ppl: float
    standard_error: float


class RunState(TypedDict, total=False):
    state: str
    chunks: int
    log: str
    pid: int
    boot_id: str
    start_ticks: str
    ppl: float
    standard_error: float


class Status(TypedDict):
    schema: str
    state: str
    pid: int
    boot_id: str
    start_ticks: str
    runs: dict[str, RunState]


class Manifest(TypedDict, total=False):
    source_commit: str
    quantize: list[str]
    command: list[str]
    source_sha256: str
    dataset_sha256: str
    binaries: dict[str, str]
    policy: str
    context: int
    scored_tokens_per_chunk: int
    activation_policy: str
    model_sha256: str


class Result(NativeResult):
    seconds: float
    policy: str


MODELS = (Model('Qwen', 'qwen_base', 'qwen2-0.5B', 584),
          Model('Llama', 'llama_base', 'llama3.2-1B', 564))


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Status | Manifest | Result) -> None:
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(log: Path, chunks: int) -> NativeResult:
    text = log.read_text(errors='replace')
    planned = re.findall(r'calculating perplexity over (\d+) chunks', text)
    rows = re.findall(r'^\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$', text, re.M)
    final = re.findall(r'Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)', text)
    if planned != [str(chunks)] or [int(r[0]) for r in rows] != list(range(0, chunks * 512, 512)) or len(final) != 1:
        raise RuntimeError('Incomplete stock llama.cpp full-corpus PPL log')
    ppl, error = map(float, final[0])
    if not math.isfinite(ppl) or ppl <= 0 or not math.isfinite(error) or float(rows[-1][1]) != ppl:
        raise RuntimeError('Invalid native PPL totals')
    offload = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
    if not offload or any(int(n) == 0 or n != total for n, total in offload) or 'CUDA0' not in text:
        raise RuntimeError('Missing CUDA model-offload evidence')
    return NativeResult(chunks=chunks, ppl=ppl, standard_error=error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status: Status = dict(schema='upstream-nvfp4-ppl-v1', state='waiting', pid=os.getpid(),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                        runs={m.name: dict(state='WAIT', chunks=m.chunks) for m in MODELS})
    write(output / 'status.json', status)
    # Share the original queue lock; an extra measurement must not overlap its GPU work.
    with (output.parent / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = {k: v for k, v in os.environ.items()
               if k != 'LD_PRELOAD' and not k.startswith(('GGML_', 'GEMMINI_', 'OMP_', 'KMP_'))}
        env['OMP_NUM_THREADS'] = '4'
        source_commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
        for model in MODELS:
            directory = output / model.name.lower()
            directory.mkdir(exist_ok=True)
            source = ROOT / 'models' / model.folder / (model.stem + '.fp16.gguf')
            target = ROOT / 'models' / model.folder / (model.stem + '.NVFP4_LLAMA_POLICY.gguf')
            command = [str(BIN / 'llama-perplexity'), '--model', str(target), '--file',
                       str(ROOT / 'wikitext-2-raw/wiki.test.raw'), '--chunks', '-1',
                       '--ctx-size', '512', '--batch-size', '512', '--ubatch-size', '512',
                       '--threads', '4', '--threads-batch', '4', '--gpu-layers', '99',
                       '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--seed', '42',
                       '--no-warmup', '--ppl-stride', '0', '--ppl-output-type', '1', '--verbosity', '5']
            quantize = [str(BIN / 'llama-quantize'), '--tensor-type', 'blk.*=nvfp4',
                        str(source), str(target), 'Q4_0', '4']
            # This upstream revision exposes NVFP4 via --tensor-type, not the ftype CLI.
            # Q4_0 supplies stock tied embedding/output selection; only body types change.
            manifest: Manifest = dict(source_commit=source_commit, quantize=quantize, command=command,
                            source_sha256=digest(source), dataset_sha256=digest(ROOT / 'wikitext-2-raw/wiki.test.raw'),
                            binaries={p.name: digest(p) for p in (BIN / 'llama-quantize', BIN / 'llama-perplexity')},
                            policy='stock NVFP4 body encoder, stock embedding/output policy, native CUDA auto precision',
                            context=512, scored_tokens_per_chunk=255, activation_policy='upstream auto; Q8_1 on SM86')
            write(directory / 'manifest.json', manifest)
            entry = status['runs'][model.name]
            entry.update(state='QUANTIZING', log=str(directory / 'ppl.log'))
            status['state'] = 'running'
            write(output / 'status.json', status)
            with (directory / 'quantize.log').open('w') as stream:
                subprocess.run(quantize, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, pass_fds=(lock.fileno(),))
            manifest['model_sha256'] = digest(target)
            write(directory / 'manifest.json', manifest)
            entry['state'] = 'RUNNING'
            started = time.monotonic()
            with (directory / 'ppl.log').open('w') as stream:
                with subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      pass_fds=(lock.fileno(),)) as child:
                    entry['pid'] = child.pid
                    entry['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                    entry['start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                    write(output / 'status.json', status)
                    code = child.wait()
            (directory / 'exit-status.txt').write_text(str(code) + '\n')
            if code != 0:
                entry['state'] = 'FAILED'
                write(output / 'status.json', status)
                raise RuntimeError(f'Native llama.cpp exited {code}: {directory}')
            try:
                checked = verify(directory / 'ppl.log', model.chunks)
            except RuntimeError:
                entry['state'] = 'FAILED'
                status['state'] = 'failed'
                write(output / 'status.json', status)
                raise
            if digest(target) != manifest['model_sha256']:
                raise RuntimeError('Quantized model changed during inference')
            result: Result = dict(**checked, seconds=time.monotonic() - started, policy=manifest['policy'])
            write(directory / 'result.json', result)
            entry.update(state='DONE', **checked)
            write(output / 'status.json', status)
            print(f'{model.name}: {checked}', flush=True)
        status['state'] = 'complete'
        write(output / 'status.json', status)


if __name__ == '__main__':
    main()
