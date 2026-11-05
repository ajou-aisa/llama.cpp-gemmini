#!/usr/bin/env python3
"""PPL with the published block16 NVFP4 formula and FP32 reconstruction/GEMM."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = Path('/home/mojoi/build/potal-nvfp4-f32/libggml-nvfp4-preload.so')
spec = importlib.util.spec_from_file_location('public_nvfp4_queue', Path(__file__).with_name('cuda-ppl-queue.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class Measurement(TypedDict):
    chunks: int
    scored_tokens: int
    nll: float
    ppl: float


def verify(log: Path, chunks: int) -> Measurement:
    family = {584: 'qwen2-0.5B', 564: 'llama3.2-1B'}[chunks]
    checked = queue.verify(log, dict(kind='rtn', family=family, bits=4), chunks)
    text = log.read_text(errors='replace')
    queue.require(len(re.findall(r'^NVFP4_PROOF\b', text, re.M)) == 1, 'Missing/duplicate NVFP4 proof')
    entry = re.search(r'^NVFP4_PROOF (\{[^{}]*\})(?=\r?$)', text, re.M)
    queue.require(entry is not None, 'Malformed NVFP4 proof')
    proof = json.loads(entry[1].replace('\n', '').replace('\r', ''))
    dense = (169 if family == 'qwen2-0.5B' else 113) * chunks
    attention = (48 if family == 'qwen2-0.5B' else 32) * chunks
    expected = dict(schema='nvfp4-fake-v1', complete=True, format='NVFP4', weight_bits=4,
                    activation_bits=4, attention_bits=16, block_size=16, scale_format='E4M3FN',
                    tensor_scale='FP32', rounding='RNE', global_scaling='dynamic_tensor_amax',
                    weight_scaling='tensor_amax_once', gemm='dequant_f32_fp32_pedantic',
                    native_fp4=False, zero_tensors=0, dense_gpu_calls=dense,
                    verified_dense_calls=dense, attempted_dense_calls=dense, attention_gpu_calls=attention,
                    embedding='original_fp16', lm_head='NVFP4')
    queue.require(all(proof.get(key) == value for key, value in expected.items()), 'Wrong public-formula FP32 profile')
    return Measurement(chunks=checked['chunks'], scored_tokens=checked['scored_tokens'],
                       nll=checked['nll'], ppl=checked['ppl'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = queue.read(ROOT / 'scripts/experiment/cuda-ppl-jobs.rtx3070.json')
    jobs = [dict(job, preload=str(LIBRARY)) for job in config['jobs'] if job['kind'] == 'nvfp4']
    queue.require(len(jobs) == 2, 'Expected both NVFP4 model jobs')
    status = dict(schema='public-nvfp4-fp32-v1', state='waiting', pid=os.getpid(),
                  boot_id=queue.boot_id(), start_ticks=queue.start_ticks(os.getpid()),
                  runs={name: dict(state='WAIT', chunks=job['full_chunks'])
                        for name, job in zip(('Qwen', 'Llama'), jobs)})
    queue.write(output / 'status.json', status)
    with (output.parent / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, job in zip(('Qwen', 'Llama'), jobs):
            directory = output / name.lower()
            directory.mkdir(exist_ok=True)
            dataset = (ROOT / config['dataset']).resolve()
            entry = status['runs'][name]
            status['state'] = 'running'
            entry.update(state='CHECKING', log=str(directory / 'ppl.log'))
            queue.write(output / 'status.json', status)
            identity = queue.inputs(job, dataset)
            manifest = queue.settings(job, dataset, 'full', identity)
            manifest.update(driver_sha256=queue.digest(Path(__file__)),
                            formula_reference=queue.read(ROOT / 'ggml/src/ggml-cuda/nvfp4-reference/reference-identity.json'),
                            reconstruction='FP32', gemm='FP32_PEDANTIC', additional_fp16_rounding=False)
            queue.write(directory / 'manifest.json', manifest)
            env = {key: value for key, value in os.environ.items()
                   if key != 'LD_PRELOAD' and not key.startswith(('GGML_', 'GEMMINI_', 'OMP_', 'KMP_'))}
            env.update(manifest['environment'])
            started = time.monotonic()
            with (directory / 'ppl.log').open('w') as stream:
                with subprocess.Popen(manifest['command'], env=env, cwd=ROOT,
                                      stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)) as child:
                    entry.update(state='RUNNING', pid=child.pid, boot_id=queue.boot_id(),
                                 start_ticks=queue.start_ticks(child.pid))
                    queue.write(output / 'status.json', status)
                    code = child.wait()
            (directory / 'exit-status.txt').write_text(str(code) + '\n')
            try:
                queue.require(code == 0, f'PPL process failed: {directory}')
                checked = verify(directory / 'ppl.log', job['full_chunks'])
                queue.require(queue.inputs(job, dataset) == identity, 'Inputs changed during PPL')
            except RuntimeError:
                entry['state'] = 'FAILED'
                status['state'] = 'failed'
                queue.write(output / 'status.json', status)
                raise
            queue.write(directory / 'result.json', dict(**checked, seconds=time.monotonic() - started))
            entry.update(state='DONE', **checked)
            queue.write(output / 'status.json', status)
            print(f'{name}: {checked}', flush=True)
        status['state'] = 'complete'
        queue.write(output / 'status.json', status)


if __name__ == '__main__':
    main()
