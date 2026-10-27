#!/usr/bin/env python3
"""Portable sequential CUDA PPL queue, adapted from the Nano transfer runner.

Only --run starts inference. Each job runs one smoke chunk before the full corpus.
Completed runs are verified and reused; interrupted runs restart from chunk zero.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    pending.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def start_ticks(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def alive(record):
    return (record.get('boot_id') == boot_id() and record.get('start_ticks') is not None
            and start_ticks(record['pid']) == record['start_ticks'])


def verify(log, job, chunks):
    text = log.read_text(errors='replace')
    totals = re.findall(r'^PPL_TOTALS (.*)$', text, re.M)
    planned = re.findall(r'calculating perplexity over (\d+) chunks', text)
    require(len(totals) == len(planned) == 1, 'Missing final PPL totals/planned chunks')
    row = dict(field.split('=', 1) for field in totals[0].split())
    count, tokens = int(row['chunks']), int(row['total_scored_tokens'])
    nll, ppl = float(row['total_nll']), float(row['corpus_ppl'])
    per_chunk = [dict(field.split('=', 1) for field in entry.split())
                 for entry in re.findall(r'\bPPL_CHUNK ([^\r\n]*)', text)]
    require(count == chunks == int(planned[0]), 'Incomplete corpus or wrong dataset/tokenizer')
    require([int(c['index']) for c in per_chunk] == list(range(count)), 'Missing/duplicate chunks')
    require(all(int(c['scored_tokens']) == 255 and math.isfinite(float(c['nll'])) for c in per_chunk),
            'Wrong scored-token population')
    require(tokens == count * 255 and math.isfinite(nll) and math.isfinite(ppl) and ppl > 0,
            'Invalid NLL/PPL')
    require(math.isclose(sum(float(c['nll']) for c in per_chunk), nll, rel_tol=1e-10)
            and math.isclose(math.log(ppl), nll / tokens, rel_tol=1e-10), 'NLL/PPL mismatch')
    layers = {'qwen2-0.5B': 24, 'llama3.2-1B': 16}[job['family']]
    dense, attention = (7 * layers + 1) * count, 2 * layers * count

    def proof(prefix):
        entries = re.findall(r'^' + prefix + r' (.*)$', text, re.M)
        require(len(entries) == 1, 'Missing/duplicate ' + prefix)
        return json.loads(entries[0])

    result = dict(chunks=count, scored_tokens=tokens, nll=nll, ppl=ppl)
    if job['kind'] == 'potal':
        p = proof('CUDA_CPU_EXACT_PROOF')
        expected = dict(schema='cuda-cpu-exact-ppl', version=3, complete=True,
                        graph='cpu_main_equivalent_cuda', compute='INT', activation='EXSIA',
                        bits=job['bits'], dim=32, scored_tokens=tokens, activation_fp16=False,
                        float_gpu_calls=0, cpu_q6_head_matmuls=0, gpu_q6_head_matmuls=0,
                        observed_matmuls=dense, verified_matmuls=dense,
                        attention_backend='cuda_rtn_baseline', attention_gpu_calls=attention,
                        observed_attention_matmuls=attention, verified_attention_matmuls=attention)
        require(all(p.get(k) == v for k, v in expected.items()), 'Wrong PoTal profile/CUDA coverage')
        require(p['integer_gpu_launches'] > 0 and p['residual_gpu_launches'] > 0, 'Missing main/residual CUDA work')
        s = proof('HP1_SCU_PROOF')
        require(s.get('schema') == 'hp1-scu-v1' and s.get('dim') == 32
                and s.get('final_merge') == 'legacy_fp32_main_plus_residual'
                and s.get('main_launches') == p['integer_gpu_launches']
                and s.get('residual_launches') == p['residual_gpu_launches'], 'Wrong SCU execution')
        for key in ('main_shift_saturations', 'main_acc_saturations',
                    'residual_shift_saturations', 'residual_acc_saturations'):
            require(s.get(key) == 0, 'SCU saturation: ' + key)
        result['scu_proof'] = s
    else:
        p = proof('CUDA_RTN_PROOF')
        expected = dict(complete=True, activation_bits=job['bits'], attention_bits=16,
                        block_size=32, compensation=False, scored_tokens=tokens,
                        observed_matmuls=dense + attention, verified_matmuls=dense + attention,
                        dense_gpu_calls=dense, attention_gpu_calls=attention)
        require(all(p.get(k) == v for k, v in expected.items()), 'Wrong RTN profile/CUDA coverage')
        if job['kind'] == 'rtn-mx':
            h = proof('RTN_WA_HP1_PROOF')
            require(h == dict(activation='BLOCK_RTN', block_size=32, outlier_selection=False,
                              stripe_folding=False, residual=False,
                              q4_hp1_matmuls=dense if job['bits'] == 4 else 0,
                              q8_hp1_matmuls=dense if job['bits'] == 8 else 0), 'Wrong no-compensation baseline')
            result['hp1_proof'] = h
        else:
            require('RTN_WA_HP1_PROOF ' not in text, 'Unexpected HP1 weights in Q4_0/E4M3 job')
    result['cuda_proof'] = p
    return result


def inputs(job, dataset):
    binary = (ROOT / job['binary']).resolve()
    paths = [binary, (ROOT / job['model']).resolve(), dataset]
    paths += sorted(binary.parent.glob('*.so*'))
    cache = binary.parent.parent / 'CMakeCache.txt'
    require(cache.is_file(), 'Missing build configuration: ' + str(cache))
    config = dict(re.findall(r'^([A-Za-z_][A-Za-z0-9_]*):[A-Za-z_]+=(.*)$', cache.read_text(), re.M))
    expected = {'GGML_GEMMINI_ENABLE_A4NKS_ATTENTION': 'OFF'}
    if job['kind'] == 'potal':
        expected.update(GGML_GEMMINI_CUDA_CPU_EXACT='ON', GGML_GEMMINI_CUDA_RTN_ATTENTION='ON',
                        GGML_GEMMINI_ACTIVATION_QUANT='EXSIA', GGML_GEMMINI_ENABLE_RMD='ON',
                        GGML_GEMMINI_ACTIVATION_BITS=str(job['bits']), GGML_GEMMINI_WEIGHT_BITS=str(job['bits']),
                        GGML_GEMMINI_DIM='32', GGML_GEMMINI_BLOCK_SIZE='32',
                        GGML_GEMMINI_EXSIA_SIGMA='2', GGML_GEMMINI_DEFAULT_RMD_BACKEND='CPU',
                        GGML_GEMMINI_DEFAULT_MATMUL_MODE='STRIPE_PIPELINE',
                        GGML_GEMMINI_COMPUTE_TYPE='INT', GGML_GEMMINI_DEQUANT_FP_TEST='OFF',
                        GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=str(job['anchor']))
    else:
        expected.update(GGML_GEMMINI_CUDA_RTN='ON', GGML_GEMMINI_ENABLE_RMD='OFF')
    require(all(config.get(k) == v for k, v in expected.items()), 'Wrong build profile: ' + job['id'])
    paths.append(cache)
    for path in paths:
        require(path.is_file(), 'Missing input: ' + str(path))
    require(os.access(binary, os.X_OK), 'Binary is not executable: ' + str(binary))
    return {str(path): digest(path) for path in paths}


def settings(job, dataset, phase, identity):
    binary = (ROOT / job['binary']).resolve()
    command = [str(binary), '--model', str((ROOT / job['model']).resolve()), '--file', str(dataset),
               '--chunks', '1' if phase == 'smoke' else '-1', '--ctx-size', '512',
               '--batch-size', '512', '--ubatch-size', '512', '--threads', '4', '--threads-batch', '4',
               '--gpu-layers', '0', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
               '--seed', '42', '--no-warmup', '--ppl-stride', '0', '--ppl-output-type', '1']
    env = dict(OMP_NUM_THREADS='4', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0',
               LD_LIBRARY_PATH=str(binary.parent), GGML_CUDA_RTN_A_BITS=str(job['bits']))
    if job['kind'] == 'potal':
        env.update(GGML_GEMMINI_CUDA_CPU_EXACT='1', GGML_GEMMINI_HP1_SCU='1',
                   GGML_GEMMINI_CUDA_CPU_EXACT_Q6_HEAD='0', GGML_CUDA_RTN_A_BITS='16')
    return dict(schema='mxint-cuda-ppl-v1', job=job, phase=phase, command=command,
                environment=env, input_sha256=identity, runner_sha256=digest(Path(__file__)))


def run_case(job, dataset, output, phase, identity):
    directory = output / job['id'] / phase
    expected = settings(job, dataset, phase, identity)
    chunks = 1 if phase == 'smoke' else job['full_chunks']
    if (directory / 'result.json').exists():
        require(read(directory / 'manifest.json') == expected, 'Completed experiment settings changed')
        require((directory / 'exit-status.txt').read_text().strip() == '0', 'Prior run failed')
        checked = verify(directory / 'ppl.log', job, chunks)
        require(all(read(directory / 'result.json').get(k) == v for k, v in checked.items()), 'Result differs from log')
        print('Verified completed:', job['id'], phase, flush=True)
        return
    if directory.exists():
        previous = directory / 'execution.json'
        require(not previous.exists() or not alive(read(previous)), 'Previous inference is still alive')
        directory.rename(directory.with_name(f'{phase}.interrupted-{time.time_ns()}'))
    directory.mkdir(parents=True)
    write(directory / 'manifest.json', expected)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'GEMMINI_', 'OMP_', 'KMP_'))}
    env.update(expected['environment'])
    started = time.time()
    write(output / 'status.json', dict(state='running', job=job['id'], phase=phase, pid=os.getpid(),
                                     start_ticks=start_ticks(os.getpid()), boot_id=boot_id(), started_at=started))
    print('Starting:', job['id'], phase, flush=True)
    with (directory / 'ppl.log').open('w') as log:
        child = subprocess.Popen(expected['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(directory / 'execution.json', dict(pid=child.pid, start_ticks=start_ticks(child.pid), boot_id=boot_id(),
                                                started_at=started, command=expected['command']))
        code = child.wait()
    elapsed = time.time() - started
    (directory / 'exit-status.txt').write_text(str(code) + '\n')
    require(code == 0, f'Inference failed ({code}): {directory / "ppl.log"}')
    require(inputs(job, dataset) == identity, 'Inputs changed during inference')
    checked = verify(directory / 'ppl.log', job, chunks)
    write(directory / 'result.json', dict(id=job['id'], family=job['family'], phase=phase,
          activation_bits=job['bits'], weight_bits=job['bits'], compensation=job['kind'] == 'potal',
          anchor=job.get('anchor'), seconds=elapsed, **checked))
    with (output / 'results.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'family', 'activation_bits', 'weight_bits',
                                'chunks', 'ppl', 'nll', 'seconds'], delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(read(p) for p in sorted(output.glob('*/full/result.json')))
    print(f'Completed {job["id"]} {phase}: {checked["chunks"]} chunks, PPL={checked["ppl"]:.9f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/experiment/cuda-ppl-jobs.example.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/cuda-ppl')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list', action='store_true')
    mode.add_argument('--check', action='store_true', help='Verify files/build profiles without inference')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append', help='Select job ID; repeat to select several in configured order')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--build', action='store_true', help='Build/test each selected runtime before inference')
    args = parser.parse_args()
    config = read(args.config)
    ids = [job['id'] for job in config['jobs']]
    require(len(ids) == len(set(ids)) and all(re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids), 'Invalid/duplicate job IDs')
    require(not args.case or set(args.case) <= set(ids), 'Unknown --case')
    jobs = [j for j in config['jobs'] if not args.case or j['id'] in args.case]
    require(jobs, 'Empty queue')
    for job in jobs:
        require(job['kind'] in ('potal', 'rtn-mx', 'rtn-e4m3') and job['bits'] in (4, 8), 'Unsupported job')
        require(job['full_chunks'] > 0 and job['family'] in ('qwen2-0.5B', 'llama3.2-1B'), 'Invalid corpus/model')
        if job['kind'] == 'potal':
            require(job['anchor'] in (2, 3), 'Invalid stripe anchor')
    if not args.run and not args.check:
        for job in jobs:
            print(job['id'], f'A{job["bits"]}W{job["bits"]}', f'{job["full_chunks"]} chunks', job['binary'])
        return
    require(not args.build or args.run, '--build requires --run')
    dataset = (ROOT / config['dataset']).resolve()
    if args.check:
        for job in jobs:
            inputs(job, dataset)
            print('PASS inputs/build:', job['id'])
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write(output / 'plan.json', dict(dataset=str(dataset), jobs=jobs))
            built = set()
            for job in jobs:
                build = (ROOT / job['binary']).resolve().parent.parent
                if args.build and build not in built:
                    write(output / 'status.json', dict(state='building', job=job['id'], pid=os.getpid(),
                                                      log=str(build / 'build.log')))
                    command = ['bash', str(ROOT / 'scripts/experiment/build-potal-cuda.sh'), '--build-dir', str(build)]
                    command += ['--bits', str(job['bits']), '--anchor', str(job['anchor'])] if job['kind'] == 'potal' else ['--rtn']
                    subprocess.run(command, cwd=ROOT, check=True)
                    built.add(build)
                write(output / 'status.json', dict(state='checking_inputs', job=job['id'], pid=os.getpid()))
                identity = inputs(job, dataset)
                run_case(job, dataset, output, 'smoke', identity)
                if not args.smoke_only:
                    run_case(job, dataset, output, 'full', identity)
            write(output / 'status.json', dict(state='complete', completed_jobs=[j['id'] for j in jobs],
                                              smoke_only=args.smoke_only, finished_at=time.time()))
        except BaseException as error:
            previous = read(output / 'status.json') if (output / 'status.json').exists() else {}
            write(output / 'status.json', dict(state='failed', job=previous.get('job'),
                                              phase=previous.get('phase'), error=repr(error), failed_at=time.time()))
            raise


if __name__ == '__main__':
    main()
