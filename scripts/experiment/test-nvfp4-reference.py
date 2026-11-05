#!/usr/bin/env python3
"""CPU comparisons with pinned NVIDIA Model-Optimizer; GPU checks are opt-in."""
import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = 'modelopt/torch/quantization/qtensor/nvfp4_tensor.py'
UPSTREAM_COMMIT = 'ff336e72129f5b2bae47fb2e3ff35112f1bdbc8d'
UPSTREAM_SHA256 = '05d9749aa3d2c8329367643df5cd783ba5869c458dc4806b4842059d0baf1f24'


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    commit = UPSTREAM_COMMIT
    url = f'https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/{commit}/{UPSTREAM_PATH}'
    data = urllib.request.urlopen(url).read()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_SHA256, 'Pinned upstream digest mismatch'
    (directory / 'nvidia-nvfp4_tensor.py').write_bytes(data)
    metadata = dict(repository='NVIDIA/Model-Optimizer', commit=commit, url=url,
                    sha256=hashlib.sha256(data).hexdigest(), license='Apache-2.0')
    (directory / 'upstream-identity.json').write_text(json.dumps(metadata, indent=2) + '\n')


def official_reference(directory, torch):
    data = (directory / 'nvidia-nvfp4_tensor.py').read_bytes()
    metadata = json.loads((directory / 'upstream-identity.json').read_text())
    assert metadata['commit'] == UPSTREAM_COMMIT
    assert hashlib.sha256(data).hexdigest() == metadata['sha256'] == UPSTREAM_SHA256, 'Upstream reference changed'
    # Execute upstream quantize/dequantize verbatim; replace only package integration helpers.
    class Storage:
        def __init__(self, shape, dtype, packed):
            self.metadata = dict(shape=shape, dtype=dtype)
            self._quantized_data = packed
    def block_amax(x, block_sizes):
        return x.reshape(*x.shape[:-1], -1, block_sizes[-1]).abs().amax(-1)
    namespace = dict(torch=torch, E2M1_MAX=6, E4M3_MAX=448, BaseQuantizedTensor=Storage,
                     reduce_amax=lambda x: x.abs().amax(), reduce_block_amax=block_amax,
                     reduce_block_padding=lambda x, block_sizes: x)
    tree = ast.parse(data)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('e2m1_bounds', 'e2m1_values') for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == '_cast_per_block_scale_to_fp8':
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'NVFP4QTensor':
            selected.append(node)
    assert len(selected) == 4
    exec(compile(ast.Module(body=selected, type_ignores=[]), metadata['url'], 'exec'), namespace)
    return namespace['NVFP4QTensor'], metadata


def configure(library):
    fptr, uptr = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
    library.nvfp4_ref_code.argtypes = [ctypes.c_float]
    library.nvfp4_ref_scale_code.argtypes = [ctypes.c_float]
    library.nvfp4_ref_scale_value.argtypes = [ctypes.c_uint]
    library.nvfp4_ref_scale_value.restype = ctypes.c_float
    library.nvfp4_ref_quantize.argtypes = [fptr, ctypes.c_uint, uptr, uptr, fptr, fptr]


def check_bindings(binary, shim, directory):
    targets = ('_Z21ggml_cuda_rtn_computeP11ggml_tensor', '_Z25ggml_cuda_rtn_dense_callsv',
               '_Z19ggml_cuda_rtn_clearv', '_Z19ggml_cuda_rtn_errorv')
    host_library = binary.parent / 'libggml-gemmini.so'
    relocations = '\n'.join(subprocess.check_output(['readelf', '-rW', str(p)], text=True)
                            for p in (binary, host_library))
    assert all(any(symbol in line and 'JUMP_SLOT' in line for line in relocations.splitlines())
               for symbol in targets), 'RTN entry points lack interposable PLT relocations'
    dynamic = subprocess.check_output(['readelf', '-dW', str(host_library)], text=True)
    assert '(SYMBOLIC)' not in dynamic
    env = dict(os.environ, GGML_CUDA_NVFP4='1', GGML_CUDA_RTN_A_BITS='4',
               LD_PRELOAD=str(shim), LD_BIND_NOW='1', LD_DEBUG='bindings')
    probe = subprocess.run([str(binary), '--help'], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
    assert probe.returncode == 0, probe.stderr[-4000:]
    lines = [line.strip() for line in probe.stderr.splitlines()
             if 'binding file ' in line and any(symbol in line for symbol in targets)]
    assert all(any(symbol in line and f'to {shim} ' in line
                   and (f'binding file {binary} ' in line or f'binding file {host_library} ' in line)
                   for line in lines) for symbol in targets), 'Loader did not bind all RTN hooks to NVFP4'
    assert 'NVFP4_DEVICE' not in probe.stderr, 'A help-only binding probe unexpectedly initialized the GPU'
    result = dict(complete=True, binary=str(binary), preload=str(shim), gpu_initialized=False,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  preload_sha256=hashlib.sha256(shim.read_bytes()).hexdigest(), bindings=lines)
    (directory / 'elf-binding-report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--reference-dir', type=Path, default=ROOT.parent / 'experiment-tools/nvfp4-qa')
    p.add_argument('--library', type=Path, default=Path('/home/mojoi/build/potal-nvfp4/libnvfp4-reference.so'))
    p.add_argument('--gpu-library', type=Path)
    p.add_argument('--base-library', type=Path)
    p.add_argument('--check-bindings', type=Path, metavar='RTN_BINARY')
    args = p.parse_args()
    if args.fetch:
        fetch(args.reference_dir)
    import torch
    assert not torch.cuda.is_available(), 'Reference environment must be CPU-only'
    official, upstream = official_reference(args.reference_dir, torch)
    lib = ctypes.CDLL(str(args.library)); configure(lib)
    boundaries = torch.tensor([0.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    fp4_vectors = torch.cat([boundaries, torch.nextafter(boundaries, torch.zeros_like(boundaries)),
                             torch.nextafter(boundaries, torch.full_like(boundaries, float('inf'))),
                             torch.tensor([0, -0., .5, 1, 1.5, 2, 3, 4, 6, 100])])
    fp4_vectors = torch.cat([fp4_vectors, -fp4_vectors])
    expected_codes = official._cast_fp4(fp4_vectors.clone()).tolist()
    assert [lib.nvfp4_ref_code(v) for v in fp4_vectors.tolist()] == expected_codes
    scale_values = torch.tensor([lib.nvfp4_ref_scale_value(i) for i in range(127)])
    scale_ties = (scale_values[1:] + scale_values[:-1]) * .5
    scale_vectors = torch.cat([scale_values, scale_ties,
                               torch.nextafter(scale_ties, torch.zeros_like(scale_ties)),
                               torch.nextafter(scale_ties, torch.full_like(scale_ties, float('inf'))),
                               torch.tensor([0, 1e-12, 448, 449, 1e10])])
    expected_scales = scale_vectors.clamp(2**-9, 448).to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    assert [lib.nvfp4_ref_scale_code(v) for v in scale_vectors.tolist()] == expected_scales
    generator = torch.Generator().manual_seed(20261027)
    tensors = [torch.randn(32, 128, generator=generator), torch.randn(4, 128, generator=generator) * 1e-8,
               torch.randn(8, 128, generator=generator) * 1e4,
               torch.cat([torch.zeros(16), torch.full((16,), 2688.), torch.full((16,), 0.0001),
                          torch.full((16,), 0.01)]).reshape(1, -1),
               torch.tensor(([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] + [0.] * 8) * 2).reshape(1, -1)]
    ties = torch.cat([fp4_vectors[fp4_vectors.abs() <= 6], torch.tensor([6.])])
    ties = torch.nn.functional.pad(ties, (0, (-len(ties)) % 16))
    # A separate block pins global scale to 1, so the other blocks exercise true ties.
    tensors.append(torch.cat([torch.full((16,), 2688.), ties]).reshape(1, -1))
    scale_cases = scale_vectors[(scale_vectors >= 0) & (scale_vectors <= 448)]
    tensors.append(torch.cat([torch.tensor([2688.]), scale_cases * 6]).repeat_interleave(16).reshape(1, -1))
    gpu = None
    if args.gpu_library:
        assert args.base_library, '--base-library is required for the GPU shared-library ABI'
        ctypes.CDLL(str(args.base_library), mode=ctypes.RTLD_GLOBAL)
        gpu = ctypes.CDLL(str(args.gpu_library))
        gpu.nvfp4_gpu_reference.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
    comparisons = []
    for index, x in enumerate(tensors):
        n = x.numel(); values = x.flatten().tolist()
        src = (ctypes.c_float * n)(*values); codes = (ctypes.c_uint8 * n)(); scales = (ctypes.c_uint8 * (n // 16))()
        global_scale = ctypes.c_float(); dequant = (ctypes.c_float * n)()
        assert lib.nvfp4_ref_quantize(src, n, codes, scales, ctypes.byref(global_scale), dequant) == 0
        packed, s, g = official.quantize(x.clone(), 16)
        expected = packed.dequantize(dtype=torch.float32, scale=s, double_scale=g, block_sizes={-1: 16})
        unpacked = torch.stack([packed._quantized_data & 15, packed._quantized_data >> 4], dim=-1).flatten().tolist()
        assert list(codes) == unpacked, f'E2M1 mismatch in tensor {index}'
        assert list(scales) == s.view(torch.uint8).flatten().tolist(), f'E4M3 mismatch in tensor {index}'
        assert global_scale.value == g.item(), f'FP32 tensor scale mismatch in tensor {index}'
        actual = torch.tensor(list(dequant)).reshape(x.shape)
        assert torch.equal(actual, expected), f'FP32 dequantization mismatch in tensor {index}'
        if gpu:
            output = (ctypes.c_float * n)()
            assert gpu.nvfp4_gpu_reference(src, n, output) == 0
            gpu_values = torch.tensor(list(output)).reshape(x.shape)
            assert torch.equal(gpu_values, expected.half().float()), f'GPU fake-quant mismatch in tensor {index}'
        comparisons.append(dict(index=index, elements=n, global_scale=g.item(), bit_exact=True))
    zeros = (ctypes.c_float * 32)(); q = (ctypes.c_uint8 * 32)(); s = (ctypes.c_uint8 * 2)(); g = ctypes.c_float(); d = (ctypes.c_float * 32)()
    assert lib.nvfp4_ref_quantize(zeros, 32, q, s, ctypes.byref(g), d) == 0 and not any(d)
    zero_reference, zs, zg = official.quantize(torch.zeros(1, 32), 16)
    zero_dequant = zero_reference.dequantize(dtype=torch.float32, scale=zs, double_scale=zg, block_sizes={-1: 16})
    assert zg.item() == 0 and not torch.isfinite(zero_dequant).all()
    gpu_edges = False
    if gpu:
        assert gpu.nvfp4_gpu_reference(zeros, 32, d) == 0 and not any(d)
    for value in (float('inf'), float('-inf'), float('nan')):
        zeros[0] = value
        assert lib.nvfp4_ref_quantize(zeros, 32, q, s, ctypes.byref(g), d) == 2
        if gpu:
            assert gpu.nvfp4_gpu_reference(zeros, 32, d) == 2
    if gpu:
        gpu_edges = True
    report = dict(schema='nvfp4-reference-qa-v1', complete=True, reference=upstream,
                  torch_version=torch.__version__, cpu_only_torch=True, gpu_checked=bool(gpu),
                  fp4_rounding_vectors=len(fp4_vectors), e4m3_rounding_vectors=len(scale_vectors),
                  full_tensor_vectors=comparisons, zero_tensor='zero-preserving extension', nonfinite='reject',
                  official_zero_tensor='nonfinite output from zero global scale; extension intentionally differs',
                  gpu_zero_and_nonfinite_checked=gpu_edges,
                  reference_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    if args.check_bindings:
        shim = args.library.parent / 'libggml-nvfp4-preload.so'
        report['interposition_checked'] = check_bindings(args.check_bindings, shim, args.reference_dir)['complete']
    if gpu:
        report['gpu_library_sha256'] = hashlib.sha256(args.gpu_library.read_bytes()).hexdigest()
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    destination = args.reference_dir / ('gpu-reference-report.json' if gpu else 'cpu-reference-report.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
