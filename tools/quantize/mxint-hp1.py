#!/usr/bin/env python3
"""Pack official Microsoft MXINT4/8 values into unchanged HP1 GGUF tensors."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gguf-py'))
from gguf import GGUFReader, GGUFEndian  # noqa: E402

REFERENCE_COMMIT = '7bc41952de394f5cc5e782baf132e7c7542eb4e4'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def official_reference(path):
    """Use the same pinned upstream implementation as the MXINT experiments."""
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == REFERENCE_COMMIT, 'Wrong microxcaling revision: ' + commit)
    require(not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no']),
            'The official reference has tracked modifications')
    sys.path.insert(0, str(path.resolve()))
    from mx import MxSpecs
    from mx.mx_ops import quantize_mx_op
    specs = MxSpecs()
    specs.update(scale_bits=8, block_size=32, shared_exp_method='max', round='even',
                 custom_cuda=False, mx_flush_fp32_subnorms=False)

    def quantize(weights, bits):
        with torch.no_grad():
            return quantize_mx_op(torch.from_numpy(weights), specs, elem_format=f'int{bits}',
                                  axes=[-1], block_size=32, round='even').numpy()
    return quantize


def pack_reference(weights, values, bits):
    """Lossless storage adapter; only upstream chooses quantized values."""
    require(bits in (4, 8), 'Expected MXINT4 or MXINT8')
    require(weights.dtype == values.dtype == np.float32 and weights.shape == values.shape,
            'Expected matching FP32 source and reference arrays')
    require(weights.ndim == 3 and weights.shape[-1] == 32, 'Expected [rows, K-blocks, 32]')
    require(np.isfinite(weights).all() and np.isfinite(values).all(), 'Nonfinite weight/reference')
    amax = np.max(np.abs(weights), axis=-1)
    zero = amax == 0
    exponent = np.frexp(amax)[1].astype(np.int32) - 1
    require(np.all(zero | ((exponent >= -127) & (exponent <= 127))), 'Weight exponent outside supported E8M0 range')
    # Absorb the MX fixed-point factor 2**-(n-2) into the HP1 integer-code scale.
    theta = exponent - (bits - 2)
    channel = np.min(np.where(zero, 32767, theta), axis=1)
    empty = np.all(zero, axis=1)
    channel[empty] = 0
    channel_scale = np.ldexp(np.ones(len(weights), np.float32), channel)
    channel_scale[empty] = 0
    offsets = theta - channel[:, None]
    require(np.all(zero | ((offsets >= 0) & (offsets < 32))), 'Unsupported SCU shift')
    code_values = np.ldexp(values, -theta[..., None])
    codes = code_values.astype(np.int16)
    limit = (1 << (bits - 1)) - 1
    require(np.array_equal(codes, code_values) and np.all(np.abs(codes) <= limit),
            'Official values are not exactly representable by symmetric HP1 codes')
    # A sufficient absolute-prefix bound for main and signed residual-limb INT32 accumulation.
    weighted_l1 = np.ldexp(np.abs(codes).sum(axis=-1).astype(np.float64), np.where(zero, 0, offsets)).sum(axis=1)
    require(np.all(weighted_l1 * (1 << (bits - 1)) <= np.iinfo(np.int32).max), 'Unproved INT32 SCU bound')
    require(np.all(weighted_l1 * ((1 << 31) + (1 << (bits - 1))) < 2**63), 'Unproved INT64 residual bound')
    offset, stride = (16, 24) if bits == 4 else (32, 40)
    raw = np.zeros((*amax.shape, stride), np.uint8)
    if bits == 4:
        raw[..., :16] = ((codes[..., :16] + 8) | ((codes[..., 16:] + 8) << 4)).astype(np.uint8)
    else:
        raw[..., :32] = codes.astype(np.int8).view(np.uint8)
    raw[zero] = 0
    raw[..., offset:offset+2] = np.where(zero, -32768, offsets).astype('<i2')[..., None].view(np.uint8)
    scales = np.broadcast_to(channel_scale[:, None], amax.shape).astype('<f4').copy()
    raw[..., offset+4:offset+8] = scales[..., None].view(np.uint8)
    return raw


def decoder(library, bits):
    lib = ctypes.CDLL(str(library.resolve()))
    function = getattr(lib, f'dequantize_row_q{bits}_hp1')
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    function.restype = None
    return lib, function


def decode(raw, function):
    values = np.empty((*raw.shape[:-1], 32), np.float32)
    function(raw.ctypes.data, values.ctypes.data, values.size)
    return values


def convert(source, template, target, bits, mx_path, decoder_lib):
    source, template, target = (p.resolve() for p in (source, template, target))
    require(bits in (4, 8) and not target.exists(), 'Output must be a new MXINT4/8 model')
    report_path = target.with_suffix(target.suffix + '.mxint.json')
    require(not report_path.exists(), 'Output report already exists')
    quantize = official_reference(mx_path)
    library, c_decode = decoder(decoder_lib, bits)
    inputs = {str(p): digest(p) for p in (source, template, decoder_lib.resolve())}
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
    os.close(fd)
    partial = Path(name)
    try:
        shutil.copyfile(template, partial)
        src, dst = GGUFReader(source), GGUFReader(partial, 'r+')
        require(src.endianess == dst.endianess == GGUFEndian.LITTLE, 'Only little-endian GGUF is supported')
        for key, field in src.fields.items():
            if key.startswith('GGUF.') or key in ('general.file_type', 'general.quantization_version'):
                continue
            require(key in dst.fields and dst.fields[key].contents() == field.contents(),
                    'Template metadata differs from source: ' + key)
        sources = {t.name: t for t in src.tensors}
        require(set(sources) == {t.name for t in dst.tensors}, 'Source/template tensor inventory mismatch')
        matrices = vectors = count = 0
        for tensor in dst.tensors:
            original = sources[tensor.name]
            require(np.array_equal(original.shape, tensor.shape), 'Tensor shape mismatch: ' + tensor.name)
            if len(tensor.shape) == 1:
                require(original.tensor_type.name == tensor.tensor_type.name == 'F32'
                        and original.data.tobytes() == tensor.data.tobytes(), 'F32 vector changed: ' + tensor.name)
                vectors += 1
                continue
            require(original.tensor_type.name == 'F16' and tensor.tensor_type.name == f'Q{bits}_HP1',
                    'Every matrix, including embedding/head, must be F16 -> HP1: ' + tensor.name)
            k = int(tensor.shape[0])
            require(k % 32 == 0, 'K must be divisible by 32')
            rows, blocks = int(tensor.n_elements) // k, k // 32
            weights = original.data.reshape(rows, blocks, 32)
            output = tensor.data.reshape(rows, blocks, 24 if bits == 4 else 40)
            for first in range(0, rows, 128):
                values = np.asarray(weights[first:first+128], dtype=np.float32)
                expected = quantize(values, bits)
                raw = pack_reference(values, expected, bits)
                require(np.array_equal(decode(raw, c_decode), expected), 'Original C decoder differs from Microsoft MX: ' + tensor.name)
                output[first:first+128] = raw
                require(np.array_equal(output[first:first+128], raw), 'GGUF write mismatch')
                count += values.size
            matrices += 1
            print(f'MXINT{bits}: {tensor.name}', flush=True)
        require(matrices > 0, 'No HP1 matrices found')
        dst.data.flush()
        with partial.open('rb') as f:
            os.fsync(f.fileno())
        require(all(digest(Path(p)) == sha for p, sha in inputs.items()), 'Conversion input changed')
        official_reference(mx_path)
        # Publish without overwriting a file created by another process.
        os.link(partial, target)
        report = dict(state='PASS', bits=bits, output=str(target), sha256=digest(target),
                      input_sha256=inputs, reference_commit=REFERENCE_COMMIT,
                      scale_bits=8, block_size=32, rounding='even', shared_exp_method='max',
                      storage=f'Q{bits}_HP1; channel PoT scale plus nonnegative block exponent offsets',
                      tensor_policy='all matrices HP1; F32 vectors unchanged', matrices=matrices, vectors=vectors,
                      weights=count, all_weights_c_decoder_exact=True, inference_changes=False,
                      int32_prefix_bound=True, int64_residual_bound=True)
        with report_path.open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
        return report
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original F16 GGUF')
    parser.add_argument('--template', type=Path, required=True, help='All-matrix HP1 GGUF produced by llama-quantize from this input')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bits', type=int, choices=(4, 8), required=True)
    parser.add_argument('--mx-path', type=Path, required=True, help='Clean microsoft/microxcaling checkout at the documented commit')
    parser.add_argument('--decoder-lib', type=Path, required=True, help='Built libggml-base.so with the existing HP1 decoder')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(json.dumps(convert(args.input, args.template, args.output, args.bits, args.mx_path, args.decoder_lib), indent=2))


if __name__ == '__main__':
    main()
