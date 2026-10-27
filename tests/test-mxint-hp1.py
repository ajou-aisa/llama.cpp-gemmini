#!/usr/bin/env python3
"""Check official rounding and real GGUF conversion through the existing C decoder."""
import argparse
import importlib.util
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mxint_hp1', ROOT / 'tools/quantize/mxint-hp1.py')
mxint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mxint)
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, LlamaFileType  # noqa: E402


def write_fixture(path, weights, raw=None, bits=4):
    writer = GGUFWriter(path, 'llama')
    writer.add_name('mxint-rounding-fixture')
    writer.add_file_type(LlamaFileType.MOSTLY_F16 if raw is None else getattr(LlamaFileType, f'MOSTLY_Q{bits}_HP1'))
    for name in ('token_embd.weight', 'blk.0.ffn_up.weight'):
        if raw is None:
            writer.add_tensor(name, weights.reshape(len(weights), -1).astype(np.float16))
        else:
            writer.add_tensor(name, raw.reshape(len(raw), -1), raw_dtype=getattr(GGMLQuantizationType, f'Q{bits}_HP1'))
    writer.add_tensor('output_norm.weight', np.array([1, .5], np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mx-path', type=Path, required=True)
    parser.add_argument('--decoder-lib', type=Path, required=True)
    args = parser.parse_args()
    mxint.torch.set_num_threads(2)
    quantize = mxint.official_reference(args.mx_path)
    with tempfile.TemporaryDirectory(prefix='mxint-hp1-') as temporary:
        scratch = Path(temporary)
        for bits in (4, 8):
            limit = (1 << (bits - 1)) - 1
            probes = np.array([.5, 1.5, 2.5, -.5, -1.5, -2.5, -limit-.75, limit+.75,
                               0, 1, -1, limit, -limit], np.float32)
            block = np.resize(probes, 32)
            row = np.stack([block, block / 8, np.zeros(32, np.float32)])
            weights = np.stack([row, np.zeros_like(row)])
            expected_codes = np.clip(np.rint(block), -limit, limit)
            expected = np.stack([np.stack([expected_codes, expected_codes / 8, np.zeros(32, np.float32)]),
                                 np.zeros_like(row)])
            values = quantize(weights, bits)
            assert np.array_equal(values, expected), 'Official tie and symmetric clipping boundaries'
            raw = mxint.pack_reference(weights, values, bits)
            library, decoder = mxint.decoder(args.decoder_lib, bits)
            assert np.array_equal(mxint.decode(raw, decoder), expected)
            source, template, output = (scratch / f'{name}-{bits}.gguf' for name in ('fp16', 'hp1', 'mxint'))
            write_fixture(source, weights)
            write_fixture(template, weights, raw, bits)
            # Deliberately corrupt a template's codes: conversion must use the F16 source.
            existing = GGUFReader(template, 'r+')
            existing.tensors[0].data.flat[0] ^= 0x01
            existing.data.flush()
            before = mxint.digest(source), mxint.digest(template)
            result = mxint.convert(source, template, output, bits, args.mx_path, args.decoder_lib)
            converted = GGUFReader(output)
            for tensor in converted.tensors:
                if len(tensor.shape) > 1:
                    assert np.array_equal(tensor.data.reshape(raw.shape), raw), 'Actual GGUF tensor bytes differ'
            assert result['all_weights_c_decoder_exact'] and result['matrices'] == 2 and result['vectors'] == 1
            assert (mxint.digest(source), mxint.digest(template)) == before, 'Input mutated'
            try:
                mxint.convert(source, template, output, bits, args.mx_path, args.decoder_lib)
            except ValueError:
                pass
            else:
                raise AssertionError('Existing output was overwritten')
            invalid = weights.copy()
            invalid.flat[0] = np.nan
            try:
                mxint.pack_reference(invalid, values, bits)
            except ValueError:
                pass
            else:
                raise AssertionError('Nonfinite source was accepted')
            print(f'PASS MXINT{bits}: official ties, clipping, zero rows/blocks, channel offsets, C decode, GGUF conversion')


if __name__ == '__main__':
    main()
