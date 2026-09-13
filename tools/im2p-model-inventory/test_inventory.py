#!/usr/bin/env python3
"""Test the native metadata tool without NumPy or model computation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

BINARY = None


def string_bytes(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def fixture():
    # A small F32 GGUF metadata/extent fixture, not a numerical model oracle.
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    kv = string_bytes('general.architecture') + struct.pack('<I', 8) + string_bytes('llama')
    tensor = (string_bytes('name\nwith"quote') + struct.pack('<IQQIQ', 2, 2, 3, 0, 0))
    data = header + kv + tensor
    return data + b'\x00' * ((-len(data)) % 32) + b'\x00' * 24


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='im2p-inventory-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, *args):
        return subprocess.run([str(BINARY), *map(str, args)], text=True,
                              capture_output=True, check=False)

    def test_help_without_device(self):
        result = self.invoke('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('no backend or model execution', result.stdout)

    def test_missing_argument(self):
        self.assertEqual(self.invoke().returncode, 2)

    def test_missing_file(self):
        result = self.invoke(self.root / 'missing.gguf')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')

    def test_directory_rejected(self):
        self.assertEqual(self.invoke(self.root).returncode, 1)

    def test_invalid_magic(self):
        path = self.root / 'invalid.gguf'
        path.write_bytes(b'not a model' * 32)
        result = self.invoke(path)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')

    def test_readonly_shape_type_and_json_escaping(self):
        path = self.root / 'tensor.gguf'
        path.write_bytes(fixture())
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        path.chmod(0o400)
        result = self.invoke(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['scope'], 'metadata_only_no_invocation')
        self.assertFalse(data['tensor_data_allocated'])
        self.assertEqual(data['architecture'], 'llama')
        self.assertEqual(data['tensor_count'], 1)
        tensor = data['tensors'][0]
        self.assertEqual(tensor['name'], 'name\nwith"quote')
        self.assertEqual(tensor['ggml_shape'], [2, 3, 1, 1])
        self.assertEqual(tensor['type_id'], 0)
        self.assertEqual(tensor['bytes'], 24)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_truncated_payload_rejected_without_partial_json(self):
        path = self.root / 'short.gguf'
        path.write_bytes(fixture()[:-1])
        result = self.invoke(path)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
