import struct

import numpy as np

from weight_io import FloatStream, GGUFReader, ROOT, hp1_values
from weight_stats import FINE_EDGES, Moments


def test_hp1_packing_scale_and_zero_sentinel() -> None:
    for bits, width, offset in ((4, 24, 16), (8, 40, 32)):
        raw = np.zeros((2, width), dtype=np.uint8)
        codes = (np.arange(-8, 24, dtype=np.int16) if bits == 8 else
                 np.concatenate((np.arange(-8, 8, dtype=np.int16), np.arange(7, -9, -1, dtype=np.int16))))
        if bits == 4:
            raw[0, :16] = (codes[:16] + 8) | ((codes[16:] + 8) << 4)
        else:
            raw[0, :32] = codes.astype(np.int8).view(np.uint8)
        raw[0, offset:offset + 2] = np.frombuffer(struct.pack("<h", 3), dtype=np.uint8)
        raw[1, offset:offset + 2] = np.frombuffer(struct.pack("<h", -32768), dtype=np.uint8)
        raw[:, -4:] = np.frombuffer(struct.pack("<f", 0.125), dtype=np.uint8)
        values, decoded = hp1_values(raw, bits)
        np.testing.assert_array_equal(values[:32], codes.astype(np.float32))
        np.testing.assert_array_equal(decoded[:32], codes)
        np.testing.assert_array_equal(values[32:], np.zeros(32))


def test_streaming_header_matches_local_gguf_reader() -> None:
    path = ROOT / "models/gpt2.fp16.gguf"
    native = GGUFReader(str(path), "r")
    with path.open("rb") as source:
        stream = FloatStream(source)
        entries = stream.header()
        offset = stream.position
        expected = {tensor.name: tensor for tensor in native.tensors}
        assert len(entries) == len(expected)
        for entry in entries:
            tensor = expected[entry.name]
            assert entry.shape == tuple(tensor.shape)
            assert offset + entry.offset == tensor.data_offset


def test_statistics_are_exact_and_histogram_conserves_mass() -> None:
    values = np.array([-4, -1, 0, 1, 2, 4], dtype=np.float32)
    reference = np.array([-3, -1, 0, 1, 2, 3], dtype=np.float32)
    whole, first, second = Moments(), Moments(), Moments()
    whole.add(values, reference, None)
    first.add(values[:3], reference[:3], None)
    second.add(values[3:], reference[3:], None)
    first.merge(second)
    assert first.summary() == whole.summary()
    assert whole.count == 6 and whole.zeros == 1
    assert whole.histogram.sum() + whole.zeros == whole.count
    assert whole.summary()["rmse"] == np.sqrt(2 / 6)
    assert whole.summary()["max_abs"] == 4
    np.testing.assert_array_equal(whole.fine_histogram.reshape(160, 8).sum(axis=1), whole.histogram)
    np.testing.assert_array_equal(first.fine_histogram, whole.fine_histogram)


def test_fine_bins_preserve_separate_observed_magnitudes() -> None:
    values = np.array([0, -1, 1.03, -1.10, 2, 4], dtype=np.float32)
    stats = Moments()
    stats.add(values, values, None)
    expected, _ = np.histogram(np.log2(np.abs(values[values != 0]).astype(np.float64)), bins=FINE_EDGES)
    np.testing.assert_array_equal(stats.fine_histogram, expected)
    np.testing.assert_array_equal(stats.fine_histogram.reshape(160, 8).sum(axis=1), stats.histogram)
    assert np.count_nonzero(stats.fine_histogram) > np.count_nonzero(stats.histogram)


def test_zeroed_weights_are_binned_by_nonzero_reference_magnitude() -> None:
    reference = np.array([0, 0.01, -0.01, 0.125, -2, 4], dtype=np.float32)
    quantized = np.array([0, 0, 0.02, 0, 0, 4], dtype=np.float32)
    whole, first, second = Moments(), Moments(), Moments()
    whole.add(quantized, reference, None)
    first.add(quantized[:3], reference[:3], None)
    second.add(quantized[3:], reference[3:], None)
    first.merge(second)
    expected, _ = np.histogram(np.log2(np.array([float(reference[1]), 0.125, 2])), bins=FINE_EDGES)
    np.testing.assert_array_equal(whole.zeroed_reference_histogram, expected)
    np.testing.assert_array_equal(first.zeroed_reference_histogram, expected)
    assert whole.summary()["zeroed_nonzero_count"] == 3
    assert whole.summary()["zeroed_nonzero_fraction"] == 0.5
    assert whole.zeros == 4
