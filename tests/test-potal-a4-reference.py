"""Run with python3 tests/test-potal-a4-reference.py."""
from fractions import Fraction
import math
import random
import struct
import unittest

from potal_a4_reference import (
    I32_MIN, I32_MAX, adapt_block, digits, effective_digits, exact_sum,
    ieee_bits, publish_f32_bits, qround, shift,
)


class ArithmeticContract(unittest.TestCase):
    def test_shift_boundaries(self):
        self.assertEqual(shift(1 << 30, -32), 0)
        self.assertEqual(shift(I32_MIN, -32), -1)
        self.assertEqual(shift(I32_MIN, -33), 0)
        self.assertEqual(shift(-1, 200), I32_MIN)
        self.assertEqual(shift(0, 200), 0)
        self.assertEqual(shift(1, 200), I32_MAX)

    def test_full_int32_digits_and_selective_boundary(self):
        rng = random.Random(1729)
        for value in [I32_MIN, I32_MAX, 0x77777778, 0, 8, -8] + [rng.randint(I32_MIN, I32_MAX) for _ in range(4096)]:
            ds = digits(value)
            self.assertTrue(all(-8 <= d <= 7 for d in ds))
            self.assertEqual(sum(d * 16 ** lane for lane, d in enumerate(ds)), value)
        self.assertEqual(len(digits(0x77777778)), 9)
        self.assertEqual(effective_digits(8, False, 0), (7,))
        self.assertEqual(effective_digits(8, False, 0, query=True), (-8, 1))
        self.assertEqual(effective_digits(8, True, 0), (-8, 1))
        self.assertEqual(effective_digits(8, False, 1), (-8, 1))

    def test_unsigned_p_contract(self):
        for q in range(256):
            main = (q & 15) - 8
            upper = digits(q - (q & 15))
            self.assertEqual(main + 8 + sum(d * 16 ** i for i, d in enumerate(upper)), q)
            self.assertEqual(upper[0], 0)
            self.assertEqual(len(upper) == 3, q >= 128)

    def test_stationary_k_contract(self):
        for code in range(-128, 120):
            low = (code + 8) % 16 - 8
            high = (code - low) // 16
            self.assertTrue(-8 <= low <= 7 and -8 <= high <= 7)
            self.assertEqual(low + 16 * high, code)

    def test_half_even_and_real_tail_population(self):
        self.assertEqual([qround(x, 0) for x in [0.5, 1.5, 2.5, -1.5]], [0, 2, 2, -2])
        exponent, codes, support = adapt_block([1.99] * 32, 2)
        self.assertEqual(exponent, 0)
        self.assertEqual(codes, (8,) * 32)
        self.assertFalse(any(support))
        self.assertEqual(adapt_block([0.0] * 3, 6)[1], (0, 0, 0))
        # Zeros do not participate; enormous q squares require Python integers.
        real = adapt_block([1.0, 1.0, 1.0, 100.0, 128.0], 6)
        padded = adapt_block([1.0, 1.0, 1.0, 100.0, 128.0] + [0.0] * 27, 6)
        self.assertEqual(real[0], padded[0])
        self.assertEqual(real[1], padded[1][:5])
        self.assertEqual(real[2], padded[2][:5])
        adapt_block([math.ldexp(1.0, -149)] * 30 + [1.0, 2.0], 6)
        for bad in [math.nan, math.inf, -math.inf]:
            with self.assertRaisesRegex(ValueError, "finite"):
                adapt_block([bad], 2)

    def test_exact_cancellation_and_double_rounding(self):
        terms = [(1, 63), (1, 0), (-1, 63)]
        self.assertEqual(exact_sum(terms), 1)
        self.assertEqual(publish_f32_bits(terms), 0x3f800000)
        # Exact lies just above an F32 midpoint; F64 lands on that midpoint.
        terms = [(1, 0), (1, -24), (1, -54)]
        self.assertEqual(ieee_bits(exact_sum(terms), 32), 0x3f800001)
        self.assertEqual(publish_f32_bits(terms), 0x3f800000)

    def test_ieee_boundaries(self):
        for terms, bits in [([], 0), ([(1, -149)], 1), ([(1, -150)], 0),
                            ([(-1, -150)], 0x80000000), ([(3, -150)], 2),
                            ([(1, 128)], 0x7f800000), ([(-1, 128)], 0xff800000),
                            ([(1, -1075)], 0), ([(-1, -1075)], 0x80000000),
                            ([(1, 2000), (-1, 2000)], 0)]:
            self.assertEqual(publish_f32_bits(terms), bits)
        self.assertEqual(ieee_bits(Fraction(1, 1 << 1074), 64), 1)
        self.assertEqual(ieee_bits(Fraction(-1, 1 << 1075), 64), 1 << 63)
        rng = random.Random(42)
        for _ in range(4096):
            value = math.ldexp(rng.uniform(-1, 1), rng.randint(-140, 125))
            expected = int.from_bytes(struct.pack(">f", value), "big")
            self.assertEqual(ieee_bits(Fraction.from_float(value), 32), expected)


if __name__ == "__main__":
    unittest.main()
