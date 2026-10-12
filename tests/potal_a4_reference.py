"""Independent scalar PoTal arithmetic oracle; no NumPy/int64 accumulation.

Contract: experiments/potal_a4/ORIGINAL_SPEC.md and IMPLEMENTATION_PLAN.md.
This module is operator evidence, not a model forward or Gemmini execution.
"""

from fractions import Fraction
import math

NEG = -32768
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def shift(q: int, delta: int) -> int:
    if not I32_MIN <= q <= I32_MAX:
        raise ValueError("shift requires an int32 code")
    if delta > 0:
        return min(I32_MAX, max(I32_MIN, q << min(delta, 32)))
    if delta == 0:
        return q
    if delta < -32:
        return 0
    magnitude = (abs(q) + (1 << (-delta - 1))) >> -delta
    return -magnitude if q < 0 else magnitude


def digits(value: int) -> tuple[int, ...]:
    result = []
    while value:
        digit = (value + 8) % 16 - 8
        result.append(digit)
        value = (value - digit) // 16
    return tuple(result) or (0,)


def effective_digits(u: int, outlier: bool, delta: int, *, query: bool = False) -> tuple[int, ...]:
    """Preserve selective linear (+8 -> 7 unless corrected), all-corrected Q."""
    return digits(u if query or outlier or delta > 0 else min(7, max(-8, u)))


def qround(value: float, exponent: int) -> int:
    if not math.isfinite(value):
        raise ValueError("quantization requires finite values")
    numerator, denominator = value.as_integer_ratio()
    if exponent >= 0:
        denominator <<= exponent
    else:
        numerator <<= -exponent
    return min(I32_MAX, max(I32_MIN, round(Fraction(numerator, denominator))))


def eps(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("exponent requires finite values")
    return math.frexp(abs(value))[1] - 1 if value else NEG


def adapt_block(values: list[float], rho: int) -> tuple[int, tuple[int, ...], tuple[bool, ...]]:
    """Original top/strict 2-sigma adaptation on a real (possibly tail) block."""
    if not 1 <= len(values) <= 32 or rho not in (2, 6):
        raise ValueError("requires 1..32 real source values and rho 2/6")
    exponents = [eps(x) for x in values]
    distinct = sorted(set(exponents) - {NEG}, reverse=True)
    pre = distinct[min(1, len(distinct) - 1)] if distinct else NEG
    top = [bool(len(distinct) > 1 and e == distinct[0]) for e in exponents]
    provisional = [qround(x, pre - rho if pre != NEG else 0) for x in values]
    population = [i for i, x in enumerate(values) if x != 0 and not top[i]]
    count = len(population)
    total = sum(abs(provisional[i]) for i in population)
    squares = sum(provisional[i] ** 2 for i in population)
    variance = count * squares - total * total
    significant = [False] * len(values)
    for i in population:
        centered = count * abs(provisional[i]) - total
        significant[i] = variance > 0 and centered > 0 and centered ** 2 > 4 * variance
    exponent = max((exponents[i] for i in population if not significant[i]), default=NEG) if any(significant) else pre
    codes = tuple(qround(x, exponent - rho if exponent != NEG else 0) for x in values)
    return exponent, codes, tuple(a or b for a, b in zip(top, significant))


def exact_sum(terms: list[tuple[int, int]]) -> Fraction:
    """Sum integer coefficient * 2**exponent without overflow or lost cancellation."""
    if not terms:
        return Fraction(0)
    anchor = min(exponent for _, exponent in terms)
    coefficient = sum(code << (exponent - anchor) for code, exponent in terms)
    return Fraction(coefficient << anchor) if anchor >= 0 else Fraction(coefficient, 1 << -anchor)


def _round_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or (2 * remainder == denominator and quotient % 2))


def ieee_bits(value: Fraction, width: int) -> int:
    """Round exact rational to binary32/binary64, independent of fenv/FTZ."""
    if width not in (32, 64):
        raise ValueError("IEEE width must be 32 or 64")
    precision, exponent_bits = (24, 8) if width == 32 else (53, 11)
    sign = int(value < 0) << (width - 1)
    if not value:
        return 0
    n, d = abs(value.numerator), value.denominator
    exponent = n.bit_length() - d.bit_length()
    below = n < d << exponent if exponent >= 0 else n << -exponent < d
    if below:
        exponent -= 1
    bias = (1 << (exponent_bits - 1)) - 1
    minimum, maximum = 1 - bias, bias
    unit = max(exponent, minimum) - (precision - 1)
    significand = _round_ratio(n, d << unit) if unit >= 0 else _round_ratio(n << -unit, d)
    if significand >= 1 << precision:
        significand >>= 1
        exponent += 1
    if exponent > maximum:
        return sign | (((1 << exponent_bits) - 1) << (precision - 1))
    if significand < 1 << (precision - 1):
        return sign | significand
    return sign | ((max(exponent, minimum) + bias) << (precision - 1)) | (significand - (1 << (precision - 1)))


def publish_f32_bits(terms: list[tuple[int, int]]) -> int:
    """The specified exact -> F64 -> F32 conversion (including double rounding)."""
    binary64 = ieee_bits(exact_sum(terms), 64)
    sign = binary64 >> 63
    exponent = (binary64 >> 52) & 0x7ff
    coefficient = binary64 & ((1 << 52) - 1)
    if exponent == 0x7ff:
        return (sign << 31) | 0x7f800000
    if exponent:
        coefficient |= 1 << 52
        power = exponent - 1023 - 52
    else:
        power = -1074
    if not coefficient:
        return sign << 31
    rounded64 = exact_sum([(-coefficient if sign else coefficient, power)])
    return ieee_bits(rounded64, 32)
