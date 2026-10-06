#ifndef GGML_METAL_SOFT_F64_H
#define GGML_METAL_SOFT_F64_H

// IEEE binary64 arithmetic for Metal, which has no native double type. All
// rounding is nearest, ties to even; no floating operation can flush subnormals.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef ulong mq_f64;
typedef long mq_i64;
typedef uint mq_u32;
inline uint mq_clz64(ulong x) { return metal::clz(x); }
inline ulong mq_mulhi64(ulong a, ulong b) { return metal::mulhi(a, b); }
#else
#include <cstdint>
typedef uint64_t mq_f64;
typedef int64_t mq_i64;
typedef uint32_t mq_u32;
inline mq_u32 mq_clz64(mq_f64 x) { return x ? __builtin_clzll(x) : 64; }
inline mq_f64 mq_mulhi64(mq_f64 a, mq_f64 b) { return mq_f64((__uint128_t(a) * b) >> 64); }
#endif

inline mq_f64 mq_f64_sign_mask() { return mq_f64(1) << 63; }
inline mq_f64 mq_f64_fraction_mask() { return (mq_f64(1) << 52) - 1; }
inline mq_f64 mq_f64_infinity() { return mq_f64(0x7ff) << 52; }
inline mq_f64 mq_f64_nan() { return mq_f64(0x7ff8) << 48; }
inline bool mq_f64_is_nan(mq_f64 a) {
    return (a & ~mq_f64_sign_mask()) > mq_f64_infinity();
}

// The discarded bits collapse to one sticky bit, including shifts >= 64.
inline mq_f64 mq_shift_right_jam(mq_f64 x, unsigned distance) {
    if (distance == 0) { return x; }
    if (distance >= 64) { return x != 0; }
    return (x >> distance) | ((x << (64 - distance)) != 0);
}

// sig has its leading bit at 55 and three guard/round/sticky bits. Subnormal
// rounding happens before encoding the exponent, including carry into normal.
inline mq_f64 mq_f64_round_pack(mq_f64 sign, int exponent, mq_f64 sig) {
    if (exponent < -1022) {
        sig = mq_shift_right_jam(sig, unsigned(-1022 - exponent));
        exponent = -1022;
    }
    mq_f64 mantissa = sig >> 3;
    unsigned remainder = unsigned(sig & 7);
    if (remainder > 4 || (remainder == 4 && (mantissa & 1))) { ++mantissa; }
    if (mantissa >= (mq_f64(1) << 53)) {
        mantissa >>= 1;
        ++exponent;
    }
    if (exponent > 1023) { return sign | mq_f64_infinity(); }
    if (!mantissa) { return sign; }
    mq_f64 biased = mantissa < (mq_f64(1) << 52) ? 0 : mq_f64(exponent + 1023);
    return sign | (biased << 52) | (mantissa & mq_f64_fraction_mask());
}

// Finite nonzero inputs become a 53-bit significand and unbiased exponent.
struct mq_f64_parts { mq_f64 sig; int exponent; };
inline mq_f64_parts mq_f64_unpack(mq_f64 a) {
    mq_f64_parts p;
    p.sig = a & mq_f64_fraction_mask();
    p.exponent = int((a >> 52) & 0x7ff) - 1023;
    if (p.exponent == -1023) {
        unsigned shift = mq_clz64(p.sig) - 11;
        p.sig <<= shift;
        p.exponent = -1022 - int(shift);
    } else {
        p.sig |= mq_f64(1) << 52;
    }
    return p;
}

inline mq_f64 mq_f64_from_i64(mq_i64 a) {
    mq_f64 sign = a < 0 ? mq_f64_sign_mask() : 0;
    mq_f64 magnitude = a < 0 ? mq_f64(0) - mq_f64(a) : mq_f64(a);
    if (!magnitude) { return 0; }
    unsigned top = 63 - mq_clz64(magnitude);
    mq_f64 sig = top <= 55 ? magnitude << (55 - top) : mq_shift_right_jam(magnitude, top - 55);
    return mq_f64_round_pack(sign, int(top), sig);
}

inline mq_f64 mq_f64_from_f32_bits(mq_u32 a) {
    mq_f64 sign = mq_f64(a & 0x80000000u) << 32;
    unsigned exponent = (a >> 23) & 255;
    mq_f64 fraction = a & 0x7fffffu;
    if (exponent == 255) {
        return sign | mq_f64_infinity() | (fraction ? (fraction << 29) | (mq_f64(1) << 51) : 0);
    }
    if (exponent == 0) {
        if (!fraction) { return sign; }
        unsigned top = 63 - mq_clz64(fraction);
        return sign | (mq_f64(int(top) - 149 + 1023) << 52) |
               ((fraction << (52 - top)) & mq_f64_fraction_mask());
    }
    return sign | (mq_f64(exponent + 896) << 52) | (fraction << 29);
}

inline mq_f64 mq_f64_mul(mq_f64 a, mq_f64 b) {
    mq_f64 sign = (a ^ b) & mq_f64_sign_mask();
    mq_f64 aa = a & ~mq_f64_sign_mask(), bb = b & ~mq_f64_sign_mask();
    if (mq_f64_is_nan(a) || mq_f64_is_nan(b)) { return mq_f64_nan(); }
    if (aa == mq_f64_infinity() || bb == mq_f64_infinity()) {
        return (!aa || !bb) ? mq_f64_nan() : sign | mq_f64_infinity();
    }
    if (!aa || !bb) { return sign; }
    mq_f64_parts pa = mq_f64_unpack(a), pb = mq_f64_unpack(b);
    mq_f64 lo = pa.sig * pb.sig, hi = mq_mulhi64(pa.sig, pb.sig);
    // The exact product has 105 or 106 bits. Retain 56 and jam its tail.
    unsigned carry = unsigned(hi >> 41);
    unsigned shift = 49 + carry;
    mq_f64 sig = (hi << (64 - shift)) | (lo >> shift) |
                 ((lo << (64 - shift)) != 0);
    return mq_f64_round_pack(sign, pa.exponent + pb.exponent + int(carry), sig);
}

inline mq_f64 mq_f64_add(mq_f64 a, mq_f64 b) {
    mq_f64 aa = a & ~mq_f64_sign_mask(), bb = b & ~mq_f64_sign_mask();
    if (mq_f64_is_nan(a) || mq_f64_is_nan(b)) { return mq_f64_nan(); }
    if (aa == mq_f64_infinity() || bb == mq_f64_infinity()) {
        if (aa == bb && ((a ^ b) & mq_f64_sign_mask())) { return mq_f64_nan(); }
        return aa == mq_f64_infinity() ? a : b;
    }
    if (!aa && !bb) { return (a & b) & mq_f64_sign_mask(); }
    if (!aa) { return b; }
    if (!bb) { return a; }
    // Magnitude order gives a nonnegative significand subtraction.
    if (aa < bb) { mq_f64 temporary = a; a = b; b = temporary; }
    mq_f64 sign = a & mq_f64_sign_mask();
    mq_f64_parts pa = mq_f64_unpack(a), pb = mq_f64_unpack(b);
    mq_f64 sa = pa.sig << 3;
    mq_f64 sb = mq_shift_right_jam(pb.sig << 3, unsigned(pa.exponent - pb.exponent));
    mq_f64 sig;
    int exponent = pa.exponent;
    if (((a ^ b) & mq_f64_sign_mask()) == 0) {
        sig = sa + sb;
        if (sig & (mq_f64(1) << 56)) { sig = mq_shift_right_jam(sig, 1); ++exponent; }
    } else {
        sig = sa - sb;
        if (!sig) { return 0; }
        unsigned shift = mq_clz64(sig) - 8;
        sig <<= shift;
        exponent -= int(shift);
    }
    return mq_f64_round_pack(sign, exponent, sig);
}

inline mq_u32 mq_f64_to_f32_bits(mq_f64 a) {
    mq_u32 sign = mq_u32(a >> 32) & 0x80000000u;
    mq_f64 magnitude = a & ~mq_f64_sign_mask();
    if (magnitude >= mq_f64_infinity()) {
        return sign | (magnitude == mq_f64_infinity() ? 0x7f800000u : 0x7fc00000u);
    }
    if (!magnitude) { return sign; }
    mq_f64_parts p = mq_f64_unpack(a);
    mq_f64 sig = mq_shift_right_jam(p.sig, 26);
    if (p.exponent < -126) {
        sig = mq_shift_right_jam(sig, unsigned(-126 - p.exponent));
        p.exponent = -126;
    }
    mq_u32 mantissa = mq_u32(sig >> 3);
    unsigned remainder = unsigned(sig & 7);
    if (remainder > 4 || (remainder == 4 && (mantissa & 1))) { ++mantissa; }
    if (mantissa >= (1u << 24)) { mantissa >>= 1; ++p.exponent; }
    if (p.exponent > 127) { return sign | 0x7f800000u; }
    if (!mantissa) { return sign; }
    mq_u32 exponent = mantissa < (1u << 23) ? 0 : mq_u32(p.exponent + 127);
    return sign | (exponent << 23) | (mantissa & 0x7fffffu);
}

inline mq_u32 mq_f32_add_bits(mq_u32 a, mq_u32 b) {
    // Binary64 exactly resolves every binary32 sum relevant to its final round.
    return mq_f64_to_f32_bits(mq_f64_add(mq_f64_from_f32_bits(a), mq_f64_from_f32_bits(b)));
}

#endif
