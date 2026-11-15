#pragma once
// Independent dot-product reference. No production executor, SDK or command model.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace golden_fp32 {
inline void check(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FP32_GOLDEN FAIL %s\n", why); std::exit(1); }
}
inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float widen(uint16_t h) {
    const unsigned e = (h >> 10) & 31, f = h & 1023;
    const double magnitude = e == 0 ? std::ldexp(double(f), -24) :
        e == 31 ? (f ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity()) :
        std::ldexp(double(1024 + f), int(e) - 25);
    return float((h & 0x8000) ? -magnitude : magnitude);
}
inline float sample(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    const uint32_t u = (state & 0x807fffff) | ((115u + (state >> 24) % 18u) << 23);
    float f; std::memcpy(&f, &u, 4); return f;
}
struct Stats {
    uint64_t values = 0, mismatches = 0;
    double max_abs = 0, squared_error = 0, squared_reference = 0;
};
// mode: 0 = ascending FMA in each 8-element partial, then RNE adds;
// 1 = one ascending FMA chain; 2 = descending FMA within each partial.
inline Stats compare(const float * a, const float * b, const float * actual,
                     size_t m, size_t n, size_t k, unsigned mode) {
    check(std::fegetround() == FE_TONEAREST && mode <= 2, "rounding mode/order");
    Stats stats;
    for (size_t row = 0; row < m; ++row) {
        // References are column vectors: their recurrence is independent of SDK tiling.
        std::vector<float> expected(n, 0.0f), partial(n);
        std::vector<double> exact(n, 0.0);
        for (size_t block = 0; block < (k + 7) / 8; ++block) {
            std::fill(partial.begin(), partial.end(), 0.0f);
            if (mode == 1) partial = expected;
            for (size_t q = 0; q < 8; ++q) {
                const size_t z = block * 8 + (mode == 2 ? 7 - q : q);
                if (z < k) {
                    for (size_t col = 0; col < n; ++col) {
                        // Independently apply the RTL unity MVIN scales, not bit copies.
                        const float av = std::fma(a[row * k + z], 1.0f, 0.0f);
                        const float bv = std::fma(b[z * n + col], 1.0f, 0.0f);
                        partial[col] = std::fma(av, bv, partial[col]);
                        exact[col] += double(a[row * k + z]) * double(b[z * n + col]);
                    }
                } else {
                    for (size_t col = 0; col < n; ++col) partial[col] = std::fma(0.0f, 0.0f, partial[col]);
                }
            }
            for (size_t col = 0; col < n; ++col)
                expected[col] = block == 0 || mode == 1 ? partial[col] : expected[col] + partial[col];
        }
        for (size_t col = 0; col < n; ++col) {
            expected[col] = std::fma(expected[col], 1.0f, 0.0f); // Unity MVOUT scale.
            const float got = actual[row * n + col];
            ++stats.values;
            if (bits(got) != bits(expected[col])) {
                if (stats.mismatches < 3)
                    std::fprintf(stderr, "mismatch row=%zu col=%zu got=%08x expected=%08x\n",
                                 row, col, bits(got), bits(expected[col]));
                ++stats.mismatches;
            }
            const double error = double(got) - exact[col];
            stats.max_abs = std::max(stats.max_abs, std::fabs(error));
            stats.squared_error += error * error;
            stats.squared_reference += exact[col] * exact[col];
        }
    }
    return stats;
}
inline void report(const char * name, const Stats & s) {
    std::printf("FP32_GOLDEN case=%s values=%llu mismatches=%llu fp64_max_abs=%.17g fp64_rmse=%.17g\n",
                name, (unsigned long long)s.values, (unsigned long long)s.mismatches,
                s.max_abs, std::sqrt(s.squared_error / double(s.values)));
    check(s.mismatches == 0, name);
}
}
