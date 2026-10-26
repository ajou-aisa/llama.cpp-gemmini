#pragma once

// Independent DIM16 attention arithmetic. No production reference, SCU helper,
// quantizer, or emulator arithmetic is used to calculate expected results.
#include "matmul/attention.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace potal_test::attention_golden {

using ggml::potal::attention::role;
using ggml::potal::attention::stationary;
using ggml::potal::attention::stream;
using wide = __int128_t;

inline int32_t clamp32(wide x) {
    return x < INT32_MIN ? INT32_MIN : x > INT32_MAX ? INT32_MAX : static_cast<int32_t>(x);
}

// Division and remainder, rather than signed shifts or production lowdigit().
inline std::array<int8_t, 5> digits(int64_t value, bool probability = false) {
    std::array<int8_t, 5> out{};
    size_t first = 0;
    if (probability) {
        out[0] = static_cast<int8_t>(value % 256 - 128);
        value /= 256;
        first = 1;
    }
    for (size_t l = first; l < out.size(); ++l) {
        int64_t remainder = value % 256;
        if (remainder < -128) remainder += 256;
        if (remainder > 127) remainder -= 256;
        out[l] = static_cast<int8_t>(remainder);
        value = (value - remainder) / 256;
    }
    return out;
}

inline int32_t shifted(wide partial, uint32_t carrier, unsigned offset) {
    if (carrier == POTAL_CARRIER_ZERO || partial == 0) return 0;
    // Above 31, any nonzero integer is outside Sat32 (except -1 at 31).
    if (carrier + offset > 31) return partial < 0 ? INT32_MIN : INT32_MAX;
    return clamp32(partial * (wide{1} << (carrier + offset)));
}

// This is an exact BLOCK column sum, not a sequence of saturating SCU
// microtiles. Cancellation happens before the single INT32 range check.
inline bool column_bias(const stationary & b, size_t column, int32_t & out) {
    wide bias = 0;
    for (size_t begin = 0; begin < b.kp; begin += 32) {
        const uint32_t carrier = b.carriers[(begin/32)*b.n+column];
        if (carrier == POTAL_CARRIER_ZERO) continue;
        wide sum = 0;
        for (size_t k = begin; k < std::min(begin+32,b.k); ++k)
            sum += b.limbs[0][k*b.n+column];
        bias += sum * (wide{1} << carrier) * 128;
    }
    if (bias < INT32_MIN || bias > INT32_MAX) return false;
    out = static_cast<int32_t>(bias);
    return true;
}

// Inputs are deliberately prepared fixtures. Stream operands are derived again
// from their integer codes, so an incorrect prepared limb cannot bless itself.
// Each pass starts a fresh ACC and uses the original carrier, including high.
inline std::vector<int32_t> qk_pass(const stream & a, const stationary & b,
                                   size_t q_limb, size_t k_limb) {
    std::vector<int32_t> out(a.rows * b.n, 0);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.n; ++j) {
            int32_t acc = 0;
            for (size_t begin = 0; begin < a.kp; begin += 16) {
                wide dot = 0;
                for (size_t k = begin; k < begin + 16 && k < a.k; ++k)
                    dot += wide{digits(a.codes[i * a.kp + k])[q_limb]} *
                           b.limbs[k_limb][k * b.n + j];
                const auto carrier = b.carriers[(begin / 32) * b.n + j];
                acc = clamp32(wide{acc} + shifted(dot, carrier, 0));
            }
            out[i * b.n + j] = acc;
        }
    }
    return out;
}

// The old shared-ACC/high+8 recurrence is available only for counterexamples.
inline bool evaluate(const stream & a, const stationary & b, role operation,
                     std::vector<int64_t> & out, bool legacy_shared_qk = false) {
    if (operation == role::qk && !legacy_shared_qk) {
        std::vector<int64_t> values(a.rows * b.n, 0);
        wide place = 1;
        for (size_t limb = 0; limb < a.limb_count; ++limb, place *= 256) {
            const auto low = qk_pass(a, b, limb, 0);
            const auto high = qk_pass(a, b, limb, 1);
            for (size_t index = 0; index < values.size(); ++index) {
                const wide combined = wide{low[index]} + wide{256} * high[index];
                if (combined < INT64_MIN || combined > INT64_MAX) return false;
                const wide reconstructed = wide{values[index]} + combined * place;
                if (reconstructed < INT64_MIN || reconstructed > INT64_MAX) return false;
                values[index] = static_cast<int64_t>(reconstructed);
            }
        }
        out = std::move(values);
        return true;
    }
    std::vector<int32_t> bias(b.n,0);
    if (operation == role::pv)
        for (size_t j = 0; j < b.n; ++j)
            if (!column_bias(b,j,bias[j])) return false;
    std::vector<int64_t> values(a.rows * b.n);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.n; ++j) {
            wide reconstructed = 0;
            wide place = 1;
            for (size_t limb = 0; limb < a.limb_count; ++limb, place *= 256) {
                int32_t acc = 0;
                const size_t passes = operation == role::qk ? 2 : 1;
                for (size_t panel = 0; panel < passes; ++panel) {
                    for (size_t begin = 0; begin < a.kp; begin += 16) {
                        wide dot = 0;
                        for (size_t k = begin; k < begin + 16; ++k) {
                            if (k >= a.k) continue;
                            const auto d = digits(a.codes[i * a.kp + k], operation == role::pv);
                            const int x = d[limb];
                            const size_t wlimb = operation == role::qk ? panel : 0;
                            dot += wide{x} * b.limbs[wlimb][k * b.n + j];
                        }
                        const unsigned offset = panel == 0 ? 0 : 8;
                        const uint32_t carrier = b.carriers[(begin / 32) * b.n + j];
                        acc = clamp32(wide{acc} + shifted(dot, carrier, offset));
                    }
                }
                // Exactly one atomic ACC addition after the complete low job.
                // No correction belongs to any upper P limb.
                if (operation == role::pv && limb == 0)
                    acc = clamp32(wide{acc} + bias[j]);
                reconstructed += wide{acc} * place;
                // Reconstruct only a completed stream limb, never a K panel.
                if (reconstructed < std::numeric_limits<int64_t>::min() ||
                    reconstructed > std::numeric_limits<int64_t>::max()) return false;
            }
            values[i * b.n + j] = static_cast<int64_t>(reconstructed);
        }
    }
    out = std::move(values);
    return true;
}

} // namespace potal_test::attention_golden
