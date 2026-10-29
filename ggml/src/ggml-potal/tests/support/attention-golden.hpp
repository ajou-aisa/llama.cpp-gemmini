#pragma once

// Independent physical-DIM attention arithmetic. No production reference, SCU helper,
// quantizer, or emulator arithmetic is used to calculate expected results.
#include "matmul/attention.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
            for (size_t begin = 0; begin < a.kp; begin += POTAL_HW_DIM) {
                wide dot = 0;
                for (size_t k = begin; k < begin + POTAL_HW_DIM && k < a.k; ++k)
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
                    for (size_t begin = 0; begin < a.kp; begin += POTAL_HW_DIM) {
                        wide dot = 0;
                        for (size_t k = begin; k < begin + POTAL_HW_DIM; ++k) {
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

// H3-7C: raw command outputs are retained even when the interval certificate
// refuses the job. This distinguishes exact dots from saturating hardware math.
struct pv_planes {
    std::vector<std::vector<int8_t>> digits;
    std::vector<std::vector<int32_t>> raw;
    std::vector<int64_t> integers;
    bool certified = true;
};

inline bool pv_metadata(const stream & a, const stationary & b) {
    if (!a.rows || !a.k || !b.n || a.k != b.k || a.kp != b.kp ||
        a.kp < a.k || a.kp % 32 || a.rows > SIZE_MAX / a.kp ||
        b.n > SIZE_MAX / b.kp || a.rows > SIZE_MAX / b.n || a.codes.size() != a.rows * a.kp ||
        b.limbs[0].size() != b.kp * b.n || b.base.size() != b.n ||
        b.carriers.size() != b.kp / 32 * b.n) return false;
    for (uint32_t carrier : b.carriers)
        if (carrier != POTAL_CARRIER_ZERO && carrier > POTAL_CARRIER_MAX_SHIFT) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t k = 0; k < a.kp; ++k)
            if (a.codes[i*a.kp+k] < 0 || a.codes[i*a.kp+k] > 65535 ||
                (k >= a.k && a.codes[i*a.kp+k] != 0)) return false;
    return true;
}

// Sparse signed binary normalization cancels large carriers BEFORE attempting
// any bounded shift. No negative shift or shift by the width of __int128.
inline bool binary_sum(std::map<uint32_t, wide> terms, wide & out) {
    for (auto it = terms.begin(); it != terms.end(); ++it) {
        const wide carry = it->second / 2;
        it->second %= 2;
        if (carry) {
            if (it->first == UINT32_MAX) return false;
            terms[it->first + 1] += carry;
        }
    }
    const wide maximum = std::numeric_limits<wide>::max();
    const wide minimum = std::numeric_limits<wide>::min();
    wide value = 0;
    uint32_t previous = terms.empty() ? 0 : terms.rbegin()->first;
    for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
        uint32_t gap = previous - it->first;
        if (value == 0) gap = 0;
        bool consumed = false;
        while (gap) {
            // 2^127 - 1 fits even though the intermediate doubling does not.
            if (gap == 1 && value == maximum/2+1 && it->second == -1) {
                value = maximum;
                consumed = true;
                break;
            }
            if (value > maximum / 2 || value < minimum / 2) return false;
            value *= 2;
            --gap;
        }
        if (!consumed) {
            if ((it->second > 0 && value == maximum) ||
                (it->second < 0 && value == minimum)) return false;
            value += it->second;
        }
        previous = it->first;
    }
    if (value != 0) {
        if (previous >= 128) return false;
        while (previous--) {
            if (value > maximum / 2 || value < minimum / 2) return false;
            value *= 2;
        }
    }
    out = value;
    return true;
}

inline bool pv_exact(const stream & a, const stationary & b, std::vector<wide> & out) {
    if (!pv_metadata(a,b)) return false;
    std::vector<wide> values(a.rows*b.n);
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < b.n; ++j) {
            std::map<uint32_t, wide> terms;
            for (size_t k = 0; k < a.k; ++k) {
                const auto carrier = b.carriers[k/32*b.n+j];
                if (carrier != POTAL_CARRIER_ZERO)
                    terms[carrier] += wide{a.codes[i*a.kp+k]} * b.limbs[0][k*b.n+j];
            }
            if (!binary_sum(std::move(terms),values[i*b.n+j])) return false;
        }
    out = std::move(values);
    return true;
}

inline bool pv_bitplanes(const stream & a, const stationary & b, pv_planes & out) {
    if (!pv_metadata(a,b)) return false;
    pv_planes result;
    int32_t maximum = *std::max_element(a.codes.begin(),a.codes.end());
    size_t count = 1;
    for (int32_t x = maximum / 2; x; x /= 2) ++count;
    result.digits.assign(count,std::vector<int8_t>(a.codes.size()));
    result.raw.assign(count,std::vector<int32_t>(a.rows*b.n));
    result.integers.assign(a.rows*b.n,0);
    int64_t place = 1;
    for (size_t p = 0; p < count; ++p, place *= 2) {
        for (size_t x = 0; x < a.codes.size(); ++x)
            result.digits[p][x] = int8_t((a.codes[x] / place) % 2);
        for (size_t i = 0; i < a.rows; ++i)
            for (size_t j = 0; j < b.n; ++j) {
                int32_t acc = 0;
                wide prefix_low = 0, prefix_high = 0;
                for (size_t begin = 0; begin < a.kp; begin += POTAL_HW_DIM) {
                    int low = 1, high = 0;
                    wide positive = 0, negative = 0, dot = 0;
                    for (size_t k = begin; k < begin+POTAL_HW_DIM; ++k) {
                        const int digit = result.digits[p][i*a.kp+k];
                        const int v = b.limbs[0][k*b.n+j];
                        dot += digit * v;
                        if (k < a.k) {
                            low = std::min(low,digit);
                            high = std::max(high,digit);
                            positive += std::max(v,0);
                            negative += std::min(v,0);
                        }
                    }
                    if (begin >= a.k) low = high = 0;
                    wide lower = low*positive + high*negative;
                    wide upper = high*positive + low*negative;
                    const auto carrier = b.carriers[begin/32*b.n+j];
                    if (carrier == POTAL_CARRIER_ZERO) lower = upper = 0;
                    else if (carrier >= 32) {
                        if (lower || upper) result.certified = false;
                        // Saturated bounds still force rejection without a wide shift.
                        lower = shifted(lower,carrier,0);
                        upper = shifted(upper,carrier,0);
                    } else {
                        lower *= wide{1} << carrier;
                        upper *= wide{1} << carrier;
                    }
                    prefix_low += lower;
                    prefix_high += upper;
                    if (lower < INT32_MIN || upper > INT32_MAX ||
                        prefix_low < INT32_MIN || prefix_high > INT32_MAX)
                        result.certified = false;
                    acc = clamp32(wide{acc} + shifted(dot,carrier,0));
                }
                result.raw[p][i*b.n+j] = acc;
                result.integers[i*b.n+j] += int64_t(acc)*place;
            }
    }
    out = std::move(result);
    return true;
}

} // namespace potal_test::attention_golden
