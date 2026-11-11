#include "attention.hpp"

#include "dense.hpp"
#include "ggml-gemmini.h"
#include "ggml-gemmini/quants/act/exsia/local.hpp"
#include "ggml-gemmini/quants/act/exsia/exsia_shift.hpp"
#include "ggml-gemmini/quants/common/math.hpp"

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <memory>
#include <stdexcept>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif

#include "trace/timeline.hpp"

namespace ggml::potal::attention {
namespace {

namespace exsia = ggml::gemmini::quants::act::exsia;
constexpr int16_t zero_exp = std::numeric_limits<int16_t>::min();
constexpr int bits = 8, radix = 256, bias = 128, rho = 14;

void require(bool condition, potal_status status = POTAL_ERR_INVALID_ARGUMENT) {
    if (!condition) {
        throw status;
    }
}

template<class F> potal_status checked(F && action) {
    try {
        action();
        return POTAL_OK;
    } catch (potal_status status) {
        return status;
    } catch (const std::bad_alloc &) {
        return POTAL_ERR_CAPACITY;
    } catch (const std::length_error &) {
        return POTAL_ERR_CAPACITY;
    }
}

size_t product(size_t a, size_t b) {
    require(a == 0 || b <= SIZE_MAX / a, POTAL_ERR_CAPACITY);
    return a * b;
}

size_t padded_k(size_t k) {
    require(k != 0 && k <= SIZE_MAX - (block_k - 1), POTAL_ERR_CAPACITY);
    return (k + block_k - 1) / block_k * block_k;
}

struct nearest_rounding {
    int previous = std::fegetround();
    nearest_rounding() {
        require(previous != -1 && std::fesetround(FE_TONEAREST) == 0);
    }
    ~nearest_rounding() {
        (void) std::fesetround(previous);
    }
};

bool floating(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

bool host_memory(const ggml_tensor & tensor) {
    const auto * base = tensor.view_src ? tensor.view_src : &tensor;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

// Accept permutations and row gaps, but not overlapping elements or overflow.
bool layout(const ggml_tensor & tensor) {
    std::array<size_t, 4> order = { 0, 1, 2, 3 };
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tensor.nb[a] < tensor.nb[b]; });
    size_t extent = ggml_type_size(tensor.type);
    for (size_t d : order) {
        if (tensor.ne[d] <= 0 || tensor.nb[d] < ggml_type_size(tensor.type)) {
            return false;
        }
        const size_t count = static_cast<size_t>(tensor.ne[d]);
        if (count > 1) {
            if (tensor.nb[d] < extent || count - 1 > (SIZE_MAX - extent) / tensor.nb[d]) {
                return false;
            }
            extent += (count - 1) * tensor.nb[d];
        }
    }
    return tensor.data == nullptr || reinterpret_cast<uintptr_t>(tensor.data) <= UINTPTR_MAX - extent;
}

void input_tensor(const ggml_tensor & tensor, size_t head, size_t batch) {
    require(floating(tensor.type) && tensor.data && layout(tensor) && host_memory(tensor));
    require(head < size_t(tensor.ne[2]) && batch < size_t(tensor.ne[3]));
}

float read_value(const ggml_tensor & tensor, size_t k, size_t row, size_t head, size_t batch) {
    const auto * p = static_cast<const char *>(tensor.data) +
                    k * tensor.nb[0] + row * tensor.nb[1] + head * tensor.nb[2] + batch * tensor.nb[3];
    float value;
    if (tensor.type == GGML_TYPE_F16) {
        ggml_fp16_t half;
        std::memcpy(&half, p, sizeof(half));
        value = ggml_fp16_to_fp32(half);
    } else {
        std::memcpy(&value, p, sizeof(value));
    }
    require(std::isfinite(value));
    return value;
}

int32_t sat32(int64_t value) {
    return value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : static_cast<int32_t>(value);
}

int32_t scu(int32_t partial, uint32_t carrier, unsigned offset) {
    require(potal_carrier_is_valid(carrier));
    if (carrier == POTAL_CARRIER_ZERO || partial == 0) {
        return 0;
    }
    const uint32_t shift = carrier + offset;
    if (shift >= 32) {
        return partial < 0 ? INT32_MIN : INT32_MAX;
    }
    return sat32(int64_t(partial) * (int64_t{1} << shift));
}

void reconstruct(int64_t & accumulator, int64_t part, size_t limb, unsigned digit_bits = bits) {
    const __int128_t value = static_cast<__int128_t>(accumulator) +
                             static_cast<__int128_t>(part) * (static_cast<__int128_t>(1) << (digit_bits * limb));
    require(value >= INT64_MIN && value <= INT64_MAX, POTAL_ERR_CAPACITY);
    accumulator = static_cast<int64_t>(value);
}

void split(stream & out, size_t index, int64_t value) {
    size_t limb = 0;
    do {
        require(limb < max_limbs, POTAL_ERR_CAPACITY);
        const int32_t digit = ggml::gemmini::lowdigit(static_cast<uint64_t>(value), bits);
        out.limbs[limb][index] = static_cast<int8_t>(digit);
        value = (value - digit) / radix;
        ++limb;
    } while (value != 0);
    out.limb_count = std::max(out.limb_count, limb);
}

void prepare_weight(const ggml_tensor & tensor, size_t head, size_t batch, role operation, stationary & out) {
    input_tensor(tensor, head, batch);
    require(operation == role::qk || operation == role::pv);
    out.k = size_t(tensor.ne[0]);
    out.n = size_t(tensor.ne[1]);
    out.kp = padded_k(out.k);
    const size_t blocks = out.kp / block_k;
    out.base.resize(out.n);
    out.carriers.resize(product(blocks, out.n));
    out.limbs[0].resize(product(out.kp, out.n));
    if (operation == role::qk) {
        out.limbs[1].resize(product(out.kp, out.n));
    }
    std::array<float, block_k> values{};
    std::vector<int32_t> codes(block_k);
    std::vector<int16_t> theta(blocks);
    exsia::ExpScanner scanner;
    exsia::WideQuantizer quantizer;
    for (size_t j = 0; j < out.n; ++j) {
        int16_t base = INT16_MAX;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t valid = std::min(block_k, out.k - block * block_k);
            int16_t e = zero_exp;
            for (size_t k = 0; k < valid; ++k) {
                values[k] = read_value(tensor, block * block_k + k, j, head, batch);
                e = std::max(e, scanner.unbiased_exp(values[k]));
            }
            theta[block] = e == zero_exp ? zero_exp : int16_t(e - (operation == role::qk ? rho : bits - 2));
            if (e == zero_exp) {
                continue;
            }
            base = std::min(base, theta[block]);
            quantizer.quantize_block(values.data(), valid, theta[block], codes);
            for (size_t k = 0; k < valid; ++k) {
                const int32_t code = std::clamp(codes[k], operation == role::qk ? -32768 : -128,
                                               operation == role::qk ? 32639 : 127);
                const size_t index = (block * block_k + k) * out.n + j;
                const int32_t low = operation == role::qk ? ggml::gemmini::lowdigit(code, bits) : code;
                out.limbs[0][index] = static_cast<int8_t>(low);
                if (operation == role::qk) {
                    out.limbs[1][index] = static_cast<int8_t>((code - low) / radix);
                }
            }
        }
        out.base[j] = base == INT16_MAX ? 0 : base;
        for (size_t block = 0; block < blocks; ++block) {
            const int difference = int(theta[block]) - int(out.base[j]);
            require(theta[block] == zero_exp || (difference >= 0 && difference <= int(POTAL_CARRIER_MAX_SHIFT) - bits));
            out.carriers[block * out.n + j] =
                theta[block] == zero_exp ? POTAL_CARRIER_ZERO : static_cast<uint32_t>(difference);
        }
    }
}

// One stripe's activation in three steps that the strict pipeline runs on different harts
// (execute_strict): init (sizes), local (disjoint row ranges: finite reads, P range, QK ExSIA
// local stage per block) and fold (stripe-wide: QK anchor and shift, P scale and quantization,
// radix limbs). prepare_activation runs the same three steps on one thread.
struct activation_parts {
    std::vector<float>   values;    // [rows][kp], zero padding
    std::vector<int16_t> exponents; // QK: [rows][kp / 32] local block exponents
};

void init_activation(const ggml_tensor & tensor, size_t head, size_t batch, size_t begin, size_t rows,
                     role operation, stream & out, activation_parts & parts) {
    input_tensor(tensor, head, batch);
    require(operation == role::qk || operation == role::pv);
    require(rows != 0 && begin < size_t(tensor.ne[1]) && rows <= size_t(tensor.ne[1]) - begin);
    out.rows = rows;
    out.k = size_t(tensor.ne[0]);
    out.kp = padded_k(out.k);
    const size_t count = product(rows, out.kp);
    parts.values.assign(count, 0.0f);
    out.codes.assign(count, 0);
    for (auto & limb : out.limbs) {
        limb.assign(count, 0);
    }
    parts.exponents.assign(operation == role::qk ? product(rows, out.kp / block_k) : 0, 0);
}

void local_activation(const ggml_tensor & tensor, size_t head, size_t batch, size_t begin, size_t row_begin,
                      size_t row_end, role operation, stream & out, activation_parts & parts) {
    require(row_begin <= row_end && row_end <= out.rows);
    const size_t blocks = out.kp / block_k;
    for (size_t i = row_begin; i < row_end; ++i) {
        for (size_t k = 0; k < out.k; ++k) {
            const float value = read_value(tensor, k, begin + i, head, batch);
            require(operation == role::qk || (value >= 0 && value <= 1));
            parts.values[i * out.kp + k] = value;
        }
    }
    if (operation != role::qk || row_begin == row_end) {
        return;
    }
    exsia::Meta meta;
    meta.rho = rho;
    exsia::ExSIAState state;
    state.B_size = block_k;
    state.K_logical = out.k;
    state.K_padded = out.kp;
    state.blocks_per_row = blocks;
    exsia::StripeScratch scratch;
    require(scratch.prepare(block_k), POTAL_ERR_CAPACITY);
    exsia::LocalStage local;
    for (size_t i = row_begin; i < row_end; ++i) {
        for (size_t block = 0; block < blocks; ++block) {
            uint64_t mask_word = 0;
            exsia::BlockMask mask(&mask_word, block_k);
#if EXSIA_BRANCH_COUNTS_ENABLED
            exsia::LocalBlockCycleSample sample;
#endif
            require(local.run_optimized(meta, state, parts.values.data() + i * out.kp + block * block_k,
                                        std::min(block_k, out.k - block * block_k), block_k, i, block,
                                        scratch, mask, out.codes.data() + i * out.kp + block * block_k,
                                        parts.exponents[i * blocks + block]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                        , sample
#endif
                                        ));
        }
    }
}

void fold_activation(role operation, stream & out, activation_parts & parts) {
    const size_t blocks = out.kp / block_k;
    exsia::ExpScanner scanner;
    if (operation == role::qk) {
        exsia::StripeState top;
        for (const int16_t e : parts.exponents) {
            scanner.update_stripe_top3_exp(top, e);
        }
        // Stripe anchor rank as in ExSIA folding (folding.cpp): TOP3 takes e3 when it exists.
        const int16_t anchor = GGML_GEMMINI_EXSIA_STRIPE_ANCHOR == 3 && top.e3 != zero_exp ? top.e3
                             : top.e2 != zero_exp ? top.e2 : top.e1 != zero_exp ? top.e1 : 0;
        out.scale = int(anchor) - rho;
        for (size_t i = 0; i < out.rows; ++i) {
            for (size_t k = 0; k < out.k; ++k) {
                const int16_t e = parts.exponents[i * blocks + k / block_k];
                out.codes[i * out.kp + k] = exsia::detail::shift_q_i32(
                    out.codes[i * out.kp + k], e == zero_exp ? 0 : int16_t(e - anchor));
            }
        }
    } else {
        const float maximum = *std::max_element(parts.values.begin(), parts.values.end());
        out.scale = (maximum > 0 ? int(scanner.unbiased_exp(maximum)) : 0) - 15;
        exsia::WideQuantizer quantizer;
        quantizer.quantize_block(parts.values.data(), parts.values.size(), int16_t(out.scale), out.codes);
    }
    for (size_t i = 0; i < out.rows; ++i) {
        for (size_t k = 0; k < out.k; ++k) {
            const size_t index = i * out.kp + k;
            if (operation == role::pv) {
                out.codes[index] = std::clamp(out.codes[index], 0, 65535);
            }
            split(out, index, int64_t(out.codes[index]) -
                  (operation == role::pv && out.algorithm == pv_algorithm::legacy ? bias : 0));
        }
    }
}

void prepare_activation(const ggml_tensor & tensor, size_t head, size_t batch, size_t begin,
                         size_t rows, role operation, stream & out) {
    activation_parts parts;
    init_activation(tensor, head, batch, begin, rows, operation, out, parts);
    local_activation(tensor, head, batch, begin, 0, rows, operation, out, parts);
    fold_activation(operation, out, parts);
}

void validate_prepared(const stream & activation, const stationary & weight, role operation) {
    require(operation == role::qk || operation == role::pv);
    require(activation.rows && weight.n && activation.k && activation.k == weight.k &&
            activation.kp == weight.kp && activation.kp % block_k == 0 && activation.k <= activation.kp &&
            activation.limb_count && activation.limb_count <= max_limbs);
    const size_t a_count = product(activation.rows, activation.kp);
    const size_t b_count = product(weight.kp, weight.n);
    require(weight.base.size() == weight.n && weight.carriers.size() == product(weight.kp / block_k, weight.n));
    require(weight.limbs[0].size() == b_count &&
            (operation != role::qk || weight.limbs[1].size() == b_count));
    for (size_t limb = 0; limb < activation.limb_count; ++limb) {
        require(activation.limbs[limb].size() == a_count);
    }
    for (uint32_t carrier : weight.carriers) {
        require(potal_carrier_is_valid(carrier));
    }
}

void make_bias(const stationary & weight, std::vector<int32_t> & out) {
    require(weight.k && weight.n && weight.k <= weight.kp && weight.kp % block_k == 0 &&
            weight.limbs[0].size() == product(weight.kp, weight.n) &&
            weight.carriers.size() == product(weight.kp / block_k, weight.n));
    out.resize(weight.n);
    for (size_t j = 0; j < weight.n; ++j) {
        uint32_t maximum = 0;
        for (size_t block = 0; block < weight.kp / block_k; ++block) {
            const uint32_t carrier = weight.carriers[block * weight.n + j];
            require(potal_carrier_is_valid(carrier));
            if (carrier != POTAL_CARRIER_ZERO) {
                maximum = std::max(maximum, carrier);
            }
        }
        // Group exact block sums by exponent. Binary carry normalization keeps
        // cancellation exact even for shifts much wider than INT64/__int128.
        std::vector<int64_t> coefficients(size_t(maximum) + 1);
        for (size_t block = 0; block < weight.kp / block_k; ++block) {
            const uint32_t carrier = weight.carriers[block * weight.n + j];
            if (carrier == POTAL_CARRIER_ZERO) {
                continue;
            }
            int64_t sum = 0;
            for (size_t k = block * block_k; k < (block + 1) * block_k && k < weight.k; ++k) {
                sum += weight.limbs[0][k * weight.n + j];
            }
            const __int128_t combined = static_cast<__int128_t>(coefficients[carrier]) + sum;
            require(combined >= INT64_MIN && combined <= INT64_MAX, POTAL_ERR_CAPACITY);
            coefficients[carrier] = static_cast<int64_t>(combined);
        }
        // bias = 128*S fits INT32 iff S is a signed25 value. Normalize through
        // 64 extra sign bits instead of clamping wide terms before cancellation.
        int64_t carry = 0;
        uint32_t low = 0;
        bool high_zero = true, high_one = true;
        for (size_t shift = 0; shift <= size_t(maximum) + 64; ++shift) {
            const __int128_t wide = static_cast<__int128_t>(carry) +
                                    (shift < coefficients.size() ? coefficients[shift] : 0);
            require(wide >= INT64_MIN && wide <= INT64_MAX, POTAL_ERR_CAPACITY);
            const int64_t coefficient = static_cast<int64_t>(wide);
            const uint32_t bit = static_cast<uint64_t>(coefficient) & 1u;
            if (shift < 25) {
                low |= bit << shift;
            } else {
                high_zero = high_zero && bit == 0;
                high_one = high_one && bit == 1;
            }
            carry = static_cast<int64_t>((wide - bit) / 2);
        }
        const bool negative = (low & (uint32_t{1} << 24)) != 0;
        require(negative ? high_one && carry == -1 : high_zero && carry == 0, POTAL_ERR_UNSUPPORTED);
        const int64_t sum = int64_t(low) - (negative ? (int64_t{1} << 25) : 0);
        out[j] = static_cast<int32_t>(sum * bias);
    }
}

// A rejected bias does not imply that another saturated decomposition is
// equivalent. Admit unsigned bit planes only when intervals prove that every
// SCU contribution and every ascending-K ACC prefix stays in signed INT32.
// This bounds arithmetic; it does not compute a CPU dot product or output.
unsigned pv_bitplanes(const stream & activation, const stationary & weight,
                      std::vector<int32_t> & correction) {
    try {
        make_bias(weight, correction);
        return 0; // Preserve the complete legacy path, including saturation.
    } catch (potal_status status) {
        if (status != POTAL_ERR_UNSUPPORTED) {
            throw;
        }
    }
    require(POTAL_HW_DIM == 16 || POTAL_HW_DIM == 32, POTAL_ERR_UNSUPPORTED);
    require(activation.codes.size() == product(activation.rows, activation.kp));
    uint32_t maximum = 0;
    for (size_t i = 0; i < activation.rows; ++i) {
        for (size_t k = 0; k < activation.kp; ++k) {
            const int32_t code = activation.codes[i * activation.kp + k];
            require(code >= 0 && code <= 65535 && (k < activation.k || code == 0));
            maximum = std::max(maximum, uint32_t(code));
        }
    }
    unsigned planes = 1;
    while (maximum >> planes) {
        ++planes;
    }
    const size_t tiles = activation.kp / POTAL_HW_DIM;
    // [negative sum, positive sum] for each physical DIM16 or DIM32 V microtile.
    std::vector<std::array<int32_t, 2>> sums(product(tiles, weight.n));
    for (size_t t = 0; t < tiles; ++t) {
        for (size_t j = 0; j < weight.n; ++j) {
            for (size_t k = t * POTAL_HW_DIM; k < std::min((t + 1) * POTAL_HW_DIM, weight.k); ++k) {
                const int32_t value = weight.limbs[0][k * weight.n + j];
                sums[t * weight.n + j][value < 0 ? 0 : 1] += value;
            }
        }
    }
    // AND/OR give the exact minimum/maximum digit of every binary plane.
    std::vector<std::array<uint32_t, 2>> limits(product(activation.rows, tiles));
    for (size_t i = 0; i < activation.rows; ++i) {
        for (size_t t = 0; t < tiles; ++t) {
            auto & range = limits[i * tiles + t];
            range = {t * POTAL_HW_DIM < activation.k ? 65535u : 0u, 0u};
            for (size_t k = t * POTAL_HW_DIM; k < std::min((t + 1) * POTAL_HW_DIM, activation.k); ++k) {
                const uint32_t code = uint32_t(activation.codes[i * activation.kp + k]);
                range[0] &= code;
                range[1] |= code;
            }
        }
    }
    for (unsigned plane = 0; plane < planes; ++plane) {
        for (size_t i = 0; i < activation.rows; ++i) {
            for (size_t j = 0; j < weight.n; ++j) {
                int64_t lower = 0, upper = 0;
                for (size_t t = 0; t < tiles; ++t) {
                    const uint32_t carrier = weight.carriers[(t * POTAL_HW_DIM / block_k) * weight.n + j];
                    if (carrier == POTAL_CARRIER_ZERO) {
                        continue;
                    }
                    const auto & range = limits[i * tiles + t];
                    const auto & sum = sums[t * weight.n + j];
                    const int64_t minimum = (range[0] >> plane) & 1u;
                    const int64_t maximum = (range[1] >> plane) & 1u;
                    int64_t lo = minimum * sum[1] + maximum * sum[0];
                    int64_t hi = maximum * sum[1] + minimum * sum[0];
                    if (carrier >= 32) {
                        require(lo == 0 && hi == 0, POTAL_ERR_UNSUPPORTED);
                        continue;
                    }
                    const int64_t factor = int64_t{1} << carrier;
                    lo *= factor;
                    hi *= factor;
                    require(lo >= INT32_MIN && hi <= INT32_MAX, POTAL_ERR_UNSUPPORTED);
                    lower += lo;
                    upper += hi;
                    require(lower >= INT32_MIN && upper <= INT32_MAX, POTAL_ERR_UNSUPPORTED);
                }
            }
        }
    }
    correction.clear();
    return planes;
}

void integer_reference(const stream & activation, const stationary & weight, role operation,
                        std::vector<int64_t> & out) {
    validate_prepared(activation, weight, operation);
    out.assign(product(activation.rows, weight.n), 0);
    std::vector<int32_t> correction;
    unsigned planes = 0;
    if (operation == role::pv && activation.algorithm == pv_algorithm::legacy) {
        planes = pv_bitplanes(activation, weight, correction);
    }
    for (size_t limb = 0; limb < (planes ? planes : activation.limb_count); ++limb) {
        for (size_t i = 0; i < activation.rows; ++i) {
            for (size_t j = 0; j < weight.n; ++j) {
                int64_t stationary_sum = 0;
                for (size_t pass = 0; pass < (operation == role::qk ? 2u : 1u); ++pass) {
                    int32_t acc = 0;
                    for (size_t begin = 0; begin < activation.kp; begin += POTAL_HW_DIM) {
                        int32_t partial = 0;
                        for (size_t k = begin; k < begin + POTAL_HW_DIM && k < activation.k; ++k) {
                            const size_t index = i * activation.kp + k;
                            const int32_t digit = planes ? int32_t((uint32_t(activation.codes[index]) >> limb) & 1u) :
                                                          int32_t(activation.limbs[limb][index]);
                            partial += digit *
                                       weight.limbs[operation == role::qk ? pass : 0][k * weight.n + j];
                        }
                        const uint32_t carrier = weight.carriers[(begin / block_k) * weight.n + j];
                        acc = sat32(int64_t(acc) + scu(partial, carrier, 0));
                    }
                    if (operation == role::pv && activation.algorithm == pv_algorithm::legacy && limb == 0 && !planes) {
                        acc = sat32(int64_t(acc) + correction[j]);
                    }
                    reconstruct(stationary_sum, acc, pass);
                }
                reconstruct(out[i * weight.n + j], stationary_sum, limb, planes ? 1 : bits);
            }
        }
    }
}

potal_tile_plan limb_plan(const stream & activation, const stationary & weight, role operation, size_t limb) {
    const size_t passes = operation == role::qk ? 2 : 1;
    const size_t selection_k = product(activation.kp, passes);
    potal_tile_plan plan;
    require(potal_tile_select(activation.rows, weight.n, selection_k, block_k / POTAL_HW_DIM, &plan) == POTAL_OK,
            POTAL_ERR_CAPACITY);
    // Preserve the original doubled-K selection's I/J/K and whole-block
    // panel boundaries. Each QK stationary limb now owns a complete GEMM.
    const size_t original_tiles = weight.kp / POTAL_HW_DIM;
    plan.tile_K = std::min(plan.tile_K, original_tiles);
    if (operation == role::pv && limb == 0) {
        plan.tile_K = std::min(plan.tile_K, size_t(POTAL_HW_BANK_ROWS) /
                                           (std::max(plan.tile_I, plan.tile_J) * POTAL_HW_DIM));
    }
    require(plan.tile_K >= block_k / POTAL_HW_DIM, POTAL_ERR_CAPACITY);
    while (original_tiles % plan.tile_K || plan.tile_K % (block_k / POTAL_HW_DIM)) {
        --plan.tile_K;
    }
    return plan;
}

void execute_stripe(const stream & activation, const stationary & weight, role operation,
                     const planned_runner & run, const corrected_runner & corrected,
                     result & report, std::vector<int64_t> & out) {
    validate_prepared(activation, weight, operation);
    out.assign(product(activation.rows, weight.n), 0);
    std::vector<int32_t> correction;
    unsigned planes = 0;
    const bool legacy = operation == role::pv && activation.algorithm == pv_algorithm::legacy;
    if (legacy) {
        require(bool(corrected), POTAL_ERR_UNSUPPORTED);
        planes = pv_bitplanes(activation, weight, correction);
        report.pv_bitplane_stripes += planes != 0;
    }
    for (size_t limb = 0; limb < (planes ? planes : activation.limb_count); ++limb) {
        const size_t passes = operation == role::qk ? 2 : 1;
        const size_t k_total = activation.kp;
        const size_t selection_k = product(activation.kp, passes);
        require(activation.rows <= UINT32_MAX && weight.n <= UINT32_MAX && selection_k <= UINT32_MAX,
                POTAL_ERR_CAPACITY);
        hp1_operands operands;
        require(operands.allocate({ int64_t(activation.rows), int64_t(weight.n), int64_t(k_total) }),
                POTAL_ERR_CAPACITY);
        std::unique_ptr<void, decltype(&std::free)> bias_dma(nullptr, std::free);
        if (legacy && limb == 0 && !planes) {
            bias_dma.reset(dma_alloc(product(operands.n_pad, sizeof(int32_t))));
            require(bool(bias_dma), POTAL_ERR_CAPACITY);
            std::memcpy(bias_dma.get(), correction.data(), weight.n * sizeof(int32_t));
        }
        for (size_t i = 0; i < activation.rows; ++i) {
            for (size_t k = 0; k < activation.k; ++k) {
                const size_t index = i * activation.kp + k;
                operands.a[i * operands.lda + k] = planes ?
                    int8_t((uint32_t(activation.codes[index]) >> limb) & 1u) : activation.limbs[limb][index];
            }
        }
        for (size_t block = 0; block < weight.kp / block_k; ++block) {
            for (size_t j = 0; j < weight.n; ++j) {
                operands.own.carriers[block * operands.cstride + j] =
                    weight.carriers[block * weight.n + j];
            }
        }
        const potal_tile_plan plan = limb_plan(activation, weight, operation, limb);
        potal_stripe_gemm_desc descriptor{};
        descriptor.work_kind = limb == 0 ? POTAL_STRIPE_WORK_MAIN : POTAL_STRIPE_WORK_RESIDUAL;
        descriptor.m = uint32_t(activation.rows);
        descriptor.n = uint32_t(weight.n);
        descriptor.k = uint32_t(k_total);
        descriptor.a = operands.a;
        descriptor.b = operands.b;
        descriptor.c = operands.c;
        descriptor.lda = operands.lda;
        descriptor.ldb = operands.ldb;
        descriptor.ldc = operands.ldc;
        descriptor.carriers = operands.carriers;
        descriptor.carrier_stride = operands.cstride;
        descriptor.scale_block_k = block_k;
        descriptor.tile_i = uint32_t(plan.tile_I);
        descriptor.tile_j = uint32_t(plan.tile_J);
        descriptor.tile_k = uint32_t(plan.tile_K);
        std::vector<int64_t> stationary_sum;
        if (operation == role::qk) {
            stationary_sum.assign(out.size(), 0);
        }
        for (size_t pass = 0; pass < passes; ++pass) {
            for (size_t k = 0; k < weight.kp; ++k) {
                for (size_t j = 0; j < weight.n; ++j) {
                    operands.own.b[k * operands.ldb + j] =
                        weight.limbs[operation == role::qk ? pass : 0][k * weight.n + j];
                }
            }
            // Regular WS overwrites the first K panel, accumulates later
            // panels, and stores C only at the final panel. Its fence covers
            // that store before Host consumption or reuse for the next limb.
            const planned_result completed = !legacy || limb != 0 || planes ? run(descriptor) :
                corrected(descriptor, static_cast<const int32_t *>(bias_dma.get()), weight.n);
            const auto & work = completed.report;
            report.jobs += work.loop_ws != 0;
            report.loops += work.loop_ws;
            report.fences += work.fences;
            report.scale_writes += work.scale_writes;
            report.scale_releases += work.scale_releases;
            report.tlb_flushes += completed.tlb_flushes;
            report.scale_s += work.scale_s;
            report.issue_s += work.issue_s;
            report.fence_s += work.fence_s;
            if (!completed.ok) {
                report.fatal = completed.fatal;
                report.stage = completed.stage ? completed.stage : "attention_ws";
                if (completed.fatal) {
                    operands.abandon(); // Unknown DMA completion: keep every operand alive.
                    (void) bias_dma.release();
                }
                require(false, completed.status != POTAL_OK ? completed.status : POTAL_ERR_BAD_RESPONSE);
            }
            require(completed.status == POTAL_OK && work.loop_ws && work.fences >= work.loop_ws &&
                    work.scale_writes == work.scale_releases, POTAL_ERR_BAD_RESPONSE);
            report.residual_jobs += limb != 0;
            report.qk_limb_jobs += operation == role::qk;
            report.pv_bitplane_jobs += planes != 0;
            report.correction_jobs += legacy && limb == 0 && !planes;
            report.bias_updates += legacy && limb == 0 && !planes ?
                ((activation.rows + POTAL_HW_DIM - 1) / POTAL_HW_DIM) *
                ((weight.n + POTAL_HW_DIM - 1) / POTAL_HW_DIM) : 0;
            const auto reconstruct_start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < activation.rows; ++i) {
                for (size_t j = 0; j < weight.n; ++j) {
                    const int32_t value = operands.c[i * operands.ldc + j];
                    if (operation == role::qk) {
                        reconstruct(stationary_sum[i * weight.n + j], value, pass);
                    } else {
                        reconstruct(out[i * weight.n + j], value, limb, planes ? 1 : bits);
                    }
                }
            }
            report.reconstruct_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reconstruct_start).count();
        }
        if (operation == role::qk) {
            const auto reconstruct_start = std::chrono::steady_clock::now();
            for (size_t index = 0; index < out.size(); ++index) {
                reconstruct(out[index], stationary_sum[index], limb);
            }
            report.reconstruct_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reconstruct_start).count();
        }
    }
}


// ---- H3-8I-25V strict five-hart attention pipeline (execute_strict) ---------------------------
// One task is one stripe of one (batch, head), in the order of execute(). Hart roles:
//   1/2 (local0/1): local_activation over the two halves of the stripe rows
//   3   (fold)    : the stationary of each (ab, ah) when it changes, fold_activation, operand
//                   staging and the per-limb plans: one prepared job of planned GEMM descriptors
//   0   (issuer)  : the job's descriptors through the owner's planned runner, nothing else
//   4   (final)   : checked INT64 reconstruction (execute_stripe's pass, then limb order) and the
//                   F32 values into the staged output
// Operands, plans, issue order and reconstruction order equal execute(): bitwise identical output.

struct strict_task {
    size_t ab = 0, ah = 0, batch = 0, head = 0, begin = 0, rows = 0;
};

struct gemm_item {
    std::unique_ptr<hp1_operands> ops;
    potal_stripe_gemm_desc         desc{};
    size_t                         limb = 0, pass = 0;
};

struct strict_job {
    size_t                            task = 0;
    size_t                            rows = 0;
    int                               scale = 0;
    std::shared_ptr<const stationary> weight;
    std::vector<gemm_item>            gemms; // limb-major, pass-minor: execute_stripe's issue order

    void abandon() {
        for (auto & g : gemms) {
            if (g.ops) {
                g.ops->abandon(); // unknown DMA completion: keep every operand alive
            }
        }
    }
};

std::unique_ptr<strict_job> stage_job(size_t task, const stream & activation,
                                      const std::shared_ptr<const stationary> & weight, role operation) {
    validate_prepared(activation, *weight, operation);
    const size_t passes = operation == role::qk ? 2 : 1;
    require(activation.rows <= UINT32_MAX && weight->n <= UINT32_MAX &&
            product(activation.kp, passes) <= UINT32_MAX, POTAL_ERR_CAPACITY);
    auto job = std::make_unique<strict_job>();
    job->task = task;
    job->rows = activation.rows;
    job->scale = activation.scale;
    job->weight = weight;
    for (size_t limb = 0; limb < activation.limb_count; ++limb) {
        const potal_tile_plan plan = limb_plan(activation, *weight, operation, limb);
        for (size_t pass = 0; pass < passes; ++pass) {
            gemm_item item;
            item.limb = limb;
            item.pass = pass;
            item.ops = std::make_unique<hp1_operands>();
            hp1_operands & operands = *item.ops;
            require(operands.allocate({ int64_t(activation.rows), int64_t(weight->n), int64_t(activation.kp) }),
                    POTAL_ERR_CAPACITY);
            for (size_t i = 0; i < activation.rows; ++i) {
                for (size_t k = 0; k < activation.k; ++k) {
                    operands.a[i * operands.lda + k] = activation.limbs[limb][i * activation.kp + k];
                }
            }
            for (size_t block = 0; block < weight->kp / block_k; ++block) {
                for (size_t j = 0; j < weight->n; ++j) {
                    operands.own.carriers[block * operands.cstride + j] = weight->carriers[block * weight->n + j];
                }
            }
            for (size_t k = 0; k < weight->kp; ++k) {
                for (size_t j = 0; j < weight->n; ++j) {
                    operands.own.b[k * operands.ldb + j] =
                        weight->limbs[operation == role::qk ? pass : 0][k * weight->n + j];
                }
            }
            potal_stripe_gemm_desc & descriptor = item.desc;
            descriptor.work_kind = limb == 0 ? POTAL_STRIPE_WORK_MAIN : POTAL_STRIPE_WORK_RESIDUAL;
            descriptor.m = uint32_t(activation.rows);
            descriptor.n = uint32_t(weight->n);
            descriptor.k = uint32_t(activation.kp);
            descriptor.a = operands.a;
            descriptor.b = operands.b;
            descriptor.c = operands.c;
            descriptor.lda = operands.lda;
            descriptor.ldb = operands.ldb;
            descriptor.ldc = operands.ldc;
            descriptor.carriers = operands.carriers;
            descriptor.carrier_stride = operands.cstride;
            descriptor.scale_block_k = block_k;
            descriptor.tile_i = uint32_t(plan.tile_I);
            descriptor.tile_j = uint32_t(plan.tile_J);
            descriptor.tile_k = uint32_t(plan.tile_K);
            job->gemms.push_back(std::move(item));
        }
    }
    return job;
}

// Hart 0: the job's GEMMs in order; the checks and counters of execute_stripe.
void issue_job(strict_job & job, role operation, const planned_runner & run, result & report) {
    for (auto & g : job.gemms) {
        const planned_result completed = run(g.desc);
        const auto & work = completed.report;
        report.jobs += work.loop_ws != 0;
        report.loops += work.loop_ws;
        report.fences += work.fences;
        report.scale_writes += work.scale_writes;
        report.scale_releases += work.scale_releases;
        report.tlb_flushes += completed.tlb_flushes;
        report.scale_s += work.scale_s;
        report.issue_s += work.issue_s;
        report.fence_s += work.fence_s;
        if (!completed.ok) {
            report.fatal = completed.fatal;
            report.stage = completed.stage ? completed.stage : "attention_ws";
            if (completed.fatal) {
                job.abandon();
            }
            require(false, completed.status != POTAL_OK ? completed.status : POTAL_ERR_BAD_RESPONSE);
        }
        require(completed.status == POTAL_OK && work.loop_ws && work.fences >= work.loop_ws &&
                work.scale_writes == work.scale_releases, POTAL_ERR_BAD_RESPONSE);
        report.residual_jobs += g.limb != 0;
        report.qk_limb_jobs += operation == role::qk;
    }
}

// Hart 4: execute_stripe's reconstruction order, then execute()'s F32 conversion.
void reconstruct_job(const strict_job & job, role operation, const strict_task & task, size_t heads,
                     size_t rows_total, std::vector<float> & staged, double & reconstruct_s) {
    const stationary & weight = *job.weight;
    const size_t n = weight.n, passes = operation == role::qk ? 2 : 1;
    const auto start = std::chrono::steady_clock::now();
    std::vector<int64_t> out(product(job.rows, n), 0);
    std::vector<int64_t> stationary_sum;
    size_t index = 0;
    while (index < job.gemms.size()) {
        const size_t limb = job.gemms[index].limb;
        if (operation == role::qk) {
            stationary_sum.assign(out.size(), 0);
        }
        for (size_t pass = 0; pass < passes; ++pass, ++index) {
            const gemm_item & g = job.gemms[index];
            require(g.limb == limb && g.pass == pass, POTAL_ERR_BAD_RESPONSE);
            for (size_t i = 0; i < job.rows; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    const int32_t value = g.ops->c[i * g.ops->ldc + j];
                    if (operation == role::qk) {
                        reconstruct(stationary_sum[i * n + j], value, pass);
                    } else {
                        reconstruct(out[i * n + j], value, limb, bits);
                    }
                }
            }
        }
        if (operation == role::qk) {
            for (size_t e = 0; e < out.size(); ++e) {
                reconstruct(out[e], stationary_sum[e], limb);
            }
        }
    }
    for (size_t i = 0; i < job.rows; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float value = static_cast<float>(std::ldexp(static_cast<double>(out[i * n + j]),
                                                              job.scale + int(weight.base[j])));
            require(std::isfinite(value), POTAL_ERR_CAPACITY);
            staged[((task.batch * heads + task.head) * rows_total + task.begin + i) * n + j] = value;
        }
    }
    reconstruct_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

uint32_t layer_of(const ggml_tensor * op) {
    for (const ggml_tensor * t : std::array<const ggml_tensor *, 3>{ { op, op->src[1], op->src[0] } }) {
        if (t == nullptr) continue;
        const char * name = t->name;
        const size_t length = std::strlen(name);
        size_t digits = 0;
        while (digits < length && name[length - 1 - digits] >= '0' && name[length - 1 - digits] <= '9') ++digits;
        if (digits != 0 && digits < 9 && digits < length && name[length - 1 - digits] == '-') {
            return uint32_t(std::strtoul(name + length - digits, nullptr, 10));
        }
    }
    return ::potal::timeline::no_id;
}
} // namespace

role get_role(const ggml_tensor * op) {
    switch (ggml_gemmini_get_attention_role(op)) {
        case GGML_GEMMINI_ATTENTION_QK: return role::qk;
        case GGML_GEMMINI_ATTENTION_PV: return role::pv;
        default: return role::none;
    }
}

ggml_potal_reject check(const ggml_tensor * op, bool require_data) {
    if (!op || op->op != GGML_OP_MUL_MAT || get_role(op) == role::none) {
        return GGML_POTAL_REJECT_OP_PARAMS;
    }
    if (op->op_params[0] != GGML_PREC_DEFAULT && op->op_params[0] != GGML_PREC_F32) {
        return GGML_POTAL_REJECT_OP_PARAMS;
    }
    for (size_t i = 2; i < sizeof(op->op_params) / sizeof(op->op_params[0]); ++i) {
        if (op->op_params[i]) {
            return GGML_POTAL_REJECT_OP_PARAMS;
        }
    }
    if (!op->src[0] || !op->src[1] || !floating(op->src[0]->type) || !floating(op->src[1]->type) ||
        op->type != GGML_TYPE_F32) {
        return GGML_POTAL_REJECT_TYPE;
    }
    const auto & a = *op->src[0];
    const auto & b = *op->src[1];
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] <= 0 || b.ne[d] <= 0 || op->ne[d] <= 0 ||
            a.ne[d] > UINT32_MAX / 2 || b.ne[d] > UINT32_MAX / 2 || op->ne[d] > UINT32_MAX / 2) {
            return GGML_POTAL_REJECT_SHAPE;
        }
    }
    if (a.ne[0] != b.ne[0] || b.ne[2] % a.ne[2] || b.ne[3] % a.ne[3] ||
        op->ne[0] != a.ne[1] || op->ne[1] != b.ne[1] || op->ne[2] != b.ne[2] || op->ne[3] != b.ne[3]) {
        return GGML_POTAL_REJECT_SHAPE;
    }
    if (!layout(a) || !layout(b) || !layout(*op)) {
        return GGML_POTAL_REJECT_LAYOUT;
    }
    for (const ggml_tensor * tensor : { &a, &b, op }) {
        if (!host_memory(*tensor) || (require_data && tensor->data == nullptr)) {
            return GGML_POTAL_REJECT_MEMORY;
        }
    }
    return GGML_POTAL_REJECT_NONE;
}

potal_status prepare_stationary(const ggml_tensor & tensor, size_t head, size_t batch,
                                 role operation, stationary & out) {
    return checked([&] {
        nearest_rounding rounding;
        stationary prepared;
        prepare_weight(tensor, head, batch, operation, prepared);
        out = std::move(prepared);
    });
}

potal_status prepare_stream(const ggml_tensor & tensor, size_t head, size_t batch,
                            size_t begin, size_t rows, role operation, stream & out, pv_algorithm algorithm) {
    return checked([&] {
        nearest_rounding rounding;
        stream prepared;
        prepared.algorithm = operation == role::pv ? algorithm : pv_algorithm::legacy;
        prepare_activation(tensor, head, batch, begin, rows, operation, prepared);
        out = std::move(prepared);
    });
}

potal_status reference(const stream & activation, const stationary & weight,
                       role operation, std::vector<int64_t> & out) {
    return checked([&] {
        std::vector<int64_t> staged;
        integer_reference(activation, weight, operation, staged);
        out = std::move(staged);
    });
}

potal_status column_bias(const stationary & weight, std::vector<int32_t> & out) {
    return checked([&] {
        std::vector<int32_t> staged;
        make_bias(weight, staged);
        out = std::move(staged);
    });
}

result execute(ggml_tensor * op, const planned_runner & run, const corrected_runner & corrected, pv_algorithm algorithm) {
    result report;
    if (check(op, true) != GGML_POTAL_REJECT_NONE || !run) {
        report.status = POTAL_ERR_INVALID_ARGUMENT;
        report.stage = "attention_contract";
        return report;
    }
    report.status = checked([&] {
        nearest_rounding rounding;
        const auto & a = *op->src[0];
        const auto & b = *op->src[1];
        const role operation = get_role(op);
        const size_t rows = size_t(b.ne[1]), n = size_t(a.ne[1]);
        potal_tile_plan plan;
        require(potal_tile_select(rows, n, padded_k(size_t(b.ne[0])), block_k / POTAL_HW_DIM, &plan) == POTAL_OK,
                POTAL_ERR_CAPACITY);
        const size_t stripe_rows = plan.tile_I * POTAL_HW_DIM;
        const size_t elements = product(product(product(n, rows), size_t(b.ne[2])), size_t(b.ne[3]));
        std::vector<float> staged(elements);
        const size_t head_repeat = size_t(b.ne[2] / a.ne[2]);
        const size_t batch_repeat = size_t(b.ne[3] / a.ne[3]);
        for (size_t ab = 0; ab < size_t(a.ne[3]); ++ab) {
            for (size_t ah = 0; ah < size_t(a.ne[2]); ++ah) {
                stationary weight;
                prepare_weight(a, ah, ab, operation, weight);
                for (size_t batch = ab * batch_repeat; batch < (ab + 1) * batch_repeat; ++batch) {
                    for (size_t head = ah * head_repeat; head < (ah + 1) * head_repeat; ++head) {
                        for (size_t begin = 0; begin < rows; begin += stripe_rows) {
                            stream activation;
                            activation.algorithm = operation == role::pv ? algorithm : pv_algorithm::legacy;
                            prepare_activation(b, head, batch, begin, std::min(stripe_rows, rows - begin),
                                               operation, activation);
                            std::vector<int64_t> integers;
                            execute_stripe(activation, weight, operation, run, corrected, report, integers);
                            for (size_t i = 0; i < activation.rows; ++i) {
                                for (size_t j = 0; j < n; ++j) {
                                    const float value = static_cast<float>(std::ldexp(
                                        static_cast<double>(integers[i * n + j]), activation.scale + int(weight.base[j])));
                                    require(std::isfinite(value), POTAL_ERR_CAPACITY);
                                    staged[((batch * size_t(b.ne[2]) + head) * rows + begin + i) * n + j] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Causal masking belongs to ggml_soft_max_ext AFTER the raw QK MatMul.
        // No unmasked MatMul output is replaced or skipped using src[2].
        for (size_t batch = 0; batch < size_t(op->ne[3]); ++batch) {
            for (size_t head = 0; head < size_t(op->ne[2]); ++head) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        auto * destination = static_cast<char *>(op->data) + batch * op->nb[3] +
                                            head * op->nb[2] + i * op->nb[1] + j * op->nb[0];
                        const float value = staged[((batch * size_t(op->ne[2]) + head) * rows + i) * n + j];
                        std::memcpy(destination, &value, sizeof(value));
                    }
                }
            }
        }
    });
    if (report.status != POTAL_OK && !report.stage) {
        report.stage = "attention_prepare_or_reconstruct";
    }
    return report;
}

result execute_hardware(ggml_tensor * op) {
    result report;
    issuer_owner owner;
    std::string why;
    potal_status status = POTAL_OK;
    if (!owner.start([&](const owner_runners & runners) {
            report = execute(op, runners.planned, [](const potal_stripe_gemm_desc & desc,
                                                      const int32_t * bias, size_t columns) {
                return run_corrected(desc, bias, columns);
            });
            return consume_outcome{ report.fatal, report.status, report.stage };
        }, why, status)) {
        report.status = status;
        report.stage = "attention_issuer";
        return report;
    }
    owner.join();
    return report;
}

result execute_strict(ggml_tensor * op, stripe_owner & owner, const std::array<int, 5> & role_cpus, bool pin,
                      pv_algorithm algorithm) {
    result report;
    if (check(op, true) != GGML_POTAL_REJECT_NONE) {
        report.status = POTAL_ERR_INVALID_ARGUMENT;
        report.stage = "attention_contract";
        return report;
    }
    const role operation = get_role(op);
    if (operation == role::pv && algorithm == pv_algorithm::legacy) {
        report.status = POTAL_ERR_UNSUPPORTED; // the corrected bias runner is not a strict pipeline route
        report.stage = "attention_strict_legacy_pv";
        return report;
    }
    const auto & a = *op->src[0];
    const auto & b = *op->src[1];
    const size_t rows = size_t(b.ne[1]), n = size_t(a.ne[1]), heads = size_t(b.ne[2]);
    std::vector<strict_task> tasks;
    std::vector<float> staged;
    report.status = checked([&] {
        potal_tile_plan plan;
        require(potal_tile_select(rows, n, padded_k(size_t(b.ne[0])), block_k / POTAL_HW_DIM, &plan) == POTAL_OK,
                POTAL_ERR_CAPACITY);
        const size_t stripe_rows = plan.tile_I * POTAL_HW_DIM;
        staged.assign(product(product(product(n, rows), heads), size_t(b.ne[3])), 0.0f);
        const size_t head_repeat = size_t(b.ne[2] / a.ne[2]);
        const size_t batch_repeat = size_t(b.ne[3] / a.ne[3]);
        for (size_t ab = 0; ab < size_t(a.ne[3]); ++ab)
            for (size_t ah = 0; ah < size_t(a.ne[2]); ++ah)
                for (size_t batch = ab * batch_repeat; batch < (ab + 1) * batch_repeat; ++batch)
                    for (size_t head = ah * head_repeat; head < (ah + 1) * head_repeat; ++head)
                        for (size_t begin = 0; begin < rows; begin += stripe_rows)
                            tasks.push_back({ ab, ah, batch, head, begin, std::min(stripe_rows, rows - begin) });
    });
    if (report.status != POTAL_OK) {
        report.stage = "attention_strict_setup";
        return report;
    }

    namespace tl = ::potal::timeline;
    const bool timeline = tl::enabled();
    uint32_t run_id = 0;
    const uint32_t layer = layer_of(op);
    if (timeline) {
        tl::register_thread(tl::role::caller, -1, -1);
        run_id = tl::begin_run(operation == role::qk ? tl::op_type::qk : tl::op_type::pv,
                               rows == 1 ? tl::phase::decode : tl::phase::prefill, layer);
    }
    tl::scoped_span node_span(tl::active(), tl::stage::attn_node, run_id, layer, tl::no_id, tl::no_id);

    const size_t T = tasks.size(), capacity = 4;
    struct shared_state {
        std::mutex m;
        std::condition_variable cv;
        std::vector<stream> streams;
        std::vector<activation_parts> parts;
        std::vector<int> stage_done; // 0 none, 1 initialized, 2/3 halves done (3 = both)
        std::deque<std::unique_ptr<strict_job>> ready, done;
        size_t final_done = 0;
        bool fold_finished = false, issue_finished = false, failed = false, fatal = false;
        potal_status status = POTAL_OK;
        const char * stage = nullptr;
        std::thread::id issuer_thread;
        std::atomic<uint64_t> cpu_work_on_issuer_thread{ 0 }, cpu_work_on_issuer_cpu{ 0 };
    } st;
    st.streams.resize(T);
    st.parts.resize(T);
    st.stage_done.assign(T, 0);
    for (size_t i = 0; i < T; ++i) {
        st.streams[i].algorithm = operation == role::pv ? algorithm : pv_algorithm::legacy;
    }
    const auto fail = [&](potal_status status, const char * stage, bool fatal) {
        std::lock_guard<std::mutex> lock(st.m);
        if (!st.failed) {
            st.failed = true;
            st.status = status == POTAL_OK ? POTAL_ERR_BAD_RESPONSE : status;
            st.stage = stage;
        }
        st.fatal = st.fatal || fatal;
        st.cv.notify_all();
    };
    // Every CPU numeric step notes where it ran: never on the issuer thread or hart 0's CPU.
    const auto note_cpu_work = [&] {
        if (std::this_thread::get_id() == st.issuer_thread) ++st.cpu_work_on_issuer_thread;
#if defined(__linux__)
        if (pin && sched_getcpu() == role_cpus[0]) ++st.cpu_work_on_issuer_cpu;
#endif
    };
    const auto wait_for = [&](std::unique_lock<std::mutex> & lock, tl::stage s, uint32_t stripe, auto ready) {
        if (ready()) return;
        if (tl::active()) tl::record(s, tl::event::wait_begin, 0, run_id, layer, stripe, tl::no_id, tl::no_id, 0, 0);
        st.cv.wait(lock, ready);
        if (tl::active()) tl::record(s, tl::event::wait_end, 0, run_id, layer, stripe, tl::no_id, tl::no_id, 0, 0);
    };

    // Hart 0: only the planned runner, job by job.
    result issue_report;
    const stripe_consumer consume = [&](const owner_runners & runners) {
        consume_outcome out;
        {
            std::lock_guard<std::mutex> lock(st.m);
            st.issuer_thread = std::this_thread::get_id();
        }
        if (timeline) tl::register_thread(tl::role::issuer, 0, role_cpus[0]);
        std::unique_ptr<strict_job> job;
        try {
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(st.m);
                    wait_for(lock, tl::stage::attn_issue, tl::no_id,
                             [&] { return st.failed || !st.ready.empty() || st.fold_finished; });
                    if (st.failed || st.ready.empty()) break;
                    job = std::move(st.ready.front());
                    st.ready.pop_front();
                }
                {
                    tl::scoped_span span(tl::active(), tl::stage::attn_issue, run_id, layer, uint32_t(job->task),
                                         tl::no_id);
                    span.work = job->gemms.size();
                    issue_job(*job, operation, runners.planned, issue_report);
                }
                std::lock_guard<std::mutex> lock(st.m);
                st.done.push_back(std::move(job));
                st.cv.notify_all();
            }
        } catch (potal_status status) {
            if (issue_report.fatal && job) (void) job.release(); // abandoned operands stay allocated
            fail(status, issue_report.stage ? issue_report.stage : "attention_strict_issue", issue_report.fatal);
        } catch (...) {
            fail(POTAL_ERR_BAD_RESPONSE, "attention_strict_issue", true);
        }
        {
            std::lock_guard<std::mutex> lock(st.m);
            st.issue_finished = true;
            out.fatal = st.fatal;
            out.status = st.failed ? st.status : POTAL_OK;
            out.stage = st.failed ? st.stage : nullptr;
            st.cv.notify_all();
        }
        return out;
    };
    std::string why;
    potal_status owner_status = POTAL_OK;
    if (!owner.start(consume, why, owner_status)) {
        report.status = owner_status != POTAL_OK ? owner_status : POTAL_ERR_NO_HARDWARE;
        report.stage = "attention_issuer";
        if (timeline) tl::end_run(run_id);
        return report;
    }

    const auto role_thread = [&](int hart, tl::role r, auto body) {
        return std::thread([&, hart, r, body] {
            try {
                nearest_rounding rounding;
                if (pin) {
                    std::string pin_why;
                    if (!pin_role_cpu(role_cpus[hart], pin_why)) {
                        fail(POTAL_ERR_INVALID_ARGUMENT, "attention_strict_pin", false);
                        return;
                    }
                }
                if (timeline) tl::register_thread(r, hart, pin ? role_cpus[hart] : -1);
                body();
            } catch (potal_status status) {
                fail(status, "attention_strict_cpu", false);
            } catch (const std::bad_alloc &) {
                fail(POTAL_ERR_CAPACITY, "attention_strict_cpu", false);
            } catch (const std::length_error &) {
                fail(POTAL_ERR_CAPACITY, "attention_strict_cpu", false);
            } catch (...) {
                fail(POTAL_ERR_BAD_RESPONSE, "attention_strict_cpu", false);
            }
        });
    };
    const auto local_body = [&](size_t worker) {
        for (size_t i = 0; i < T; ++i) {
            const strict_task & task = tasks[i];
            {
                std::unique_lock<std::mutex> lock(st.m);
                wait_for(lock, tl::stage::attn_local, uint32_t(i), [&] {
                    return st.failed || (i < st.final_done + capacity && (worker == 0 || st.stage_done[i] >= 1));
                });
                if (st.failed) return;
            }
            const size_t half = (task.rows + 1) / 2;
            const size_t row_begin = worker == 0 ? 0 : half, row_end = worker == 0 ? half : task.rows;
            {
                tl::scoped_span span(tl::active(), tl::stage::attn_local, run_id, layer, uint32_t(i), uint32_t(worker));
                note_cpu_work();
                if (worker == 0) {
                    init_activation(b, task.head, task.batch, task.begin, task.rows, operation, st.streams[i],
                                    st.parts[i]);
                    std::lock_guard<std::mutex> lock(st.m);
                    st.stage_done[i] = 1;
                    st.cv.notify_all();
                }
                local_activation(b, task.head, task.batch, task.begin, row_begin, row_end, operation, st.streams[i],
                                 st.parts[i]);
                span.work = row_end - row_begin;
            }
            std::lock_guard<std::mutex> lock(st.m);
            ++st.stage_done[i]; // 1 + worker0 + worker1 = 3
            st.cv.notify_all();
        }
    };
    const auto fold_body = [&] {
        std::shared_ptr<const stationary> weight;
        size_t current_ab = SIZE_MAX, current_ah = SIZE_MAX;
        for (size_t i = 0; i < T; ++i) {
            const strict_task & task = tasks[i];
            {
                std::unique_lock<std::mutex> lock(st.m);
                wait_for(lock, tl::stage::attn_fold, uint32_t(i), [&] { return st.failed || st.stage_done[i] == 3; });
                if (st.failed) return;
            }
            note_cpu_work();
            if (task.ab != current_ab || task.ah != current_ah) {
                tl::scoped_span span(tl::active(), tl::stage::attn_stationary, run_id, layer, uint32_t(i), tl::no_id);
                auto prepared = std::make_shared<stationary>();
                prepare_weight(a, task.ah, task.ab, operation, *prepared);
                weight = std::move(prepared);
                current_ab = task.ab;
                current_ah = task.ah;
            }
            std::unique_ptr<strict_job> job;
            {
                tl::scoped_span span(tl::active(), tl::stage::attn_fold, run_id, layer, uint32_t(i), tl::no_id);
                fold_activation(operation, st.streams[i], st.parts[i]);
                st.parts[i] = activation_parts{};
                job = stage_job(i, st.streams[i], weight, operation);
                st.streams[i] = stream{};
                span.work = job->gemms.size();
            }
            std::lock_guard<std::mutex> lock(st.m);
            st.ready.push_back(std::move(job));
            st.cv.notify_all();
        }
        std::lock_guard<std::mutex> lock(st.m);
        st.fold_finished = true;
        st.cv.notify_all();
    };
    double reconstruct_s = 0;
    const auto final_body = [&] {
        for (size_t i = 0; i < T; ++i) {
            std::unique_ptr<strict_job> job;
            {
                std::unique_lock<std::mutex> lock(st.m);
                wait_for(lock, tl::stage::attn_recon, uint32_t(i),
                         [&] { return st.failed || !st.done.empty() || st.issue_finished; });
                if (st.failed || st.done.empty()) return;
                job = std::move(st.done.front());
                st.done.pop_front();
            }
            require(job->task == i, POTAL_ERR_BAD_RESPONSE);
            {
                tl::scoped_span span(tl::active(), tl::stage::attn_recon, run_id, layer, uint32_t(i), tl::no_id);
                note_cpu_work();
                reconstruct_job(*job, operation, tasks[i], heads, rows, staged, reconstruct_s);
            }
            job.reset(); // completed and read: the operands are free
            std::lock_guard<std::mutex> lock(st.m);
            ++st.final_done;
            st.cv.notify_all();
        }
    };
    std::thread local0 = role_thread(1, tl::role::exsia_local0, [&] { local_body(0); });
    std::thread local1 = role_thread(2, tl::role::exsia_local1, [&] { local_body(1); });
    std::thread fold = role_thread(3, tl::role::exsia_fold, fold_body);
    std::thread finalizer = role_thread(4, tl::role::finalizer, final_body);
    local0.join();
    local1.join();
    fold.join();
    finalizer.join();
    {
        // A CPU stage that stopped early leaves the issuer waiting: release it.
        std::lock_guard<std::mutex> lock(st.m);
        if (st.final_done != T && !st.failed) {
            st.failed = true;
            st.status = POTAL_ERR_BAD_RESPONSE;
            st.stage = "attention_strict_incomplete";
        }
        st.cv.notify_all();
    }
    owner.join();

    report = issue_report;
    report.reconstruct_s = reconstruct_s;
    report.strict_tasks = T;
    report.cpu_work_on_issuer_thread = st.cpu_work_on_issuer_thread.load();
    report.cpu_work_on_issuer_cpu = st.cpu_work_on_issuer_cpu.load();
    report.fatal = report.fatal || st.fatal;
    if (st.failed) {
        report.status = st.status;
        report.stage = st.stage;
    } else {
        report.status = POTAL_OK;
        report.stage = nullptr;
        // Causal masking belongs to ggml_soft_max_ext AFTER the raw QK MatMul (as execute()).
        for (size_t batch = 0; batch < size_t(op->ne[3]); ++batch) {
            for (size_t head = 0; head < size_t(op->ne[2]); ++head) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        auto * destination = static_cast<char *>(op->data) + batch * op->nb[3] +
                                            head * op->nb[2] + i * op->nb[1] + j * op->nb[0];
                        const float value = staged[((batch * size_t(op->ne[2]) + head) * rows + i) * n + j];
                        std::memcpy(destination, &value, sizeof(value));
                    }
                }
            }
        }
    }
    node_span.close();
    if (timeline) tl::end_run(run_id);
    return report;
}

result execute_strict_hardware(ggml_tensor * op) {
    result report;
    std::array<int, 5> cpus{};
    std::string why;
    if (!cached_role_cpus(cpus, why)) {
        report.status = POTAL_ERR_INVALID_ARGUMENT;
        report.stage = "attention_strict_role_cpus";
        return report;
    }
    issuer_owner owner;
    return execute_strict(op, owner, cpus, true);
}

} // namespace ggml::potal::attention
