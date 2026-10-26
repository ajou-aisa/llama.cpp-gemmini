#include "attention.hpp"

#include "dense.hpp"
#include "ggml-gemmini.h"
#include "ggml-gemmini/quants/act/exsia/local.hpp"
#include "ggml-gemmini/quants/act/exsia/exsia_shift.hpp"
#include "ggml-gemmini/quants/common/math.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <memory>
#include <stdexcept>
#include <utility>

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

void reconstruct(int64_t & accumulator, int64_t part, size_t limb) {
    const __int128_t value = static_cast<__int128_t>(accumulator) +
                             static_cast<__int128_t>(part) * (static_cast<__int128_t>(1) << (bits * limb));
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

void prepare_activation(const ggml_tensor & tensor, size_t head, size_t batch, size_t begin,
                         size_t rows, role operation, stream & out) {
    input_tensor(tensor, head, batch);
    require(operation == role::qk || operation == role::pv);
    require(rows != 0 && begin < size_t(tensor.ne[1]) && rows <= size_t(tensor.ne[1]) - begin);
    out.rows = rows;
    out.k = size_t(tensor.ne[0]);
    out.kp = padded_k(out.k);
    const size_t count = product(rows, out.kp), blocks = out.kp / block_k;
    std::vector<float> values(count);
    out.codes.resize(count);
    for (auto & limb : out.limbs) {
        limb.resize(count);
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < out.k; ++k) {
            const float value = read_value(tensor, k, begin + i, head, batch);
            require(operation == role::qk || (value >= 0 && value <= 1));
            values[i * out.kp + k] = value;
        }
    }
    exsia::ExpScanner scanner;
    if (operation == role::qk) {
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
        exsia::StripeState top;
        std::vector<int16_t> exponents(product(rows, blocks));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t block = 0; block < blocks; ++block) {
                uint64_t mask_word = 0;
                exsia::BlockMask mask(&mask_word, block_k);
#if EXSIA_BRANCH_COUNTS_ENABLED
                exsia::LocalBlockCycleSample sample;
#endif
                require(local.run_optimized(meta, state, values.data() + i * out.kp + block * block_k,
                                            std::min(block_k, out.k - block * block_k), block_k, i, block,
                                            scratch, mask, out.codes.data() + i * out.kp + block * block_k,
                                            exponents[i * blocks + block]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                            , sample
#endif
                                            ));
                scanner.update_stripe_top2_exp(top, exponents[i * blocks + block]);
            }
        }
        const int16_t anchor = top.e2 != zero_exp ? top.e2 : top.e1 != zero_exp ? top.e1 : 0;
        out.scale = int(anchor) - rho;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < out.k; ++k) {
                const int16_t e = exponents[i * blocks + k / block_k];
                out.codes[i * out.kp + k] = exsia::detail::shift_q_i32(
                    out.codes[i * out.kp + k], e == zero_exp ? 0 : int16_t(e - anchor));
            }
        }
    } else {
        const float maximum = *std::max_element(values.begin(), values.end());
        out.scale = (maximum > 0 ? int(scanner.unbiased_exp(maximum)) : 0) - 15;
        exsia::WideQuantizer quantizer;
        quantizer.quantize_block(values.data(), values.size(), int16_t(out.scale), out.codes);
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < out.k; ++k) {
            const size_t index = i * out.kp + k;
            if (operation == role::pv) {
                out.codes[index] = std::clamp(out.codes[index], 0, 65535);
            }
            split(out, index, int64_t(out.codes[index]) - (operation == role::pv ? bias : 0));
        }
    }
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

void integer_reference(const stream & activation, const stationary & weight, role operation,
                        std::vector<int64_t> & out) {
    validate_prepared(activation, weight, operation);
    out.assign(product(activation.rows, weight.n), 0);
    std::vector<int32_t> correction;
    if (operation == role::pv) {
        make_bias(weight, correction);
    }
    for (size_t limb = 0; limb < activation.limb_count; ++limb) {
        for (size_t i = 0; i < activation.rows; ++i) {
            for (size_t j = 0; j < weight.n; ++j) {
                int64_t stationary_sum = 0;
                for (size_t pass = 0; pass < (operation == role::qk ? 2u : 1u); ++pass) {
                    int32_t acc = 0;
                    for (size_t begin = 0; begin < activation.kp; begin += POTAL_HW_DIM) {
                        int32_t partial = 0;
                        for (size_t k = begin; k < begin + POTAL_HW_DIM && k < activation.k; ++k) {
                            partial += int32_t(activation.limbs[limb][i * activation.kp + k]) *
                                       weight.limbs[operation == role::qk ? pass : 0][k * weight.n + j];
                        }
                        const uint32_t carrier = weight.carriers[(begin / block_k) * weight.n + j];
                        acc = sat32(int64_t(acc) + scu(partial, carrier, 0));
                    }
                    if (operation == role::pv && limb == 0) {
                        acc = sat32(int64_t(acc) + correction[j]);
                    }
                    reconstruct(stationary_sum, acc, pass);
                }
                reconstruct(out[i * weight.n + j], stationary_sum, limb);
            }
        }
    }
}

void execute_stripe(const stream & activation, const stationary & weight, role operation,
                     const planned_runner & run, const corrected_runner & corrected,
                     result & report, std::vector<int64_t> & out) {
    validate_prepared(activation, weight, operation);
    out.assign(product(activation.rows, weight.n), 0);
    std::vector<int32_t> correction;
    if (operation == role::pv) {
        require(bool(corrected), POTAL_ERR_UNSUPPORTED);
        make_bias(weight, correction);
    }
    for (size_t limb = 0; limb < activation.limb_count; ++limb) {
        const size_t passes = operation == role::qk ? 2 : 1;
        const size_t k_total = activation.kp;
        const size_t selection_k = product(activation.kp, passes);
        require(activation.rows <= UINT32_MAX && weight.n <= UINT32_MAX && selection_k <= UINT32_MAX,
                POTAL_ERR_CAPACITY);
        hp1_operands operands;
        require(operands.allocate({ int64_t(activation.rows), int64_t(weight.n), int64_t(k_total) }),
                POTAL_ERR_CAPACITY);
        std::unique_ptr<void, decltype(&std::free)> bias_dma(nullptr, std::free);
        if (operation == role::pv && limb == 0) {
            bias_dma.reset(dma_alloc(product(operands.n_pad, sizeof(int32_t))));
            require(bool(bias_dma), POTAL_ERR_CAPACITY);
            std::memcpy(bias_dma.get(), correction.data(), weight.n * sizeof(int32_t));
        }
        for (size_t i = 0; i < activation.rows; ++i) {
            for (size_t k = 0; k < activation.k; ++k) {
                operands.a[i * operands.lda + k] = activation.limbs[limb][i * activation.kp + k];
            }
        }
        for (size_t block = 0; block < weight.kp / block_k; ++block) {
            for (size_t j = 0; j < weight.n; ++j) {
                operands.own.carriers[block * operands.cstride + j] =
                    weight.carriers[block * weight.n + j];
            }
        }
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
            const planned_result completed = operation == role::qk || limb != 0 ? run(descriptor) :
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
            report.correction_jobs += operation == role::pv && limb == 0;
            report.bias_updates += operation == role::pv && limb == 0 ?
                ((activation.rows + POTAL_HW_DIM - 1) / POTAL_HW_DIM) *
                ((weight.n + POTAL_HW_DIM - 1) / POTAL_HW_DIM) : 0;
            for (size_t i = 0; i < activation.rows; ++i) {
                for (size_t j = 0; j < weight.n; ++j) {
                    const int32_t value = operands.c[i * operands.ldc + j];
                    if (operation == role::qk) {
                        reconstruct(stationary_sum[i * weight.n + j], value, pass);
                    } else {
                        reconstruct(out[i * weight.n + j], value, limb);
                    }
                }
            }
        }
        if (operation == role::qk) {
            for (size_t index = 0; index < out.size(); ++index) {
                reconstruct(out[index], stationary_sum[index], limb);
            }
        }
    }
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
                            size_t begin, size_t rows, role operation, stream & out) {
    return checked([&] {
        nearest_rounding rounding;
        stream prepared;
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

result execute(ggml_tensor * op, const planned_runner & run, const corrected_runner & corrected) {
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

} // namespace ggml::potal::attention
