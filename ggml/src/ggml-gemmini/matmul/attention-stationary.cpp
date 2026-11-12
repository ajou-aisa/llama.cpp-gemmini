#include "attention-stationary.hpp"
#include "ggml-gemmini-matmul-config.hpp"
#include "../quants/act/exsia/local.hpp"
#include "../quants/common/hp1_scu.hpp"
#include "../quants/common/math.hpp"
#include "../residual/rmd/rmd-types.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>

#if GGML_GEMMINI_ENABLE_A4NKS_ATTENTION
namespace ggml::gemmini {
namespace {
constexpr size_t block_size = 32;
constexpr int bits = GGML_GEMMINI_ACTIVATION_BITS;
constexpr int radix = 1 << bits;
constexpr int16_t zero_exp = std::numeric_limits<int16_t>::min();

void require(bool condition, rmd::RmdStatus status = rmd::RmdStatus::invalid_arguments) {
    if (!condition) throw status;
}

struct RoundingScope {
    int previous = std::fegetround();
    RoundingScope() { require(previous != -1 && std::fesetround(FE_TONEAREST) == 0); }
    ~RoundingScope() { std::fesetround(previous); }
};

float read(const AttentionOperand & input, size_t k, size_t column) {
    const auto & t = input.tensor;
    const auto * p = static_cast<const char *>(t.data) +
        k*t.nb[0] + column*t.nb[1] + input.head*t.nb[2] + input.batch*t.nb[3];
    float value;
    if (t.type == GGML_TYPE_F16) {
        ggml_fp16_t half;
        std::memcpy(&half, p, sizeof(half));
        value = ggml_fp16_to_fp32(half);
    } else {
        std::memcpy(&value, p, sizeof(value));
    }
    require(std::isfinite(value));
    return value;
}

void add_scaled(int64_t & accumulator, int64_t value, uint32_t shift) {
    if (!value) return;
    require(shift <= 63, rmd::RmdStatus::overflow);
    const __int128_t sum = static_cast<__int128_t>(accumulator) +
        static_cast<__int128_t>(value) * (static_cast<__int128_t>(1) << shift);
    require(sum >= std::numeric_limits<int64_t>::min() &&
            sum <= std::numeric_limits<int64_t>::max(), rmd::RmdStatus::overflow);
    accumulator = static_cast<int64_t>(sum);
}
} // namespace

AttentionPreparation prepare_attention_stationary(
        const AttentionOperand & input, AttentionStationary & w, int workers) {
    require(workers >= 1 && workers <= 3);
    const size_t k = input.tensor.ne[0], n = input.tensor.ne[1];
    const size_t blocks = (k + block_size - 1)/block_size, kp = blocks*block_size;
    const bool reshape = !w.packed_valid || w.k != k || w.n != n;
    size_t stride = input.qk && input.reuse ? w.packed_columns() : n;
    if (input.qk && stride < n) {
        require(stride <= SIZE_MAX / 2, rmd::RmdStatus::overflow);
        stride = std::max(n, stride * 2);
    }
    if (input.qk) {
        require(stride <= SIZE_MAX - 31, rmd::RmdStatus::overflow);
        stride = (stride + 31) / 32 * 32;
    }
    require(stride >= n && kp <= SIZE_MAX / stride, rmd::RmdStatus::overflow);
    const bool repack = !w.packed_valid || !input.reuse || stride != w.packed_columns() ||
        (input.qk ? w.k != k || n < w.n : w.n != n || k < w.k);
    w.packed_valid = false;
    w.columns.resize(n);
    for (auto & column : w.columns) column.resize(blocks);
    std::vector<AttentionPreparation> counts(n);
    std::vector<std::exception_ptr> errors(n);
    // Columns are independent: no scale/reduction order is changed. Only
    // preparation runs in workers; accelerator submission stays on its owner.
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1 && n > 1)
#endif
    for (size_t j = 0; j < n; ++j) {
        try {
            RoundingScope rounding;
            if (input.reuse && input.qk && (j < input.begin || j >= input.end)) {
                counts[j].reused_blocks = blocks;
                continue;
            }
            quants::act::exsia::ExpScanner scanner;
            quants::act::exsia::WideQuantizer quantizer;
            std::vector<int32_t> codes(block_size);
            for (size_t b = 0; b < blocks; ++b) {
                auto & block = w.columns[j][b];
                if (input.reuse && !input.qk &&
                    (b*block_size >= input.end || (b+1)*block_size <= input.begin)) {
                    ++counts[j].reused_blocks;
                    continue;
                }
                const size_t valid = std::min(block_size, k-b*block_size);
                std::array<float, block_size> values{};
                for (size_t c = 0; c < valid; ++c)
                    values[c] = read(input, b*block_size+c, j);
                AttentionBlock next;
                int16_t exponent = zero_exp;
                for (size_t c = 0; c < valid; ++c)
                    exponent = std::max(exponent, scanner.unbiased_exp(values[c]));
                next.theta = exponent == zero_exp ? zero_exp :
                    exponent - (input.qk ? 2*bits-2 : bits-2);
                if (exponent != zero_exp) {
                    quantizer.quantize_block(values.data(), valid, next.theta, codes);
                    for (size_t c = 0; c < valid; ++c) {
                        const int32_t code = std::clamp(codes[c],
                            input.qk ? -(1 << (2*bits-1)) : -radix/2,
                            input.qk ? (1 << (2*bits-1))-radix/2-1 : radix/2-1);
                        next.low[c] = input.qk ? lowdigit(code, bits) : code;
                        if (input.qk) next.high[c] = (code-next.low[c])/radix;
                        next.sum += next.low[c];
                    }
                }
                block = next;
                ++counts[j].quantized_blocks;
            }
        } catch (...) {
            // Exceptions cannot cross an OpenMP region. Rethrow on the owner
            // before publishing any packed operand or submitting device work.
            errors[j] = std::current_exception();
        }
    }
    for (const auto & error : errors)
        if (error) {
            w = AttentionStationary{};
            std::rethrow_exception(error);
        }
    AttentionPreparation count;
    for (const auto & item : counts) {
        count.quantized_blocks += item.quantized_blocks;
        count.reused_blocks += item.reused_blocks;
    }
    if (!reshape && count.quantized_blocks == 0) {
        w.packed_valid = true;
        return count;
    }
    w.k = k;
    w.n = n;
    w.kp = kp;
    w.column_stride = stride;
    w.base.resize(n);
    w.carriers.resize(blocks*stride);
    w.codes[0].resize(kp*stride);
    w.codes[1].resize(input.qk ? kp*stride : 0);
    w.colsum.assign(input.qk ? 0 : n, 0);
#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
    w.gpu.reset();
#endif
    // Grow K's physical pitch geometrically. Logical append no longer rewrites
    // old limbs; V keeps its pitch and only overwrites changed token blocks.
    for (size_t j = 0; j < n; ++j) {
        if (!repack && input.reuse && input.qk && (j < input.begin || j >= input.end)) continue;
        int16_t base = std::numeric_limits<int16_t>::max();
        for (const auto & block : w.columns[j])
            if (block.theta != zero_exp) base = std::min(base, block.theta);
        w.base[j] = base == std::numeric_limits<int16_t>::max() ? 0 : base;
        for (size_t b = 0; b < blocks; ++b) {
            const auto & block = w.columns[j][b];
            require(quants::hp1::encode_carrier(
                block.theta == zero_exp ? zero_exp : block.theta-w.base[j], w.carriers[b*stride+j]));
            if (repack || input.qk ||
                (b*block_size < input.end && (b+1)*block_size > input.begin))
                for (size_t c = 0; c < block_size; ++c) {
                    const size_t index = (b*block_size+c)*stride+j;
                    w.codes[0][index] = block.low[c];
                    if (input.qk) w.codes[1][index] = block.high[c];
                }
            if (!input.qk && w.carriers[b*stride+j] != quants::hp1::zero_carrier)
                add_scaled(w.colsum[j], block.sum, w.carriers[b*stride+j]);
        }
    }
    w.packed_valid = true;
    return count;
}
} // namespace ggml::gemmini
#endif
