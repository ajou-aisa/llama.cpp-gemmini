#include "ethos.hpp"
#include "../../common/fp16_util.hpp"
#include "../../common/math.hpp"
#include "../../../ggml-gemmini-args.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml::gemmini::quants::act::ethos { static inline int16_t unbiased_exp(float x)
{
    if (x == 0.0f)
        return -126;

    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const int exp = static_cast<int>((u >> 23) & 0xFF);

    if (exp == 0)
        return -126;
    if (exp == 255)
        return 128;
    return static_cast<int16_t>(exp - 127);
}

static inline float load_value(
    const char *data_ptr,
    size_t stride_i_bytes,
    size_t stride_k_bytes,
    size_t row,
    size_t col)
{
    const char *row_ptr = data_ptr + row * stride_i_bytes;
    return *reinterpret_cast<const float *>(row_ptr + col * stride_k_bytes);
}

// initialize metadata for ethos
bool Initializer::init(
    Config &cfg,
    Metadata &meta,
    const char *data_ptr,
    size_t stride_i_bytes,
    size_t stride_k_bytes,
    size_t rows,
    size_t cols)
{
    if (!data_ptr || rows == 0 || cols == 0)
        return false;

    const size_t N = cfg.block_size;
    if (N != 32)
        // Current ethos assumes QK8_0=32. Extend later if needed.
        return false;
    if (cols % N != 0)
        return false;

    meta.block.clear();
    meta.stripe = StripeState{};

    StripeState &stripe = meta.stripe;
    size_t blk_per_row = cols / N; // block per row
    stripe.blk_num = rows * blk_per_row;
    const size_t computed_num_real_blocks = rows * blk_per_row;
    stripe.num_real_blocks = cfg.num_real_blocks > 0 ? cfg.num_real_blocks : computed_num_real_blocks;

    meta.block.resize(stripe.blk_num); // block memory init

    // init block's state
    for (size_t i = 0; i < rows; ++i)
    {
        const char *row_ptr = data_ptr + i * stride_i_bytes;

        for (size_t b = 0; b < blk_per_row; ++b)
        {
            const size_t blk_idx = i * blk_per_row + b; // blk index

            BlockState &blk = meta.block[blk_idx]; // reference of current block
            blk = BlockState{};
            blk.e.resize(N);
            blk.x.resize(N);
            blk.q.resize(N);

            // block init
            for (size_t j = 0; j < N; ++j)
            {
                const size_t k = b * N + j;
                float v = *reinterpret_cast<const float *>(row_ptr + k * stride_k_bytes);
                blk.x[j] = v;
                blk.e[j] = unbiased_exp(v);

                int16_t tmp = blk.e[j];
                if (tmp >= blk.e_max)
                {
                    blk.e_max2 = blk.e_max;
                    blk.e_max = tmp;
                }
                else if (tmp > blk.e_max2)
                    blk.e_max2 = tmp;
            }
            blk.s_e = blk.e_max - cfg.m;
        }
    }
    return true;
}

// quantize float-point based on shift operation
int Quantizer::quantize(float x, int16_t s_e)
{
    // round-away-from-zero 정책도 추후 고려
    if (x == 0.0f)
        return 0;

    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));

    const bool neg = (u >> 31) != 0;
    const int16_t e = unbiased_exp(x);
    const uint32_t mant = (1u << 23) | (u & 0x7FFFFF);

    const int shamt = e - 23 - s_e;

    int64_t q;
    if (shamt >= 0)
        q = int64_t(mant) << shamt;
    else
    {
        int rs = -shamt;
        q = (int64_t(mant) + (int64_t(1) << (rs - 1))) >> rs;
    }

    if (neg)
        q = -q;
    return static_cast<int>(q);
}

// clip to int8 and return whether clipped
std::pair<int8_t, bool> Clipper::clip(int q, Config &cfg)
{
    if (q > cfg.qmax)
        return {cfg.qmax, true};
    else if (q < -cfg.qmax)
        return {-cfg.qmax, true};
    // not clipped
    return {static_cast<int8_t>(q), false};
}

// detect L1-Outlier and pre-quantize
void L1Detector::detect_l1(Config &cfg, Metadata &meta, int blk_idx, Quantizer &quantizer, Clipper &clipper)
{
    const int64_t N = static_cast<int64_t>(cfg.block_size);
    BlockState &blk = meta.block[blk_idx];
    blk.S = 0;
    blk.S2 = 0;

    // for a block
    for (size_t i = 0; i < cfg.block_size; ++i)
    {
        // Q0, S, S2
        blk.q[i] = quantizer.quantize(blk.x[i], blk.s_e);
        blk.S += blk.q[i];
        blk.S2 += static_cast<int64_t>(blk.q[i]) * static_cast<int64_t>(blk.q[i]);
    }

    // outlier 판정
    for (size_t i = 0; i < cfg.block_size; ++i)
    {
        // (NQ0 - S)^2 > 9(NS2 - S^2)
        const int64_t centered = N * static_cast<int64_t>(blk.q[i]) - blk.S;
        const int64_t lhs = centered * centered;
        const int64_t rhs = 9 * (N * blk.S2 - blk.S * blk.S);
        if (lhs > rhs)
        {
            blk.outlier_mask |= (1u << i); // index i의 비트를 1로 set
        }
    }
}

void L2Detector::detect_l2(Config &cfg, Metadata &meta, int blk_idx)
{
    BlockState &blk = meta.block[blk_idx];

    const int16_t E1 = blk.e_b;  // L1 이후 inlier의 최대 exponent
    const int16_t E2 = blk.e_b2; // L1 이후 inlier의 두 번째 exponent

    // 유효한 inlier peak가 없으면 종료
    if (E1 == std::numeric_limits<int16_t>::min())
        return;

    // Python: I.sum() >= 2 와 유사한 guard
    int inlier_count = 0;
    int c1 = 0;

    for (size_t i = 0; i < cfg.block_size; ++i)
    {
        if (blk.outlier_mask & (1u << i))
            continue;

        ++inlier_count;
        if (blk.e[i] == E1)
            ++c1;
    }

    if (inlier_count < 2)
        return;

    if (E2 == std::numeric_limits<int16_t>::min())
        return;

    // if c1 <= L2_C and (E1 - E2) >= L2_D:
    if (c1 <= cfg.l2.c && (static_cast<int>(E1) - static_cast<int>(E2)) >= cfg.l2.d)
    {
        // E1에 해당하는 기존 inlier들을 L2 outlier로 승격
        for (size_t i = 0; i < cfg.block_size; ++i)
        {
            if (blk.outlier_mask & (1u << i))
                continue;

            if (blk.e[i] == E1)
                blk.outlier_mask |= (1u << i);
        }

        // 남은 inlier 기준으로 e_b / e_b2 재계산
        blk.e_b = std::numeric_limits<int16_t>::min();
        blk.e_b2 = std::numeric_limits<int16_t>::min();

        for (size_t i = 0; i < cfg.block_size; ++i)
        {
            if (blk.outlier_mask & (1u << i))
                continue;

            const int16_t tmp = blk.e[i];
            if (tmp >= blk.e_b)
            {
                blk.e_b2 = blk.e_b;
                blk.e_b = tmp;
            }
            else if (tmp > blk.e_b2)
                blk.e_b2 = tmp;
        }
    }
}

bool Ethos::run(
    Config &cfg,
    const char *data_ptr,
    size_t stride_i_bytes,
    size_t stride_k_bytes,
    size_t rows,
    size_t cols,
    size_t row_offset,
    size_t col_offset,
    int8_t *dst)
{
    cfg.result.outliers.clear();
    cfg.result.e_s = std::numeric_limits<int16_t>::min();

    if (!dst)
        return false;

    // Step 1.0: initialize meta data.
    if (!unit_i_.init(cfg, meta_, data_ptr, stride_i_bytes, stride_k_bytes, rows, cols))
        return false;

    const size_t blk_per_row = cols / cfg.block_size;

    // Step 2.0. Block-wise quantize for real blocks
    StripeState &stripe = meta_.stripe;
    if (stripe.num_real_blocks == 0)
        return false;

    for (size_t blk_idx = 0; blk_idx < stripe.num_real_blocks; ++blk_idx)
    {
        // Step 2.1: Detect L1-Outlier & pre-Quantize
        unit_l1_.detect_l1(cfg, meta_, static_cast<int>(blk_idx), unit_q_, unit_c_);

        // Step 2.2: update metadata for inliers
        BlockState &blk = meta_.block[blk_idx];
        // if outlier exists
        if (blk.outlier_mask)
        {
            // for a block
            for (size_t i = 0; i < cfg.block_size; ++i)
            {
                // skip outlier
                if (blk.outlier_mask & (1u << i))
                    continue;

                // update metadata
                int16_t tmp = blk.e[i];
                if (tmp >= blk.e_b)
                {
                    blk.e_b2 = blk.e_b;
                    blk.e_b = tmp;
                }
                else if (tmp > blk.e_b2)
                    blk.e_b2 = tmp;
            }
        }
        else // reuse metadata without outliers
        {
            blk.e_b = blk.e_max;
            blk.e_b2 = blk.e_max2;
        }

        // Step 2.3: Detect L2-outlier
        if (cfg.l2_on)
            unit_l2_.detect_l2(cfg, meta_, static_cast<int>(blk_idx));

        // All elements may be marked outlier, leaving no inlier-derived block exponent.
        if (blk.e_b == std::numeric_limits<int16_t>::min())
        {
            blk.e_b = blk.e_max;
            blk.e_b2 = blk.e_max2;
        }

        stripe.e_min = std::min(stripe.e_min, blk.e_b);

        // update s_e
        blk.s_e = blk.e_b - cfg.m;

        // Step 3.0: finalize block quantization
        if (blk.outlier_mask)
        {
            for (size_t i = 0; i < cfg.block_size; ++i)
            {
                const int q0 = unit_q_.quantize(blk.x[i], blk.s_e);
                auto [q2, clipped] = unit_c_.clip(q0, cfg);
                (void)clipped;
                blk.q[i] = q2;
            }
        }
        else
        {
            for (size_t i = 0; i < cfg.block_size; ++i)
            {
                auto [q1, clipped] = unit_c_.clip(blk.q[i], cfg);
                (void)clipped;
                blk.q[i] = q1; // Q1
            }
        }
    }

    // Step 2.4: Padding blocks are already represented by T5's padded input.
    // They are excluded from exponent detection and recoding.
    for (size_t blk_idx = stripe.num_real_blocks; blk_idx < stripe.blk_num; ++blk_idx)
    {
        for (size_t i = 0; i < cfg.block_size; ++i)
            dst[blk_idx * cfg.block_size + i] = 0;
    }

        // e_min must be updated by at least one valid block before stripe recoding.
    if (stripe.e_min == std::numeric_limits<int16_t>::max())
        return false;

stripe.e_s = stripe.e_min + cfg.delta;
cfg.result.e_s = stripe.e_s;
cfg.result.e_s_per_stripe.push_back(stripe.e_s);
    cfg.result.num_padding_blocks = stripe.blk_num - stripe.num_real_blocks;

    for (size_t blk_idx = 0; blk_idx < stripe.num_real_blocks; ++blk_idx)
    {
        // Step 3.1: set shamt for a block
        BlockState &blk = meta_.block[blk_idx];
        const int shamt = blk.e_b - stripe.e_s;
        const size_t row = blk_idx / blk_per_row;
        const size_t block_col = blk_idx % blk_per_row;

        // Step 3.2: shift Q2 as shamt
        for (size_t i = 0; i < cfg.block_size; ++i)
        {
            const size_t out_idx = blk_idx * cfg.block_size + i;
            const size_t col = block_col * cfg.block_size + i;
            int qt = 0;
            bool clipped = false;

            if (shamt >= 0) // left shift
            {
                auto recoded = unit_c_.clip(blk.q[i] << shamt, cfg);
                qt = recoded.first;
                clipped = recoded.second;
            }
            else // right shift
            {
                const int rs = -shamt;
                qt = (blk.q[i] + (int64_t(1) << (rs - 1))) >> rs;
            }

            if (clipped)
                blk.outlier_mask |= (1u << i);

            dst[out_idx] = static_cast<int8_t>(qt);

            if (blk.outlier_mask & (1u << i))
            {
                const float orig = load_value(data_ptr, stride_i_bytes, stride_k_bytes, row, col);
                const int scale_shift = static_cast<int>(stripe.e_s) - static_cast<int>(cfg.m);
                float sat;
                if (scale_shift >= 0)
                    sat = static_cast<float>(static_cast<int64_t>(qt) << scale_shift);
                else
                    sat = static_cast<float>(qt) / static_cast<float>(int64_t(1) << (-scale_shift));
                cfg.result.outliers.push_back({
                    static_cast<int>(row + row_offset),
                    static_cast<int>(col + col_offset),
                    orig,
                    sat,
                });
            }
        }

        std::vector<float>().swap(blk.x);
    }

    return true;
}

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    if (!args.f_out || !acc32 || block_k == 0 || args.I == 0 || args.J == 0)
        return;

    const size_t weight_block_size = args.block_size_k > 0 ? args.block_size_k : static_cast<size_t>(QK8_0);
    const size_t k_begin = k_offset;
    const size_t k_end = k_offset + block_k;
    const size_t total_weight_scale_elems = args.blocks_K * args.blocks_J;
    const size_t out_col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
    const size_t out_row_stride = args.stride_f_out ? args.stride_f_out : args.J;

    for (size_t i = 0; i < args.I; ++i) {
        const int32_t *row_acc32 = acc32 + i * acc_stride;
        float *row_out = args.f_out + i * out_row_stride;

        for (size_t j = 0; j < args.J; ++j) {
            float scale_w = 1.0f;

            if (args.B_scales && args.blocks_K > 0) {
                const size_t weight_blk = k_begin / weight_block_size;
                if (weight_blk < args.blocks_K) {
                    const size_t weight_scale_idx = j * args.blocks_K + weight_blk;
                    if (weight_scale_idx < total_weight_scale_elems)
                        scale_w = args.B_scales[weight_scale_idx];
                }
            }

            float contrib = static_cast<float>(row_acc32[j]) * scale_w;
            contrib = ggml::gemmini::apply_activation_exponent(contrib, args.act_quant.ethos.e_s, args.act_quant.ethos.m);
            row_out[j * out_col_stride] += contrib;
        }
    }
}
} // namespace ggml::gemmini::quants::act::ethos
