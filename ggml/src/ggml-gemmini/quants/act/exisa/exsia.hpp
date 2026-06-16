#pragma once

#include "types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia
{
struct BitMask
{
    size_t rows = 0;
    size_t cols = 0;
    std::vector<uint64_t> words;

    void resize(size_t row_count, size_t col_count)
    {
        rows = row_count;
        cols = col_count;
        words.assign((rows * cols + 63) / 64, 0);
    }

    void clear()
    {
        words.clear();
        rows = 0;
        cols = 0;
    }

    void set(size_t row, size_t col)
    {
        assert(row < rows && col < cols);
        const size_t idx = row * cols + col;
        words[idx / 64] |= uint64_t(1) << (idx % 64);
    }

    void reset(size_t row, size_t col)
    {
        assert(row < rows && col < cols);
        const size_t idx = row * cols + col;
        words[idx / 64] &= ~(uint64_t(1) << (idx % 64));
    }
};

struct BlockState
{
    std::vector<int16_t> e;
    int16_t e1 = std::numeric_limits<int16_t>::min(); // top-1 exponent of a block
    int16_t e2 = std::numeric_limits<int16_t>::min(); // top-2 exponent of a block
    int16_t e_b = std::numeric_limits<int16_t>::min(); // final max exponent of a block
    int16_t s_e = std::numeric_limits<int16_t>::min(); // final scale exponent of a block

    std::vector<float> x; // temporary original activations for a block
    std::vector<int32_t> q_wide; // wide-quantized values of a block

    int64_t S = 0; // sum of inliers
    int64_t SS = 0; // square sum of inliers
};

struct StripeState
{
    size_t row_start = 0;
    size_t row_end = 0;
    size_t blk_num = 0;
    size_t num_real_blocks = 0;

    int16_t e1 = std::numeric_limits<int16_t>::min(); // top-1 exponent among all blocks in a stripe
    int16_t e2 = std::numeric_limits<int16_t>::min(); // top-2 exponent among all blocks in a stripe
    int16_t e_s = std::numeric_limits<int16_t>::min(); // final scale exponent of a stripe
    bool promote_top_block = false;

    std::vector<BlockState> block;
    BitMask outlier_mask;
};

struct ExSIAState
{
    std::vector<StripeState> stripe;
    std::vector<int16_t> shamt;
};

class Initializer
{
public:
    bool init(
        Config &cfg,
        ExSIAState &state,
        const char *data_ptr,
        size_t stride_i_bytes,
        size_t stride_k_bytes,
        size_t rows,
        size_t cols);
};

class ExpScanner
{
public:
    void scan_top2_exp(BlockState &blk);
    void masked_top2_exp(const BitMask &mask, size_t row, size_t col_begin, BlockState &blk);
    void update_stripe_top2_exp(StripeState &stripe, int16_t e_b);
};

class OutlierMarker
{
public:
    void mark_outlier(Config &cfg,StripeState &stripe, BitMask &d_mask) const;
};

class WideQuantizer
{
public:
    int32_t quantize(float x, int16_t s_e);
    std::tuple<int32_t, int64_t, int64_t> quantize(float x, BitMask mask, int16_t s_e);
};

class SigmaDetector
{
public:
    bool detect_3sigma(int32_t q, int64_t S, int64_t SS);
};

class ResidualClipper
{
public:
    std::pair<int8_t, int32_t> clip_with_residual(int32_t q, Config &cfg);
};

class LocalStage
{
public:
    bool run(
        Config &cfg,
        ExSIAState &state,
        size_t stripe_idx);

private:
    ExpScanner unit_exp_;
    OutlierMarker unit_outlier_;
    WideQuantizer unit_quant_;
    SigmaDetector unit_sigma_;
};

class StripeFolding
{
public:
    bool run(
        Config &cfg,
        ExSIAState &state,
        size_t stripe_idx,
        size_t cols,
        size_t row_offset,
        size_t col_offset,
        int8_t *dst,
        int32_t *residual = nullptr);

private:
    OutlierMarker unit_outlier_;
    ResidualClipper unit_clip_;
};

class ExSIA
{
private:
    ExSIAState state_;
    Initializer unit_i_;
    LocalStage unit_local_;
    StripeFolding unit_folding_;

public:
    bool run(
        Config &cfg,
        const char *data_ptr,
        size_t stride_i_bytes,
        size_t stride_k_bytes,
        size_t rows,
        size_t cols,
        size_t row_offset,
        size_t col_offset,
        int8_t *dst,
        int32_t *residual = nullptr);

    const ExSIAState &state() const
    {
        return state_;
    }
};

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride);

} // namespace ggml::gemmini::quants::act::exsia
