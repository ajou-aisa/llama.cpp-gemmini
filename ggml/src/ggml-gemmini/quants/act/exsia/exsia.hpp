#pragma once

#include "types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia
{
struct BitMask
{
    size_t rows = 0;
    size_t cols = 0;
    std::vector<uint64_t> words;

    // resize the bitmask matrix to the specified number of rows and columns
    void resize(size_t row_count, size_t col_count)
    {
        rows = row_count;
        cols = col_count;
        words.assign((rows * cols + 63) / 64, 0);
    }

    // clear the bitmask matrix, setting all bits to 0
    void clear()
    {
        words.clear();
        rows = 0;
        cols = 0;
    }

    // set the bit at the specified row and column to 1
    void set(size_t row, size_t col)
    {
        assert(row < rows && col < cols);
        const size_t idx = row * cols + col;
        words[idx / 64] |= uint64_t(1) << (idx % 64);
    }

    // reset the bit at the specified row and column to 0
    void reset(size_t row, size_t col)
    {
        assert(row < rows && col < cols);
        const size_t idx = row * cols + col;
        words[idx / 64] &= ~(uint64_t(1) << (idx % 64));
    }

    // test whether the bit at the specified row and column is set to 1
    bool is_set(size_t row, size_t col) const
    {
        assert(row < rows && col < cols);
        const size_t idx = row * cols + col;
        return (words[idx / 64] >> (idx % 64)) & 1;
    }
};

struct BlockState
{
    size_t blk_size;
    std::vector<int16_t> e;
    int16_t e1 = std::numeric_limits<int16_t>::min(); // top-1 exponent of a block
    int16_t e2 = std::numeric_limits<int16_t>::min(); // top-2 exponent of a block
    int16_t e_b = std::numeric_limits<int16_t>::min(); // final max exponent of a block
    int16_t theta_b = std::numeric_limits<int16_t>::min(); // final scale exponent of a block

    std::vector<float> x; // temporary original activations for a block
};

struct StripeState
{
    size_t row_start = 0; // starting row index of a stripe
    size_t row_end = 0; // ending row index of a stripe (exclusive)
    size_t blk_num = 0; // number of blocks in a stripe
    size_t num_real_blocks = 0; // number of real blocks in a stripe (excluding padding blocks)

    int16_t e1 = std::numeric_limits<int16_t>::min(); // top-1 exponent among all blocks in a stripe
    int16_t e2 = std::numeric_limits<int16_t>::min(); // top-2 exponent among all blocks in a stripe
    int16_t e_s = std::numeric_limits<int16_t>::min(); // final scale exponent of a stripe
    bool promote_top_block = false; // flag indicating whether to promote the top block's exponent to the stripe level

    BitMask outlier_mask; // bitmask indicating outlier positions in a stripe
};

struct ExSIAState
{
    size_t B_size;
    std::vector<StripeState> stripe; // vector of stripe states for the entire matrix
    std::vector<int32_t> q_wide; // Q_X_i32 shared by local wide-quantization and stripe folding
    std::vector<int16_t> block_top1_exp; // E_s[r, b / B] shared block max exponents
};

class ExpScanner
{
public:
    int16_t unbiased_exp(const float &x);
    void scan_top2_exp(const Meta &meta, const std::vector<float> &x, BlockState &blk); // scan top-2 exponents for a block and store them in the block state

    // scan top-2 exponents for a block with a bitmask, only considering unmasked positions
    void update_block_top2_exp(const BitMask &mask, size_t row, size_t blk_idx, BlockState &blk); 

    // update the stripe-level top-2 exponents based on a block's max exponent, and determine whether to promote the block exponent to the stripe level
    void update_stripe_top2_exp(StripeState &stripe, int16_t e_b); 
};

class OutlierMarker
{
public:
    // mark outliers in a stripe based on the final stripe scale exponent and a bitmask, setting the corresponding bits in the bitmask for outlier positions
    void mark_outlier(StripeState &stripe, const BitMask &d_mask) const;
};

class WideQuantizer
{
public:
    int32_t quantize(float x, int16_t s_e); // 
    std::tuple<int32_t, int64_t, int64_t> quantize(float x, size_t row, size_t col, const BitMask &mask, int16_t s_e);
};

class SigmaDetector
{
public:
    bool detect_3sigma(int32_t q, int64_t S, int64_t SS); 
};

class ResidualClipper
{
public:
    std::pair<int8_t, int32_t> clip_with_residual(int32_t q);
};

class LocalStage
{
public:
    bool run(
        Meta &meta,
        ExSIAState &state,
        ggml_gemmini_args_t &args,
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
        Meta &meta,
        ExSIAState &state,
        ggml_gemmini_args_t &args,
        size_t stripe_idx,
        size_t cols,
        size_t row_offset,
        size_t col_offset,
        int8_t *dst,
        int32_t *residual);

private:
    OutlierMarker unit_outlier_;
    ResidualClipper unit_clip_;
};

class ExSIA
{
private:
    ExSIAState state_;
    LocalStage local_;
    StripeFolding folding_;

public:
    bool run(
        Meta &meta,
        const ggml_tensor *src,
        ggml_gemmini_args_t &args);

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
