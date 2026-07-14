#pragma once

#include "types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include <gemmini/layer.hpp>

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE GGML_GEMMINI_BLOCK_SIZE
#endif

#ifndef GGML_GEMMINI_EXSIA_SIGMA
#define GGML_GEMMINI_EXSIA_SIGMA 3
#endif

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia
{
    struct BitMask
    {
        size_t rows = 0;
        size_t cols = 0;
        std::vector<uint64_t> words;

        static bool checked_mul(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
                return false;

            out = lhs * rhs;
            return true;
        }

        static bool checked_add(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs > std::numeric_limits<size_t>::max() - rhs)
                return false;

            out = lhs + rhs;
            return true;
        }

        // resize the bitmask matrix to the specified number of rows and columns
        bool resize(size_t row_count, size_t col_count)
        {
            size_t bit_count = 0;
            size_t rounded_bit_count = 0;
            if (!checked_mul(row_count, col_count, bit_count) ||
                !checked_add(bit_count, 63, rounded_bit_count))
            {
                clear();
                return false;
            }

            rows = row_count;
            cols = col_count;
            words.assign(rounded_bit_count / 64, 0);
            return true;
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
        size_t blk_size = 0;
        std::vector<int16_t> e;
        int16_t e1 = std::numeric_limits<int16_t>::min();      // top-1 distinct exponent of a block
        int16_t e2 = std::numeric_limits<int16_t>::min();      // top-2 distinct exponent of a block
        int16_t e_b = std::numeric_limits<int16_t>::min();     // final max exponent of a block
        int16_t theta_b = std::numeric_limits<int16_t>::min(); // final scale exponent of a block

        std::vector<float> x; // temporary original activations for a block
    };

    struct StripeState
    {
        size_t row_start = 0;       // starting row index of a stripe
        size_t row_end = 0;         // ending row index of a stripe (exclusive)
        size_t blk_num = 0;         // number of blocks in a stripe
        size_t num_real_blocks = 0; // number of real blocks in a stripe (excluding padding blocks)

        int16_t e1 = std::numeric_limits<int16_t>::min();  // top-1 distinct exponent among all blocks in a stripe
        int16_t e2 = std::numeric_limits<int16_t>::min();  // top-2 distinct exponent among all blocks in a stripe
        int16_t e_s = std::numeric_limits<int16_t>::min(); // final scale exponent of a stripe
        bool promote_top_block = false;                    // set by StripeFolding when e2 != -inf; promotes top-1 exponent blocks to outliers

        BitMask outlier_mask; // bitmask indicating outlier positions in a stripe
    };

    struct ExSIAState
    {
        size_t B_size = BLOCK_SIZE;
        size_t K_logical = 0;
        size_t K_padded = 0;
        size_t blocks_per_row = 0;
        std::vector<StripeState> stripe;     // vector of stripe states for the entire matrix
        std::vector<float> x_f32;            // original activations, padded to I * K_padded
        std::vector<int32_t> q_wide;         // Q_X_i32 shared by local wide-quantization and stripe folding
        std::vector<int16_t> block_exp;      // E_s[r, b / B] shared block max exponents
        std::vector<int32_t> residual;       // shared residual buffer for stripe folding, same shape as q_wide (I * K_padded)
    };

    class ExpScanner
    {
    public:
        int16_t unbiased_exp(const float &x);
        void scan_top2_exp(const std::vector<float> &x,
                           BlockState &blk); // scan top-2 distinct exponents for a block and store them in the block state

        // scan top-2 distinct exponents for a block with a bitmask, only considering unmasked positions
        void update_block_top2_exp(const BitMask &mask,
                                   size_t row,
                                   size_t blk_idx,
                                   BlockState &blk);

        // update the stripe-level top-2 distinct exponents based on a block's max exponent
        void update_stripe_top2_exp(StripeState &stripe, int16_t e_b);
    };

    class OutlierMarker
    {
    public:
        // mark outliers in a stripe based on the final stripe scale exponent and a bitmask, setting the corresponding bits in the bitmask for outlier positions
        void mark_outlier(StripeState &stripe,
                          size_t row,
                          size_t blk_idx,
                          size_t blk_size,
                          const BitMask &d_mask) const;
    };

    class WideQuantizer
    {
    public:
        std::vector<int32_t> quantize_block(const std::vector<float> &x, int16_t theta_b); //
        std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
        quantize_block(const std::vector<float> &x,
                       size_t row,
                       size_t col,
                       const BitMask &mask,
                       int16_t theta_b);
    };

    class SigmaDetector
    {
    public:
        bool detect_sigma(int32_t q, __int128_t S, __int128_t SS, size_t N);
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
            StripeState &stripe,
            const std::vector<float> &x,
            size_t row,
            size_t blk_idx,
            uint64_t &cycle_delta);

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
            StripeState &stripe,
            ggml_gemmini_args_t &args,
            size_t stripe_idx,
            int8_t *dst,
            int32_t *residual,
            uint64_t &cycle_delta);

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

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args);

} // namespace ggml::gemmini::quants::act::exsia
