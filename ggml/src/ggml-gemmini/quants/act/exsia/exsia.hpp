#pragma once

#include "types.hpp"

#include <algorithm>
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
#define GGML_GEMMINI_EXSIA_SIGMA 2
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

        size_t active_word_count() const
        {
            size_t bit_count = 0;
            size_t rounded_bit_count = 0;
            return checked_mul(rows, cols, bit_count) &&
                           checked_add(bit_count, 63, rounded_bit_count)
                       ? rounded_bit_count / 64
                       : 0;
        }

        bool prepare(size_t row_count, size_t col_count)
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
            const size_t word_count = rounded_bit_count / 64;
            if (words.size() < word_count)
                words.resize(word_count);

            std::fill(words.begin(), words.begin() + word_count, 0);
            return true;
        }

        // clear the bitmask matrix, setting all bits to 0
        void clear()
        {
            words.clear();
            rows = 0;
            cols = 0;
        }

        void clear_active_bits()
        {
            std::fill(words.begin(), words.begin() + active_word_count(), 0);
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

        bool prepare(size_t block_size)
        {
            blk_size = block_size;
            if (e.size() < block_size)
                e.resize(block_size);
            if (x.size() < block_size)
                x.resize(block_size);
            reset();
            return true;
        }

        void reset()
        {
            e1 = std::numeric_limits<int16_t>::min();
            e2 = std::numeric_limits<int16_t>::min();
            e_b = std::numeric_limits<int16_t>::min();
            theta_b = std::numeric_limits<int16_t>::min();
        }
    };

    struct StripeScratch
    {
        BlockState block;
        std::vector<int32_t> q_tmp;
        std::vector<int32_t> q_final;
        BitMask top1_exp_mask;
        BitMask int_outlier_mask;
        BitMask folding_inlier_mask;

        bool prepare(size_t block_size)
        {
            if (!block.prepare(block_size) ||
                !top1_exp_mask.prepare(1, block_size) ||
                !int_outlier_mask.prepare(1, block_size) ||
                !folding_inlier_mask.prepare(1, block_size))
            {
                return false;
            }

            if (q_tmp.size() < block_size)
                q_tmp.resize(block_size);
            if (q_final.size() < block_size)
                q_final.resize(block_size);
            return true;
        }
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
        StripeScratch scratch;

        size_t row_count() const
        {
            return row_end - row_start;
        }

        size_t local_row(size_t global_row) const
        {
            assert(global_row >= row_start && global_row < row_end);
            return global_row - row_start;
        }
    };

    // Per-stripe cycle aggregates. Sequential today; a future pipeline slot keeps its own.
    struct StripeCycleStats
    {
        size_t stripe_idx = 0;
        uint64_t local_total = 0;
        uint64_t folding_total = 0;

        void reset()
        {
            local_total = 0;
            folding_total = 0;
        }
    };

    // Self-contained owner of one stripe's transient state. This is the unit a future
    // stripe-level pipeline would replicate into N slots; today exactly one is used.
    struct StripeWorkspace
    {
        size_t stripe_idx = 0;
        size_t row_start = 0;
        size_t row_end = 0;

        StripeState stripe; // owns outlier/promotion mask + block scratch

        std::vector<int32_t> q_wide;    // max_stripe_rows * K_padded
        std::vector<int16_t> block_exp; // max_stripe_rows * blocks_per_row

        std::vector<ggml_gemmini_qact_outlier> outliers; // stripe-local, merged after folding

        StripeCycleStats cycle_stats;

        // One-time capacity allocation sized for the largest stripe. Counts are
        // overflow-checked by the caller before this is invoked.
        bool prepare(size_t elem_capacity,
                     size_t block_capacity,
                     size_t max_rows,
                     size_t K_padded,
                     size_t block_size)
        {
            if (q_wide.size() < elem_capacity)
                q_wide.resize(elem_capacity);
            if (block_exp.size() < block_capacity)
                block_exp.resize(block_capacity);
            if (!stripe.outlier_mask.prepare(max_rows, K_padded) ||
                !stripe.scratch.prepare(block_size))
                return false;

            outliers.reserve(elem_capacity);
            return true;
        }

        // Active-range reset only: no allocation, resize, reserve, or BitMask alloc.
        void reset_for_stripe(size_t idx,
                              size_t r0,
                              size_t r1,
                              size_t K_padded,
                              size_t blocks_per_row)
        {
            const int16_t neg_inf = std::numeric_limits<int16_t>::min();
            stripe_idx = idx;
            row_start = r0;
            row_end = r1;

            const size_t rows = r1 - r0;
            const size_t elem = rows * K_padded;
            const size_t blocks = rows * blocks_per_row;

            stripe.row_start = r0;
            stripe.row_end = r1;
            stripe.e1 = neg_inf;
            stripe.e2 = neg_inf;
            stripe.e_s = neg_inf;
            stripe.promote_top_block = false;
            stripe.outlier_mask.prepare(rows, K_padded); // reuses capacity, zeroes active words

            std::fill(q_wide.begin(), q_wide.begin() + elem, 0);
            std::fill(block_exp.begin(), block_exp.begin() + blocks, neg_inf);
            outliers.clear();
            cycle_stats.reset();
            cycle_stats.stripe_idx = idx;
        }
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
        void quantize_block(const std::vector<float> &x,
                            int16_t theta_b,
                            std::vector<int32_t> &q) const;
        std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
        quantize_block(const std::vector<float> &x,
                       size_t row,
                       size_t col,
                       const BitMask &mask,
                       int16_t theta_b); // returns q, sum(|q|), and sum(|q|^2) for inliers
        void quantize_block(const std::vector<float> &x,
                            size_t row,
                            size_t col,
                            const BitMask &mask,
                            int16_t theta_b,
                            std::vector<int32_t> &q,
                            __int128_t &S,
                            __int128_t &SS) const;
    };

    class SigmaDetector
    {
    public:
        // Detect a one-sided upper-tail outlier from magnitude statistics.
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
            std::vector<int32_t> &stripe_q_wide,
            std::vector<int16_t> &stripe_block_exp,
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
            const std::vector<int32_t> &stripe_q_wide,
            const std::vector<int16_t> &stripe_block_exp,
            std::vector<int32_t> &residual,                       // dense global output, I * K_padded
            std::vector<ggml_gemmini_qact_outlier> &out_outliers, // stripe-local, merged by caller
            uint64_t &cycle_delta);

    private:
        OutlierMarker unit_outlier_;
        ResidualClipper unit_clip_;
    };

    class ExSIA
    {
    private:
        ExSIAState state_;
        // Exactly one workspace today. To pipeline stripes later this becomes a small
        // std::vector<StripeWorkspace> of slots; no other data structure has to change.
        StripeWorkspace workspace_;
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
