// error_compensation_activation_DEC.h
#pragma once
#include <orca/layer.h>
#include <orca/log.hpp>

#include "../ggml-gemmini-args.h"
#include "ggml.h"
#include <algorithm>
#include <vector>
#include <cstdint>

#ifndef DEC_ALPHA_RATIO
#define DEC_ALPHA_RATIO 0.05 // 5% salient channels
#endif

namespace aisa
{
    /**
     * Dynamic Error Compensation for activations (input-side)
     *
     * Implements: Y_com = Σ_k Σ_{r∈R_k} δ_{r,k} · Ŵ[k,j]
     * - Optimized for single-pass W[k,:] access per salient channel
     * - Zero-overhead R_k construction (no S_/delta_ rescan)
     * - Unified path for both I=1 and I>1 cases
     */
    class ActivationDEC
    {
    public:
        static void compensate(const ggml_tensor *A, ggml_gemmini_args_t *args);

    private:
        const ggml_tensor *A_ = nullptr; // F32 activation
        const ggml_gemmini_args_t *args_ = nullptr;

        const char *layer_ = "others";

        // dims
        size_t I_; // number of rows
        size_t K_; // input channels
        size_t J_; // output channels

        // act/weight buffers & layout
        const int8_t *qx_ = nullptr; // args->A
        size_t sA_ = 0;
        const int8_t *qW_ = nullptr; // args->B
        size_t sB_ = 0;
        bool B_is_KxJ_ = true;

        // scales
        float scale_A_ = 1.0f;
        const float *A_scales_ = nullptr;
        size_t A_scale_rows_ = 0;
        size_t A_scale_cols_ = 0;
        size_t act_block_size_ = QK8_0;
        bool activation_block_scaled_ = false;
        const float *B_scales_ = nullptr;
        size_t blocks_K_ = 0, blocks_J_ = 0, block_size_k_ = QK8_0;

        size_t alpha_; // number of salient channels per row

        // Per-row salient channels and residuals
        std::vector<std::vector<int>> S_;       // S[r]: salient indices
        std::vector<std::vector<float>> delta_; // delta[r][i]: residuals (float)

        // R_k: CSC-style reverse mapping {k → [(r, δ[r,k])]}
        std::vector<size_t> rk_offs_;                 // K_+1 prefix sums
        std::vector<std::pair<int, float>> rk_pairs_; // (row, delta) pairs
        std::vector<int> unique_k_;                   // non-empty k indices

        // On-the-fly staging for R_k construction (eliminates rescan overhead)
        struct Triplet
        {
            int k;
            int r;
            float d;
        };
        std::vector<Triplet> rk_stage_;

        // Scratch buffers for row-wise operations (reused across rows)
        std::vector<int> scratch_idx_;   // K_ indices for sorting
        std::vector<float> scratch_abs_; // cached |x_r[k]| values

        ActivationDEC(const ggml_tensor *A, ggml_gemmini_args_t *args)
            : A_(A), args_(args), 
              I_(args->I), J_(args->J), K_(args->K),
              sA_(args->sA), sB_(args->sB),
              scale_A_(args->scale_A),
              A_scales_(args->A_scales),
              A_scale_rows_(args->A_scale_rows),
              A_scale_cols_(args->A_scale_cols),
              act_block_size_(infer_act_block_size(args)),
              activation_block_scaled_(args->activation_block_scaled),
              B_scales_(args->B_scales),
              blocks_J_(args->blocks_J), blocks_K_(args->blocks_K)
        {
            qx_ = reinterpret_cast<const int8_t *>(args->A);
            qW_ = reinterpret_cast<const int8_t *>(args->B);
            B_is_KxJ_ = !args->transpose_B;
            block_size_k_ = args->block_size_k ? args->block_size_k : QK8_0;
            layer_ = args->layer_name ? args->layer_name : "others";
        }

        void prepare();
        void selectTopKandComputeResidual(size_t row_idx,
                                          const float *row_fp,
                                          const int8_t *row_q);
        void selectSalientandComputeResidual();
        void buildRk();
        void computeCompensation_unrolled(float *Y_com);
        void applyCompensation(float *out,
                               size_t row_stride,
                               size_t col_stride,
                               const std::vector<float> &Y_com);

        inline void load_W_row_scaled(int k, std::vector<float> &Wk_f) const;

        static constexpr bool kActBlockScaleEnabled = (ACTIVATION_BLOCK_SCALE != 0);
        static size_t infer_act_block_size(const ggml_gemmini_args_t *args);
    };
}

inline size_t aisa::ActivationDEC::infer_act_block_size(const ggml_gemmini_args_t *args)
{
    if (!kActBlockScaleEnabled || args == nullptr)
        return QK8_0;

    const size_t blocks_per_row = args->A_scale_cols;
    const size_t k = args->K;
    if (blocks_per_row == 0 || k == 0)
        return QK8_0;

    const size_t derived = k / blocks_per_row;
    return std::max<size_t>(size_t{1}, derived);
}

#include "activation_DEC.tpp"
