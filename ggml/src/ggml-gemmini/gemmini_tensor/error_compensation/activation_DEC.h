#pragma once
#include "../bench_tensor/bench_tensor.h"
#include "../../labeling/label.h"
#include "ggml.h"
#include <vector>
#include <cstdint>

#define SCALE   1.0f // s_x (activation quantization scale)
#define SCALE_W 1.0f // s_w (weight quantization scale)
#ifndef DEC_ALPHA_RATIO
#define DEC_ALPHA_RATIO 0.05    // 5% salient channels
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
        static void compensate(const ggml_tensor *A,
                               const BenchTensor<int8_t> *qA,
                               const BenchTensor<int8_t> *qW,
                               ggml_tensor *C_out);

    private:
        const ggml_tensor *A_;          // F32 activation
        const BenchTensor<int8_t> *qA_; // quantized activation
        const BenchTensor<int8_t> *qW_; // quantized weight

        const char* layer_ = "others";
        size_t I_;      // number of rows
        size_t K_;      // input channels
        size_t J_;      // output channels
        size_t alpha_;  // number of salient channels per row

        // Per-row salient channels and residuals
        std::vector<std::vector<int>>   S_;      // S[r]: salient indices
        std::vector<std::vector<float>> delta_;  // delta[r][i]: residuals (float)

        // R_k: CSC-style reverse mapping {k → [(r, δ[r,k])]}
        std::vector<size_t> rk_offs_;                 // K_+1 prefix sums
        std::vector<std::pair<int, float>> rk_pairs_; // (row, delta) pairs
        std::vector<int> unique_k_;                   // non-empty k indices

        // On-the-fly staging for R_k construction (eliminates rescan overhead)
        struct Triplet { int k; int r; float d; };
        std::vector<Triplet> rk_stage_;

        // Scratch buffers for row-wise operations (reused across rows)
        std::vector<int>   scratch_idx_;              // K_ indices for sorting
        std::vector<float> scratch_abs_;              // cached |x_r[k]| values

        ActivationDEC(const ggml_tensor *A,
                      const BenchTensor<int8_t> *qA,
                      const BenchTensor<int8_t> *qW)
            : A_(A), qA_(qA), qW_(qW) {}

        void prepare();
        void selectTopKandComputeResidual(size_t row_idx,
                                         const float *row_fp,
                                         const int8_t *row_q);
        void buildRk();
        void computeCompensation(const int8_t *W, float *y_out);
        void computeCompensation_unrolled(const int8_t *W, float *Y_com);
        void applyCompensation(ggml_tensor *C_out,
                              const std::vector<float> &y_com);
    };
}

#include "activation_DEC.tpp"