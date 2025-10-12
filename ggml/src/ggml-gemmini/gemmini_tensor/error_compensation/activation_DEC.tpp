#include "activation_DEC.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <include/gemmini.h>

namespace aisa
{
    void ActivationDEC::compensate(const ggml_tensor *A,
                                   const BenchTensor<int8_t> *qA,
                                   const BenchTensor<int8_t> *qW,
                                   ggml_tensor *C_out)
    {
        ActivationDEC dec(A, qA, qW);
        dec.layer_ = labelFromWeight(qW->getName().c_str());

        // Prepare: select top-K, compute residuals, build R_k
        dec.prepare();

        // Compute compensation matrix Y_com (I×J)
        std::vector<float> Y_com(dec.I_ * dec.J_, 0.f);
        const int8_t *W = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation(W, Y_com.data());
        dec.applyCompensation(C_out, Y_com);
    }

    void ActivationDEC::prepare()
    {
        K_ = static_cast<size_t>(A_->ne[0]); // columns (channels)
        I_ = static_cast<size_t>(A_->ne[1]); // rows
        J_ = qW_->getCols();                 // output channels

        // Top-K salient channels per row (at least 1, at most K)
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        S_.assign(I_, {});
        delta_.assign(I_, {});

        // Allocate scratch buffer once (reused across all rows)
        scratch_idx_.resize(K_);

        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        // Select top-K and compute residuals for each row
        for (size_t r = 0; r < I_; ++r)
        {
            const float *x_r = x + r * K_;
            const int8_t *qx_r = qx + r * stride_qA;
            selectTopKandComputeResidual(r, x_r, qx_r);
        }

        // Build R_k from S[r] and delta[r]
        buildRk();
    }

    void ActivationDEC::selectTopKandComputeResidual(size_t r,
                                                     const float *x_r,
                                                     const int8_t *qx_r)
    {
        uint64_t start, end;

        // Step 1: Initialize scratch index array (reused, no allocation)
        start = read_cycles();
        auto &indices = scratch_idx_; // reference to member scratch buffer
        for (size_t k = 0; k < K_; ++k)
            indices[k] = static_cast<int>(k);
        end = read_cycles();
        printf("[layer=%s][Initialize index array] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);

        // Step 2: nth_element to find top-alpha channels (O(n) instead of O(n log k))
        start = read_cycles();
        std::nth_element(indices.begin(), indices.begin() + alpha_, indices.end(),
                         [&x_r](int a, int b)
                         {
                             return std::abs(x_r[a]) > std::abs(x_r[b]);
                         });
        end = read_cycles();
        printf("[layer=%s][nth_element to find top-alpha channels] start = %lu, end = %lu, elapsed = %lu\n",
               layer_, start, end, end - start);

        // Step 3: Store indices and compute residuals (float path)
        start = read_cycles();
        S_[r].resize(alpha_);
        delta_[r].resize(alpha_);
        for (size_t i = 0; i < alpha_; ++i)
        {
            const int k = indices[i];
            S_[r][i] = k;
            // δ[r,k] = x[r,k] - x̂[r,k] * s_x
            delta_[r][i] = x_r[k] - static_cast<float>(qx_r[k]) * SCALE;
        }
        end = read_cycles();
        printf("[layer=%s][Store indices and compute residual] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::buildRk()
    {
        uint64_t start = read_cycles();

        rk_offs_.assign(K_ + 1, 0);

        // 1) Count occurrences of each k
        for (int r = 0; r < (int)I_; ++r)
            for (int k : S_[r])
                rk_offs_[k + 1]++;

        // 2) Prefix sum to get offsets
        for (size_t k = 1; k <= K_; ++k)
            rk_offs_[k] += rk_offs_[k - 1];

        const size_t nnz = rk_offs_[K_];
        rk_pairs_.resize(nnz);

        // 3) Fill (row, delta) pairs
        std::vector<size_t> pos = rk_offs_;
        for (int r = 0; r < (int)I_; ++r)
        {
            const auto &Sr = S_[r];
            const auto &Dr = delta_[r];
            for (size_t i = 0; i < Sr.size(); ++i)
            {
                const int k = Sr[i];
                rk_pairs_[pos[k]++] = {r, Dr[i]};
            }
        }

        // 4) Collect non-empty k indices
        unique_k_.clear();
        unique_k_.reserve(K_);
        for (size_t k = 0; k < K_; ++k)
            if (rk_offs_[k] != rk_offs_[k + 1])
                unique_k_.push_back((int)k);

        uint64_t end = read_cycles();
        printf("[layer=%s][Build R_k] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation(const int8_t *W, float *Y_com)
    {
        // Unified path for both I=1 and I>1
        // Formula: Y[r,j] += Σ_{k} Σ_{(r,δ)∈R_k} δ · Ŵ[k,j]

        uint64_t start = read_cycles();

        // Temporary buffer for int8→float conversion (per k)
        std::vector<float> Wk_f(J_);

        // Iterate over each salient channel k
        for (int k : unique_k_)
        {
            const size_t beg = rk_offs_[k];
            const size_t end = rk_offs_[k + 1];
            const int8_t *Wk = W + (size_t)k * J_;

            // Convert Ŵ[k,:] once per k (critical optimization)
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = static_cast<float>(Wk[j]);

            // Accumulate for all (r, δ[r,k]) in R_k
            for (size_t t = beg; t < end; ++t)
            {
                const int r = rk_pairs_[t].first;
                const float d = rk_pairs_[t].second;
                float *Yr = Y_com + (size_t)r * J_;

                for (size_t j = 0; j < J_; ++j)
                    Yr[j] += d * Wk_f[j];
            }
        }

        uint64_t end_cycle = read_cycles();
        printf("[layer=%s][Compute and accumulate compensation] start = %lu, end = %lu, elapsed = %lu\n",
               layer_, start, end_cycle, end_cycle - start);
    }

    void ActivationDEC::applyCompensation(ggml_tensor *C_out,
                                          const std::vector<float> &Y_com)
    {
        uint64_t start = read_cycles();

        float *C = static_cast<float *>(C_out->data);
        const size_t stride_C = C_out->nb[1] / sizeof(float);

        // C[r,j] += Y_com[r,j] * s_w
        for (size_t r = 0; r < I_; ++r)
        {
            const float *Yr = Y_com.data() + r * J_;
            float *Cr = C + r * stride_C;
            for (size_t j = 0; j < J_; ++j)
                Cr[j] += Yr[j] * SCALE_W;
        }

        uint64_t end = read_cycles();
        printf("[layer=%s][Apply compensation to output] start = %lu, end = %lu, elapsed = %lu\n",
               layer_, start, end, end - start);
    }
}