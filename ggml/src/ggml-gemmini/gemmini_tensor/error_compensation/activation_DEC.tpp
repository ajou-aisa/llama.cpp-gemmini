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

        // Prepare: select top-K, compute residuals, build R_k (zero-rescan)
        dec.prepare();

        // Compute compensation matrix Y_com (I×J)
        std::vector<float> Y_com(dec.I_ * dec.J_, 0.f);
        const int8_t *W = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation_unrolled(W, Y_com.data());
        dec.applyCompensation(C_out, Y_com);
    }

    void ActivationDEC::prepare()
    {
        uint64_t start, end;

        // Step 1: Initialize dimensions and buffers
        start = read_cycles();
        K_ = static_cast<size_t>(A_->ne[0]);
        I_ = static_cast<size_t>(A_->ne[1]);
        J_ = qW_->getCols();
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        S_.assign(I_, {});
        delta_.assign(I_, {});
        scratch_idx_.resize(K_);
        scratch_abs_.resize(K_);
        rk_offs_.assign(K_ + 1, 0);
        rk_stage_.clear();
        rk_stage_.reserve(I_ * std::min(alpha_, K_));
        end = read_cycles();
        printf("[layer=%s][DEC: Initialize dimensions and buffers] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);

        // Step 2: Select top-K and compute residuals for all rows
        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        start = read_cycles();
        for (size_t r = 0; r < I_; ++r)
        {
            const float *x_r = x + r * K_;
            const int8_t *qx_r = qx + r * stride_qA;
            selectTopKandComputeResidual(r, x_r, qx_r);
        }
        end = read_cycles();
        printf("[layer=%s][DEC: Select top-K and stage R_k for all rows] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);

        // Step 3: Build R_k CSC structure
        buildRk();
    }

    void ActivationDEC::selectTopKandComputeResidual(size_t r,
                                                     const float *x_r,
                                                     const int8_t *qx_r)
    {
        // Initialize index array and cache absolute values
        auto &idx = scratch_idx_;
        auto &ax = scratch_abs_;
        for (size_t k = 0; k < K_; ++k)
        {
            idx[k] = static_cast<int>(k);
            ax[k] = std::fabs(x_r[k]);
        }

        // Select top-alpha channels (skip if alpha == K)
        const size_t topk = std::min(alpha_, K_);
        if (topk < K_)
        {
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&ax](int a, int b) { return ax[a] > ax[b]; });
        }

        // Store S_[r], delta_[r] + Stage R_k + Count occurrences
        S_[r].resize(topk);
        delta_[r].resize(topk);
        for (size_t i = 0; i < topk; ++i)
        {
            const int k = idx[i];
            const float d = x_r[k] - static_cast<float>(qx_r[k]) * SCALE;
            
            S_[r][i] = k;
            delta_[r][i] = d;

            rk_stage_.push_back({k, static_cast<int>(r), d});
            rk_offs_[k + 1]++;
        }
    }

    void ActivationDEC::buildRk()
    {
        uint64_t start = read_cycles();

        // Prefix-sum to convert counts to offsets
        for (size_t k = 1; k <= K_; ++k)
            rk_offs_[k] += rk_offs_[k - 1];
        
        const size_t nnz = rk_offs_[K_];
        rk_pairs_.assign(nnz, {0, 0.f});

        // Scatter staged triplets to CSC pairs
        std::vector<size_t> pos = rk_offs_;
        for (const auto &t : rk_stage_)
        {
            const size_t dst = pos[t.k]++;
            rk_pairs_[dst] = {t.r, t.d};
        }

        // Collect non-empty k indices
        unique_k_.clear();
        unique_k_.reserve(K_);
        for (size_t k = 0; k < K_; ++k)
            if (rk_offs_[k] != rk_offs_[k + 1])
                unique_k_.push_back(static_cast<int>(k));

        // Free staging memory
        rk_stage_.clear();
        rk_stage_.shrink_to_fit();

        uint64_t end = read_cycles();
        printf("[layer=%s][DEC: Build R_k CSC structure] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation(const int8_t *W, float *Y_com)
    {
        uint64_t start = read_cycles();

        std::vector<float> Wk_f(J_);

        // For each salient channel k, convert W and accumulate
        for (int k : unique_k_)
        {
            const size_t beg = rk_offs_[k];
            const size_t end = rk_offs_[k + 1];
            const int8_t *Wk = W + static_cast<size_t>(k) * J_;

            // Convert Ŵ[k,:] once per k
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = static_cast<float>(Wk[j]);

            // Accumulate for all (r, δ[r,k]) in R_k
            for (size_t t = beg; t < end; ++t)
            {
                const int r = rk_pairs_[t].first;
                const float d = rk_pairs_[t].second;
                float *Yr = Y_com + static_cast<size_t>(r) * J_;

                for (size_t j = 0; j < J_; ++j)
                    Yr[j] += d * Wk_f[j];
            }
        }

        uint64_t end_cycle = read_cycles();
        printf("[layer=%s][DEC: Compute and accumulate compensation] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end_cycle, end_cycle - start);
    }

    void ActivationDEC::computeCompensation_unrolled(const int8_t *W, float *Y_com)
    {
        uint64_t start = read_cycles();

        std::vector<float> Wk_f(J_);

        // For each salient channel k, convert W and accumulate with unrolling
        for (int k : unique_k_)
        {
            const size_t beg = rk_offs_[k];
            const size_t end = rk_offs_[k + 1];
            const int8_t *Wk = W + static_cast<size_t>(k) * J_;

            // Convert Ŵ[k,:] to float once per k
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = static_cast<float>(Wk[j]);

            // Accumulate compensation with 8-way unrolling
            for (size_t t = beg; t < end; ++t)
            {
                const int r = rk_pairs_[t].first;
                const float d = rk_pairs_[t].second;
                float *Yr = Y_com + static_cast<size_t>(r) * J_;

                size_t j = 0;
                for (; j + 7 < J_; j += 8)
                {
                    Yr[j + 0] += d * Wk_f[j + 0];
                    Yr[j + 1] += d * Wk_f[j + 1];
                    Yr[j + 2] += d * Wk_f[j + 2];
                    Yr[j + 3] += d * Wk_f[j + 3];
                    Yr[j + 4] += d * Wk_f[j + 4];
                    Yr[j + 5] += d * Wk_f[j + 5];
                    Yr[j + 6] += d * Wk_f[j + 6];
                    Yr[j + 7] += d * Wk_f[j + 7];
                }
                for (; j < J_; ++j)
                    Yr[j] += d * Wk_f[j];
            }
        }

        uint64_t end = read_cycles();
        printf("[layer=%s][DEC: Compute and accumulate compensation (unrolled)] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);
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
        printf("[layer=%s][DEC: Apply compensation to output] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);
    }
}