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

        // Prepare: select top-K, compute residuals, build R_k (zero-overhead)
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

        // ========== Phase 1: Initialize dimensions ==========
        start = read_cycles();
        K_ = static_cast<size_t>(A_->ne[0]); // input channels
        I_ = static_cast<size_t>(A_->ne[1]); // number of rows
        J_ = qW_->getCols();                 // output channels
        alpha_ = std::max<size_t>(1, std::min(K_,
                                              static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:init:dims] I=%zu K=%zu J=%zu alpha=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, I_, K_, J_, alpha_, start, end, end - start);

        // ========== Phase 2: Initialize buffers ==========
        start = read_cycles();
        S_.assign(I_, {});
        delta_.assign(I_, {});
        scratch_idx_.resize(K_);
        scratch_abs_.resize(K_);

        // R_k buffers: rk_offs_ as count array (will be prefix-summed later)
        rk_offs_.assign(K_ + 1, 0);
        rk_stage_.clear();
        rk_stage_.reserve(I_ * std::min(alpha_, K_)); // pre-allocate for staging
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:init:buffers] S/delta/scratch/rk allocated start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);

        // ========== Phase 3: Per-row Top-K selection + staging ==========
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
        printf("[layer=%s][DEC:prepare:rows:loop-end] total_rows=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, I_, start, end, end - start);

        // ========== Phase 4: Build R_k from staged data ==========
        buildRk();
    }

    void ActivationDEC::selectTopKandComputeResidual(size_t r,
                                                     const float *x_r,
                                                     const int8_t *qx_r)
    {
        uint64_t start, end;

        // Step 1: Initialize scratch arrays + cache |x_r[k]|
        start = read_cycles();
        auto &idx = scratch_idx_;
        auto &ax = scratch_abs_;
        for (size_t k = 0; k < K_; ++k)
        {
            idx[k] = static_cast<int>(k);
            ax[k] = std::fabs(x_r[k]);
        }
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rows:r=%zu:create-idx-abs] K=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, r, K_, start, end, end - start);

        // Step 2: Find top-alpha channels (skip if alpha == K)
        const size_t topk = std::min(alpha_, K_);
        if (topk < K_)
        {
            start = read_cycles();
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&ax](int a, int b) { return ax[a] > ax[b]; });
            end = read_cycles();
            printf("[layer=%s][DEC:prepare:rows:r=%zu:top-alpha-select] alpha=%zu start=%lu end=%lu elapsed=%lu\n",
                   layer_, r, topk, start, end, end - start);
        }

        // Step 3: Store S_[r], delta_[r] + stage (k,r,δ) + count R_k
        start = read_cycles();
        S_[r].resize(topk);
        delta_[r].resize(topk);
        for (size_t i = 0; i < topk; ++i)
        {
            const int k = idx[i];
            const float d = x_r[k] - static_cast<float>(qx_r[k]) * SCALE;
            
            // Store in per-row structures (for compatibility/debugging)
            S_[r][i] = k;
            delta_[r][i] = d;

            // OPTIMIZATION: Stage for R_k and count simultaneously
            rk_stage_.push_back({k, static_cast<int>(r), d});
            rk_offs_[static_cast<size_t>(k) + 1]++; // count for prefix-sum
        }
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rows:r=%zu:store-stage-count] topk=%zu staged=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, r, topk, rk_stage_.size(), start, end, end - start);
    }

    void ActivationDEC::buildRk()
    {
        uint64_t start, end;

        // ========== Step 1: Prefix-sum to compute offsets ==========
        start = read_cycles();
        for (size_t k = 1; k <= K_; ++k)
            rk_offs_[k] += rk_offs_[k - 1];
        const size_t nnz = rk_offs_[K_];
        rk_pairs_.assign(nnz, {0, 0.f});
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rk:prefix-sum] nnz=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, nnz, start, end, end - start);

        // ========== Step 2: Scatter staged triplets into R_k ==========
        start = read_cycles();
        std::vector<size_t> pos = rk_offs_; // copy for scatter indexing
        for (const auto &t : rk_stage_)
        {
            const int k = t.k;
            const size_t dst = pos[static_cast<size_t>(k)]++;
            rk_pairs_[dst] = {t.r, t.d};
        }
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rk:scatter-stage] staged=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, rk_stage_.size(), start, end, end - start);

        // ========== Step 3: Collect non-empty k indices ==========
        start = read_cycles();
        unique_k_.clear();
        unique_k_.reserve(K_);
        for (size_t k = 0; k < K_; ++k)
            if (rk_offs_[k] != rk_offs_[k + 1])
                unique_k_.push_back(static_cast<int>(k));
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rk:unique-k] count=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, unique_k_.size(), start, end, end - start);

        // ========== Step 4: Reclaim staging memory ==========
        start = read_cycles();
        rk_stage_.clear();
        rk_stage_.shrink_to_fit();
        end = read_cycles();
        printf("[layer=%s][DEC:prepare:rk:free-stage] start=%lu end=%lu elapsed=%lu\n",
               layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation_unrolled(const int8_t *W, float *Y_com)
    {
        uint64_t start, end;

        start = read_cycles();

        // Temporary buffer for int8→float conversion (per k)
        std::vector<float> Wk_f(J_);

        // Iterate over each salient channel k
        for (int k : unique_k_)
        {
            const size_t beg = rk_offs_[static_cast<size_t>(k)];
            const size_t end_idx = rk_offs_[static_cast<size_t>(k) + 1];
            const int8_t *Wk = W + static_cast<size_t>(k) * J_;

            // Convert Ŵ[k,:] once per k (critical optimization)
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = static_cast<float>(Wk[j]);

            // Accumulate for all (r, δ[r,k]) in R_k
            for (size_t t = beg; t < end_idx; ++t)
            {
                const int r = rk_pairs_[t].first;
                const float d = rk_pairs_[t].second;
                float *Yr = Y_com + static_cast<size_t>(r) * J_;

                size_t j = 0;

                // 8-way unroll for vectorization
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

                // Handle remaining elements
                for (; j < J_; ++j)
                    Yr[j] += d * Wk_f[j];
            }
        }

        end = read_cycles();
        printf("[layer=%s][DEC:compute:compensation-unrolled] unique_k=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, unique_k_.size(), start, end, end - start);
    }

    void ActivationDEC::applyCompensation(ggml_tensor *C_out,
                                          const std::vector<float> &Y_com)
    {
        uint64_t start, end;

        start = read_cycles();

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

        end = read_cycles();
        printf("[layer=%s][DEC:apply:compensation] I=%zu J=%zu start=%lu end=%lu elapsed=%lu\n",
               layer_, I_, J_, start, end, end - start);
    }
}