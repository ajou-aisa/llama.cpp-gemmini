#include "activation_DEC.h"
#include <algorithm>
#include <cmath>
#include <include/gemmini.h>

namespace aisa
{
    void ActivationDEC::compensate(const ggml_tensor *A,
                                   const BenchTensor<int8_t> *qA,
                                   const BenchTensor<int8_t> *qW,
                                   ggml_tensor *C_out)
    {
        ActivationDEC dec(A, qA);
        dec.layer_= labelFromWeight(qW->getName().c_str()); // layer 이름 추출

        // 전처리(Top-K 선택 및 residual 계산) + 로깅
        dec.prepare();

        const size_t J = qW->getCols();
        std::vector<float> y_com(J, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation_kGrouped(What, J, y_com.data());

        start = read_cycles();
        float *y_fp = static_cast<float *>(C_out->data);
        for (size_t i = 0; i < J; ++i)
            y_fp[i] += y_com[i] * SCALE_W; // W scale 로 dequantize
        end = read_cycles();
        printf("[layer=%s][Apply compensation to output] start = %lu, end = %lu, elapsed = %lu\n", dec.layer_, start, end, end - start);
    }

    void ActivationDEC::prepare()
    {
        // A의 dimension: ne[0]=K (columns), ne[1]=I (rows)
        K_ = static_cast<size_t>(A_->ne[0]); // cols
        I_ = static_cast<size_t>(A_->ne[1]); // rows

        // row별 Top-K 개수
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        // 버퍼 준비
        S_.assign(I_, {});
        delta_.assign(I_, {});

        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        // 각 row마다 Top-K 선택 + Residual 계산
        for (size_t r = 0; r < I_; ++r)
        {
            const float *row_fp = x + r * K_;
            const int8_t *row_q = qx + r * stride_qA;
            selectTopKandComputeResidual(r, row_fp, row_q);
        }

    }

    void ActivationDEC::selectTopKandComputeResidual(size_t row_idx, 
                                                      const float *row_fp, 
                                                      const int8_t *row_q)
    {
        //Step 1: Create temporary vector with absolute values
        std::vector<std::pair<float, int>> temp(K_);
        for (size_t k = 0; k < K_; ++k)
            temp[k] = {std::abs(row_fp[k]), static_cast<int>(k)};

        uint64_t start = read_cycles();
        // Step 2: Partial sort to find top-alpha channels
        std::partial_sort(temp.begin(), temp.begin() + alpha_, temp.end(), [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        uint64_t end = read_cycles();
        printf("[layer=%s][Partial sort to find top-alpha channels] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);


        // Step 3: Store indices and compute residual
        S_[row_idx].resize(alpha_);
        delta_[row_idx].resize(alpha_);

        start = read_cycles();
        for (size_t i = 0; i < alpha_; ++i)
        {
            const int k = temp[i].second;
            S_[row_idx][i] = k;
            
            // Residual 계산 
            // origin: 0.811 -> quantized: 127 ~ 127 
            const float xhat = static_cast<float>(row_q[k]) * SCALE;
            delta_[row_idx][i] = row_fp[k] - xhat; // delat: dequantize 오차 
        }
        end = read_cycles();
        printf("[layer=%s][Store indices and compute residual] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation(const int8_t *W, size_t J, float *y_com)
    {
        const size_t stride_W = J; // W는 K x J (row-major)

        for (size_t r = 0; r < I_; ++r)
        {
            for (size_t k = 0; k < alpha_; ++k)
            {
                const int channel = S_[r][k]; // row 내 local index (실제 channel 인덱스) (0 ~ K-1)
                const float d = delta_[r][k];

                // W의 channel 행
                const int8_t *wrow = W + static_cast<size_t>(channel) * stride_W;

                for (size_t j = 0; j < J; ++j)
                    y_com[j] += d * static_cast<float>(wrow[j]); // d에 해당하는 보정이 모든 채널에 적용
            }
        }
    }

    void ActivationDEC::computeCompensation_kGrouped(const int8_t *W, size_t J, float *y_com)
    {
        const size_t stride_W = J;

        uint64_t start = read_cycles();
        // step. 1) k별 d-합 d_sum[k] 계산 (sparse)
        std::vector<int> uniq_k; // uniq_k[t]: 등장한 고유 channel k_t 
        std::vector<float> d_sum; // d_sum[uniq_k[t]]: 해당 channel의 누적 보상값

        // 메모리 예약
        uniq_k.reserve(I_ * alpha_);
        d_sum.reserve(I_ * alpha_);

        // unique_k 내 인덱스 => map(k, t)
        // mark[k] == t <-> uniq_k[t] == k
        std::vector<int> mark(K_, -1); // mark[k] = t이면 k가 uniq_k[t]에 존재. 없으면 -1

        for (size_t r = 0; r < I_; ++r)
        {
            for (size_t i = 0; i < alpha_; ++i)
            {
                int k = S_[r][i];
                float d = delta_[r][i];
                int idx = mark[k];
                if (idx < 0)
                { // 첫 등장
                    idx = (int)uniq_k.size();
                    mark[k] = idx;
                    uniq_k.push_back(k);
                    d_sum.push_back(d); // d_sum[idx] = d
                }
                else
                    d_sum[idx] += d; // 누적
            }
        }
        uint64_t end = read_cycles();
        printf("[layer=%s][Compute per-k sum d_{sum}[k]] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);

        // step. 2) 고유 k만 스캔하여 y_com 누적
        start = read_cycles();
        for (size_t t = 0; t < uniq_k.size(); ++t)
        {
            int k = uniq_k[t];
            const int8_t *wrow = W + (size_t)k * stride_W; // W[k,:]
            float s = d_sum[t];

            for (size_t j = 0; j < J; ++j)
                y_com[j] += s * (float)wrow[j];
        }
        end = read_cycles();
        printf("[layer=%s][Accumulate y_{com} using unique k values] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }
}