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

        const size_t J = qW->getCols();
        dec.layer_= labelFromWeight(qW->getName().c_str()); // layer 이름 추출
        std::vector<float> y_com(J, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());

        uint64_t start = read_cycles();
        dec.computeCompensation_kGrouped(What, J, y_com.data());
        uint64_t end = read_cycles();
        printf("[layer=%s][computeCompensation] start = %lu, end = %lu, elapsed = %lu\n", dec.layer_, start, end, end - start);

        start = read_cycles();
        float *y_fp = static_cast<float *>(C_out->data);
        for (size_t i = 0; i < J; ++i)
            y_fp[i] += y_com[i] * SCALE_W; // W scale 로 dequantize
        end = read_cycles();
        printf("[layer=%s][applyCompensation] start = %lu, end = %lu, elapsed = %lu\n", dec.layer_, start, end, end - start);
    }

    ActivationDEC::ActivationDEC(const ggml_tensor *A, const BenchTensor<int8_t> *qA)
        : A_(A), qA_(qA)
    {
        // A의 dimension: ne[0]=K (columns), ne[1]=I (rows)
        K_ = static_cast<size_t>(A_->ne[0]);
        I_ = static_cast<size_t>(A_->ne[1]);

        // row 별 Top_K 개수
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        S_.resize(I_);
        delta_.resize(I_);

        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        // 각 row마다 Top-K 선택 + Residual 계산 병합
        uint64_t start = read_cycles();
        for (size_t r = 0; r < I_; ++r)
        {
            const float *row_fp = x + r * K_;
            const int8_t *row_q = qx + r * stride_qA;
            selectTopKandComputeResidual(r, row_fp, row_q);
        }
        uint64_t end = read_cycles();
        printf("[layer=%s][selectTopKandComputeResidual] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::selectTopKandComputeResidual(size_t row_idx, 
                                                      const float *row_fp, 
                                                      const int8_t *row_q)
    {
        // 임시 벡터: 절대값과 인덱스
        std::vector<std::pair<float, int>> temp(K_);
        for (size_t k = 0; k < K_; ++k)
            temp[k] = {std::abs(row_fp[k]), static_cast<int>(k)};

        // partial_sort: 상위 alpha_개만 정렬 (O(K log alpha))
        std::partial_sort(temp.begin(), temp.begin() + alpha_, temp.end(), [](const auto &a, const auto &b)
                          { return a.first > b.first; });

        // Top-K 인덱스 저장 + Residual 계산
        S_[row_idx].resize(alpha_);
        delta_[row_idx].resize(alpha_);

        for (size_t i = 0; i < alpha_; ++i)
        {
            const int k = temp[i].second;
            S_[row_idx][i] = k;
            
            // Residual 계산 
            // origin: 0.811 -> quantized: 127 ~ 127 
            const float xhat = static_cast<float>(row_q[k]) * SCALE;
            delta_[row_idx][i] = row_fp[k] - xhat; // delat: dequantize 오차 
        }
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

        // 1) k별 d-합 s_k 계산 (sparse)
        std::vector<int> uniq_k;
        std::vector<float> s_k;

        // 메모리 예약
        uniq_k.reserve(I_ * alpha_);
        s_k.reserve(I_ * alpha_);

        // unique_k 내 인덱스 (없으면 -1)
        std::vector<int> mark(K_, -1);

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
                    s_k.push_back(d); // s_k[idx] = d
                }
                else
                    s_k[idx] += d; // 누적
            }
        }

        // 2) 고유 k만 스캔하여 y_com 누적
        for (size_t t = 0; t < uniq_k.size(); ++t)
        {
            int k = uniq_k[t];
            const int8_t *wrow = W + (size_t)k * stride_W; // W[k,:]
            float s = s_k[t];

            for (size_t j = 0; j < J; ++j)
                y_com[j] += s * (float)wrow[j];
        }

        // 3) mark 리셋 (sparse reset)
        for (int k : uniq_k)
            mark[k] = -1;
    }
}