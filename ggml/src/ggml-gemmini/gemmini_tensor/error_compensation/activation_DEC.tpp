#include "activation_DEC.h"
#include <algorithm>
#include <cmath>

namespace aisa
{
    void ActivationDEC::compensate(const ggml_tensor *A,
                                   const BenchTensor<int8_t> *qA,
                                   const BenchTensor<int8_t> *qW,
                                   ggml_tensor *C_out)
    {
        ActivationDEC dec(A, qA);

        const size_t J = qW->getCols();
        std::vector<float> y_com(J, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());
        dec.computeCompensation(What, J, y_com.data());

        float *y_fp = static_cast<float *>(C_out->data);
        for (size_t i = 0; i < J; ++i)
            y_fp[i] += y_com[i]; // W scale 보정은 없는 상태
    }

    ActivationDEC::ActivationDEC(const ggml_tensor *A, const BenchTensor<int8_t> *qA, double ratio)
        : A_(A), qA_(qA)
    {
        // A의 dimension: ne[0]=K (columns), ne[1]=I (rows)
        K_ = static_cast<size_t>(A_->ne[0]);
        I_ = static_cast<size_t>(A_->ne[1]);

        // row 별 Top_K 개수
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * ratio))));

        S_.resize(I_);
        delta_.resize(I_);

        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        // 각 row마다 Top-K 선택 + Residual 계산 병합
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
            const float xhat = static_cast<float>(row_q[k]) * SCALE;
            delta_[row_idx][i] = row_fp[k] - xhat;
        }
    }

    void ActivationDEC::computeCompensation(const int8_t *W, size_t J, float *y_com)
    {
        const size_t stride_W = J; // W는 K x J (row-major)

        for (size_t r = 0; r < I_; ++r)
        {
            for (size_t i = 0; i < alpha_; ++i)
            {
                const int k = S_[r][i]; // row 내 local index (0 ~ K-1)
                const float d = delta_[r][i];

                // W의 k번째 행
                const int8_t *wrow = W + static_cast<size_t>(k) * stride_W;

                for (size_t j = 0; j < J; ++j)
                    y_com[j] += d * static_cast<float>(wrow[j]);
            }
        }
    }

}