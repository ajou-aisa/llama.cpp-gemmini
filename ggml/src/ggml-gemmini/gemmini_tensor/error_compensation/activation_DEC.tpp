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
        dec.computeResidual();

        const size_t J = qW->getCols();
        std::vector<float> y_com(J, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());
        dec.computeCompensation(What, J, y_com.data());

        float *y_fp = static_cast<float *>(C_out->data);
        for (size_t i = 0; i < qW->getCols(); i++)
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

        // 각 row마다 독립적으로 Top-K 선택
        for (size_t r = 0; r < I_; ++r)
        {
            const float *row_ptr = x + r * K_;
            selectTopKForRow(r, row_ptr, ratio);
        }
    }

    void ActivationDEC::selectTopKForRow(size_t row_idx, const float *row_data, double ratio)
    {
        // row의 K개 원소를 절대값 기준으로 정렬
        std::vector<std::pair<float, int>> sorted(K_);
        for (size_t k = 0; k < K_; ++k)
            sorted[k] = {row_data[k], static_cast<int>(k)};

        // 내림차순(절댓값)
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b)
                  {
                      return std::abs(a.first) > std::abs(b.first);
                  });

        // Top-alpha_ 개 인덱스 저장 (row 내 local index)
        S_[row_idx].resize(alpha_);
        for (size_t i = 0; i < alpha_; ++i)
            S_[row_idx][i] = sorted[i].second;
    }

    void ActivationDEC::computeResidual()
    {
        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());
        const size_t stride_qA = qA_->getStride();

        for (size_t r = 0; r < I_; ++r)
        {
            const float *row_fp = x + r * K_;
            const int8_t *row_q = qx + r * stride_qA;

            delta_[r].resize(alpha_);

            for (size_t i = 0; i < alpha_; ++i)
            {
                const int k = S_[r][i]; // row 내 local index
                const float xhat = static_cast<float>(row_q[k]) * SCALE;
                delta_[r][i] = row_fp[k] - xhat;
            }
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