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
        
        float* y_fp = static_cast<float*>(C_out->data);
        for(int i=0; i<qW->getCols(); i++)
            y_fp[i] += y_com[i]; // W scale 보정은 없는 상태
    }

    ActivationDEC::ActivationDEC(const ggml_tensor *A, const BenchTensor<int8_t> *qA, double ratio)
        : A_(A), qA_(qA)
    {
        const size_t N = A_->ne[0] * A_->ne[1];
        alpha_ = std::max<size_t>(1, std::min(N, static_cast<size_t>(std::llround(N * ratio))));
        const float *x = static_cast<const float *>(A_->data);

        // 값과 인덱스 쌍으로 정렬
        sorted_.resize(N);
        for (size_t i = 0; i < N; ++i)
            sorted_[i] = {x[i], i};

        // 내림차순 정렬 (절대값 기준)
        std::sort(sorted_.begin(), sorted_.end(),
                  [](const auto &a, const auto &b)
                  { return std::abs(a.first) > std::abs(b.first); });

        // Top-K index 추출
        S_.resize(alpha_);
        for (size_t i = 0; i < alpha_; ++i)
            S_[i] = sorted_[i].second;
    }

    void ActivationDEC::computeResidual()
    {
        const float *x = static_cast<const float *>(A_->data);
        const int8_t *qx = static_cast<const int8_t *>(qA_->get());

        delta_.resize(alpha_);

        for (size_t i = 0; i < alpha_; ++i)
        {
            const int s = S_[i];
            // dequantize
            const float xhat = static_cast<float>(qx[s]) * SCALE;
            delta_[i] = x[s] - xhat;
        }
    }

    void ActivationDEC::computeCompensation(const int8_t *W, size_t J, float *y_com)
    {
        // S.size() == D.size() == alpha
        for (int i = 0; i < alpha_; ++i)
        {
            const int s = S_[i];                                   // 입력 채널 인덱스
            const float d = delta_[i];                             // 잔차 (alpha 길이)
            const int8_t *wrow = &W[s * J]; // W_{s,:}  (row-major)
            for (int j = 0; j < J; ++j)
                y_com[j] += d * static_cast<float>(wrow[j]);
        }
    }

}