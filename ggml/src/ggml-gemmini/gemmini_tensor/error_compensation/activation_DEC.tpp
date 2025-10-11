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
        ActivationDEC dec(A, qA, qW);
        dec.layer_ = labelFromWeight(qW->getName().c_str()); // layer 이름 추출

        // 전처리(Top-K 선택 및 residual 계산) + 로깅
        dec.prepare();

        // y_com: IxJ 보상 행렬
        std::vector<float> Y_com(dec.I_ * dec.J_, 0.f);

        const int8_t *W = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation(W, Y_com.data());

        dec.applyCompensation(C_out, Y_com);
    }

    void ActivationDEC::prepare()
    {
        // dimension: I (rows), K (channels) J (outputs) α (salient 개수)
        // ne[0]=K (columns), ne[1]=I (rows)
        K_ = static_cast<size_t>(A_->ne[0]); // cols
        I_ = static_cast<size_t>(A_->ne[1]); // rows
        J_ = qW_->getCols();
        // row별 Top-K 개수
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        // S[r], δ[r,k] 초기화
        S_.assign(I_, {});
        delta_.assign(I_, {});

        const float *x = static_cast<const float *>(A_->data);      // x (F32)
        const int8_t *qx = static_cast<const int8_t *>(qA_->get()); // x̂ (int8)
        // getStide는 element 단위 <- BehcnTensor<int8_t>이므로 byte와 동일
        const size_t stride_qA_elems = qA_->getStride();

        // 각 행 r에 대해 Top-K 선택 + δ[r,k] 계산
        for (size_t r = 0; r < I_; ++r)
        {
            const float *x_r = x + r * K_;
            const int8_t *qx_r = qx + r * stride_qA_elems;
            selectTopKandComputeResidual(r, x_r, qx_r);
        }
    }

    void ActivationDEC::selectTopKandComputeResidual(size_t r,
                                                     const float *x_r,   // x[r,:]
                                                     const int8_t *qx_r) // x̂[r,:]
    {
        uint64_t start, end;
        // Step 1: Create temporary vector with absolute values
        start = read_cycles();
        
        std::vector<std::pair<float, int>> temp(K_);
        for (size_t k = 0; k < K_; ++k)
            temp[k] = {std::abs(x_r[k]), static_cast<int>(k)};

        end = read_cycles();
        printf("[layer=%s][Create temporary vector] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);

        // Step 2: Partial sort to find top-alpha channels
        start = read_cycles();
        std::partial_sort(temp.begin(), temp.begin() + alpha_, temp.end(), [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        uint64_t end = read_cycles();
        printf("[layer=%s][Partial sort to find top-alpha channels] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);

        // Step 3: Store indices and compute residual
        start = read_cycles();
        
        S_[r].resize(alpha_);
        delta_[r].resize(alpha_);
        
        for (size_t i = 0; i < alpha_; ++i)
        {
            const int k = temp[i].second;
            S_[r][i] = k;

            // Residual 계산
            // δ[r,i] = x_i - x̂_i·s_x
            delta_[r][i] = x_r[k] - static_cast<float>(qx_r[k]) * SCALE; // delta: dequantize 오차
        }
        end = read_cycles();
        printf("[layer=%s][Store indices and compute residual] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation(const int8_t *W, float *y_com)
    {
        if (I_ == 1)
            computeCompensation_RowMajor(W, y_com);
        else
            computeCompensation_CSC(W, y_com);
    }

    void ActivationDEC::computeCompensation_RowMajor(const int8_t *W, float *Y_com)
    {
        uint64_t start, end;
        const auto &S_r = S_[0];         // S[0]
        const auto &delta_r = delta_[0]; // δ[r,:]

        const size_t JB = (J_ >= 8192 ? 4096 : J_);
        std::vector<float> W_k_float(JB); // Ŵ[k, j_tile] float 변환 버퍼

        // Step 1: Compute and accumulate compensation
        start = read_cycles();

        for (size_t j0 = 0; j0 < J_; j0 += JB)
        {
            const size_t jb = std::min(JB, J_ - j0);
            float *Y_tile = Y_com + j0;

            for (size_t i = 0; i < S_r.size(); ++i)
            {
                const int k = S_r[i];
                const float d = delta_r[i];
                const int8_t *W_k = W + (size_t)k * J_ + j0; // Ŵ[k, j_tile]

                // Ŵ -> float 타입 캐스팅
                for (size_t j = 0; j < jb; ++j)
                    W_k_float[j] = (float)W_k[j];

                // Y_com[0,j] += δ[0,k] * Ŵ[k,j]
                for (size_t j = 0; j < jb; ++j)
                    Y_tile[j] += d * W_k_float[j];
            }
        }

        end = read_cycles();
        printf("[layer=%s][Compute and accumulate compensation] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::computeCompensation_CSC(const int8_t *W, float *Y_com)
    {
        uint64_t start, end;

        const size_t JB = (J_ >= 8192 ? 4096 : J_);
        std::vector<float> W_k_float(JB);

        // Step 1: Build unique_k from S_
        start = read_cycles();
        std::vector<int> unique_k;
        unique_k.reserve(I_ * alpha_);
        std::vector<uint8_t> seen(K_, 0);
        size_t nnz = 0; // number of non-zero

        for (int r = 0; r < (int)I_; ++r)
        {
            nnz += S_[r].size();
            const auto &Sr = S_[r];
            for (size_t i = 0; i < Sr.size(); ++i)
            {
                const int k = Sr[i];
                if (!seen[k])
                {
                    seen[k] = 1;
                    unique_k.push_back(k);
                }
            }
        }
        end = read_cycles();

        printf("[layer=%s][Build unique_k] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);

        // Step 2: Compute and accumulate compensation
        start = read_cycles();

        for (size_t j0 = 0; j0 < J_; j0 += JB)
        {
            const size_t jb = std::min(JB, J_ - j0);

            for (int k : unique_k)
            {
                // int8->float 변환: k, tile 당 1회
                const int8_t *W_k = W + (size_t)k * J_ + j0; // Ŵ[k, j_tile]
                for (size_t j = 0; j < jb; ++j)
                    W_k_float[j] = (float)W_k[j];

                // 선택 행만 누적
                // Σ_{r∈R_k} where R_k = {r | k∈S[r]}
                for (int r = 0; r < (int)I_; ++r)
                {
                    const auto &S_r = S_[r];
                    const auto &delta_r = delta_[r];

                    // k ∈ S[r] 인가? (R_k 확인)
                    float d = 0.f;
                    bool hit = false;
                    for (size_t i = 0; i < S_r.size(); ++i)
                    {
                        if (S_r[i] == k)
                        {
                            d = delta_r[i]; // δ[r,k]
                            hit = true;
                            break;
                        }
                    }

                    if (!hit) // r ∉ R_k
                        continue;

                    // 누적
                    // Y_com[r,j] += δ[r,k]·Ŵ[k,j]
                    float *Y_r = Y_com + (size_t)r * J_ + j0;
                    for (size_t j = 0; j < jb; ++j)
                        Y_r[j] += d * W_k_float[j];
                }
            }
        }

        end = read_cycles();
        printf("[layer=%s][Compute and accumulate compensation] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }

    void ActivationDEC::applyCompensation(ggml_tensor *C_out, const std::vector<float> &Y_com)
    {
        uint64_t start = read_cycles();

        float *C = static_cast<float *>(C_out->data);
        const size_t stride_C = C_out->nb[1] / sizeof(float); // ggml row stride
        for (size_t r = 0; r < I_; ++r)
        {
            const float *Y_r = Y_com.data() + r * J_;
            float *C_r = C + r * stride_C;
            // s_w는 여기서 한 번만 적용하거나, 위 누적 루프에서 d에 곱해도 됨(둘 중 하나만)
            for (size_t j = 0; j < J_; ++j)
                C_r[j] += Y_r[j] * SCALE_W;
        }

        uint64_t end = read_cycles();
        printf("[layer=%s][Apply compensation to output] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }
}