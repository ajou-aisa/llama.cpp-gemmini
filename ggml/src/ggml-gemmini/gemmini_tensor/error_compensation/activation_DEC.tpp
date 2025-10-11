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

        std::vector<float> y_com(dec.I_ * dec.J_, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation(What, dec.J_, y_com.data());

        dec.applyCompensation(C_out, y_com);
    }

    void ActivationDEC::prepare()
    {
        // A의 dimension: ne[0]=K (columns), ne[1]=I (rows)
        K_ = static_cast<size_t>(A_->ne[0]); // cols
        I_ = static_cast<size_t>(A_->ne[1]); // rows
        J_ = qW_->getCols();
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
        // Step 1: Create temporary vector with absolute values
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
        if (I_ == 1)
            computeCompensation_RowMajor(W, J, y_com);
        else
            computeCompensation_CSC(W, J, y_com);
    }

    void ActivationDEC::computeCompensation_RowMajor(const int8_t *W, size_t J, float *y_com)
    {
        const size_t stride_W = J;
        const auto &Sr = S_[0];
        const auto &Dr = delta_[0];

        const size_t JB = (J >= 8192 ? 4096 : J); // J가 작으면 무타일
        std::vector<float> wbuf(JB);

        uint64_t t0 = read_cycles();
        uint64_t conv_time = 0, acc_time = 0;

        for (size_t j0 = 0; j0 < J; j0 += JB)
        {
            const size_t jb = std::min(JB, J - j0);
            float *y = y_com + j0;

            for (size_t i = 0; i < Sr.size(); ++i)
            {
                const int k = Sr[i];
                const float d = Dr[i];
                const int8_t *wrow_i8 = W + (size_t)k * stride_W + j0;

                uint64_t c0 = read_cycles();
                for (size_t j = 0; j < jb; ++j)
                    wbuf[j] = (float)wrow_i8[j];
                conv_time += (read_cycles() - c0);

                uint64_t a0 = read_cycles();
                for (size_t j = 0; j < jb; ++j)
                    y[j] += d * wbuf[j];
                acc_time += (read_cycles() - a0);
            }
        }

        uint64_t t1 = read_cycles();
        printf("[layer=%s][Row-major I=1] J=%zu, alpha=%zu, JB=%zu | total=%lu, conv=%lu (%.1f%%), acc=%lu (%.1f%%)\n",
               layer_, J, alpha_, (J >= 8192 ? (size_t)4096 : J),
               (t1 - t0), conv_time, 100.0 * conv_time / (t1 - t0), acc_time, 100.0 * acc_time / (t1 - t0));
    }

    void ActivationDEC::computeCompensation_CSC(const int8_t *W, size_t J, float *y_com)
    {
        const size_t stride_W = J;
        const size_t JB = (J >= 8192 ? 4096 : J);
        std::vector<float> wbuf(JB);

        // 0) unique_k 추출 (O(I*alpha))
        uint64_t u0 = read_cycles();
        std::vector<int> unique_k;
        unique_k.reserve(I_ * alpha_);
        std::vector<uint8_t> seen(K_, 0);
        size_t nnz = 0;
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
        uint64_t u1 = read_cycles();

        printf("[layer=%s][DEC setup] I=%zu, J=%zu, K=%zu, alpha=%zu, nnz=%zu, unique_k=%zu, JB=%zu | uniq_build=%lu\n",
               layer_, I_, J_, K_, alpha_, nnz, unique_k.size(), JB, (u1 - u0));

        // 1) 메인 루프
        uint64_t t0 = read_cycles();
        uint64_t conv_time = 0, acc_time = 0, lookup_time = 0;

        for (size_t j0 = 0; j0 < J; j0 += JB)
        {
            const size_t jb = std::min(JB, J - j0);

            for (int k : unique_k)
            {
                // 변환: k, tile 당 1회
                uint64_t c0 = read_cycles();
                const int8_t *wrow_i8 = W + (size_t)k * stride_W + j0;
                for (size_t j = 0; j < jb; ++j)
                    wbuf[j] = (float)wrow_i8[j];
                conv_time += (read_cycles() - c0);

                // 선택 행만 누적 (I <= 6 → 선형 탐색으로도 충분)
                for (int r = 0; r < (int)I_; ++r)
                {
                    const auto &Sr = S_[r];
                    const auto &Dr = delta_[r];

                    // lookup을 hit/미스 모두 포함해서 측정
                    uint64_t l0 = read_cycles();
                    float d = 0.f;
                    bool hit = false;
                    for (size_t i = 0; i < Sr.size(); ++i)
                    {
                        if (Sr[i] == k)
                        {
                            d = Dr[i];
                            hit = true;
                            break;
                        }
                    }
                    uint64_t l1 = read_cycles();
                    lookup_time += (l1 - l0);

                    if (!hit)
                        continue;

                    // 누적
                    uint64_t a0 = read_cycles();
                    float *y_row = y_com + (size_t)r * J + j0;
                    for (size_t j = 0; j < jb; ++j)
                        y_row[j] += d * wbuf[j];
                    acc_time += (read_cycles() - a0);
                }
            }
        }

        uint64_t t1 = read_cycles();
        uint64_t total = (t1 - t0);
        uint64_t overhead = (total > conv_time + acc_time + lookup_time)
                                ? (total - conv_time - acc_time - lookup_time)
                                : 0;

        printf("[layer=%s][Column on-the-fly] total=%lu | conv=%lu (%.1f%%), acc=%lu (%.1f%%), lookup=%lu (%.1f%%), overhead=%lu (%.1f%%)\n",
               layer_, total,
               conv_time, 100.0 * conv_time / total,
               acc_time, 100.0 * acc_time / total,
               lookup_time, 100.0 * lookup_time / total,
               overhead, 100.0 * overhead / total);
    }

    void ActivationDEC::applyCompensation(ggml_tensor *C_out, const std::vector<float> &y_com)
    {
        uint64_t start = read_cycles();

        float *C = static_cast<float *>(C_out->data);
        const size_t ldc = C_out->nb[1] / sizeof(float); // ggml row stride
        for (size_t r = 0; r < I_; ++r)
        {
            const float *y = y_com.data() + r * J_;
            float *crow = C + r * ldc;
            // s_w는 여기서 한 번만 적용하거나, 위 누적 루프에서 d에 곱해도 됨(둘 중 하나만)
            for (size_t j = 0; j < J_; ++j)
                crow[j] += y[j] * SCALE_W;
        }

        uint64_t end = read_cycles();
        printf("[layer=%s][Apply compensation to output] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }
}