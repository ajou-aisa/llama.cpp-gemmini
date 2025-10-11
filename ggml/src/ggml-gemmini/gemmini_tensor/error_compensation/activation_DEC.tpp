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
        dec.layer_= labelFromWeight(qW->getName().c_str()); // layer 이름 추출

        // 전처리(Top-K 선택 및 residual 계산) + 로깅
        dec.prepare();

        std::vector<float> y_com(dec.J_, 0.f);

        const int8_t *What = static_cast<const int8_t *>(qW->get());

        dec.computeCompensation_CSC(What, dec.J_, y_com.data());

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

    void ActivationDEC::computeCompensation_CSC(const int8_t *W, size_t J, float *y_com)
    {
        uint64_t start, end;
        const size_t stride_W = J;

        // ===== 1) S_에서 k별 데이터 구조 구성 =====
        start = read_cycles();
        
        // k별 등장 횟수 카운트
        std::vector<int> k_count(K_, 0);
        for (int r = 0; r < (int)I_; ++r)
        {
            for (size_t i = 0; i < alpha_; ++i)
            {
                ++k_count[S_[r][i]];
            }
        }
        
        // Prefix sum으로 각 k의 데이터 시작 위치 계산
        std::vector<int> k_offset(K_ + 1, 0);
        for (int k = 0; k < (int)K_; ++k)
        {
            k_offset[k + 1] = k_offset[k] + k_count[k];
        }
        const int nnz = k_offset[K_];
        
        // (r, delta) 쌍을 저장할 배열
        std::vector<int> row_idx(nnz);
        std::vector<float> d_val(nnz);
        
        // 데이터 채우기
        std::fill(k_count.begin(), k_count.end(), 0);
        for (int r = 0; r < (int)I_; ++r)
        {
            for (size_t i = 0; i < alpha_; ++i)
            {
                const int k = S_[r][i];
                const int p = k_offset[k] + k_count[k]++;
                row_idx[p] = r;
                d_val[p] = delta_[r][i];
            }
        }
        
        end = read_cycles();
        printf("[layer=%s][Build CSC structure from S_] start=%lu, end=%lu, elapsed=%lu (nnz=%d)\n",
            layer_, start, end, end - start, nnz);

        // ===== 2) Salient channel만 순회하며 처리 =====
        start = read_cycles();

        uint64_t conversion_time = 0;
        uint64_t accumulation_time = 0;
        int salient_k_count = 0;
        
        // W row를 float로 변환할 재사용 버퍼
        std::vector<float> wrow_float(J);

        for (int k = 0; k < (int)K_; ++k)
        {
            const int begin = k_offset[k];
            const int end_idx = k_offset[k + 1];
            
            // 이 k가 선택되지 않았으면 건너뜀
            if (begin == end_idx) continue;
            
            salient_k_count++;
            const int8_t *wrow_int8 = W + (size_t)k * stride_W;

            // W[k,:]를 float로 한 번만 변환
            uint64_t conv_start = read_cycles();
            for (size_t j = 0; j < J; ++j)
            {
                wrow_float[j] = (float)wrow_int8[j];
            }
            conversion_time += read_cycles() - conv_start;

            // 이 k를 선택한 모든 행에 적용
            uint64_t acc_start = read_cycles();
            for (int p = begin; p < end_idx; ++p)
            {
                const int r = row_idx[p];
                const float d = d_val[p];
                float *y_row = y_com + (size_t)r * J;

                // 단순 루프 (컴파일러 자동 최적화 의존)
                for (size_t j = 0; j < J; ++j)
                {
                    y_row[j] += d * wrow_float[j];
                }
            }
            accumulation_time += read_cycles() - acc_start;
        }

        end = read_cycles();
        uint64_t total_time = end - start;
        uint64_t overhead = total_time - conversion_time - accumulation_time;
        
        printf("[layer=%s][Main loop (salient k only)] start=%lu, end=%lu, elapsed=%lu\n",
            layer_, start, end, total_time);
        printf("[layer=%s][  ├─ Salient channels] %d / %d (%.1f%%)\n",
            layer_, salient_k_count, (int)K_, 100.0 * salient_k_count / K_);
        printf("[layer=%s][  ├─ Conversion time] %lu (%.1f%%)\n",
            layer_, conversion_time, 100.0 * conversion_time / total_time);
        printf("[layer=%s][  ├─ Accumulation time] %lu (%.1f%%)\n",
            layer_, accumulation_time, 100.0 * accumulation_time / total_time);
        printf("[layer=%s][  └─ Overhead time] %lu (%.1f%%)\n",
            layer_, overhead, 100.0 * overhead / total_time);
    }

    void ActivationDEC::applyCompensation(ggml_tensor *C_out, const std::vector<float> &y_com) {
        uint64_t start = read_cycles();

        float *y_fp = static_cast<float *>(C_out->data);
        for (size_t i = 0; i < J_; ++i)
            y_fp[i] += y_com[i] * SCALE_W; // W scale 로 dequantize
        
        uint64_t end = read_cycles();
        printf("[layer=%s][Apply compensation to output] start = %lu, end = %lu, elapsed = %lu\n", layer_, start, end, end - start);
    }
}