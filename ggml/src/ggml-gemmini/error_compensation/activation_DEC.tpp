// gemmini_tensor/error_compensation/activation_DEC.tpp
#include "activation_DEC.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <include/gemmini.h>
#include "../ggml-gemmini-util.h"

namespace aisa
{
    /**
     * @brief DEC(Dynamic Error Compensation) 메인 함수
     *
     * 양자화 오차를 보상하여 정확도를 향상시키는 함수
     *
     * @param A 원본 활성화 텐서 (float)
     */

    // args를 활용하는 per-block scale 적용
    void ActivationDEC::compensate(const ggml_tensor *A, ggml_gemmini_args_t *args)
    {
        // DEC 객체 초기화
        ActivationDEC dec(A, args);

        // 준비 단계: Top-K 채널 선택, 잔차 계산, R_k CSC 구조 구축
        dec.prepare();

        // 보상 행렬 Y_com (I×J) 계산
        std::vector<float> Y_com(dec.I_ * dec.J_, 0.f);

        // 언롤링된 버전으로 보상 계산
        dec.computeCompensation_unrolled(Y_com.data()); // 내부에서 per-block scale 적용

        // 계산된 보상을 출력에 적용
        dec.applyCompensation(args->f_out,
                              args->stride_f_out,
                              args->col_stride_f_out ? args->col_stride_f_out : 1,
                              Y_com);
    }

    /**
     * @brief DEC 알고리즘 준비 단계
     *
     * 3단계로 구성:
     * 1. 차원 및 버퍼 초기화
     * 2. 각 행에 대해 Top-K 채널 선택 및 잔차 계산
     * 3. R_k CSC(Compressed Sparse Column) 구조 구축
     */
    void ActivationDEC::prepare()
    {
        uint64_t start, end;

        // ========== 1단계: 차원 및 버퍼 초기화 ==========
        start = read_cycles();

        // alpha: 각 행에서 선택할 top-K 채널 수 (K의 DEC_ALPHA_RATIO 비율)
        alpha_ = std::max<size_t>(1, std::min(K_, static_cast<size_t>(std::llround(K_ * DEC_ALPHA_RATIO))));

        // S_[r]: 각 행 r의 선택된 채널 인덱스들
        S_.assign(I_, {});

        // delta_[r]: 각 행 r의 선택된 채널별 잔차(residual) 값들
        delta_.assign(I_, {});

        // 임시 버퍼들
        scratch_idx_.resize(K_); // 인덱스 정렬용
        scratch_abs_.resize(K_); // 절댓값 캐싱용

        // rk_offs_: CSC 형식의 열 오프셋 배열 (크기 K+1)
        rk_offs_.assign(K_ + 1, 0);

        // rk_stage_: R_k 구축 전 임시 저장소
        rk_stage_.clear();
        rk_stage_.reserve(I_ * std::min(alpha_, K_));

        end = read_cycles();
        PRINT_CYCLE(layer_, "DEC: Initialize dimensions and buffers", start, end, end - start);

        // ========== 2단계: 모든 행에 대해 Top-K 선택 및 잔차 계산 ==========
        const float *x = static_cast<const float *>(A_->data); // 원본 활성화

        start = read_cycles();
        for (size_t r = 0; r < I_; ++r)
        {
            const float *x_r = x + r * K_;
            const int8_t *qx_r = qx_ + r * sA_;
            selectTopKandComputeResidual(r, x_r, qx_r);
        }
        end = read_cycles();
        PRINT_CYCLE(layer_, "DEC: Select top-K and stage R_k for all rows", start, end, end - start);

        // ========== 3단계: R_k CSC 구조 구축 ==========
        buildRk();
    }

    /**
     * @brief 특정 행에 대해 Top-K 채널 선택 및 잔차 계산
     *
     * 절댓값이 큰 상위 alpha개의 채널을 선택하고,
     * 각 채널의 양자화 오차(잔차)를 계산하여 저장
     *
     * @param r 행 인덱스
     * @param x_r 원본 활성화 값 배열 (크기 K)
     * @param qx_r 양자화된 활성화 값 배열 (크기 K)
     */
    void ActivationDEC::selectTopKandComputeResidual(size_t r,
                                                     const float *x_r,
                                                     const int8_t *qx_r)
    {
        // 인덱스 배열 초기화 및 절댓값 캐싱
        auto &idx = scratch_idx_;
        auto &ax = scratch_abs_;
        for (size_t k = 0; k < K_; ++k)
        {
            idx[k] = static_cast<int>(k);
            ax[k] = std::fabs(x_r[k]); // 절댓값을 미리 계산하여 캐싱
        }

        // Top-alpha 채널 선택 (partial_sort 사용)
        // alpha가 K와 같으면 모든 채널 선택하므로 정렬 생략
        const size_t topk = std::min(alpha_, K_);
        if (topk < K_)
        {
            // 절댓값이 큰 순서로 상위 topk개만 부분 정렬
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&ax](int a, int b)
                              { return ax[a] > ax[b]; });
        }

        // 선택된 채널에 대한 처리
        S_[r].resize(topk);
        delta_[r].resize(topk);
        for (size_t i = 0; i < topk; ++i)
        {
            const int k = idx[i]; // 선택된 채널 인덱스

            // 잔차 계산: δ[r,k] = x[r,k] - q̂x[r,k] * scale
            // 여기서 q̂x는 양자화된 값을 역양자화한 것
            const float d = x_r[k] - static_cast<float>(qx_r[k]) * scale_A_;

            // 채널 인덱스와 잔차 저장
            S_[r][i] = k;
            delta_[r][i] = d;

            // R_k 구축을 위한 스테이징: (채널 k, 행 r, 잔차 d)
            rk_stage_.push_back({k, static_cast<int>(r), d});

            // 각 채널 k의 출현 횟수 카운트 (나중에 prefix-sum으로 오프셋 변환)
            rk_offs_[k + 1]++;
        }
    }

    /**
     * @brief R_k를 CSC(Compressed Sparse Column) 형식으로 구축
     *
     * R_k[k]: 채널 k에 대해 non-zero 잔차를 가진 모든 (행 인덱스, 잔차) 쌍의 리스트
     * CSC 형식:
     *   - rk_offs_[k]: 채널 k의 데이터 시작 위치
     *   - rk_pairs_: 모든 (행, 잔차) 쌍을 연속 배열로 저장
     */
    void ActivationDEC::buildRk()
    {
        uint64_t start = read_cycles();

        // Prefix-sum으로 카운트를 오프셋으로 변환
        // rk_offs_[k]는 채널 k의 데이터가 시작하는 위치를 가리킴
        for (size_t k = 1; k <= K_; ++k)
            rk_offs_[k] += rk_offs_[k - 1];

        // 전체 non-zero 원소 개수
        const size_t nnz = rk_offs_[K_];
        rk_pairs_.assign(nnz, {0, 0.f});

        // 스테이징된 triplet (k, r, d)을 CSC 형식의 pairs (r, d)로 산포(scatter)
        std::vector<size_t> pos = rk_offs_; // 각 채널별 현재 삽입 위치
        for (const auto &t : rk_stage_)
        {
            const size_t dst = pos[t.k]++; // 채널 t.k의 다음 삽입 위치
            rk_pairs_[dst] = {t.r, t.d};   // (행, 잔차) 쌍 저장
        }

        // Non-empty 채널 인덱스 수집 (최적화: 빈 채널은 스킵)
        unique_k_.clear();
        unique_k_.reserve(K_);
        for (size_t k = 0; k < K_; ++k)
            if (rk_offs_[k] != rk_offs_[k + 1]) // 이 채널에 데이터가 있는 경우
                unique_k_.push_back(static_cast<int>(k));

        // 스테이징 메모리 해제
        rk_stage_.clear();
        rk_stage_.shrink_to_fit();

        uint64_t end = read_cycles();
        PRINT_CYCLE(layer_, "DEC: Build R_k CSC structure", start, end, end - start);
    }

    /**
     * @brief 보상 행렬 계산 (8-way 언롤링 최적화 버전)
     *
     * computeCompensation과 동일한 로직이지만,
     * 내부 루프를 8-way 언롤링하여 성능 향상
     *
     * 언롤링의 이점:
     *   - 루프 오버헤드 감소
     *   - 명령어 수준 병렬성(ILP) 향상
     *   - 파이프라인 효율 증가
     *
     * @param W 양자화된 가중치 행렬 (K×J)
     * @param Y_com 출력 보상 행렬 (I×J)
     */

    void ActivationDEC::computeCompensation_unrolled(float *Y_com)
    {
        uint64_t start = read_cycles();

        std::vector<float> Wk_f(J_);

        if (B_scales_)
        {
            GGML_ASSERT(blocks_K_ == (K_ / block_size_k_));
            GGML_ASSERT(blocks_J_ == J_ && "Non-2D weight: ensure j->logical_col mapping matches pack order");
        }

        // Non-zero 잔차를 가진 각 채널 k에 대해
        for (int k : unique_k_)
        {
            const size_t beg = rk_offs_[k];
            const size_t end = rk_offs_[k + 1];

            // int8 -> float 변환 + per-block scale 1회 적용
            load_W_row_scaled(k, Wk_f);

            // 각 (행, 잔차) 쌍에 대해 보상 누적
            for (size_t t = beg; t < end; ++t)
            {
                const int r = rk_pairs_[t].first;
                const float d = rk_pairs_[t].second;
                float *Yr = Y_com + static_cast<size_t>(r) * J_;

                // 8-way 언롤링: 한 번에 8개 원소 처리
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
                // 나머지 원소 처리 (J가 8의 배수가 아닌 경우)
                for (; j < J_; ++j)
                    Yr[j] += d * Wk_f[j];
            }
        }

        uint64_t end = read_cycles();
        PRINT_CYCLE(layer_, "DEC: Compute and accumulate compensation", start, end, end - start);
    }

    /**
     * @brief 계산된 보상을 출력 텐서에 적용
     *
     * C_out[r,j] += Y_com[r,j] * scale_w
     *
     * scale_w: 가중치의 양자화 스케일 팩터
     *
     * @param C_out 출력 텐서 (양자화된 행렬곱 결과)
     * @param Y_com 보상 행렬 (I×J)
     */

    void ActivationDEC::applyCompensation(float *out,
                                          size_t row_stride,
                                          size_t col_stride,
                                          const std::vector<float> &Y_com)
    {
        uint64_t start = read_cycles();

        float *C = out;

        // 각 원소에 보상 적용: C[r,j] += Y_com[r,j]
        for (size_t r = 0; r < I_; ++r)
        {
            const float *Yr = Y_com.data() + r * J_;
            float *Cr = C + r * row_stride;
            for (size_t j = 0; j < J_; ++j)
                Cr[j * col_stride] += Yr[j];
        }

        uint64_t end = read_cycles();
        PRINT_CYCLE(layer_, "DEC: Apply compensation to output", start, end, end - start);
    }

    // int8 -> float (W의 k번째 행) + per-block scale 적용 (있으면)
    inline void ActivationDEC::load_W_row_scaled(int k, std::vector<float> &Wk_f) const
    {
        if (B_is_KxJ_)
        {
            const int8_t *row = qW_ + (size_t)k * sB_; // 연속 J_
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = (float)row[j];
        }
        else
        {
            // 물리 JxK (열우선) 형태: 각 j에서 k를 stride로 접근
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] = (float)qW_[j * sB_ + k];
        }
        if (B_scales_)
        {
            const size_t blk = (size_t)k / block_size_k_;
            const size_t base = blk * blocks_J_;
            for (size_t j = 0; j < J_; ++j)
                Wk_f[j] *= B_scales_[base + j];
        }
    }
}
