// ggml-gemmini-test.cpp
#include "ggml-gemmini-test.h"
#include <algorithm>
#include <memory>

using zerogod::ggml_gemmini_tensor;

// ===== 최상위 테스트 엔트리 =====
void ggml_gemmini_test(ggml_backend_gemmini_context *ctx,
                       struct ggml_tensor *dst,  // FP32 output (I×J)
                       struct ggml_tensor *bias) // optional FP32 bias (->int32)
{
    GGML_ASSERT(ctx && dst);
    const ggml_tensor *src0 = dst->src[0]; // weight (J×K)
    const ggml_tensor *src1 = dst->src[1]; // act    (I×K)
    GGML_ASSERT(src0 && src1);

    // 0) Shape 검증/확정(논리 I,J,K)
    const mm_shape shp = extract_and_check_shapes(dst);
    const int I_log = shp.I;
    const int J_log = shp.J;
    const int K_log = shp.K;

#if TEST_SHAPE // 원본 ggml 차원 검증 (출력)
    DBG0("[TEST_SHAPE] I=%d, J=%d, K=%d\n", I_log, J_log, K_log);
    DBG0(" dst ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)dst->ne[0], (unsigned long long)dst->ne[1],
         (unsigned long long)dst->nb[0], (unsigned long long)dst->nb[1]);
    DBG0(" src0(W) ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)src0->ne[0], (unsigned long long)src0->ne[1],
         (unsigned long long)src0->nb[0], (unsigned long long)src0->nb[1]);
    DBG0(" src1(A) ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)src1->ne[0], (unsigned long long)src1->ne[1],
         (unsigned long long)src1->nb[0], (unsigned long long)src1->nb[1]);
#endif

    /* 1) Gemmini용 캐스팅/배치 (ggml-gemmini-tensor.cpp 사용)
        - A : I×K (transpose=false)
        - B : TRANSPOSE_B
              true -> 물리 전치하여 K×J 배치(tB transpose=true)
              false -> 전치 없이 J×K 배치(tB transpose=false)
        - C : I×J (acc=true로 zero-fill)
        이후 경로에서 항상 tA = (IxK), tB = (KxJ)*/

    ggml_gemmini_tensor<int8_t> tA(ctx->tmp_ctx, src1, ".i8", /*acc=*/false, /*transpose=*/false);
    ggml_gemmini_tensor<int8_t> tB(ctx->tmp_ctx, src0, ".i8", /*acc=*/false, /*transpose=*/TRANSPOSE_B);
    ggml_gemmini_tensor<int8_t> tC(ctx->tmp_ctx, dst, ".i8", /*acc=*/true);

    // 2) 런타임 차원/stride (패딩 포함)
    GGML_ASSERT(tA.get_rows() == (size_t)I_log);
    GGML_ASSERT(tA.get_cols() == tB.get_rows());
    GGML_ASSERT(tC.get_rows() == tA.get_rows());
    GGML_ASSERT(tC.get_cols() == tB.get_cols());
    GGML_ASSERT(dst->ne[0] <= tC.get_cols());

    // dimension
    const size_t I = tC.get_rows(); // = I_log
    const size_t J = tC.get_cols(); // 패딩 포함 J
    const size_t K = tA.get_cols();

    // stride
    const size_t sA = tA.get_stride();
    const size_t sB = tB.get_stride();
    const size_t sC = tC.get_stride();

    GGML_ASSERT(sA % 16 == 0 && sB % 16 == 0 && sC % 16 == 0);

#if TEST_SLICE // 원본 텐서와 변환 텐서를 일정 크기로 잘라내어 출력 - shape, 구조 확인
    const size_t b_rows = tB.get_rows();
    const size_t b_cols = tB.get_cols();
    // a) 원본 텐서 관찰용 view (src1/src0/dst)
    DBG0("[SLICE] I=%zu J(p)=%zu K=%zu | tB(KxJ)=(%zu,%zu)\n", I, J, K, b_rows, b_cols);

    {
        std::vector<acc_t> zero_bias_view;
        auto views = make_and_dump_mm_views(ctx->tmp_ctx, src1, src0, dst, bias, I_log, J_log, K_log, zero_bias_view);
        (void)views;
    }
    // b) 변환된 내부 버퍼 일부 dump
    {
        // A(IxK): I는 1로 제한, K는 SLICE_K까지
        const int rA = std::min<int>((int)I, SLICE_I);
        const int cA = std::min<int>((int)K, SLICE_K);

        // B(KxJ): K는 SLICE_K까지, J는 논리 J와 SLICE_J 중 작은 값 (패딩은 출력 안 함)
        const int rB = std::min<int>((int)b_rows, SLICE_K);                        // rows=K
        const int cB = std::min<int>((int)std::min<size_t>(b_cols, (size_t)J_log), // cols=J (no padding)
                                     SLICE_J);

        // C(IxJ): I는 1로 제한, J는 논리 J와 SLICE_J 중 작은 값
        const int rC = std::min<int>((int)I, SLICE_I);
        const int cC = std::min<int>(J_log, SLICE_J);

        dump_matrix<elem_t>("tA (IxK)", (const elem_t *)tA.get(), rA, cA, (int)sA);
        dump_matrix<elem_t>("tB (KxJ)", (const elem_t *)tB.get(), rB, cB, (int)sB);
        dump_matrix<elem_t>("tC (IxJ)", (const elem_t *)tC.get(), rC, cC, (int)sC);
    }
#endif

    // 3) Bias 준비 (int32). 없으면 1×J(패딩) zero-bias를 repeating으로 사용
    // 현재 확인된 구조는 bias는 항상 없음
    const int32_t *bias_data = nullptr;
    size_t sD = 0;
    bool repeating = true;
    std::vector<int32_t> zero_bias;
    std::unique_ptr<ggml_gemmini_tensor<int32_t>> tD;

    if (bias)
    {
        tD = std::make_unique<ggml_gemmini_tensor<int32_t>>(ctx->tmp_ctx, bias, ".i32", /*acc=*/false, /*transpose=*/false);
        bias_data = static_cast<const int32_t *>(tD->get());
        sD = tD->get_stride();
        repeating = (tD->get_rows() == 1);
    }
    else
    {
        zero_bias.assign(tC.get_cols(), 0); // 1행(패딩 포함 J)만 있으면 됨
        bias_data = zero_bias.data();
        sD = 0; // 1행 반복 시 stride 불필요
        repeating = true;
    }

    // 4) (선택) CPU 레퍼런스
#if TEST_CPU_REF || TEST_COMPARE
    // CPU 참조는 논리 J까지만 계산/검증
    const size_t sC_exp = sC; // 동일 stride 사용
    std::vector<elem_t> C_exp(I * sC_exp, (elem_t)0);
    cpu_reference_C((const elem_t *)tA.get(), sA,
                    (const elem_t *)tB.get(), sB,
                    (const acc_t *)bias_data, sD,
                    C_exp.data(), sC_exp,
                    (int)I, J_log, K_log);
#endif

    // 5) (선택) Gemmini 실행
#if TEST_GEMMINI
    tiled_matmul_auto(I, J, K,
                      (elem_t *)tA.get(),
                      (elem_t *)tB.get(),
                      (void *)bias_data,
                      (elem_t *)tC.get(),
                      sA, sB, sD, sC,
                      1.f, 1.f, 1.f,
                      NO_ACTIVATION,
                      1, 1,
                      repeating,
                      false, // transpose_A
                      false, // transpose_B
                      false, false,
                      0, OPTION);
#endif

    // 6) (선택) 결과 비교
#if TEST_COMPARE
    (void)compare_C_and_report((const elem_t *)tC.get(), sC,
                               (const elem_t *)C_exp.data(), sC_exp,
                               (int)I, J_log);
#endif

    // 7) (선택) dst(F32)로 반영 — 논리 J만 복사
#if TEST_WRITEBACK
    test_writeback_f32_from_i8((const elem_t *)tC.get(), sC,
                               (int)I, J_log, dst);
#endif

    // (미사용 경고 방지)
    (void)I;
    (void)J;
    (void)I_log;
    (void)J_log;
    (void)K;
    (void)sA;
    (void)sB;
    (void)sC;
    (void)sD;
    (void)repeating;
}

// ===== Shape 추출/검사 =====
mm_shape extract_and_check_shapes(const ggml_tensor *dst)
{
    const ggml_tensor *src0 = dst->src[0]; // weight (J×K) layout: ne0=K, ne1=J
    const ggml_tensor *src1 = dst->src[1]; // act    (I×K) layout: ne0=K, ne1=I
    GGML_ASSERT(src0 && src1);

    const int J = (int)dst->ne[0];
    const int I = (int)dst->ne[1];

    const int J0 = (int)src0->ne[1];
    const int K0 = (int)src0->ne[0];

    const int K1 = (int)src1->ne[0];
    const int I1 = (int)src1->ne[1];

    GGML_ASSERT(J0 == J);
    GGML_ASSERT(I1 == I);
    GGML_ASSERT(K0 == K1);

    return mm_shape{I, J, K1};
}

// ===== CPU 참조/검증 =====
void cpu_reference_C(const elem_t *A, size_t sA,
                     const elem_t *B, size_t sB,
                     const acc_t *D, size_t sD,
                     elem_t *C_exp, size_t sC,
                     int I, int J, int K)
{
    for (int i = 0; i < I; ++i)
    {
        for (int j = 0; j < J; ++j)
        {
            int acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int)A[i * sA + k] * (int)B[k * sB + j];
            acc += (sD == 0) ? (int)D[j] : (int)D[i * (int)sD + j];
            C_exp[i * sC + j] = (elem_t)sat_i8(acc);
        }
    }
}

bool compare_C_and_report(const elem_t *C, size_t sC,
                          const elem_t *C_exp, size_t sE,
                          int I, int J)
{
    bool ok = true;
    for (int i = 0; i < I; ++i)
    {
        for (int j = 0; j < J; ++j)
        {
            elem_t got = C[i * sC + j];
            elem_t exp = C_exp[i * sE + j];
            if (got != exp)
            {
                DBG0("[NG] mismatch (%d,%d): got=%d exp=%d\n", i, j, (int)got, (int)exp);
                ok = false;
            }
        }
    }
    DBG0(ok ? "[OK] Gemmini matmul(+bias) matches expected\n"
            : "[FAIL] mismatch detected\n");
    return ok;
}

// ===== ggml 텐서 덤프 (nb 기반 직접 접근) =====
void dump_tensor_auto_2d(const char *name, const ggml_tensor *t,
                         int max_rows, int max_cols)
{
#if DUMP
    if (!t)
        return;
    const int rows = (int)t->ne[1];
    const int cols = (int)t->ne[0];
    const int rlim = (max_rows < 0 ? rows : std::min(rows, max_rows));
    const int clim = (max_cols < 0 ? cols : std::min(cols, max_cols));
    const char *base = (const char *)t->data;
    const size_t nb0 = t->nb[0];
    const size_t nb1 = t->nb[1];

    DBG0("%s (type=%s) rows=%d cols=%d row_stride(bytes)=%zu\n",
         name, ggml_type_name(t->type), rows, cols, nb1);

    if (t->type == GGML_TYPE_I8)
    {
        for (int i = 0; i < rlim; ++i)
        {
            DBG0("[ ");
            for (int j = 0; j < clim; ++j)
            {
                const int8_t *p = (const int8_t *)(base + (size_t)i * nb1 + (size_t)j * nb0);
                DBG0("%d ", (int)*p);
            }
            if (clim < cols)
                DBG0("...");
            DBG0("]\n");
        }
    }
    else if (t->type == GGML_TYPE_F32)
    {
        for (int i = 0; i < rlim; ++i)
        {
            DBG0("[ ");
            for (int j = 0; j < clim; ++j)
            {
                const float *p = (const float *)(base + (size_t)i * nb1 + (size_t)j * nb0);
                DBG0("%g ", (double)*p);
            }
            if (clim < cols)
                DBG0("...");
            DBG0("]\n");
        }
    }
    else if (t->type == GGML_TYPE_I32)
    {
        for (int i = 0; i < rlim; ++i)
        {
            DBG0("[ ");
            for (int j = 0; j < clim; ++j)
            {
                const int32_t *p = (const int32_t *)(base + (size_t)i * nb1 + (size_t)j * nb0);
                DBG0("%d ", (int)*p);
            }
            if (clim < cols)
                DBG0("...");
            DBG0("]\n");
        }
    }
    else
    {
        DBG0("%s: dump not implemented for type=%d\n", name, (int)t->type);
    }
    if (rlim < rows)
        DBG0("...\n");
#else
    (void)name;
    (void)t;
    (void)max_rows;
    (void)max_cols;
#endif
}

// ===== slicing & 관찰 =====
sliced_views make_and_dump_mm_views(ggml_context *ctx,
                                    const ggml_tensor *A_in,
                                    const ggml_tensor *B_in,
                                    ggml_tensor *C_out,
                                    const ggml_tensor *D_bias,
                                    int I, int J, int K,
                                    std::vector<acc_t> &zero_bias_out)
{
    GGML_ASSERT(A_in && A_in->data && "A_in is invalid");
    GGML_ASSERT(B_in && B_in->data && "B_in is invalid");
    GGML_ASSERT(C_out && C_out->data && "C_out is invalid");

    // SLICE 한도 적용
    const int vI = std::min(I, SLICE_I); // 1
    const int vK = std::min(K, SLICE_K);
    const int vJ = std::min(J, SLICE_J);

    // A_sliced: (ne0=K, ne1=I) -> rows=vI, cols=vK
    ggml_tensor *A_sliced = ggml_view_2d(ctx,
                                         const_cast<ggml_tensor *>(A_in),
                                         /*ne0=*/vK, /*ne1=*/vI,
                                         /*nb1=*/A_in->nb[1], /*offset=*/0);

    // B_sliced: (ne0=J, ne1=K) -> rows=vK, cols=vJ  (W^T 관찰용)
    ggml_tensor *B_sliced = ggml_view_2d(ctx,
                                         const_cast<ggml_tensor *>(B_in),
                                         /*ne0=*/vJ, /*ne1=*/vK,
                                         /*nb1=*/B_in->nb[1], /*offset=*/0);

    // C_sliced: (ne0=J, ne1=I) -> rows=vI, cols=vJ
    ggml_tensor *C_sliced = ggml_view_2d(ctx,
                                         C_out,
                                         /*ne0=*/vJ, /*ne1=*/vI,
                                         /*nb1=*/C_out->nb[1], /*offset=*/0);

    ggml_tensor *D_sliced = nullptr;
    const void *d_data_ptr = nullptr;
    size_t sD = 0;

    if (D_bias)
    {
        GGML_ASSERT(D_bias->data && "D_bias->data is invalid");
        D_sliced = ggml_view_2d(
            ctx, const_cast<ggml_tensor *>(D_bias),
            /*ne0=*/vJ, /*ne1=*/vI,
            /*nb1=*/D_bias->nb[1], /*offset=*/0);
        d_data_ptr = D_sliced->data;
        sD = (size_t)(D_sliced->nb[1] / sizeof(acc_t));
    }
    else
    {
        // zero-bias도 slice 크기에 맞춰 최소만 준비
        const size_t sD_elem = align_up((size_t)vJ, (size_t)(GEMMINI_ALIGN / sizeof(acc_t)));
        zero_bias_out.assign((size_t)vI * sD_elem, 0);
        d_data_ptr = zero_bias_out.data();
        sD = sD_elem;
    }

#if DUMP
    // view 자체가 잘린 크기이므로 전체 출력(-1,-1)로 충분
    dump_tensor_auto_2d("A_sliced (I x K view of act)", A_sliced, -1, -1);
    dump_tensor_auto_2d("B_sliced (K x J view of W^T)", B_sliced, -1, -1);
    dump_tensor_auto_2d("C_sliced (I x J view of dst)", C_sliced, -1, -1);
    if (D_bias)
        dump_tensor_auto_2d("D_sliced (I x J bias)", D_sliced, -1, -1);
#endif

    return sliced_views{A_sliced, B_sliced, C_sliced, D_sliced, d_data_ptr, sD};
}

void test_writeback_f32_from_i8(const elem_t *C_i8, size_t sC,
                                int I, int J, ggml_tensor *dst)
{
    if (dst->type != GGML_TYPE_F32)
    {
        DBG0("[TEST_WRITEBACK] skip: dst type is not F32 (type=%d)\n", (int)dst->type);
        return;
    }
    const size_t nb1 = dst->nb[1];
    uint8_t *out_base = static_cast<uint8_t *>(dst->data);
    for (int r = 0; r < I; ++r)
    {
        const elem_t *row_c = C_i8 + (size_t)r * sC;
        float *row_out = reinterpret_cast<float *>(out_base + (size_t)r * nb1);
        for (int j = 0; j < J; ++j)
            row_out[j] = (float)row_c[j];
    }
}
