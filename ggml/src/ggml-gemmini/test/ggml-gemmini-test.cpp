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
    int I_log = 0, J_log = 0, K_log = 0;
    extract_and_check_shapes(dst, I_log, J_log, K_log);

#if TEST_SHAPE
    log_shapes(dst, src0, src1, I_log, J_log, K_log);
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
    test_dump_slices(ctx->tmp_ctx,
                     src1, src0, dst,
                     I_log, J_log, K_log,
                     (const elem_t *)tA.get(), sA,
                     (const elem_t *)tB.get(), sB,
                     (const elem_t *)tC.get(), sC);
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
#if TEST_CPU_REF
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
#if TEST_CPU_REF && TEST_GEMMINI && TEST_COMPARE
    (void)compare_C_and_report((const elem_t *)tC.get(), sC,
                               (const elem_t *)C_exp.data(), sC_exp,
                               (int)I, J_log);
#endif

    // 7) (선택) dst(F32)로 반영 — 논리 J만 복사
#if TEST_GEMMINI && TEST_DEQUANTIZE
    test_dequantize_output((const elem_t *)tC.get(), sC,
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
void extract_and_check_shapes(const ggml_tensor *dst, int &I, int &J, int &K)
{
    DBG0("[extract_and_check_shapes] called: \n");
    const ggml_tensor *src0 = dst->src[0]; // weight stored KxJ (W^T): ne0=K, ne1=J
    const ggml_tensor *src1 = dst->src[1]; // act    stored KxI      : ne0=K, ne1=I
    GGML_ASSERT(src0 && src1);

    J = (int)dst->ne[0];
    I = (int)dst->ne[1];

    const int K0 = (int)src0->ne[0];
    const int J0 = (int)src0->ne[1];
    const int K1 = (int)src1->ne[0];
    const int I1 = (int)src1->ne[1];

    GGML_ASSERT(J0 == J);
    GGML_ASSERT(I1 == I);
    GGML_ASSERT(K0 == K1);

    K = K1;
}

void log_shapes(const ggml_tensor *dst,
                const ggml_tensor *src0,
                const ggml_tensor *src1,
                int I, int J, int K)
{
    DBG0("[log_shapes] called\n");

    DBG0("[TEST_SHAPE] I=%d, J=%d, K=%d\n", I, J, K);
    DBG0(" dst  (logical I x J) stored ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)dst->ne[0], (unsigned long long)dst->ne[1],
         (unsigned long long)dst->nb[0], (unsigned long long)dst->nb[1]);
    DBG0(" src0 (W^T stored K x J, logical W J x K) ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)src0->ne[0], (unsigned long long)src0->ne[1],
         (unsigned long long)src0->nb[0], (unsigned long long)src0->nb[1]);
    DBG0(" src1 (A stored K x I,   logical A I x K) ne=[%llu,%llu], nb=[%llu,%llu]\n",
         (unsigned long long)src1->ne[0], (unsigned long long)src1->ne[1],
         (unsigned long long)src1->nb[0], (unsigned long long)src1->nb[1]);
}

// int * int 조합 확인
void check_types_equal(const ggml_tensor *dst)
{
    DBG0("[check_types_equal] called\n");

    if (!dst || !dst->src[0] || !dst->src[1])
    {
        DBG0("dst/src0/src1 is NULL\n");
        return;
    }

    const struct ggml_tensor *s0 = dst->src[0];
    const struct ggml_tensor *s1 = dst->src[1];

    const char *name_dst = (dst->name ? dst->name : "(dst)");
    const char *name_s0 = (s0->name ? s0->name : "(src0)");
    const char *name_s1 = (s1->name ? s1->name : "(src1)");

    const char *tn_dst = ggml_type_name(dst->type);
    const char *tn_s0 = ggml_type_name(s0->type);
    const char *tn_s1 = ggml_type_name(s1->type);

    DBG0("dst  %s : %s\n", name_dst, tn_dst);
    DBG0("src0 %s : %s\n", name_s0, tn_s0);
    DBG0("src1 %s : %s\n", name_s1, tn_s1);

    if (s0->type == s1->type)
        DBG0("TYPE OK: %s and %s are the same type (%s)\n", name_s0, name_s1, tn_s0);
    else
        DBG0("TYPE MISMATCH: %s (%s) vs %s (%s)\n", name_s0, tn_s0, name_s1, tn_s1);
}

// ===== CPU 참조/검증 =====
void cpu_reference_C(const elem_t *A, size_t sA,
                     const elem_t *B, size_t sB,
                     const acc_t *D, size_t sD,
                     elem_t *C_exp, size_t sC,
                     int I, int J, int K)
{
    DBG0("[cpu_reference_C] called\n");

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
    DBG0("[compare_C_and_report] called\n");
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

// test_dump_slices: dump_matrix만 사용 (원본 텐서는 타입 분기)
void test_dump_slices(ggml_context *tmp_ctx,
                      const ggml_tensor *src1, // act: stored (ne0=K, ne1=I)
                      const ggml_tensor *src0, // W^T: stored (ne0=K, ne1=J)
                      ggml_tensor *dst,        // dst: stored (ne0=J, ne1=I) (F32)
                      int I, int J, int K,
                      const elem_t *A_i8, size_t sA, // tA (I x K), stride elems
                      const elem_t *B_i8, size_t sB, // tB (K x J), stride elems
                      const elem_t *C_i8, size_t sC) // tC (I x J), stride elems
{
    DBG0("[test_dump_slices] called\n");
#if DUMP
    // SLICE 한도 (논리 크기 기준 clamp)
    const int vI = std::min(I, SLICE_I); // 보통 1
    const int vK = std::min(K, SLICE_K);
    const int vJ = std::min(J, SLICE_J);

    // ===== 원본 ggml 텐서 slice view 만들기 =====
    ggml_tensor *A_slice = ggml_view_2d(tmp_ctx,
                                        const_cast<ggml_tensor *>(src1),
                                        /*ne0=*/vK, /*ne1=*/vI,
                                        /*nb1=*/src1->nb[1], /*offset=*/0);

    ggml_tensor *B_slice = ggml_view_2d(tmp_ctx,
                                        const_cast<ggml_tensor *>(src0),
                                        /*ne0=*/vJ, /*ne1=*/vK,
                                        /*nb1=*/src0->nb[1], /*offset=*/0);

    ggml_tensor *C_slice = ggml_view_2d(tmp_ctx,
                                        dst,
                                        /*ne0=*/vJ, /*ne1=*/vI,
                                        /*nb1=*/dst->nb[1], /*offset=*/0);

    DBG0("[SLICE] vI=%d vK=%d vJ=%d | I=%d K=%d J=%d\n", vI, vK, vJ, I, K, J);
    DBG0("  A_slice: type=%s nb0=%zu nb1=%zu\n", ggml_type_name(A_slice->type),
         (size_t)A_slice->nb[0], (size_t)A_slice->nb[1]);
    DBG0("  B_slice: type=%s nb0=%zu nb1=%zu\n", ggml_type_name(B_slice->type),
         (size_t)B_slice->nb[0], (size_t)B_slice->nb[1]);
    DBG0("  C_slice: type=%s nb0=%zu nb1=%zu\n", ggml_type_name(C_slice->type),
         (size_t)C_slice->nb[0], (size_t)C_slice->nb[1]);

    // stride
    const int sA_view = (int)(A_slice->nb[1] / sizeof(elem_t));
    const int sB_view = (int)(B_slice->nb[1] / sizeof(elem_t));
    const int sC_view = (int)(C_slice->nb[1] / sizeof(float));

    // ===== 타입 분기 덤프 유틸
    auto dump_any = [](const char *tag, const ggml_tensor *t, int r, int c, int s)
    {
        switch (t->type)
        {
        case GGML_TYPE_I8:
            GGML_ASSERT(t->nb[0] == sizeof(int8_t));
            dump_matrix<int8_t>(tag, (const int8_t *)t->data, r, c, s);
            break;
        case GGML_TYPE_F32:
            GGML_ASSERT(t->nb[0] == sizeof(float));
            dump_matrix<float>(tag, (const float *)t->data, r, c, s);
            break;
        case GGML_TYPE_I32: // acc_t 경로(필요 시)
            GGML_ASSERT(t->nb[0] == sizeof(acc_t));
            dump_matrix<acc_t>(tag, (const acc_t *)t->data, r, c, s);
            break;
        default:
            DBG0("%s: unsupported ggml type=%d (nb0=%zu, nb1=%zu) — skipped\n",
                 tag, (int)t->type, (size_t)t->nb[0], (size_t)t->nb[1]);
            break;
        }
    };

    // ===== 원본 ggml slice 덤프 =====
    dump_any("A_slice (I x K, from src1)", A_slice, vI, vK, sA_view);
    dump_any("B_slice (K x J, from src0)", B_slice, vK, vJ, sB_view);
    dump_any("C_slice (I x J, from dst )", C_slice, vI, vJ, sC_view);

    // ===== 변환된 내부 버퍼(tA/tB/tC) 일부 덤프 — 항상 I8 =====
    dump_matrix<elem_t>("tA (IxK)", A_i8, vI, vK, (int)sA);
    dump_matrix<elem_t>("tB (KxJ)", B_i8, vK, vJ, (int)sB);
    dump_matrix<elem_t>("tC (IxJ)", C_i8, vI, vJ, (int)sC);
#else
    (void)tmp_ctx;
    (void)src1;
    (void)src0;
    (void)dst;
    (void)I;
    (void)J;
    (void)K;
    (void)A_i8;
    (void)sA;
    (void)B_i8;
    (void)sB;
    (void)C_i8;
    (void)sC;
#endif
}

void test_dequantize_output(const elem_t *C_i8, size_t sC,
                            int I, int J, ggml_tensor *dst)
{
    DBG0("[test_dequantize_output] called\n");
    if (dst->type != GGML_TYPE_F32)
    {
        DBG0("[TEST_DEQUANTIZE] skip: dst type is not F32 (type=%d)\n", (int)dst->type);
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
