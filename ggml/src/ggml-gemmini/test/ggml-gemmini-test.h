// test/ggml-gemmini-test.h
#pragma once

#ifndef OPTION
#define OPTION CPU
#endif

// 1: src0(JxK) -> 변환 시 물리 전치(KxJ)로 배치(현재 기본)
// 0: src0(JxK) -> 전치 없이 JxK로 배치
#ifndef TRANSPOSE_B
#define TRANSPOSE_B 1  
#endif
#ifndef TEST_SHAPE
#define TEST_SHAPE 0
#endif
#ifndef TEST_SLICE
#define TEST_SLICE 0
#endif
#ifndef TEST_CPU_REF
#define TEST_CPU_REF 0
#endif
#ifndef TEST_GEMMINI
#define TEST_GEMMINI 0
#endif
#ifndef TEST_COMPARE
#define TEST_COMPARE 0
#endif
#ifndef TEST_WRITEBACK
#define TEST_WRITEBACK 0
#endif
#ifndef DUMP
#define DUMP (TEST_SLICE || TEST_CPU_REF || TEST_COMPARE)
#endif

// slice 크기
#ifndef SLICE_I
#define SLICE_I 1      // A/C는 행 1만 출력
#endif
#ifndef SLICE_K
#define SLICE_K 4     // K 방향 최대 출력 열
#endif
#ifndef SLICE_J
#define SLICE_J 6     // J 방향 최대 출력 열(논리 J 기준)
#endif

#include "include/gemmini.h"
#include "../ggml-gemmini-util.h"
#include "../ggml-gemmini-tensor.h"
#include <cstdio>
#include <type_traits>
#include <vector>
#include <cmath>
#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

using namespace zerogod;

// ====== 최상위 테스트 엔트리 (ggml 백엔드 경로에서 호출) ======
void ggml_gemmini_test(ggml_backend_gemmini_context *ctx,
                       struct ggml_tensor *dst,   // FP32 output (I×J)
                       struct ggml_tensor *bias); // optional FP32 bias (->int32)

// ====== 유틸 ======
static inline int8_t sat_i8(int x)
{
    return x > 127 ? 127 : (x < -128 ? -128 : (int8_t)x);
}

template <typename T>
static inline void dump_matrix(const char *name, const T *m, int r, int c, int s)
{
#if DUMP
    DBG0("%s =\n", name);
    for (int i = 0; i < r; ++i)
    {
        DBG0("[ ");
        for (int j = 0; j < c; ++j)
        {
            const T v = m[i * s + j];
            if constexpr (std::is_integral_v<T>)
            {
                if constexpr (sizeof(T) <= sizeof(int))
                    DBG0("%d ", (int)v);
                else
                    DBG0("%lld ", (long long)v);
            }
            else
            {
                DBG0("%g ", (double)v);
            }
        }
        DBG0("]\n");
    }
#else
    (void)name;
    (void)m;
    (void)r;
    (void)c;
    (void)s;
#endif
}

// ====== Shape 추출 & 검사 ======
struct mm_shape
{
    int I; // rows of C (and A)
    int J; // cols of C (and B)
    int K; // inner dim
};

// dst: (I x J) => ggml의 관례상 ne[0]=J, ne[1]=I
// src0(weight): (J x K) layout (ne[0]=J, ne[1]=K) — 실행 시 B는 KxJ로 multiply
// src1(act): (I x K) layout (ne[0]=K, ne[1]=I) — 실행 시 A는 IxK
mm_shape extract_and_check_shapes(const ggml_tensor *dst);

// ====== CPU 참조 계산 & 검증 ======
void cpu_reference_C(const elem_t *A, size_t sA,
                     const elem_t *B, size_t sB,
                     const acc_t *D, size_t sD,
                     elem_t *C_exp, size_t sC,
                     int I, int J, int K);

bool compare_C_and_report(const elem_t *C, size_t sC,
                          const elem_t *C_exp, size_t sE,
                          int I, int J);

// ====== ggml 텐서 slicing & 관찰 ======
// ggml 2D 텐서를 (nb1/sizeof(T))를 stride로 보아 일부만 덤프
void dump_tensor_auto_2d(const char *name, const ggml_tensor *t,
                         int max_rows = -1, int max_cols = -1);

// A_in: act(K×I), B_in: weight(J×K), C_out: dst(J×I), D_bias: bias(J×I)
// I,J,K를 기준으로 view_2d를 만들고, bias 없을 경우 0으로 채운 가상 버퍼를 준비.
// 반환: d_data_ptr(누산 버퍼 포인터), sD(요소 단위 stride), zero_bias(생성 시 소유).
struct sliced_views
{
    ggml_tensor *A_sliced; // (ne0=K, ne1=I) -> 관찰 시 rows=I, cols=K, stride=nb1/sizeof(T)
    ggml_tensor *B_sliced; // (ne0=J, ne1=K) -> 관찰 시 rows=K, cols=J, stride=nb1/sizeof(T)
    ggml_tensor *C_sliced; // (ne0=J, ne1=I) -> 관찰 시 rows=I, cols=J, stride=nb1/sizeof(T)
    ggml_tensor *D_sliced; // nullable, same layout as C

    const void *d_data_ptr; // acc_t* (bias or zero buffer)
    size_t sD;              // stride in elements for D (nb1/sizeof(acc_t))
};

sliced_views make_and_dump_mm_views(ggml_context *ctx,
                                    const ggml_tensor *A_in,
                                    const ggml_tensor *B_in,
                                    ggml_tensor *C_out,
                                    const ggml_tensor *D_bias,
                                    int I, int J, int K,
                                    std::vector<acc_t> &zero_bias_out);

void test_writeback_f32_from_i8(const elem_t *C_i8, size_t sC,
                                int I, int J, ggml_tensor *dst);