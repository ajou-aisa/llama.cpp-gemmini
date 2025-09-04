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
#ifndef TEST_DEQUANTiZE
#define TEST_DEQUANTIZE 0
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
    if (!m) { DBG0("%s: <null>\n", name); return; }
    DBG0("%s (r=%d, c=%d, ld=%d) =\n", name, r, c, s);

    for (int i = 0; i < r; ++i) {
        DBG0("[ ");
        for (int j = 0; j < c; ++j) {
            const T v = m[(size_t)i * (size_t)s + (size_t)j];

            if constexpr (std::is_same_v<T, float>) {
                // fp32
                DBG0("%.6g ", (double)v);
            } else if constexpr (std::is_same_v<T, int8_t>) {
                // int8 -> 가독성을 위해 int로 승격 출력
                DBG0("%d ", (int)v);
            } else if constexpr (std::is_same_v<T, acc_t>) {
                // acc_t (일반적으로 int32_t)
                if constexpr (std::numeric_limits<acc_t>::is_signed)
                    DBG0("%lld ", (long long)v);
                else
                    DBG0("%llu ", (unsigned long long)v);
            } else {
                // 컴파일 타임 가드: 지원 타입 외 사용 방지
                static_assert(std::is_same_v<T, void>,
                              "dump_matrix: supported types are float, int8_t, acc_t only.");
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

// dst: (I x J) => ggml의 관례상 ne[0]=J, ne[1]=I
// src0(weight): (K x J) layout (ne[0]=K, ne[1]=J) — 실행 시 B는 transpose하여 KxJ로 multiply
// src1(act): (I x K) layout (ne[0]=K, ne[1]=I) — 실행 시 A는 IxK
void extract_and_check_shapes(const ggml_tensor *dst, int &I, int &J, int &K);

void log_shapes(const ggml_tensor *dst,
                const ggml_tensor *src0,
                const ggml_tensor *src1,
                int I, int J, int K);

// ====== CPU 참조 계산 & 검증 ======
void cpu_reference_C(const elem_t *A, size_t sA,
                     const elem_t *B, size_t sB,
                     const acc_t *D, size_t sD,
                     elem_t *C_exp, size_t sC,
                     int I, int J, int K);

bool compare_C_and_report(const elem_t *C, size_t sC,
                          const elem_t *C_exp, size_t sE,
                          int I, int J);

// TEST_SLICE: 원본 ggml 텐서와 변환 버퍼(tA,tB,tC)를 SLICE_* 규칙으로 일부만 덤프
void test_dump_slices(ggml_context* tmp_ctx,
                      const ggml_tensor* src1, const ggml_tensor* src0, ggml_tensor* dst,
                      int I, int J, int K,
                      const elem_t* A_i8, size_t sA,
                      const elem_t* B_i8, size_t sB,
                      const elem_t* C_i8, size_t sC);

void test_dequantize_output(const elem_t *C_i8, size_t sC,
                                int I, int J, ggml_tensor *dst);