// test/ggml-gemmini-test.h
#ifndef __GGML_GEMMINI_TEST_H__
#define __GGML_GEMMINI_TEST_H__

#include "include/gemmini.h"
#include <cstdio>
#include <type_traits>

template <typename T>
static inline void dump_matrix(const char* name, const T* m, int r, int c, int s) {
    printf("%s =\n", name);
    for (int i = 0; i < r; ++i)
    {
        printf("[ ");
        for (int j = 0; j < c; ++j)
        {
            const T v = m[i * s + j];
            if constexpr (std::is_integral_v<T>)
            {
                if constexpr (sizeof(T) <= sizeof(int))
                    printf("%d ", (int)v);
                else
                    printf("%lld ", (long long)v);
            }
            else
            {
                printf("%g ", (double)v);
            }
        }
        printf("]\n");
    }
};

static inline int8_t sat_i8(int x) {
    return x > 127 ? 127 : (x < -128 ? -128 : (int8_t)x);
}

static void ggml_backend_gemmini_mul_mat_test() {
    DBG("[Gemmini] mul_mat test called");

    constexpr int I = 2, J = 2, K = 2;
    constexpr size_t sA = K, sB = J, sD = J, sC = J;

    alignas(16) elem_t A[I * sA] = {}; // IxK
    alignas(16) elem_t B[K * sB] = {}; // KxJ
    alignas(16) elem_t C[I * sC] = {}; // IxJ

    alignas(64) elem_t C_expected[I * sC] = {}; // 기대값
    alignas(16) acc_t D[I * sC] = {1, 2, 3, 4}; // IxJ, bias

    int e = 1;
    auto init_matrix = [&e](elem_t *mat, int row, int stride)
    {
        for (int i = 0; i < row; i++)
            for (int j = 0; j < stride; j++)
                mat[i * stride + j] = e++;
    };

    init_matrix(A, I, sA); // [[1, 2], [3, 4]]
    init_matrix(B, K, sB); // KxJ, [[5, 6], [7, 8]]

    // expected
    for (int i = 0; i < I; ++i)
    {
        for (int j = 0; j < J; ++j)
        {
            int acc = 0; // 필요시 acc_t
            for (int k = 0; k < K; ++k)
            {
                acc += (int)A[i * sA + k] * (int)B[k * sB + j];
            }
            acc += (int)D[i * sD + j];
            C_expected[i * sC + j] = (elem_t)sat_i8(acc);
        }
    }

    dump_matrix("A (I x K)", A, I, K, sA);
    dump_matrix("B (K x J)", B, K, J, sB);
    dump_matrix("D (I x J), acc_t", D, I, J, sD);
    dump_matrix("Expected C (I x J)", C_expected, I, J, sC);

    tiled_matmul_auto(I, J, K,
                      A, B, (const void *)D, (void *)C,
                      sA, sB, sD, sC,
                      1.f, 1.f, 1.f,
                      NO_ACTIVATION,
                      1, 1,
                      false,
                      false, // transpose_A
                      false, // transpose_B
                      false, false,
                      0, CPU);

    dump_matrix("C (result from gemmini)", C, I, J, sC);

    // compare
    bool ok = true;
    for (int i = 0; i < I; ++i)
    {
        for (int j = 0; j < J; ++j)
        {
            int8_t got = C[i * sC + j];
            int8_t exp = C_expected[i * sC + j];
            if (got != exp)
            {
                printf("[NG] mismatch (%d, %d): got=%d exp=%d\n", i, j, got, exp);
                ok = false;
            }
        }
    }
    printf(ok ? "[OK] Gemmini matmul(+bias) matches expected\n"
              : "[FAIL] mismatch detected\n");
}
#endif // __GGML_GEMMINI_TEST_H__