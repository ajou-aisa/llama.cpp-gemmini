// test/ggml-gemmini-test.h
#ifndef __GGML_GEMMINI_TEST_H__
#define __GGML_GEMMINI_TEST_H__

#include "include/gemmini.h"
#include <cstdio>
#include <type_traits>

using namespace zerogod;

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

static void ggml_backend_gemmini_mul_mat_test(const int i, const int j, const int k, tiled_matmul_type_t PATH) {
    GGML_ASSERT(i > 0 && j > 0 && k > 0);

    DBG("\n[Gemmini] mul_mat test called");

    const size_t I = (size_t)i, J = (size_t)j, K = (size_t)k;
    // stride
    const size_t sA = align_up(K, 16 / sizeof(elem_t));
    const size_t sB = align_up(J, 16 / sizeof(elem_t)); 
    const size_t sC = align_up(J, 16 / sizeof(elem_t));
    const size_t sD = align_up(J, 16 / sizeof(acc_t));

    DBG("\nI=%zu, J=%zu, K=%zu\n", I, J, K);

    auto alloc16 = [](size_t bytes) -> void *
    {
        void *p = std::aligned_alloc(16, bytes);
        if(!p) {
            DBG("aligned_alloc failed (bytes=%zu)\n", bytes);
            std::abort();
        }
        std::memset(p, 0, bytes);
        return p;
    };

    elem_t *A = (elem_t *)alloc16(I * sA * sizeof(elem_t)); // IxK
    elem_t *B = (elem_t *)alloc16(K * sB * sizeof(elem_t)); // K×J
    elem_t *C = (elem_t *)alloc16(I * sC * sizeof(elem_t)); // I×J

    elem_t *C_expected = (elem_t *)alloc16(I * sC * sizeof(elem_t)); // expected value
    acc_t *D = (acc_t *)alloc16(I * sD * sizeof(acc_t));             // I×J bias

    int e = 1;
    // 포화 방지
    int t = (int)std::floor(std::sqrt(127.0 / (double)K));
    int leftover = 127 - (int)(K * t * (long long)t); // 0..127
    int db = std::min(4, std::max(0, leftover));

    // init A
    for (size_t r = 0; r < I; ++r)
        for (size_t c = 0; c < K; ++c)
            A[r * sA + c] = (elem_t)(e++ % (2 * t + 1) - t);

    // init B
    for (size_t r = 0; r < K; ++r)
        for (size_t c = 0; c < J; ++c)
            B[r * sB + c] = (elem_t)(e++ % (2 * t + 1) - t);

    // init D
    for (size_t r = 0; r < I; ++r)
        for (size_t c = 0; c < J; ++c)
            D[r * sD + c] = (acc_t)((e++ % (2*db + 1)) - db);

    // expected
    for (size_t r = 0; r < I; ++r)
    {
        for (size_t c = 0; c < J; ++c)
        {
            int acc = 0; // 필요시 acc_t
            for (int k = 0; k < K; ++k)
                acc += (int)A[r * sA + k] * (int)B[k * sB + c];

            acc += (int)D[r * sD + c];
            C_expected[r * sC + c] = (elem_t)sat_i8(acc);
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
                      0, PATH);

    dump_matrix("C (result from gemmini)", C, I, J, sC);

    // compare
    bool ok = true;
    for (size_t r = 0; r < I; ++r)
    {
        for (size_t c = 0; c < J; ++c)
        {
            elem_t got = C[r * sC + c];
            elem_t exp = C_expected[r * sC + c];
            if (got != exp)
            {
                printf("[NG] mismatch (%zu, %zu): got=%d exp=%d\n", r, c, (int)got, (int)exp);
                ok = false;
            }
        }
    }
    printf(ok ? "[OK] Gemmini matmul(+bias) matches expected\n"
              : "[FAIL] mismatch detected\n");
}
#endif // __GGML_GEMMINI_TEST_H__