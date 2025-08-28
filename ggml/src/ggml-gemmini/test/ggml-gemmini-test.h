// test/ggml-gemmini-test.h
#ifndef __GGML_GEMMINI_TEST_H__
#define __GGML_GEMMINI_TEST_H__

#ifndef OPTION
#define OPTION CPU
#endif

#include "include/gemmini.h"
#include <cstdio>
#include <type_traits>
#include <vector>
#include <cmath>
#include "ggml.h"

using namespace zerogod;

template <typename T>
static inline void dump_matrix(const char* name, const T* m, int r, int c, int s) {
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
};

static inline int8_t sat_i8(int x) {
    return x > 127 ? 127 : (x < -128 ? -128 : (int8_t)x);
}

// 원본 텐서를 원하는 크기로 slicing하여 연산 테스트
static void ggml_backend_gemmini_mul_mat_test(struct ggml_context * ctx, const struct ggml_tensor *A_in, const struct ggml_tensor *B_in, struct ggml_tensor *C_out, const struct ggml_tensor *D_bias, const int i, const int j, const int k) {
    GGML_ASSERT(i > 0 && j > 0 && k > 0);

    DBG0("\n[Gemmini] mul_mat test called");

    const size_t I = (size_t)i, J = (size_t)j, K = (size_t)k;
    // stride
    const size_t sA = align_up(K, 16 / sizeof(elem_t));
    const size_t sB = align_up(J, 16 / sizeof(elem_t));
    const size_t sC = align_up(J, 16 / sizeof(elem_t));
    const size_t sD_elem = align_up(J, 16 / sizeof(acc_t));

    DBG0("\nI=%zu, J=%zu, K=%zu\n", I, J, K);

    auto alloc16 = [](size_t bytes) -> void *
    {
        void *p = std::aligned_alloc(16, bytes);
        if(!p) {
            DBG0("aligned_alloc failed (bytes=%zu)\n", bytes);
            std::abort();
        }
        std::memset(p, 0, bytes);
        return p;
    };

    elem_t *C_expected = (elem_t *)alloc16(I * sC * sizeof(elem_t)); // expected value

    GGML_ASSERT(A_in && A_in->data && "A_in is invalid");
    GGML_ASSERT(B_in && B_in->data && "B_in is invalid");
    GGML_ASSERT(C_out && C_out->data && "C_out is invalid");

    struct ggml_tensor *A_sliced = ggml_view_2d(ctx, const_cast<struct ggml_tensor *>(A_in), I, K, A_in->nb[1], 0);
    struct ggml_tensor *B_sliced = ggml_view_2d(ctx, const_cast<struct ggml_tensor *>(B_in), J, K, B_in->nb[1], 0);
    struct ggml_tensor *C_sliced = ggml_view_2d(ctx, C_out, I, J, C_out->nb[1], 0);

    const void* d_data_ptr;
    size_t sD;
    std::vector<acc_t> zero_bias;

    if (D_bias) {
        GGML_ASSERT(D_bias->data && "D_bias->data is invalid");
        struct ggml_tensor* D_sliced = ggml_view_2d(ctx, const_cast<struct ggml_tensor *>(D_bias), I, J, D_bias->nb[1], 0);
        d_data_ptr = D_sliced->data;
        sD = D_sliced->nb[1] / sizeof(acc_t);
    } else {
        zero_bias.assign(I * sD_elem, 0);
        d_data_ptr = zero_bias.data();
        sD = sD_elem;
    }

    const float test_scale = 0.5f;

    // expected
    for (size_t r = 0; r < I; ++r)
    {
        for (size_t c = 0; c < J; ++c)
        {
            int acc = 0; // 필요시 acc_t
            for (int k_idx = 0; k_idx < K; ++k_idx) // Renamed k to k_idx to avoid conflict with function parameter k
                acc += (int)((elem_t*)A_sliced->data)[r * (A_sliced->nb[1] / sizeof(elem_t)) + k_idx] * (int)((elem_t*)B_sliced->data)[c * (B_sliced->nb[1] / sizeof(elem_t)) + k_idx]; // Access B as transposed

            acc += (int)((const acc_t*)d_data_ptr)[r * sD + c];

            int32_t scaled_acc = roundf((float)acc * test_scale);
            C_expected[r * sC + c] = (elem_t)sat_i8(scaled_acc);
        }
    }

    dump_matrix("A (I x K)", (elem_t*)A_sliced->data, I, K, A_sliced->nb[1] / sizeof(elem_t));
    dump_matrix("B (J x K, stored transposed)", (elem_t*)B_sliced->data, J, K, B_sliced->nb[1] / sizeof(elem_t)); // Dump as JxK
    dump_matrix("D (I x J), acc_t", (const acc_t*)d_data_ptr, I, J, sD);
    dump_matrix("Expected C (I x J)", C_expected, I, J, sC);

    tiled_matmul_auto(I, J, K,
                      (elem_t*)A_sliced->data, (elem_t*)B_sliced->data, d_data_ptr, (void *)C_sliced->data,
                      sA, sB, sD, sC, // Use original strides for tiled_matmul_auto
                      test_scale, test_scale, test_scale,
                      NO_ACTIVATION,
                      1, 1,
                      false,
                      false, // transpose_A
                      true, // transpose_B
                      false, false,
                      0, OPTION);

    dump_matrix("C (result from gemmini)", (elem_t*)C_sliced->data, I, J, C_sliced->nb[1] / sizeof(elem_t));

    // compare
    bool ok = true;
    for (size_t r = 0; r < I; ++r)
    {
        for (size_t c = 0; c < J; ++c)
        {
            elem_t got = ((elem_t*)C_sliced->data)[r * (C_sliced->nb[1] / sizeof(elem_t)) + c];
            elem_t exp = C_expected[r * sC + c];
            if (got != exp)
            {
                DBG0("[NG] mismatch (%zu, %zu): got=%d exp=%d\n", r, c, (int)got, (int)exp);
                ok = false;
            }
        }
    }
    if (ok)
        DBG0("[OK] Gemmini matmul(+bias) matches expected\n");
    else    
        DBG0("[FAIL] mismatch detected\n");
}
#endif // __GGML_GEMMINI_TEST_H__
