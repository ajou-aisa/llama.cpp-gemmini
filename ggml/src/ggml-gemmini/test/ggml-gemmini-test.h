// test/ggml-gemmini-test.h
#ifndef __GGML_GEMMINI_TEST_H__
#define __GGML_GEMMINI_TEST_H__

#include "include/gemmini_params.h"
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

#endif // __GGML_GEMMINI_TEST_H__