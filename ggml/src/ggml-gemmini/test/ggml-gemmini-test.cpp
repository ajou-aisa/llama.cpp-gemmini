// ggml-gemmini-test.cpp
#include "ggml-gemmini-test.h"
#include <cstdio>
#include <type_traits>

template <typename T>
static inline void dump_matrix(const char *name, const T *m, int r, int c, int s)
{
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
}