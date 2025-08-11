#ifndef __GGML_GEMMINI_TEST_H__
#define __GGML_GEMMINI_TEST_H__

#include "../ggml-gemmini-util.h"
#include "include/gemmini.h"

template <typename T>
static inline void dump_matrix(const char* name, const T* m, int r, int c, int s);

static inline int8_t sat_i8(int x) {
    return x > 127 ? 127 : (x < -128 ? -128 : (int8_t)x);
}

void ggml_backend_gemmini_mul_mat_test();

#endif // __GGML_GEMMINI_TEST_H__