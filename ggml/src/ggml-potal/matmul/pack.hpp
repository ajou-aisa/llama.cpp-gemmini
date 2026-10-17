#pragma once

// GGML MUL_MAT <-> PoTal WS GEMM contract for the validated DIM16 scope.
//
//   GGML  dst[m][n] = sum_k src1[m][k] * src0[n][k]
//         src0 (weights)     I8,  ne = {K, N}: row n holds K values
//         src1 (activations) I8,  ne = {K, M}: row m holds K values
//         dst                F32, ne = {N, M}: row m holds N values
//   PoTal C[m][n] = sum_k A[m][k] * B[k][n]          (potal.h, row-major)
//         A = src1 (int8 [M][K]), B = src0 transposed (int8 [K][N]),
//         C = dst before the INT32 -> F32 conversion (int32 [M][N])
//
// With every scale carrier x1 and |sum| <= 32 * 128 * 128 the INT32 result is
// exact and exactly representable in F32.

#include "ggml.h"
#include "ggml-potal.h"

#include <cstdint>

namespace ggml::potal {

constexpr int64_t TILE  = 16;       // M = N = DIM
constexpr int64_t K_MAX = 2 * TILE; // K = 16 or 32

struct mul_mat_shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

// supports_op contract for any op.
ggml_potal_reject check_op(const ggml_tensor * op);

// MUL_MAT contract. require_data additionally demands allocated src/dst data
// (graph_compute). On success fills *shape when it is not null.
ggml_potal_reject check_mul_mat(const ggml_tensor * op, bool require_data, mul_mat_shape * shape);

// The functions below need a tensor that passed check_mul_mat(.., true, ..).
// src1 -> A, int8 [M][K], row stride K.
void pack_a(const ggml_tensor * src1, const mul_mat_shape & s, int8_t * a);
// src0 -> B, int8 [K][N], row stride N (transposes the GGML weight rows).
void pack_b(const ggml_tensor * src0, const mul_mat_shape & s, int8_t * b);
// C, int32 [M][N], row stride N -> dst (F32).
void unpack_c(const int32_t * c, const mul_mat_shape & s, ggml_tensor * dst);

// Fill a contiguous host F32 dst with NaN so a failed node never exposes stale
// or partial data as a result. Other tensors are left untouched.
void poison_dst(ggml_tensor * dst);

} // namespace ggml::potal
