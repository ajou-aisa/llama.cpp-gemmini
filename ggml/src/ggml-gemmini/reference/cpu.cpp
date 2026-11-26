// Derived from Gemmini SDK develop fcfcf5ce; see matmul/README.md and LICENSE.gemmini*.
#include "cpu.hpp"
#include "matmul/reference.hpp"
#include <cstdio>
#include <cstdlib>
namespace ggml::gemmini {
void matmul_cpu_fp(bool transA, bool transB,
                          size_t DIM_I, size_t DIM_J, size_t DIM_K,
                          const float* A, const float* B, const float* D,
                          float* C,
                          size_t stride_A, size_t stride_B,
                          size_t stride_D, size_t stride_C) {
  const size_t A_i_stride = transA ? 1        : stride_A;
  const size_t A_k_stride = transA ? stride_A : 1;
  const size_t B_j_stride = transB ? stride_B : 1;
  const size_t B_k_stride = transB ? 1        : stride_B;

  if (DIM_K == 0) {
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        C[i * stride_C + j] = D == NULL ? 0.0f : D[i * stride_D + j];
      }
    }
    return;
  }

  const size_t I4 = DIM_I & ~(size_t)3;
  const size_t J4 = DIM_J & ~(size_t)3;

  // Main 4x4 tile loop
  for (size_t i = 0; i < I4; i += 4) {
    for (size_t j = 0; j < J4; j += 4) {
      float r00 = 0.0f, r01 = 0.0f, r02 = 0.0f, r03 = 0.0f;
      float r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f;
      float r20 = 0.0f, r21 = 0.0f, r22 = 0.0f, r23 = 0.0f;
      float r30 = 0.0f, r31 = 0.0f, r32 = 0.0f, r33 = 0.0f;

      const float* a0 = A + (i + 0) * A_i_stride;
      const float* a1 = A + (i + 1) * A_i_stride;
      const float* a2 = A + (i + 2) * A_i_stride;
      const float* a3 = A + (i + 3) * A_i_stride;

      const float* b0 = B + (j + 0) * B_j_stride;
      const float* b1 = B + (j + 1) * B_j_stride;
      const float* b2 = B + (j + 2) * B_j_stride;
      const float* b3 = B + (j + 3) * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float av0 = *a0;
        const float av1 = *a1;
        const float av2 = *a2;
        const float av3 = *a3;

        const float bv0 = *b0;
        const float bv1 = *b1;
        const float bv2 = *b2;
        const float bv3 = *b3;

        r00 += av0 * bv0; r01 += av0 * bv1; r02 += av0 * bv2; r03 += av0 * bv3;
        r10 += av1 * bv0; r11 += av1 * bv1; r12 += av1 * bv2; r13 += av1 * bv3;
        r20 += av2 * bv0; r21 += av2 * bv1; r22 += av2 * bv2; r23 += av2 * bv3;
        r30 += av3 * bv0; r31 += av3 * bv1; r32 += av3 * bv2; r33 += av3 * bv3;

        a0 += A_k_stride;
        a1 += A_k_stride;
        a2 += A_k_stride;
        a3 += A_k_stride;

        b0 += B_k_stride;
        b1 += B_k_stride;
        b2 += B_k_stride;
        b3 += B_k_stride;
      }

      if (D != NULL) {
        const float* d0 = D + (i + 0) * stride_D + j;
        const float* d1 = D + (i + 1) * stride_D + j;
        const float* d2 = D + (i + 2) * stride_D + j;
        const float* d3 = D + (i + 3) * stride_D + j;

        r00 += d0[0]; r01 += d0[1]; r02 += d0[2]; r03 += d0[3];
        r10 += d1[0]; r11 += d1[1]; r12 += d1[2]; r13 += d1[3];
        r20 += d2[0]; r21 += d2[1]; r22 += d2[2]; r23 += d2[3];
        r30 += d3[0]; r31 += d3[1]; r32 += d3[2]; r33 += d3[3];
      }

      float* c0 = C + (i + 0) * stride_C + j;
      float* c1 = C + (i + 1) * stride_C + j;
      float* c2 = C + (i + 2) * stride_C + j;
      float* c3 = C + (i + 3) * stride_C + j;

      c0[0] = r00; c0[1] = r01; c0[2] = r02; c0[3] = r03;
      c1[0] = r10; c1[1] = r11; c1[2] = r12; c1[3] = r13;
      c2[0] = r20; c2[1] = r21; c2[2] = r22; c2[3] = r23;
      c3[0] = r30; c3[1] = r31; c3[2] = r32; c3[3] = r33;
    }

    // Right tail: rows are still 4-wide, columns are scalar
    for (size_t j = J4; j < DIM_J; j++) {
      float r0 = 0.0f;
      float r1 = 0.0f;
      float r2 = 0.0f;
      float r3 = 0.0f;

      const float* a0 = A + (i + 0) * A_i_stride;
      const float* a1 = A + (i + 1) * A_i_stride;
      const float* a2 = A + (i + 2) * A_i_stride;
      const float* a3 = A + (i + 3) * A_i_stride;
      const float* b = B + j * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float bv = *b;

        r0 += (*a0) * bv;
        r1 += (*a1) * bv;
        r2 += (*a2) * bv;
        r3 += (*a3) * bv;

        a0 += A_k_stride;
        a1 += A_k_stride;
        a2 += A_k_stride;
        a3 += A_k_stride;
        b += B_k_stride;
      }

      if (D != NULL) {
        r0 += D[(i + 0) * stride_D + j];
        r1 += D[(i + 1) * stride_D + j];
        r2 += D[(i + 2) * stride_D + j];
        r3 += D[(i + 3) * stride_D + j];
      }

      C[(i + 0) * stride_C + j] = r0;
      C[(i + 1) * stride_C + j] = r1;
      C[(i + 2) * stride_C + j] = r2;
      C[(i + 3) * stride_C + j] = r3;
    }
  }

  // Bottom tail: remaining rows, 4 columns at a time
  for (size_t i = I4; i < DIM_I; i++) {
    for (size_t j = 0; j < J4; j += 4) {
      float r0 = 0.0f;
      float r1 = 0.0f;
      float r2 = 0.0f;
      float r3 = 0.0f;

      const float* a = A + i * A_i_stride;
      const float* b0 = B + (j + 0) * B_j_stride;
      const float* b1 = B + (j + 1) * B_j_stride;
      const float* b2 = B + (j + 2) * B_j_stride;
      const float* b3 = B + (j + 3) * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        const float av = *a;

        r0 += av * (*b0);
        r1 += av * (*b1);
        r2 += av * (*b2);
        r3 += av * (*b3);

        a += A_k_stride;
        b0 += B_k_stride;
        b1 += B_k_stride;
        b2 += B_k_stride;
        b3 += B_k_stride;
      }

      if (D != NULL) {
        const float* d = D + i * stride_D + j;
        r0 += d[0];
        r1 += d[1];
        r2 += d[2];
        r3 += d[3];
      }

      float* c = C + i * stride_C + j;
      c[0] = r0;
      c[1] = r1;
      c[2] = r2;
      c[3] = r3;
    }

    // Bottom-right scalar tail
    for (size_t j = J4; j < DIM_J; j++) {
      float sum = 0.0f;

      const float* a = A + i * A_i_stride;
      const float* b = B + j * B_j_stride;

      for (size_t k = 0; k < DIM_K; k++) {
        sum += (*a) * (*b);
        a += A_k_stride;
        b += B_k_stride;
      }

      if (D != NULL) {
        sum += D[i * stride_D + j];
      }

      C[i * stride_C + j] = sum;
    }
  }
}

void matmul_cpu_int32(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
                               const elem_t *A, const elem_t *B, const acc_t *D,
                               void *C,
                               size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                               scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                               int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,
                               bool full_C)
  {

    if (!full_C)
    {
      matmul_cpu(transA, transB, DIM_I, DIM_J, DIM_K,
                 A, B, D, (elem_t *)C,
                 stride_A, stride_B, stride_D, stride_C,
                 A_scale_factor, B_scale_factor, D_scale_factor,
                 act, scale, bert_scale, repeating_bias);
      return;
    }

    if (act != GEMMINI_ACT_NONE)
    {
      printf("[matmul_cpu_int32] Expected GEMMINI_ACT_NONE but got act=%d (full_C-only path)\n", act);
      exit(1);
    }

    const int no_bias = D == NULL;
    acc_t *C_acc = (acc_t *)C;

    if (!transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0)
    {
      for (size_t i = 0; i < DIM_I; i += 4)
      {
        for (size_t j = 0; j < DIM_J; j += 4)
        {
          acc_t result[4][4];

          for (size_t ii = 0; ii < 4; ii++)
            for (size_t jj = 0; jj < 4; jj++)
            {
              const size_t bias_row = repeating_bias ? 0 : i + ii;
              result[ii][jj] = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j + jj), D_scale_factor);
            }

          for (size_t k = 0; k < DIM_K; k++)
          {
            result[0][0] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[0][1] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[0][2] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[0][3] +=
                GEMMINI_SCALE(*(A + i * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[1][0] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[1][1] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[1][2] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[1][3] +=
                GEMMINI_SCALE(*(A + (i + 1) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[2][0] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[2][1] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[2][2] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[2][3] +=
                GEMMINI_SCALE(*(A + (i + 2) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
            result[3][0] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j), B_scale_factor);
            result[3][1] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 1), B_scale_factor);
            result[3][2] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 2), B_scale_factor);
            result[3][3] +=
                GEMMINI_SCALE(*(A + (i + 3) * stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k * stride_B + j + 3), B_scale_factor);
          }

          for (size_t ii = 0; ii < 4; ++ii)
            for (size_t jj = 0; jj < 4; ++jj)
            {
              const size_t idx = (i + ii) * stride_C + (j + jj);
              C_acc[idx] = result[ii][jj];
            }
        }
      }
    }
    else
    {
      size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A};
      size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1};

      static acc_t c_buffer[1024];
      const size_t c_buffer_sz = sizeof(c_buffer) / sizeof(c_buffer[0]);
      if ((act == GEMMINI_ACT_LAYERNORM || act == GEMMINI_ACT_SOFTMAX) && DIM_J > c_buffer_sz)
      {
        printf("Matmul is too large to normalize\n");
        exit(1);
      }

      for (size_t i = 0; i < DIM_I; i++)
      {
        for (size_t j = 0; j < DIM_J; j++)
        {
          const size_t idx = (i * stride_C) + j;

          const size_t bias_row = repeating_bias ? 0 : i;
          acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);
          const acc_t bias_term = sum;

          for (size_t k = 0; k < DIM_K; k++)
          {
            const size_t a_off = i * A_dim_strides[0] + k * A_dim_strides[1];
            const size_t b_off = j * B_dim_strides[0] + k * B_dim_strides[1];
            const elem_t a_raw = A[a_off];
            const elem_t b_raw = B[b_off];
            const acc_t a_scaled = GEMMINI_SCALE(a_raw, A_scale_factor);
            const acc_t b_scaled = GEMMINI_SCALE(b_raw, B_scale_factor);
            const acc_t prod = a_scaled * b_scaled;
            sum += prod;
          }

          C_acc[idx] = sum;
        }
      }
    }
  }
} // namespace ggml::gemmini
