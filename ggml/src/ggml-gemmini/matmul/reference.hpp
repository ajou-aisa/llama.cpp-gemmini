// CPU reference matmul of the legacy Gemmini SDK, moved verbatim (lines 982-1244).
// Provenance: RISC-V-DynDNN-gemmini-include develop fcfcf5ce7c4670bd789365f11810177f97016f81
// gemmini.h (208174 bytes, sha256 a6d7207ddee555f0fbcd00b9952a12507b458ee43a899c7c1e60a131af53b15f),
// blob cd15b5479a59c62e549e14103ae2386440ffeb60, derived from gemmini-rocc-tests (BSD-3-Clause;
// matmul/LICENSE.gemmini-rocc-tests, matmul/LICENSE.gemmini, matmul/README.md).
// Numeric contract: elem_t/acc_t and the scale macros of the generated
// gemmini_params.h, exactly as the SDK compiled them.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gemmini_params.h"

#include "gemmini.h"

static acc_t int_sqrt(acc_t n) {
  if (n == 0) return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1) {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev) return x_prev;
    x_prev = x_next;
  };
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == GEMMINI_ACT_IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == GEMMINI_ACT_RELU) {
    x = x < 0 ? 0 : x;
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

  const int no_bias = D == NULL;
  if (act != GEMMINI_ACT_LAYERNORM && act != GEMMINI_ACT_SOFTMAX && !transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, bert_scale);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, bert_scale);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, bert_scale);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, bert_scale);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride

    // We also create a buffer that we can use for layernorms and softmaxes
    static acc_t c_buffer[1024];
    const size_t c_buffer_sz = sizeof(c_buffer)/sizeof(c_buffer[0]);
    if ((act == GEMMINI_ACT_LAYERNORM || act == GEMMINI_ACT_SOFTMAX) && DIM_J > c_buffer_sz) {
      printf("Matmul is too large to normalize\n");
      exit(1);
    }

    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }

        if (act == GEMMINI_ACT_LAYERNORM || act == GEMMINI_ACT_SOFTMAX)
          c_buffer[j] = sum;
        else
          *c = scale_and_sat(sum, act, scale, bert_scale);
      }

      if (act == GEMMINI_ACT_LAYERNORM) {
        acc_t sum = 0;
        for (size_t j = 0; j < DIM_J; j++)
          sum += c_buffer[j];
        acc_t mean = sum / (acc_t)DIM_J;

        acc_t total_err_sq = 0;
        for (size_t j = 0; j < DIM_J; j++)
          total_err_sq += (c_buffer[j] - mean)*(c_buffer[j] - mean);
        acc_t variance = total_err_sq / (acc_t)DIM_J;

        acc_t stddev = int_sqrt(variance);
        if (variance == 0) stddev = 1;

        for (size_t j = 0; j < DIM_J; j++) {
          c_buffer[j] -= mean;
          c_buffer[j] /= stddev;

          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, scale, bert_scale);
        }
      } else if (act == GEMMINI_ACT_SOFTMAX) {
        const scale_t a = 0.3585;
        const scale_t b = 1.353;
        const scale_t c = 0.344;

        // is SCALE supposed to be input scale?
        const acc_t qln2 = (acc_t) (0.693147 / bert_scale);
        const acc_t qln2_inv = 65536 / qln2;
        const acc_t qb = b / bert_scale;
        const acc_t qc = c / (a*bert_scale*bert_scale);

        // pass 1: get max_q
        acc_t max_q = -2147483648;
        for (size_t j = 0; j < DIM_J; j++) {
          if (c_buffer[j] > max_q) max_q = c_buffer[j];
        }

        // pass 2: calculate iexp(q_tilde) and sum(q_tilde)
        acc_t sum_exp = 0;
        for (size_t j = 0; j < DIM_J; j++) {
          acc_t q = c_buffer[j] - max_q;
          // [HS] matmul_cpu not used. It cannot be used for float
          acc_t z = 0;
          // acc_t z = (acc_t) (-q * qln2_inv) >> 16;
          acc_t qp = q + z * qln2;
          // [HS] also cannot be used for Float
          acc_t q_exp = 0;
          // acc_t q_exp = (qp + qb)*(qp + qb) + qc;
          c_buffer[j] = q_exp + z;
          // -- ends. Make some lines dummy.
          sum_exp += c_buffer[j];
        }

        // pass 3: divide by sum
        scale_t factor = (127.f) / (float) sum_exp; // what corresponds to 1 in output?
        for (size_t j = 0; j < DIM_J; j++) {
          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, factor, bert_scale);
        }
      }
    }
  }
}
