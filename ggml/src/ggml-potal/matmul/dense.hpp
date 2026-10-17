#pragma once

// GGML MUL_MAT <-> PoTal tiled GEMM contract for Q8_HP1 weights (W4-5).
//
//   GGML  dst[m][n] = sum_k src1[m][k] * src0[n][k]
//         src0 (weights)     Q8_HP1, ne = {K, N}: row n holds K/32 blocks
//         src1 (activations) F32,    ne = {K, M}: row m holds K values
//         dst                F32,    ne = {N, M}: row m holds N values
//   PoTal A[m][k] = tensor-quantized src1 (int8), B[k][n] = qs of src0 row n,
//         carrier[k/32][n] = SCU carrier of block (n, k/32),
//         C[m][n] = Sat32 sum over DIM-wide K passes p of
//                   sat32((sum_{k in p} A[m][k] * B[k][n]) << carrier[p*DIM/32][n])
//   dst[m][n] = float(double(C[m][n]) * double(activation_scale) * double(channel_scale[n]))
//
// The HP1 exponent is applied once, by the SCU; reconstruction uses only the
// two linear scales. No error compensation term exists in this contract.

#include "ggml.h"
#include "ggml-potal.h"
#include "potal.h"

#include "../ggml-potal-args.h"
#include "../quants/weight/hp1/hp1.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml::potal {

struct hp1_shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

// supports_op / graph_compute contract of an F32 x Q8_HP1 MUL_MAT. require_data
// additionally demands allocated src/dst data. On success fills *shape.
ggml_potal_reject check_mul_mat_hp1(const ggml_tensor * op, bool require_data, hp1_shape * shape);

// DMA staging operands of one MUL_MAT, POTAL_HW_MAX_BYTES aligned, padding zeroed:
//   a        int8  [m_pad][lda]         lda multiple of MAX_BYTES
//   b        int8  [k][ldb]             ldb multiple of MAX_BYTES
//   carriers u32   [k/32][cstride]      cstride = n_pad
//   scale    f32   [n_pad]              channel_scale per column
//   c        int32 [m_pad][ldc]         ldc = n_pad
// Buffers handed to the accelerator must outlive the last POTAL_FENCE that
// covers them: after a fatal hardware failure call abandon() instead of
// letting the destructor free them.
class hp1_operands {
  public:
    hp1_operands() = default;
    hp1_operands(const hp1_operands &) = delete;
    hp1_operands & operator=(const hp1_operands &) = delete;
    ~hp1_operands();

    bool allocate(const hp1_shape & s); // false if out of memory
    void abandon();                     // leak the buffers on purpose

    int8_t *   a        = nullptr;
    int8_t *   b        = nullptr;
    uint32_t * carriers = nullptr;
    float *    scale    = nullptr;
    int32_t *  c        = nullptr;
    size_t     lda = 0, ldb = 0, cstride = 0, ldc = 0;
    size_t     m_pad = 0, n_pad = 0;

  private:
    void * blocks[5] = {};
};

// The ggml_potal_args_t of one MUL_MAT (Gemmini args layout): I = M, J = N,
// K, the Q8_HP1 view of src0, the F32 view of dst and the layer identity.
// false if a size does not fit the contract.
bool init_args(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const hp1_shape & s,
               const char * model_arch, const char * layer, ggml_potal_args_t & args);

// src1 -> args.A (quants::quantize_activation, GGML_GEMMINI_ACTIVATION_QUANT)
// -> ops.a; *activation_scale = the tensor scale. false if quantization fails.
bool quantize_activations(const ggml_tensor * src1, ggml_potal_args_t & args, hp1_operands & ops,
                          float * activation_scale);

// args' Q8_HP1 blocks -> ops.b / ops.carriers / ops.scale (quants::weight::hp1::unpack).
quants::weight::hp1::status unpack_weights(const ggml_potal_args_t & args, hp1_operands & ops,
                                           quants::weight::hp1::stats * stats,
                                           quants::wreader::WeightReaderStatus * reader = nullptr);

// ops.c -> args.f_out (F32).
void reconstruct(const hp1_operands & ops, const ggml_potal_args_t & args, float activation_scale);

// Independent CPU integer reference of the PoTal GEMM (no tiling, K passes of
// POTAL_HW_DIM in order, SCU per pass and column, Sat32 accumulation):
// c[m * ldc + n] for m < M, n < N. block_k is the K extent of one carrier.
void reference_gemm_i32(const int8_t * a, size_t lda, const int8_t * b, size_t ldb, const uint32_t * carriers,
                        size_t cstride, size_t block_k, size_t M, size_t N, size_t K, int32_t * c, size_t ldc);

} // namespace ggml::potal
