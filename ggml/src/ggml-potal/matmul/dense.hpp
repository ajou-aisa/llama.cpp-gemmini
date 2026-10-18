#pragma once

// GGML MUL_MAT <-> PoTal tiled GEMM for Q8_HP1 weights.
//
//   GGML  dst[m][n] = sum_k src1[m][k] * src0[n][k]
//         src0 (weights)     Q8_HP1, ne = {K, N}: row n holds K/32 blocks
//         src1 (activations) F32,    ne = {K, M}: row m holds K values
//         dst                F32,    ne = {N, M}: row m holds N values
//
// Activation: ggml::gemmini::quants::quantize_activation, i.e. the algorithm
// GGML_GEMMINI_ACTIVATION_QUANT selects (EXSIA, TENSOR, TOKEN, BLOCK, STRIPE),
// with its own metadata (act::Meta) and, under GGML_GEMMINI_ENABLE_RMD, its
// residual payloads. Weight: the HP1 qs payload, one SCU carrier per
// (32-wide K block, column) and one channel_scale per column.
//
// The route is chosen from the activation metadata the quantizer produced,
// crossed with the HP1 block-scaled weight ("<activation>+block"):
//
//   Tensor+block, Token+block, Stripe+block, ExSIA+block
//       activation scale constant along K for every row:
//       one potal_tiled_matmul_auto over the whole K (SCU carriers per block,
//       Sat32 accumulation over K in the ACC), then
//       dst[m][n] = C[m][n] * row_scale[m] * channel_scale[n]
//   Block+block
//       activation scale per (row, 32-wide K group): one
//       potal_tiled_matmul_auto per K group, then
//       dst[m][n] = sum_g C_g[m][n] * scale[m][g] * channel_scale[n]   (double)
//
// where C = PoTal(A, B, carriers):
//   C[m][n] = Sat32 sum over DIM-wide K passes p of
//             sat32((sum_{k in p} A[m][k] * B[k][n]) << carrier[p*DIM/32][n])
// The HP1 exponent is applied once, by the SCU; reconstruction uses only the
// linear scales. With compensation ON (cpu_direct residual route) the residual
// correction of the original Gemmini code is added to dst afterwards.

#include "ggml.h"
#include "ggml-potal.h"
#include "potal.h"

#include "execution.hpp"
#include "ggml-gemmini-args.h"
#include "quants/weight/hp1/hp1.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ggml::potal {

struct hp1_shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

// supports_op / graph_compute contract of an F32 x Q8_HP1 MUL_MAT. require_data
// additionally demands allocated src/dst data. On success fills *shape.
ggml_potal_reject check_mul_mat_hp1(const ggml_tensor * op, bool require_data, hp1_shape * shape);

// Static operands of one Q8_HP1 weight (N x K), POTAL_HW_MAX_BYTES aligned,
// padding zeroed; filled once by prepare_weight() and read-only afterwards, so
// one instance can serve every MUL_MAT of that weight (weight_cache.hpp):
//   b        int8  [k][ldb]             ldb multiple of MAX_BYTES
//   carriers u32   [k/32][cstride]      cstride = n_pad
//   scale    f32   [n_pad]              channel_scale per column
// Buffers handed to the accelerator must outlive the last POTAL_FENCE that
// covers them: after a fatal hardware failure call abandon() instead of
// letting the destructor free them.
class hp1_weight {
  public:
    hp1_weight() = default;
    hp1_weight(const hp1_weight &) = delete;
    hp1_weight & operator=(const hp1_weight &) = delete;
    ~hp1_weight();

    bool   allocate(int64_t n, int64_t k); // false if out of memory
    void   abandon();                      // leak the buffers on purpose
    size_t bytes() const;                  // allocated bytes

    int8_t *   b        = nullptr;
    uint32_t * carriers = nullptr;
    float *    scale    = nullptr;
    size_t     n = 0, k = 0;
    size_t     ldb = 0, cstride = 0, n_pad = 0;
    quants::weight::hp1::stats stats; // of the blocks prepare_weight() unpacked

  private:
    void * blocks[3] = {};
};

// DMA staging operands of one MUL_MAT, POTAL_HW_MAX_BYTES aligned, padding zeroed:
//   a        int8  [m_pad][lda]         lda multiple of MAX_BYTES
//   b, carriers, scale                  of the weight (hp1_weight layout)
//   c        int32 [m_pad][ldc]         ldc = n_pad
// allocate(s) also allocates its own weight operands (filled by
// unpack_weights); allocate(s, w) points b/carriers/scale at the prepared
// weight w, which must outlive these operands and is never written here.
// Same buffer lifetime rule as hp1_weight: abandon() after a fatal failure
// (it does not touch a shared weight; its owner abandons that).
class hp1_operands {
  public:
    hp1_operands() = default;
    hp1_operands(const hp1_operands &) = delete;
    hp1_operands & operator=(const hp1_operands &) = delete;
    ~hp1_operands();

    bool allocate(const hp1_shape & s);                        // false if out of memory
    bool allocate(const hp1_shape & s, const hp1_weight & w); // false if out of memory or w is not s.n x s.k
    void abandon();                                            // leak the buffers on purpose

    int8_t *         a        = nullptr;
    const int8_t *   b        = nullptr;
    const uint32_t * carriers = nullptr;
    const float *    scale    = nullptr;
    int32_t *        c        = nullptr;
    size_t           lda = 0, ldb = 0, cstride = 0, ldc = 0;
    size_t           m_pad = 0, n_pad = 0;
    hp1_weight       own; // weight operands of allocate(s)

  private:
    bool   allocate_dynamic(const hp1_shape & s);
    void * blocks[2] = {};
};

// The ggml_gemmini_args_t of one MUL_MAT, set as the Gemmini backend sets it
// (ops.cpp): I = M, J = N, K, sA, sC, the Q8_HP1 view of src0, the F32 view of
// dst, the layer identity, the residual route of the resolved matmul options
// and the tile counts of the PoTal tile plan (activation_rows_per_stripe =
// its stripe rows). false if a size does not fit or a buffer cannot be allocated.
bool init_args(const ggml_tensor * src0, ggml_tensor * dst, const hp1_shape & s, const char * model_arch,
               const char * layer, ggml::gemmini::residual::ResidualRoute residual_route,
               const potal_tile_plan & plan, ggml_gemmini_args_t & args);

// src1 -> args.A and args.act_quant (ggml::gemmini::quants::quantize_activation,
// the configured algorithm) -> ops.a. false if quantization fails.
bool quantize_activations(const ggml_tensor * src1, ggml_gemmini_args_t & args, hp1_operands & ops);

// args' Q8_HP1 blocks -> ops.own (b / carriers / scale) (quants::weight::hp1::unpack).
// ops must come from allocate(s).
quants::weight::hp1::status unpack_weights(const ggml_gemmini_args_t & args, hp1_operands & ops,
                                           quants::weight::hp1::stats *                   stats,
                                           ggml::gemmini::quants::wreader::WeightReaderStatus * reader = nullptr);

// args' Q8_HP1 blocks -> w (allocated for args.J x args.K), w.stats included.
// The full validation and unpacking of one weight; w is complete only when ok.
quants::weight::hp1::status prepare_weight(const ggml_gemmini_args_t & args, hp1_weight & w,
                                           ggml::gemmini::quants::wreader::WeightReaderStatus * reader = nullptr);

// ---- route -----------------------------------------------------------------

enum class act_route : uint8_t { tensor, token, stripe, exsia, block };
static_assert(static_cast<int>(act_route::tensor) == GGML_POTAL_ROUTE_TENSOR_BLOCK &&
                  static_cast<int>(act_route::block) == GGML_POTAL_ROUTE_BLOCK_BLOCK,
              "act_route indexes ggml_backend_potal_stats::route_completed");

struct route_plan {
    act_route    activation = act_route::tensor;
    const char * name       = nullptr; // "Tensor+block", ...
    bool         k_groups   = false;   // one PoTal GEMM per 32-wide K group (Block+block)
};

// Route of the activation metadata in args (after quantize_activations). false
// if there is none (quantization did not run or failed).
bool select_route(const ggml_gemmini_args_t & args, route_plan & route);

const char * act_route_name(act_route r);

// Executes one PoTal job (the issuer in the backend, an emulator in tests).
using tiled_runner = std::function<tiled_report(const tiled_job &)>;

struct route_result {
    bool         ok         = false;
    bool         fatal      = false;   // hardware state unknown: the staging buffers must be abandoned
    const char * stage      = nullptr; // failing step
    tiled_report last;                 // last PoTal job (plan, status)
    potal_ws_report work;              // all PoTal jobs of this route: counts and guest seconds
    uint64_t     jobs       = 0;       // potal_tiled_matmul_auto calls
    size_t       k_groups   = 0;       // Block+block: K groups executed
};

// The route on staged operands: PoTal GEMM(s) through `run`, then the F32
// reconstruction into args.f_out. The output is complete only when ok.
route_result execute_route(const route_plan & route, const ggml_gemmini_args_t & args, hp1_operands & ops,
                           const tiled_runner & run);

// ---- error compensation (GGML_GEMMINI_ENABLE_RMD) ---------------------------

struct compensation_result {
    bool         ok           = true;
    const char * detail       = nullptr; // failure reason
    size_t       payloads     = 0;       // residual payloads the quantizer produced
    size_t       executed     = 0;       // corrections computed and merged into dst
    size_t       residuals    = 0;       // residual events consumed
    size_t       nonzero      = 0;       // non-zero correction entries merged
};

// After execute_route: the residual correction the quantizer captured, added
// to args.f_out by the original Gemmini cpu_direct code (direct-executor +
// rmd-compose). A ws_packet payload needs the Gemmini WS residual executor,
// which PoTal does not have: it fails, nothing is dropped. With
// GGML_GEMMINI_ENABLE_RMD=0 the quantizers capture nothing and this returns
// ok with zero payloads.
compensation_result apply_compensation(const ggml_gemmini_args_t & args);

// Independent CPU integer reference of the PoTal GEMM (no tiling, K passes of
// POTAL_HW_DIM in order, SCU per pass and column, Sat32 accumulation):
// c[m * ldc + n] for m < M, n < N. block_k is the K extent of one carrier.
void reference_gemm_i32(const int8_t * a, size_t lda, const int8_t * b, size_t ldb, const uint32_t * carriers,
                        size_t cstride, size_t block_k, size_t M, size_t N, size_t K, int32_t * c, size_t ldc);

// ---- shared by the FULL route and the stripe pipeline (matmul/stripe.hpp) -----

size_t align_up(size_t v, size_t to);

// A DMA buffer: `bytes` rounded up to POTAL_HW_MAX_BYTES, aligned to it and
// zeroed; nullptr if out of memory. Free with std::free, except after a fatal
// hardware failure: then leak it (the accelerator may still reach it).
void * dma_alloc(size_t bytes);

// Main reconstruction of one output row (FULL routes with a constant-along-K
// scale and every stripe of the pipeline):
//   out[n * col_stride] = float(double(c[n]) * activation_scale * double(channel_scale[n]))
void reconstruct_row(const int32_t * c, size_t n, double activation_scale, const float * channel_scale, float * out,
                     size_t col_stride);

// Rows [row_begin, row_begin + rows) of the A8 codes in args.A -> dst rows of
// stride lda (K codes each). false if args.A is not the dense A8 I x K buffer
// or the rows are outside it.
bool copy_activation_rows(const ggml_gemmini_args_t & args, size_t row_begin, size_t rows, int8_t * dst, size_t lda);

} // namespace ggml::potal
