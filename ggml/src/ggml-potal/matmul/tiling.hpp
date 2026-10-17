#pragma once

// Gemmini-compatible tiled WS matmul on PoTal.
//
//   gemmini.h                      ggml-potal
//   tiled_matmul_auto()       ->   potal_tiled_matmul_auto()
//   tiled_matmul()            ->   potal_tiled_matmul()
//   tiled_matmul_outer()      ->   potal_tiled_matmul_outer()
//   sp_tiled_matmul_ws()      ->   potal_sp_tiled_matmul_ws()
//   gemmini_loop_ws (RoCC)    ->   potal.h: potal_gemm_ws_issue / potal_loop_ws_issue
//
// Argument order, dimension/stride meaning (elements, row-major) and the
// tile_I/J/K meaning (DIM blocks per LOOP_WS) follow gemmini.h. PoTal adds one
// operand, the per-(K scale block, output column) SCU carriers, and the
// session that owns scale generations and counts the issued work.
//
// Supported subset: WS, int8 A/B, int32 C (full_C), no D/bias, no activation,
// identity scale factors, no transpose. Any other request returns
// POTAL_ERR_UNSUPPORTED before a command is issued; it is never ignored.

#include "potal.h"
#include "potal_hw_params.h"

#include <cstddef>
#include <cstdint>

namespace ggml::potal {

// gemmini.h: enum tiled_matmul_type_t {OS, WS, CPU}.
enum potal_tiled_matmul_type_t : int { POTAL_OS = 0, POTAL_WS = 1, POTAL_CPU = 2 };

// gemmini.h activation codes; PoTal implements NO_ACTIVATION only.
constexpr int POTAL_NO_ACTIVATION = 0;

// Scale operand types of the PoTal parameter header (gemmini_potal_params.h).
using potal_scale_t     = int32_t;
using potal_scale_acc_t = int32_t;
using potal_acc_scale_t = float;
// potal.c programs CONFIG_LD with scale 0 (MVIN_SCALE_IDENTITY of the PoTal
// header) and no accumulator scaling: these are the only accepted values.
constexpr potal_scale_t     POTAL_MVIN_SCALE_IDENTITY = 0;
constexpr potal_acc_scale_t POTAL_ACC_SCALE_IDENTITY  = 0;

// carriers[b * stride + n]: SCU carrier of K scale block b (K elements
// [b*block_k, (b+1)*block_k)) for output column n. block_k = scaleBlockK.
struct potal_scale_operand {
    const uint32_t * carriers;
    size_t           stride;
    uint32_t         block_k;
};

// Hardware operations of the leaf. potal_hw_ops() binds them to potal.h;
// host tests bind an RTL-semantics emulator.
struct potal_ws_ops {
    potal_status (*scale_write)(void * ctx, uint32_t slot, uint32_t generation, uint32_t column, uint32_t carrier);
    potal_status (*scale_release)(void * ctx, uint32_t slot, uint32_t generation, uint32_t column);
    potal_status (*gemm_ws_issue)(void * ctx, const potal_gemm_ws * gemm);
    potal_status (*loop_ws_issue)(void * ctx, const potal_loop_ws * loop);
    potal_status (*fence)(void * ctx);
    void *       ctx;
};

const potal_ws_ops & potal_hw_ops();

// Work and failure accounting of one session (cumulative).
struct potal_ws_report {
    potal_status status = POTAL_OK;
    const char * stage  = nullptr; // failing step
    bool         fatal  = false;   // hardware state unknown: no further work may be issued

    uint64_t invocations     = 0; // LOOP_WS invocations issued (both APIs)
    uint64_t validated_gemms = 0; // of which through potal_gemm_ws_issue (validated scope)
    uint64_t fences          = 0; // POTAL_FENCE == POTAL_OK
    uint64_t scale_writes    = 0;
    uint64_t scale_releases  = 0;
    uint32_t last_generation = 0;
    uint32_t max_scale_slots = 0; // largest slot window of one invocation
    double   scale_s         = 0; // guest seconds in SCALE_WRITE/RELEASE
    double   issue_s         = 0;
    double   fence_s         = 0;
};

// Issuer-owned state: one session per accelerator owner thread.
struct potal_ws_session {
    const potal_ws_ops * ops        = nullptr;
    uint32_t             generation = 0; // last generation used, 1..255 after the first invocation
    potal_ws_report      report;
};

// Tile selection of tiled_matmul_auto (WS branch) with the PoTal limits.
struct potal_tile_plan {
    size_t   tile_I = 0, tile_J = 0, tile_K = 0;
    size_t   I0 = 0, J0 = 0, K0 = 0;            // LOOP_WS invocations per dimension
    size_t   pad_I = 0, pad_J = 0, pad_K = 0;   // padding of the last tile, elements
    size_t   spad_rows = 0, max_spad_rows = 0;  // A+B rows of one invocation / per-loop SPAD half
    size_t   acc_rows = 0, max_acc_rows = 0;    // C rows of one invocation / per-loop ACC half
    uint32_t passes_per_block = 0;
    uint32_t scale_slots      = 0;              // scale slots of the largest invocation
};

// Same starting point and growth loop as tiled_matmul_auto (WS, double
// buffered: half of SPAD and of ACC, which is also LoopMatmul's per-loop
// partition). PoTal changes: tile_K starts and grows in passes_per_block steps
// (scale blocks are never split across invocations) and every candidate must
// also fit WorkIdsPerLoop and the scale window. POTAL_ERR_CAPACITY if even
// one block-sized tile does not fit.
potal_status potal_tile_select(size_t dim_I, size_t dim_J, size_t dim_K, uint32_t passes_per_block,
                               potal_tile_plan * plan);

potal_status potal_tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                                     const int8_t * A, const int8_t * B,
                                     const void * D, int32_t * C,
                                     size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                     potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                     potal_scale_acc_t D_scale_factor,
                                     int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                     bool repeating_bias,
                                     bool transpose_A, bool transpose_B,
                                     bool full_C, bool low_D,
                                     uint8_t weightA,
                                     potal_tiled_matmul_type_t tiled_matmul_type,
                                     const potal_scale_operand & scales, potal_ws_session & session,
                                     potal_tile_plan * plan_out = nullptr);

potal_status potal_tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
                                const int8_t * A, const int8_t * B,
                                const void * D, int32_t * C,
                                size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                potal_scale_acc_t D_scale_factor,
                                int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                bool repeating_bias,
                                size_t tile_I, size_t tile_J, size_t tile_K,
                                bool transpose_A, bool transpose_B,
                                bool full_C, bool low_D,
                                uint8_t weightA,
                                potal_tiled_matmul_type_t tiled_matmul_type,
                                const potal_scale_operand & scales, potal_ws_session & session);

potal_status potal_tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
                                      const int8_t * A, const int8_t * B,
                                      const void * D, int32_t * C,
                                      size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                      potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                      potal_scale_acc_t D_scale_factor,
                                      size_t tile_I, size_t tile_J, size_t tile_K,
                                      int act, potal_acc_scale_t scale, potal_acc_scale_t bert_scale,
                                      bool repeating_bias,
                                      bool a_transpose, bool b_transpose,
                                      bool full_C, bool low_D,
                                      uint8_t weightA,
                                      int dataflow,
                                      const potal_scale_operand & scales, potal_ws_session & session);

// One LOOP_WS invocation: SCALE_WRITE every carrier it reads -> issue ->
// POTAL_FENCE -> SCALE_RELEASE. `carriers` points at (first K block of this
// chunk, first column of this J tile group); passes_per_block = block_k / DIM.
// ex_accumulate follows gemmini: !no_bias || D == NULL.
potal_status potal_sp_tiled_matmul_ws(const int8_t * A, const int8_t * B,
                                      const void * D, int32_t * C,
                                      potal_scale_t A_scale_factor, potal_scale_t B_scale_factor,
                                      potal_scale_acc_t D_scale_factor,
                                      size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                      size_t A_row_stride, size_t B_row_stride, size_t D_row_stride,
                                      size_t C_row_stride,
                                      bool a_transpose, bool b_transpose,
                                      bool full_C, bool low_D,
                                      bool no_bias, bool repeating_bias,
                                      int act,
                                      int a_spad_id, int b_spad_id,
                                      const uint32_t * carriers, size_t carrier_stride, uint32_t passes_per_block,
                                      potal_ws_session & session);

} // namespace ggml::potal
