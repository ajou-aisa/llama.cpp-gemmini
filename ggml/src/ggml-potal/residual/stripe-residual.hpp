#pragma once

// PoTal WS residual of one stripe (STRIPE_PIPELINE, GGML_GEMMINI_ENABLE_RMD,
// ws_packet route): the stripe's sealed RMD packet becomes one logical
// Residual GEMM on PoTal, whose INT32 result the CPU recomposes and merges.
//
// Run-aware request (the meaning of Gemmini rmd-run-aware.cpp, rebuilt here
// from the packet alone, without its gemmini.h tiler):
//   runs  the packet's original weight blocks, block id ascending; the K of a
//         run are the block's selected local K, ascending (compact order)
//   rows  every (original lane id, stripe-local source row) with a nonzero
//         digit in any run, lane-major then source-row-major; a run without
//         a digit for that lane/row contributes zeros
//   A_r   [row][run K]       balanced radix-256 digits (read_packet_digit)
//   B_r   [run K][n]         prepared HP1 qs of (block * 32 + local K, n)
//   car_r [run][n]           prepared HP1 carrier of (block, n)
// B_r and the carriers are gathered from the prepared weight (weight_cache),
// never from the GGUF blocks again.
//
// K32 padded lowering: PoTal LOOP_WS selects a scale per scale block with one
// passes_per_block for the whole invocation, so run r gets an execution K
// extent of 32: its c_r compact K first, then 32 - c_r zero K in A_r and B_r.
//   K_compact = sum c_r, K_execution = 32 * runs, scale_block_k = 32,
//   passes_per_block = 32 / DIM (2 at DIM16)
// No DIM-wide partial mixes two runs, the compact order of each run is kept,
// and a padded K or an all-zero pass adds SCU(0) = 0 (Sat32(acc + 0) = acc), so
// every (row, n) accumulates, in run order with Sat32,
//   q = Sat32(HP1_shift(exact INT8 dot of one <= DIM fragment, run carrier))
// exactly as the compact request would. E.g. compact K 31 is one logical GEMM:
// passes of 16 and 15 (+1 zero) K, not two host GEMMs.
//
// Reconstruction (rmd-executor.cpp execute_rmd_stripe_im2p_run_aware):
//   R[source_row][n] = sum over rows of C_r[row][n] * 256^lane    (checked __int128)
// then the INT64 range check. C_r already went through the SCU and Sat32: the
// HP1 exponent is never applied again on the CPU.
//
// Merge (merge_prepared_hp1): the checked merge of Gemmini rmd-compose.cpp
// (merge_rmd_correction_checked), cell for cell, with the stripe's own
// post-fold theta and the prepared weight's channel scales:
//   out = out + float(double(R) * double(channel_scale) * double(2^theta)),
// NaN/Inf and missing metadata failing, all or nothing. The weight's HP1
// metadata is not validated again per stripe: prepare_weight() validated the
// whole weight once (weight_cache), and the prepared weight is immutable.

#include "matmul/dense.hpp"
#include "matmul/tiling.hpp"

#include "quants/act/exsia/exsia-event.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::potal::residual {

namespace rmd = ggml::gemmini::rmd;

// A8 residuals: balanced radix 256 (rmd::balanced_radix_contract(8)).
constexpr uint8_t digit_bits = 8;

struct run {
    uint32_t              block_id = 0;  // original weight block (K / 32)
    std::vector<uint16_t> local_k;       // selected block-local K, ascending
};

struct lane_row {
    uint8_t  lane       = 0; // original radix lane id: contributes 256^lane
    uint32_t source_row = 0; // stripe-local activation row
};

// DMA-ready operands of one Residual GEMM (K32 padded lowering), each buffer
// POTAL_HW_MAX_BYTES aligned with its padding zeroed:
//   a        int8  [m_pad][lda]          A_r
//   b        int8  [k_execution][ldb]    B_r
//   carriers u32   [runs][cstride]
//   c        int32 [m_pad][ldc]          written by the Residual GEMM
// The same lifetime rule as hp1_operands: abandon() after a fatal failure.
class request {
  public:
    request() = default;
    request(const request &) = delete;
    request & operator=(const request &) = delete;
    ~request();

    void abandon(); // leak the buffers on purpose
    void reset();   // free the buffers, forget runs and rows

    // The four buffers for m rows over k_execution with the weight's ldb and
    // carrier stride (n columns); false if out of memory.
    bool allocate(size_t m_rows, size_t n_cols, size_t k_exec, size_t runs_count, size_t weight_ldb,
                  size_t weight_cstride);

    std::vector<run>      runs;
    std::vector<lane_row> rows;      // GEMM row i = rows[i]
    size_t                m = 0;     // rows.size(): active lane rows
    size_t                n = 0;
    size_t                k_compact   = 0;
    size_t                k_execution = 0;

    int8_t *   a        = nullptr;
    int8_t *   b        = nullptr;
    uint32_t * carriers = nullptr;
    int32_t *  c        = nullptr;
    size_t     lda = 0, ldb = 0, cstride = 0, ldc = 0;

    // Selected once, here, after the packet was sealed and before any
    // residual command: potal_tile_select(m, n, k_execution, 32 / DIM).
    potal_tile_plan plan;

  private:
    void * blocks[4] = {};
};

// The run-aware view of one sealed packet, before any lowering: the runs, the
// active (lane, source row) rows and their compact digits
//   digits int8 [rows][k_compact]   run r's compact K at the run's compact offset
// (validate_packet, A8 radix range, shape against the prepared weight w).
// Shared by build_request (K32 padded lowering) and the paired executor
// (matmul/paired.hpp, indexed weight): both see the same rows, runs and order.
struct decoded_packet {
    std::vector<run>      runs;
    std::vector<lane_row> rows;
    size_t                k_compact = 0;
    std::vector<int8_t>   digits;
};

// RmdStatus::success only with at least one active row; a packet without any
// nonzero digit is invalid_packet; unsupported_route: not an A8 packet.
rmd::RmdStatus decode_packet(const rmd::StripePacket & packet, const hp1_weight & w, decoded_packet & out);

// validate_packet, the run-aware rows/runs, the padded operands gathered from
// the prepared weight and the residual plan, into `out` (reset first).
// RmdStatus::success only with a complete request; a packet without any
// nonzero digit is invalid_packet; unsupported_route: not an A8 packet.
rmd::RmdStatus build_request(const rmd::StripePacket & packet, const hp1_weight & w, request & out);

// The Residual GEMM's C (req.c) -> the stripe's BlockScaledInt64Correction,
// row_count x n values (stripe-local rows). Rows must be strictly increasing
// in (lane, source row): a duplicate or missing ordering is invalid_packet.
// Transactional: out is unchanged on failure.
rmd::RmdStatus reconstruct(const rmd::StripePacket & packet, const request & req,
                           rmd::BlockScaledInt64Correction & out);

// reconstruct() on any Residual C: c[i * ldc + col] is the INT32 of rows[i].
rmd::RmdStatus reconstruct_rows(const rmd::StripePacket & packet, const std::vector<lane_row> & rows,
                                const int32_t * c, size_t ldc, size_t n, rmd::BlockScaledInt64Correction & out);

// Adds `correction` (stripe-local rows x weight.n, row-major) to rows
// [row_begin, row_end) of the F32 output out[row * row_stride + col *
// col_stride] (out_rows rows) with the stripe's post-fold theta and
// weight.scale, in merge_rmd_correction_checked's order per cell:
//   domain = double(R) * double(weight.scale[col])
//   scaled = domain * double(std::ldexp(1.0f, theta))
//   delta  = float(scaled), merged = out + delta
// The route-plan and reader checks of the Gemmini merge are replaced by the
// weight's own preparation; what the merge itself checks stays: bounds, sizes
// and strides (invalid_arguments), a non-finite channel scale
// (unsupported_route), theta missing or without a finite positive F32 scale
// (invalid_arguments), any non-finite domain/scaled/delta/merged value
// (overflow; bit tests, valid under -ffast-math). Transactional: every cell is
// staged first; out and *nonzero (raw nonzero corrections) are written only
// on success.
rmd::RmdStatus merge_prepared_hp1(const hp1_weight & weight, size_t row_begin, size_t row_end,
                                  const ggml::gemmini::quants::act::exsia::StripeMetadataSnapshot & metadata,
                                  const rmd::BlockScaledInt64Correction & correction, float * out, size_t out_rows,
                                  size_t row_stride, size_t col_stride, size_t * nonzero);

} // namespace ggml::potal::residual
