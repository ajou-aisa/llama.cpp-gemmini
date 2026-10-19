#pragma once

// Paired Main/Residual GEMM: the software execution model of one F32 x Q8_HP1
// MUL_MAT whose Main GEMM and Residual GEMM share one stationary weight and
// run as alternating micro-tiles (docs/paired-gemm-software-contract.md).
// Nothing here issues a RoCC command: the current PoTal RTL has no paired
// execution, so the only backend is PAIRED_EMULATED (a functional model of
// the SP/ACC/weight/scale state below) and a hardware request fails with
// PAIRED_HARDWARE_UNSUPPORTED before any work. The serialized STRIPE_PIPELINE
// (matmul/stripe.hpp) is untouched and stays the hardware path.
//
//   paired_tiled_matmul_auto
//     validate the request (shape, weight, stripes, theta, every carrier)
//     -> plan: paired_tile_select, or the caller's frozen Main tiles
//        (paired_frozen_plan); t_I / t_J / stripe height never change after it
//     -> per stripe: build_paired_residual_request (residual groups of at most
//        stripe_rows lane rows, indexed into the shared prepared weight)
//     -> common K windows of whole K32 blocks (only the K window shrinks)
//     -> build_paired_schedule + validate_paired_schedule, spill capacity
//     -> execute_paired_emulated, stripe by stripe, in schedule order
//     -> radix INT64 reconstruction, then one merge mode
//     -> commit (outputs are written only when everything succeeded)
//
// Memory model (POTAL_HW_* geometry; DIM-element rows):
//   SP pair p (p = window instance % 2, current/next K window ping-pong):
//     bank 2p      [0, R/2)   Main activation     R = POTAL_HW_BANK_ROWS
//                  [R/2, R)   Residual limbs      (fixed 1:1, never borrowed)
//     bank 2p+1    weight, rear-placed: the window's B ends at the bank end
//                  (LoopMatmul: b_addr_end = (loop + 1) * half, A from the front)
//   ACC bank 0 Main partial sums, ACC bank 1 the resident Residual group's
//   partial sums (POTAL_HW_ACC_ROWS / 2 each: the per-loop half
//   potal_tile_select already uses, not halved again). Other Residual groups
//   are spilled as raw INT32 and restored; the SCU is never applied twice.
//
// Numerics (bit-exact with the serialized path):
//   Main      per output, DIM K passes in K order: C = Sat32(C + SCU(dot, carrier(block, n)))
//   Residual  per output (lane row, n): runs in block order, each run's compact
//             K in fragments of at most DIM (the run-aware compact contract;
//             never zero-expanded to original K), SCU by the run's block carrier,
//             Sat32 in that order. A fragment whose digits are all zero for a
//             group is not executed: SCU(0) = 0 and Sat32(acc + 0) = acc.
//   INT64     R[row][n] = sum C_r[(lane, row)][n] * 256^lane (checked __int128)
//   merge     LEGACY_F32: the pipeline's Main F32 + merge_prepared_hp1 (bitwise
//             the STRIPE_PIPELINE output); INT64: Y = check64(C_main + R),
//             out = float(double(Y) * (double(channel_scale) * double(2^theta))).

#include "dense.hpp"
#include "tiling.hpp"

#include "residual/stripe-residual.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::potal {

namespace prmd = ggml::gemmini::rmd;

// ---- geometry ------------------------------------------------------------------

constexpr size_t paired_dim            = POTAL_HW_DIM;
constexpr size_t paired_sp_banks       = POTAL_HW_BANK_NUM;
constexpr size_t paired_sp_bank_rows   = POTAL_HW_BANK_ROWS;
constexpr size_t paired_sp_pairs       = 2; // current / next K window (LoopMatmul concurrent_loops)
constexpr size_t paired_partition_rows = paired_sp_bank_rows / 2; // Main | Residual activation halves
// PoTalConfigs acc_banks = 2; the generated parameter header carries only
// POTAL_HW_ACC_ROWS (hardware-required: export the bank count).
constexpr size_t paired_acc_banks      = 2;
constexpr size_t paired_acc_bank_rows  = (size_t) POTAL_HW_ACC_ROWS / paired_acc_banks;
constexpr size_t paired_block_k        = 32; // HP1 scale block = one residual run (rmd::kBlockSize)

static_assert(paired_sp_banks == 2 * paired_sp_pairs, "paired layout: one activation and one weight bank per pair");
static_assert(paired_block_k % paired_dim == 0, "a scale block is a whole number of DIM passes");

// SP row addresses of one pair (rows of paired_dim int8).
struct paired_sp_layout {
    size_t main_a_base     = 0;
    size_t residual_a_base = 0;
    size_t weight_bank     = 0; // first row of the weight bank
    size_t weight_end      = 0; // one past the last weight row (rear placement)
};

paired_sp_layout paired_layout(size_t pair);

// ---- plan ------------------------------------------------------------------------

struct paired_tile_plan {
    size_t   m = 0, n = 0, k = 0;
    size_t   tile_I = 0, tile_J = 0;  // DIM tiles; frozen once selected
    size_t   tile_K = 0;              // DIM passes of one common K window
    size_t   requested_tile_K = 0;    // before the common K-window fit
    bool     frozen           = false; // tiles given by the caller (paired_frozen_plan)
    uint32_t passes_per_block = 0;     // paired_block_k / DIM
    size_t   stripe_rows      = 0;     // tile_I * DIM: the ExSIA stripe height
    size_t   stripes = 0, j_groups = 0;
    size_t   window_blocks = 0;        // K32 blocks per window (last may be fewer)
    size_t   k_windows     = 0;
    // Largest window, DIM-element rows, and the capacity each must fit.
    size_t   main_a_rows = 0, residual_a_rows = 0, weight_rows = 0, acc_rows = 0;
    size_t   partition_rows = paired_partition_rows, weight_bank_rows = paired_sp_bank_rows,
             acc_bank_rows  = paired_acc_bank_rows;
    uint32_t scale_slots    = 0; // window blocks * tile_J, shared by both paths
};

// gemmini.h tiled_matmul_auto's start and growth order (J, I, K; K in whole
// scale blocks) with the paired limits: tile_I*tile_K*DIM <= R/2 (Main
// activation), tile_K*tile_J*DIM <= R (weight bank), tile_I*tile_J*DIM <= one
// ACC bank, tile_I*tile_J <= WorkIdsPerLoop, window scale slots <= the scale
// window. A residual group has at most stripe_rows rows and a window at most
// window_blocks runs of at most passes_per_block fragments, so the Residual
// partition (the same size) fits whenever the Main one does.
// POTAL_ERR_INVALID_ARGUMENT for a shape (K a multiple of 32), POTAL_ERR_CAPACITY.
potal_status paired_tile_select(size_t m, size_t n, size_t k, paired_tile_plan & plan);

// Frozen Main tiles (e.g. the STRIPE_PIPELINE plan that fixed the ExSIA stripe
// height): tile_I (limited to the I tiles m needs) and tile_J are kept; only
// the common K window shrinks, in whole K32 blocks, until both paths fit.
// POTAL_ERR_CAPACITY if not even one block fits: never a smaller tile_I/J and
// never a fallback to the serialized GEMMs.
potal_status paired_frozen_plan(size_t m, size_t n, size_t k, size_t tile_I, size_t tile_J, size_t tile_K,
                                paired_tile_plan & plan);

// Rows of stripe s; K blocks [first, end) of window w; columns of J group g.
size_t paired_stripe_rows(const paired_tile_plan & plan, size_t stripe);
void   paired_window_blocks(const paired_tile_plan & plan, size_t window, size_t & first, size_t & end);
size_t paired_group_cols(const paired_tile_plan & plan, size_t j_group);

// ---- residual descriptor (shared, indexed weight) -------------------------------

// The spec's run: one original weight block and its selected local K.
using paired_residual_run = residual::run;

// One DIM-wide compact partial: `count` consecutive compact K of one run.
struct paired_fragment {
    uint32_t run           = 0;
    uint32_t block_id      = 0; // original weight block = carrier / scale block
    uint32_t compact_begin = 0; // compact K position (over all runs) of the first K
    uint32_t local_begin   = 0; // index into the run's local_k
    uint32_t count         = 0; // 1..DIM
};

struct paired_residual_group {
    size_t               first_row = 0; // into rows / limbs
    size_t               rows      = 0; // 1..stripe_rows
    std::vector<uint8_t> nonzero;       // per fragment: some digit of the group's rows is nonzero
};

// No copy of B and no gathered carriers: fragments name original K indices of
// the prepared weight, which both paths read in place.
struct paired_residual_request {
    size_t                                stripe_id = 0, row_begin = 0, row_count = 0, n = 0;
    size_t                                k_compact = 0;
    std::vector<paired_residual_run>      runs;
    std::vector<residual::lane_row>       rows;      // Residual C row i
    std::vector<int8_t>                   limbs;     // [rows][k_compact] radix digits
    std::vector<paired_fragment>          fragments; // run order, compact order
    std::vector<paired_residual_group>    groups;
    const hp1_weight *                    weight = nullptr;

    bool     empty() const { return rows.empty(); }
    // Original weight K of the x-th K of fragment f.
    uint32_t original_k(const paired_fragment & f, size_t x) const;
};

// decode_packet (the run-aware rows/runs of build_request), fragments, groups
// of at most group_rows rows (lane-major order kept), per-group nonzero
// fragments. Transactional: out is reset first and complete only on success.
prmd::RmdStatus build_paired_residual_request(const prmd::StripePacket & packet, const hp1_weight & w,
                                              size_t group_rows, paired_residual_request & out);

// ---- schedule ----------------------------------------------------------------------

enum class paired_path : uint8_t { none, main, residual };

enum class paired_op_kind : uint8_t {
    main_a_load,     // DRAM -> SP Main activation partition of a pair (logical DMA)
    residual_a_load, // DRAM -> SP Residual partition: one group's limbs for one window
    weight_load,     // DRAM -> SP weight bank: the window's shared weight, once
    scale_load,      // the window's carriers (blocks x J tiles) into its scale context
    window_begin,    // pair READY -> ACTIVE; a K-window switch when not the first
    weight_preload,  // stationary weight of the next micro-tile (indexed for Residual)
    main_tile,       // one (K pass, N tile) over the stripe's I tiles
    residual_tile,   // one (fragment, N tile) over the group's I tiles
    mode_switch,     // the next micro-tile is of the other path
    group_init,      // ACC-R <- 0 for a group's first contribution
    group_save,      // ACC-R -> spill, raw INT32
    group_restore,   // spill -> ACC-R, raw INT32
    acc_complete,    // MVOUT of a finished tile (Main stripe tile or Residual group)
    window_end,      // pair ACTIVE -> DRAINED -> EMPTY
};

const char * paired_op_name(paired_op_kind k);
const char * paired_path_name(paired_path p);

constexpr uint32_t paired_none = UINT32_MAX;

struct paired_op {
    paired_op_kind kind     = paired_op_kind::mode_switch;
    paired_path    path     = paired_path::none;
    uint32_t       stripe   = 0;
    uint32_t       j_group  = 0;          // N tile group (the ACC epoch is j_group + 1)
    uint32_t       window   = 0;          // common K window
    uint32_t       pair     = 0;          // SP pair
    uint32_t       group    = paired_none; // Residual group
    uint32_t       block_id = paired_none; // original weight block (carrier identity)
    uint32_t       fragment = paired_none; // Residual compact fragment
    uint32_t       k_begin  = 0;           // Main: original K of the pass
    uint32_t       k_count  = 0;           // K of the micro-tile (Main DIM, Residual the fragment's)
    uint32_t       j        = 0;           // DIM column tile in the J group
    uint32_t       i_tiles  = 0;           // DIM row tiles streamed
    uint32_t       rows = 0, cols = 0;     // valid rows / columns
    uint32_t       sp_a = 0, sp_a_stride = 0; // SP row of row tile 0 and the row-tile stride
    uint32_t       sp_w = 0;               // Main: weight tile row; Residual: the window's weight base
    uint32_t       acc_bank = 0, acc_addr = 0;
    uint32_t       scale_slot = 0;         // window block * J tiles + j
    bool           first = false, last = false; // first / last contribution to this ACC tile
    bool           tail  = false; // beyond the 2 * min(Main, Residual) alternating micro-tiles of its window
};

struct paired_schedule {
    size_t                             stripe = 0;
    std::vector<paired_op>             ops;
    // Residual group order of each window instance (j_group * k_windows + window).
    std::vector<std::vector<uint32_t>> group_order;
};

// Deterministic: the same plan and request always give the same ops.
paired_schedule build_paired_schedule(const paired_tile_plan & plan, size_t stripe,
                                      const paired_residual_request & residual);

// Static checks of a schedule, independent of the emulator: partition bounds
// and overlap, one weight load per window instance, per-output K order and
// first/last flags, strict 1:1 alternation while both paths have work, no
// dummy micro-tile, saves matched by restores. Also the peak spill bytes.
potal_status validate_paired_schedule(const paired_tile_plan & plan, const paired_residual_request & residual,
                                      const paired_schedule & schedule, size_t * peak_spill_bytes,
                                      std::string * why);

std::string paired_op_text(const paired_op & op);

// ---- execution ----------------------------------------------------------------------

enum class paired_backend : uint8_t { emulated, hardware };
enum class paired_execution_backend : uint8_t { none, paired_emulated, paired_hardware_unsupported };
enum class paired_merge_mode : uint8_t { legacy_f32, int64 };
enum class paired_check : uint8_t { not_checked, match, mismatch };

const char * paired_execution_backend_name(paired_execution_backend b);

// Emulated (logical) counts: never hardware cycles or FireSim DMA traffic.
struct paired_counts {
    uint64_t ops                            = 0;
    uint64_t main_micro_tiles               = 0;
    uint64_t residual_micro_tiles           = 0;
    uint64_t mode_switches                  = 0;
    uint64_t main_only_tail_micro_tiles     = 0; // beyond the alternating pairs of their window
    uint64_t residual_only_tail_micro_tiles = 0;
    uint64_t dummy_micro_tiles              = 0; // K = 0 or an all-zero Residual fragment: 0 in a valid schedule
    uint64_t skipped_zero_fragments         = 0; // (group, fragment, N tile) not executed: all digits zero
    uint64_t window_instances               = 0;
    uint64_t shared_weight_windows          = 0; // windows whose one weight load both paths read
    uint64_t logical_weight_loads           = 0;
    uint64_t logical_weight_dma_bytes       = 0;
    uint64_t logical_activation_dma_bytes   = 0; // Main A + Residual limbs
    uint64_t logical_weight_preloads        = 0;
    uint64_t weight_identity_changes        = 0;
    uint64_t indexed_weight_reads           = 0; // SP weight rows read by original K index
    uint64_t scale_loads                    = 0;
    uint64_t residual_group_inits           = 0;
    uint64_t residual_acc_saves             = 0;
    uint64_t residual_acc_restores          = 0;
    uint64_t residual_acc_save_bytes        = 0;
    uint64_t residual_acc_restore_bytes     = 0;
    uint64_t acc_completions                = 0;
    uint64_t main_acc_peak_bytes            = 0;
    uint64_t residual_acc_peak_bytes        = 0;
    uint64_t scratchpad_peak_bytes          = 0;
    uint64_t spill_peak_bytes               = 0;
};

void paired_add(paired_counts & total, const paired_counts & add);

struct paired_stripe_report {
    size_t        stripe_id       = 0;
    size_t        row_begin       = 0;
    size_t        rows            = 0;
    size_t        residual_rows   = 0; // active (lane, row) pairs
    size_t        residual_groups = 0;
    size_t        k_compact       = 0;
    paired_counts counts;
};

struct paired_execution_report {
    potal_status             status            = POTAL_OK;
    const char *             stage             = nullptr; // failing step
    std::string              detail;
    paired_execution_backend execution_backend = paired_execution_backend::none;
    paired_merge_mode        merge             = paired_merge_mode::legacy_f32;
    const char *             traffic_basis     = "emulated"; // every count below is logical
    paired_tile_plan         plan;
    size_t                   stripe_rows = 0, common_k_windows = 0, residual_groups = 0;
    uint64_t                 executed_ops = 0; // emulator ops actually run (0: refused before execution)
    paired_counts            totals;
    std::vector<paired_stripe_report> stripes;
    // Filled by a verifier (host tests); the executor leaves them not_checked.
    paired_check main_int32_match           = paired_check::not_checked;
    paired_check residual_int32_match       = paired_check::not_checked;
    paired_check reconstruction_int64_match = paired_check::not_checked;
};

// Host-test fault injection; nullptr in every real use.
struct paired_faults {
    int64_t fail_at_op    = -1;    // the emulator fails at this op index (over the whole call)
    bool    corrupt_spill = false; // flip one saved INT32 after the first group_save
};

struct paired_stripe_input {
    int16_t                     theta  = INT16_MIN; // ExSIA post-fold theta of the stripe
    const prmd::StripePacket *  packet = nullptr;   // sealed residual packet, nullptr: none
};

struct paired_gemm_desc {
    size_t                      m = 0, n = 0, k = 0;
    const int8_t *              a = nullptr; // Main A8 codes [m][lda]
    size_t                      lda = 0;
    const hp1_weight *          weight = nullptr; // the shared prepared weight
    size_t                      stripe_rows = 0;  // ExSIA stripe height of `stripes`: must equal the plan's
    const paired_stripe_input * stripes      = nullptr;
    size_t                      stripe_count = 0;
    size_t                      tile_I = 0, tile_J = 0, tile_K = 0; // frozen Main tiles, all 0: select
    paired_backend              backend = paired_backend::emulated;
    paired_merge_mode           merge   = paired_merge_mode::legacy_f32;
    size_t                      spill_capacity_bytes = SIZE_MAX; // Residual ACC spill backing granted
    const paired_faults *       faults = nullptr;
};

struct paired_stripe_output {
    std::vector<residual::lane_row> rows;       // Residual C rows
    std::vector<int32_t>            residual_c; // rows.size() x n
    std::vector<int64_t>            correction; // stripe rows x n (empty without residual)
};

struct paired_output {
    std::vector<int32_t>              main_c; // m x n
    std::vector<paired_stripe_output> stripes;
    std::vector<float>                out;    // m x n in the requested merge mode
};

// One stripe's emulated execution in schedule order (Main C rows of the
// stripe, Residual C of its lane rows). op_index counts ops over the call
// (fault injection). Nothing is written to `main_c` / `residual_c` unless
// POTAL_OK; every temporary state is local and released on return.
potal_status execute_paired_emulated(const paired_gemm_desc & desc, const paired_tile_plan & plan,
                                     const paired_residual_request & residual, const paired_schedule & schedule,
                                     std::vector<int32_t> & main_c, std::vector<int32_t> & residual_c,
                                     paired_counts & counts, uint64_t & op_index, std::string & why);

// The whole request (see the top of this file). POTAL_OK: out is complete. On
// failure out is untouched and report.status/stage/detail say why.
potal_status paired_tiled_matmul_auto(const paired_gemm_desc & desc, paired_output & out,
                                      paired_execution_report & report,
                                      std::vector<paired_schedule> * schedules = nullptr);

} // namespace ggml::potal
