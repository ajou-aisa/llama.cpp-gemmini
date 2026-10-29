#pragma once

// potal_paired_matmul_auto(): one ExSIA stripe's Main GEMM and Residual GEMM as
// one paired execution on the H2 PoTal hardware (generators/potal, branch potal:
// PoTalScaleCommandIngress INDEX_CONFIG / INDEX_WRITE, LoopMatmul paired
// LOOP_WS, ExecuteController indexed PRELOAD; validated by the H2 BareMetalC
// regressions). It replaces nothing of the serialized STRIPE_PIPELINE path and
// never falls back to it: a request it cannot plan is refused with a status.
//
// Stages (docs/potal-paired-matmul-auto.md):
//   H3-0  potal_make_tile_plan        Main-derived I/J, final common K, every capacity  [implemented]
//   H3-1  potal_make_execution_plan   ordered Main / Residual / load / Index / scale / ACC
//                                     operations with explicit dependencies, bank-pair
//                                     residency, and potal_validate_execution_plan  [implemented]
//   H3-2A include/potal.h             the checked C11 command API and the paired session
//   H3-2B potal_paired_matmul_auto    one stripe: plans, residual staging, the plan's ops
//                                     through the paired session, ACC readback, drain  [implemented]
//
// Hardware model the plan is made for (the validated H2 command contract):
//   scratchpad   BANK_NUM banks of BANK_ROWS rows (DIM int8 each) in two physical
//                pairs: pair p = bank 2p (activation) + bank 2p+1 (weight) =
//                LoopMatmul half p, selected by a/b_ex_spad_id = p + 1
//   activation   Main A from the bank start; Residual A behind it at the
//                bank-local offset LOOP_WS rs2[31:3] = tile_I * tile_K tiles
//   weight       the original B panel, rear-placed in the weight bank: sp rows
//                [bank_end - k_tiles * j_tiles * DIM, bank_end), loaded by the
//                Main LOOP_WS and read by Residual through INDEX_CONFIG(bBase =
//                that first row, jTiles, sourceKRows = panel K rows) + INDEX_WRITE
//   ACC          Main -> ACC0 = rows [0, ACC_ROWS/2); Residual -> ACC1 = rows
//                [ACC_ROWS/2, ACC_ROWS) (LOOP_WS rs1[61] paired, rs1[62] residual)
//   index table  one per weight buffer (= pair), POTAL_HW_INDEX_ENTRIES slots; an
//                Indexed LOOP_WS reads slots [0, activeRows) in DIM-wide passes,
//                one scale slot per pass (passes_per_block = 1)
//   scale cache  POTAL_HW_SCALE_ENTRIES slots shared by every LOOP_WS in flight
//
// Address units (never mixed):
//   sp row       physical scratchpad row of DIM int8, 0 .. BANK_NUM * BANK_ROWS - 1
//   tile         DIM-row tile offset or count (LOOP_WS A offset, footprints)
//   original_k   element index into the weight's K, 0 .. k - 1
//   resident_k   original_k - panel.k_begin: the K row inside the resident B
//                panel; Index Table entries and INDEX_CONFIG sourceKRows use it
//   compact k    position in the stripe's compact residual K (decode_packet order)
//
// Residual numerics: a fragment is at most DIM consecutive compact K of one run
// (one original weight block), the DIM partial the serialized K32 lowering also
// forms; the SCU applies the run's block carrier to it and Sat32 accumulates in
// run order. Lane rows stay separate A rows (radix 256^lane is applied by the
// existing INT64 reconstruction on the CPU); nothing is collapsed into one INT8 row.

#include "tiling.hpp"

#include "residual/stripe-residual.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::potal {

static_assert(POTAL_HW_SCALE_ENTRIES == POTAL_SCALE_ENTRIES, "potal.h scale-key encoding assumes the header's scale cache");

// ---- geometry ----------------------------------------------------------------------

// The hardware the plan is checked against. potal_paired_hw_geometry() is the
// validated parameter header; host tests pass smaller values to reach every refusal.
struct potal_paired_geometry {
    size_t dim           = 0; // systolic array / scratchpad row width
    size_t bank_num      = 0;
    size_t bank_rows     = 0; // sp rows per bank
    size_t acc_rows      = 0; // whole accumulator; ACC0 and ACC1 are halves
    size_t index_entries = 0; // POTAL_HW_INDEX_ENTRIES
    size_t scale_entries = 0; // POTAL_HW_SCALE_ENTRIES
    size_t work_ids      = 0; // output tiles one LOOP_WS may own (WorkIdsPerLoop)
    size_t block_k       = 0; // K elements of one SCU scale block (HP1: 32)
};

potal_paired_geometry potal_paired_hw_geometry();

constexpr size_t potal_paired_pairs = 2; // physical bank pairs = Index Tables = weight buffers

// sp rows of one physical pair.
struct potal_paired_pair_layout {
    size_t pair               = 0;
    size_t activation_bank    = 0; // bank index of Main / Residual A
    size_t weight_bank        = 0; // bank index of the resident B panel
    size_t activation_begin   = 0; // first sp row of the activation bank
    size_t activation_end     = 0; // one past its last row
    size_t weight_begin       = 0; // first sp row of the weight bank
    size_t weight_end         = 0; // one past its last row: LoopMatmul b_addr_end of the half
    size_t index_buffer       = 0; // INDEX_CONFIG selects it from bBase: weight buffer id = pair
    size_t spad_id            = 0; // LOOP_WS a/b_ex_spad_id selector = pair + 1
};

potal_paired_pair_layout potal_paired_pair(size_t pair, const potal_paired_geometry & hw = potal_paired_hw_geometry());

// ---- status ------------------------------------------------------------------------

// potal_make_tile_plan() outcomes. Anything but ok is an explicit refusal with a
// diagnostic; the Main I/J tiles are never changed to make a request fit.
enum class potal_paired_status : uint8_t {
    ok,
    invalid_argument,         // shape, Main plan or residual geometry inconsistent
    allocation_failure,       // out of memory while building the plan
    main_acc_capacity,        // tile_I x tile_J output tiles exceed ACC0 or WorkIdsPerLoop
    index_window_capacity,    // the Index Table cannot hold one DIM fragment
    residual_acc_capacity,    // residual lane-row tiles x tile_J exceed ACC1 or WorkIdsPerLoop
    weight_bank_capacity,     // not even one scale block of B with tile_J fits the weight bank
    activation_bank_capacity, // not even one scale block of Main A beside the Residual A fits the activation bank
    scale_window_capacity,    // one Main panel + one Residual window cannot share the scale cache
    invalid_execution_plan,   // potal_validate_execution_plan found a violated ordering / residency / identity rule
    execution_failed,         // a paired-session call failed (report.hw_status / stage say which)
};

const char * potal_paired_status_name(potal_paired_status s);

// ---- residual fragments --------------------------------------------------------------

// `count` (1..DIM) consecutive compact K of one run: one DIM-wide K pass of an
// Indexed LOOP_WS, scaled by carrier(block_id, n). Fragment order = run order =
// block order; original K of its x-th K = block_id * block_k + local_k[local_begin + x].
struct potal_paired_fragment {
    uint32_t run           = 0;
    uint32_t block_id      = 0;
    uint32_t compact_begin = 0; // compact k of its first K
    uint32_t local_begin   = 0; // index into the run's local_k
    uint32_t count         = 0;
};

// Fragments of a decoded packet (residual::decode_packet) with the geometry
// checks potal_make_tile_plan() applies: runs with ascending block ids below
// k / block_k, ascending block-local K, lane rows ascending in (lane, source
// row) below `rows`, k_compact and digits consistent. invalid_argument otherwise.
// When the packet carries its digits, a fragment whose digits are zero in every
// lane row is left out (SCU(0) = 0, Sat32(acc + 0) = acc: no contribution) and
// counted in *skipped; a sealed packet never has one (every selected K has a
// nonzero digit in some lane row).
potal_paired_status potal_paired_fragments(const residual::decoded_packet & residual, size_t rows, size_t k,
                                           const potal_paired_geometry & hw, std::vector<potal_paired_fragment> & out,
                                           std::string * why, size_t * skipped = nullptr);

uint32_t potal_paired_fragment_original_k(const residual::decoded_packet & residual, const potal_paired_fragment & f,
                                          size_t x, const potal_paired_geometry & hw);

// ---- plan ----------------------------------------------------------------------------

// One Index Window of a K panel: the fragments of one Indexed LOOP_WS. Slots
// [0, active_rows) of the pair's Index Table hold resident_k values; the last
// fragment's tail is hardware padding (pad_k), a shorter earlier fragment is
// padded with filler slots (a valid resident_k whose A digits are zero).
// H3-1 adds the entries, the pair, the issue order and the ACC1 first/last flags.
struct potal_paired_index_window {
    size_t index          = 0; // within its panel
    size_t fragment_begin = 0; // into the plan's fragments
    size_t fragment_end   = 0;
    size_t k_tiles        = 0; // DIM passes = fragments
    size_t active_rows    = 0; // Index Table slots used: <= index_entries
    size_t pad_k          = 0; // k_tiles * DIM - active_rows, < DIM
    size_t filler_slots   = 0; // slots inside [0, active_rows) with zero A digits
    size_t a_rows         = 0; // Residual A sp rows of this window: residual_row_tiles * k_tiles * DIM
};

// One original K panel: the K range of one resident B panel, its blocks and
// its share of the residual. Residual K of a panel = the fragments whose block
// lies in it (a block never straddles two panels).
struct potal_paired_k_panel {
    size_t index              = 0;
    size_t k_begin            = 0; // original_k of the panel's first K
    size_t k_rows             = 0; // original K elements = INDEX_CONFIG sourceKRows
    size_t k_tiles            = 0; // DIM tiles of the panel (= Main max_k)
    size_t block_begin        = 0; // scale blocks [block_begin, block_end)
    size_t block_end          = 0;
    size_t weight_rows        = 0; // k_tiles * tile_J * DIM sp rows (largest J group)
    size_t main_a_rows        = 0; // tile_I * k_tiles * DIM sp rows
    size_t fragment_begin     = 0; // residual fragments [fragment_begin, fragment_end)
    size_t fragment_end       = 0;
    size_t residual_k_compact = 0;
    size_t residual_a_rows    = 0; // largest window's a_rows (0: no residual in this panel)
    std::vector<potal_paired_index_window> windows;
};

struct potal_paired_tile_plan {
    // Request.
    size_t   rows = 0, n = 0, k = 0; // stripe rows (A rows of this stripe), columns, original K
    uint32_t passes_per_block = 0;   // block_k / DIM: Main K tiles per scale block
    size_t   i_tiles = 0, j_tiles = 0, k_tiles = 0;

    // Main tiles (DIM tiles): tile_I and tile_J are the Main auto-tiler's (tile_I
    // limited to the I tiles `rows` needs), tile_K_main its K tile.
    size_t tile_I = 0, tile_J = 0, tile_K_main = 0;

    // K limits in DIM tiles, whole scale blocks (0: not even one block).
    size_t tile_K_weight_limit     = 0; // weight bank: tile_K * tile_J * DIM <= bank_rows
    size_t tile_K_activation_limit = 0; // activation bank, Main A beside the reserved Residual A
    size_t tile_K_scale_limit      = 0; // scale cache, a Main panel beside one Residual window
    size_t tile_K_hardware_limit   = 0; // the three above without any residual
    size_t tile_K_residual_limit   = 0; // the three above with this stripe's residual (= hardware without one)
    size_t tile_K                  = 0; // min(tile_K_main, tile_K_residual_limit, tile_K_hardware_limit)

    // Derived Main geometry with the final K: potal_frozen_plan(rows, n, k, tile_I, tile_J, tile_K).
    potal_tile_plan main;
    size_t          j_groups = 0, k_panels = 0; // main.J0, main.K0
    size_t          pad_I = 0, pad_J = 0;       // elements missing from the last I / J tile

    // Footprints of the largest panel / window (sp rows, ACC rows) and the limits they met.
    size_t   main_a_rows              = 0;
    size_t   residual_a_rows          = 0; // reserved for the largest possible window
    size_t   residual_a_offset_tiles  = 0; // tile_I * tile_K: LOOP_WS rs2[31:3] of every Residual LOOP_WS
    size_t   weight_rows              = 0;
    size_t   main_acc_rows            = 0; // tile_I * tile_J * DIM <= acc_rows / 2 (ACC0)
    size_t   residual_acc_rows        = 0; // residual_row_tiles * tile_J * DIM <= acc_rows / 2 (ACC1)
    uint32_t scale_slots_main         = 0; // (tile_K / passes_per_block) * tile_J
    uint32_t scale_slots_residual     = 0; // fragments_per_window * tile_J

    // Residual geometry of the stripe (all zero without a packet).
    bool   residual               = false;
    size_t residual_lane_rows     = 0; // active (lane, source row) rows = Residual A / C rows
    size_t residual_row_tiles     = 0; // ceil(lane rows / DIM) = Residual max_i
    size_t residual_pad_I         = 0;
    size_t residual_k_compact     = 0;
    size_t fragments_per_window   = 0; // min(index_entries / DIM, fragments)
    size_t residual_windows       = 0; // over all panels
    size_t residual_filler_slots  = 0;
    size_t residual_empty_panels  = 0; // panels without residual work: no Indexed LOOP_WS
    size_t residual_zero_fragments = 0; // all-zero fragments left out (digits known)
    std::vector<potal_paired_fragment> fragments; // the executed fragments
    std::vector<potal_paired_k_panel>  panels;
};

// H3-0. The paired plan of one stripe:
//   1. `main` is the existing Main auto-tiler's plan (potal_tile_select, or the
//      frozen STRIPE_PIPELINE plan); its tile_I (limited to the I tiles `rows`
//      needs) and tile_J are kept as they are.
//   2. The stripe's residual (residual::decode_packet output; nullptr: none) is
//      measured: lane rows, fragments, the largest Index Window.
//   3. tile_K = min(tile_K_main, tile_K_residual_limit, tile_K_hardware_limit),
//      whole scale blocks; the Main execution plan is rebuilt with it.
//   4. Original K is partitioned into panels of tile_K tiles; each panel's
//      residual fragments are split into Index Windows of at most
//      index_entries / DIM fragments (never fewer windows than the compact K needs).
// Every capacity is checked in physical units with overflow-safe arithmetic.
// Returns ok with the complete plan, else the first refusal (plan fields
// computed before it stay filled for diagnostics; *why says the numbers).
potal_paired_status potal_make_tile_plan(const potal_tile_plan & main, size_t rows, size_t n, size_t k,
                                         const residual::decoded_packet * residual, potal_paired_tile_plan & plan,
                                         std::string * why, const potal_paired_geometry & hw = potal_paired_hw_geometry());

// resident_k of an original K inside a panel; false when the K is not resident there.
bool potal_paired_resident_k(const potal_paired_k_panel & panel, size_t original_k, size_t * resident_k);

// ---- execution plan (H3-1) ---------------------------------------------------------------
//
// The ordered operations of one stripe. No hardware bitfield: the H3-2 encoder
// maps them onto the validated command forms. A `residency` is one (J group, K
// panel) instance of the original B panel in one bank pair: residency r = j_group
// * k_panels + panel, pair = r % 2 (ping-pong), weight generation = r.
//
// Issue order per residency (K panels ascending inside a J group, J groups ascending):
//   pair_retire(r - 2)     the pair's previous residency: every consumer complete  [if any]
//   scale_program(Main r)  slots [0, blocks * j_tiles), new generation
//   load_main_a(r) load_weight(r) main_gemm(r) [acc_complete(ACC0)]
//   per Index Window w:    index_program(r, w) scale_program(Residual r, w) load_residual_a(r, w)
//                          residual_gemm(r, w) [acc_complete(ACC1)] scale_release(Residual r, w)
//   scale_release(Main r)  after the windows: an accepted SCALE_RELEASE is a drain point
//                          (releaseSafe == fenceDrained), so Main and its first Residual
//                          window share the stream before anything drains
//   ...                    pair_retire of the last two residencies at the end
//
// In H3-2 load_main_a + load_weight + main_gemm are one Main LOOP_WS (its LdA /
// LdB), load_residual_a + residual_gemm one Indexed LOOP_WS (LdA at the Residual
// A offset, no LdB), index_program one INDEX_CONFIG followed by its INDEX_WRITEs
// (indivisible), scale_program / scale_release the SCALE_WRITE / SCALE_RELEASE of
// a slot window, acc_complete the C store of the last contribution.
//
// Dependencies: every op lists the earlier ops it needs. `complete == false`:
// issuing after the dependency in the in-order command stream is enough (the
// hardware orders it: a LOOP_WS after its SCALE_WRITEs, a same-buffer INDEX
// update stalls while that buffer is busy, SCALE_RELEASE is held until older
// work drained). `complete == true`: the dependency must have *completed*
// before this op is issued; H3-2 realizes it with POTAL_FENCE (or a validated
// ordering argument) and never by assuming a submitted command finished.

enum class potal_paired_op_kind : uint8_t {
    load_main_a,     // DRAM -> pair activation bank, tiles [0, tile_I * k_tiles)
    load_weight,     // DRAM -> pair weight bank, rear-placed: once per residency
    scale_program,   // SCALE_WRITE of one scale window
    main_gemm,       // Main LOOP_WS: Main A x resident B -> ACC0 (+=)
    load_residual_a, // DRAM -> pair activation bank at the Residual A offset: one window's packed digits
    index_program,   // INDEX_CONFIG + INDEX_WRITE x active_rows of one window: one transaction
    residual_gemm,   // Indexed LOOP_WS: Residual A x resident B (Index Table) -> ACC1 (+=)
    scale_release,   // SCALE_RELEASE of one scale window
    acc_complete,    // the ACC0 / ACC1 tile of a J group is final (C store); ACC1: reconstruction follows on the CPU
    pair_retire,     // every consumer of the pair's residency completed: the pair may take another panel
};

const char * potal_paired_op_name(potal_paired_op_kind k);

enum class potal_paired_pair_phase : uint8_t { free, loading, resident, retired };

const char * potal_paired_pair_phase_name(potal_paired_pair_phase p);

struct potal_paired_dep {
    size_t op       = 0;
    bool   complete = false; // see above
};

constexpr size_t potal_paired_no_index = SIZE_MAX;

// One Index Table slot of a window: `original_k` of the weight, `resident_k` =
// original_k - panel.k_begin (the INDEX_WRITE value), `compact_k` the column of
// the stripe's compact digits packed into this slot (potal_paired_no_index for a
// filler slot, whose A digits are zero).
struct potal_paired_index_entry {
    size_t slot       = 0;
    size_t fragment   = 0; // into the tile plan's fragments
    size_t original_k = 0;
    size_t resident_k = 0;
    size_t compact_k  = potal_paired_no_index;
    bool   filler     = false;
};

// One scale-cache window: slots [slot_base, slot_base + slots) carrying the
// carriers of `blocks` scale blocks x j_tiles (Main: block = scale block of the
// panel, passes_per_block passes; Residual: block = fragment, one pass each).
struct potal_paired_scale_window {
    size_t index       = 0;
    bool   residual    = false;
    size_t residency   = 0;
    size_t window      = potal_paired_no_index; // Residual: the Index Window
    size_t slot_base   = 0;
    size_t slots       = 0;
    size_t blocks      = 0;
    size_t j_tiles     = 0;
    size_t generation  = 0; // plan-local sequence 1, 2, ...: H3-2 maps it to the session's 8-bit generation
    size_t program_op  = 0, consumer_op = 0, release_op = 0;
};

struct potal_paired_op {
    size_t                 id        = 0;
    potal_paired_op_kind   kind      = potal_paired_op_kind::pair_retire;
    size_t                 j_group   = 0;
    size_t                 panel     = 0;
    size_t                 residency = 0;
    size_t                 pair      = 0;
    size_t                 window    = potal_paired_no_index; // Residual ops: Index Window of the panel
    // Loop geometry (DIM tiles) of a gemm / load.
    size_t                 i_tiles = 0, j_tiles = 0, k_tiles = 0, pad_i = 0, pad_j = 0, pad_k = 0;
    size_t                 a_offset_tiles = 0; // bank-local A offset: 0 (Main), tile_I * tile_K (Residual)
    // sp rows read / written: A region and the resident B region of the pair.
    size_t                 a_sp_begin = 0, a_sp_rows = 0, b_sp_begin = 0, b_sp_rows = 0;
    size_t                 weight_generation = 0; // = residency
    // index_program: the INDEX_CONFIG fields and the entries [entry_begin, entry_end).
    size_t                 index_buffer = 0, index_b_base = 0, index_j_tiles = 0, index_source_k_rows = 0,
                           index_active_rows = 0, entry_begin = 0, entry_end = 0;
    size_t                 same_buffer_predecessor = potal_paired_no_index; // residual_gemm this update waits for (hardware-stalled)
    // Scale window of a scale_program / scale_release / gemm.
    size_t                 scale_window = potal_paired_no_index;
    // Accumulation: ACC bank 0 (Main) / 1 (Residual); epoch = J group; one work ID per output tile.
    size_t                 acc_bank = 0, acc_epoch = 0, work_ids = 0;
    bool                   first = false, last = false; // first / last contribution to this ACC identity
    bool                   reconstruction_required = false; // acc_complete(ACC1): raw INT32 lane rows, CPU radix step
    std::vector<potal_paired_dep> deps;
};

// One (J group, K panel) residency of the original B panel in a pair.
struct potal_paired_residency {
    size_t index = 0, j_group = 0, panel = 0, pair = 0;
    size_t j_tiles = 0, cols = 0;      // J tiles / valid columns of the group
    size_t load_weight_op = 0, main_gemm_op = 0, retire_op = 0;
    size_t first_residual_op = potal_paired_no_index, last_residual_op = potal_paired_no_index;
    size_t windows = 0;                // Index Windows = residual_gemm ops
    size_t consumers = 0;              // main_gemm + residual_gemm ops the retire waits for
};

struct potal_paired_pair_transition {
    size_t                  op   = 0;
    size_t                  pair = 0;
    potal_paired_pair_phase from = potal_paired_pair_phase::free, to = potal_paired_pair_phase::free;
    size_t                  residency = potal_paired_no_index;
};

// Final state of a pair after the plan (every pair ends retired or free).
struct potal_paired_pair_state {
    size_t                  pair      = 0;
    potal_paired_pair_phase phase     = potal_paired_pair_phase::free;
    size_t                  residency = potal_paired_no_index; // last residency
    size_t                  residencies = 0;
    bool                    weight_valid = false, main_a_valid = false, residual_a_valid = false;
    size_t                  index_window = potal_paired_no_index; // last programmed window (op id)
    size_t                  pending_main = 0, pending_residual = 0;
};

struct potal_paired_execution_plan {
    potal_paired_tile_plan                    tile;
    std::vector<residual::lane_row>           residual_rows; // Residual A / C row i = (lane, source row): provenance
    std::vector<potal_paired_op>              ops;
    std::vector<potal_paired_index_entry>     index_entries;
    std::vector<potal_paired_scale_window>    scale_windows;
    std::vector<potal_paired_residency>       residencies;
    std::vector<potal_paired_pair_transition> transitions;
    potal_paired_pair_state                   pairs[potal_paired_pairs];
    // Counts.
    size_t main_gemms = 0, residual_gemms = 0, weight_loads = 0, index_programs = 0, index_writes = 0,
           scale_programs = 0, scale_releases = 0, acc_completes = 0, pair_retires = 0, complete_deps = 0;
};

// H3-1. The execution plan of one stripe from its tile plan (potal_make_tile_plan,
// status ok) and the same decoded packet (nullptr without residual). Pure,
// deterministic; `out` is written only when the plan was built and passed
// potal_validate_execution_plan (invalid_execution_plan otherwise, *why says the rule).
potal_paired_status potal_make_execution_plan(const potal_paired_tile_plan & tile,
                                              const residual::decoded_packet * residual,
                                              potal_paired_execution_plan & out, std::string * why,
                                              const potal_paired_geometry & hw = potal_paired_hw_geometry());

// Static checks of an execution plan, independent of its generator: op ids and
// dependency direction, residency order and pair alternation, pair reuse only
// after a retire whose completion deps cover every consumer, sp regions inside
// their banks without Main / Residual A overlap, INDEX_CONFIG fields against the
// resident panel, entries complete / in order / in range / resident_k =
// original_k - k_begin (and the packet's K when given), same-buffer index
// updates after the previous Indexed gemm and never between a window's program
// and its gemm, scale windows inside the cache and never overlapping while live,
// release after a completed consumer, ACC identities with exactly one first and
// one last in order and an acc_complete after the last, work IDs within
// WorkIdsPerLoop, every Main panel and residual window exactly once, and a drain
// point (an accepted SCALE_RELEASE, or the fence a not yet covered `complete`
// dependency needs) between any two gemms two positions apart in issue order
// (they share one of LoopMatmul's two loop slots and its work IDs). With today's
// layout (every Main window at slot 0, every Residual window behind it) the
// scale-slot reuse rules already force that drain; the rule guards other layouts.
potal_paired_status potal_validate_execution_plan(const potal_paired_execution_plan & plan,
                                                  const residual::decoded_packet * residual, std::string * why,
                                                  const potal_paired_geometry & hw = potal_paired_hw_geometry());

std::string potal_paired_op_text(const potal_paired_op & op);

// ---- execution (H3-2B) -------------------------------------------------------------------
//
// potal_paired_matmul_auto() runs one stripe's Main and Residual GEMMs from a
// validated execution plan through the H3-2A paired session (include/potal.h):
//   LOAD_MAIN_A + LOAD_WEIGHT + MAIN_GEMM  one Main potal_paired_loop_ws: A = the stripe
//                                          codes at the panel's K, B = the prepared weight
//                                          rows of the panel (original layout, ldb), ACC0
//   INDEX_PROGRAM                          potal_paired_index_config + one
//                                          potal_paired_index_write per entry (slot order)
//   LOAD_RESIDUAL_A + RESIDUAL_GEMM        one Indexed potal_paired_loop_ws: A = the window's
//                                          staged digits, B DRAM address 0 (resident panel),
//                                          a_offset_tiles = the plan's Residual A offset, ACC1
//   SCALE_PROGRAM / SCALE_RELEASE          potal_paired_scale_write / _release of the window's
//                                          slots with the next hardware generation (1..255)
//   ACC_COMPLETE                           potal_paired_acc_readback_issue of the epoch's tiles
//                                          into c_main / c_residual (+ the J group's columns)
//   PAIR_RETIRE, `complete` dependencies   satisfied by the drain points already passed
//                                          (accepted SCALE_RELEASE / FENCE), else potal_paired_fence
//   end                                    potal_paired_fence: only then the CPU may read c_*
// Results are raw INT32: Main rows x n in c_main, one row per (lane, source row) of
// plan.residual_rows in c_residual. Reconstruction (reconstruct_row, residual::
// reconstruct_rows) and the checked merge stay with the caller (stripe.cpp).

struct potal_paired_request {
    uint64_t run_id    = 0; // identity, recorded only
    size_t   stripe_id = 0;
    size_t   row_begin = 0;

    size_t                  rows = 0, n = 0, k = 0; // stripe rows, columns, original K
    const potal_tile_plan * main_plan = nullptr;    // the frozen Main plan (tile_I / tile_J / tile_K)
    const int8_t *          a         = nullptr;    // Main A8 codes [rows][lda], DIM-byte aligned
    size_t                  lda       = 0;
    const hp1_weight *      weight    = nullptr;    // prepared, immutable, outlives the call
    const residual::decoded_packet * residual = nullptr; // decode_packet output with digits; nullptr: none
    int32_t *               c_main    = nullptr;    // [rows][ldc_main] INT32, DIM*4-byte aligned, DMA-reachable
    size_t                  ldc_main  = 0;
    int32_t *               c_residual = nullptr;   // [residual lane rows][ldc_residual], same rules
    size_t                  ldc_residual = 0;
    bool                    tlb_flush = true;       // TLB FLUSH before the first DMA (the session must be drained)
    uint32_t *              generation = nullptr;   // in/out: the owner's 8-bit scale generation counter
    const potal_paired_geometry * hw = nullptr;     // nullptr: potal_paired_hw_geometry()
    // The execution plan the caller already made for this request (potal_paired_plan_stripe,
    // e.g. to choose the route): used as is after potal_validate_execution_plan and a check
    // that it describes this request; nullptr: planned here.
    const potal_paired_execution_plan * plan = nullptr;
    // Validation aid (GGML_POTAL_PAIRED_FENCES=CONSERVATIVE): a POTAL_FENCE before every
    // Indexed LOOP_WS, ACC readback and scale release that would otherwise rely on the
    // RTL ordering of the stream (resident-B RAW, ACC RAW, release drain): the H2-validated
    // fence placement, at the cost of the Main / Residual overlap.
    bool                    conservative_fences = false;
    // Explicit DIM32 K32 specialization of the validated H2 schedule. Eligible
    // consecutive-original-block Residual windows read their live Main Scale
    // allocation; their logical release becomes a checked global Fence.
    bool                    ktile_scale_reuse = false;
    bool                    next_k_prefetch = false; // explicit managed-H2 load admission; publication still requires global drain
};

struct potal_paired_counts {
    uint64_t main_loop_ws = 0, residual_loop_ws = 0, index_configs = 0, index_writes = 0, scale_writes = 0,
             scale_releases = 0, readbacks = 0, readback_tiles = 0, fences = 0, drains = 0, tlb_flushes = 0,
             ops = 0;
    uint64_t scale_reused_windows = 0, scale_reused_columns = 0;
    uint64_t prefetch_loads = 0, prefetch_mvins = 0, prefetch_bytes_declared = 0, resident_main_loop_ws = 0;
    double residual_prefetch_issue_s = 0; // Combined CPU API interval; not DMA/compute duration
    // Owner-thread seconds inside the session calls (steady clock, as tiling.cpp's
    // potal_ws_report): SCALE_WRITE / RELEASE, command issue (LOOP_WS, INDEX_*, MVOUT,
    // FLUSH), POTAL_FENCE. A release also waits for the drain it implies.
    double   scale_s = 0, issue_s = 0, fence_s = 0;
};

struct potal_paired_report {
    potal_paired_status status    = potal_paired_status::invalid_argument;
    const char *        stage     = nullptr; // failing step, nullptr when ok
    std::string         detail;
    potal_status        hw_status = POTAL_OK; // status of the failing paired-session call
    bool                issued    = false;    // at least one command reached the session
    bool                fatal     = false;    // work may be outstanding: buffers must not be freed or reused
    bool                cleaned   = false;    // after an answered failure: drained and scale entries released
    uint32_t            last_generation = 0;
    potal_paired_counts counts;
    potal_paired_tile_plan      tile;
    potal_paired_execution_plan plan;
};

// The Residual A staging of one stripe: per Index Window one DMA-ready int8 matrix
// [lane rows][lda] whose column s holds the digits of the window's Index slot s
// (filler slots and the pad_k tail zero), shared by every J group.
class potal_paired_staging {
  public:
    struct window {
        size_t panel = 0, index = 0; // the plan's panel and window
        size_t offset = 0;           // bytes into the buffer
        size_t lda = 0, cols = 0;    // row stride (elements) and valid columns (k_tiles * DIM)
    };
    potal_paired_staging() = default;
    potal_paired_staging(const potal_paired_staging &) = delete;
    potal_paired_staging & operator=(const potal_paired_staging &) = delete;
    ~potal_paired_staging();

    void           abandon(); // after a fatal failure: the accelerator may still read it
    const int8_t * a(size_t window) const { return buffer_ + windows[window].offset; }
    size_t         bytes() const { return bytes_; }

    std::vector<window> windows; // plan order (panel-major)
    size_t              rows = 0;

  private:
    friend potal_paired_status potal_paired_pack_residual(const potal_paired_execution_plan &,
                                                          const residual::decoded_packet &, potal_paired_staging &,
                                                          std::string *, const potal_paired_geometry &);
    int8_t * buffer_ = nullptr;
    size_t   bytes_  = 0;
};

// Packs the decoded digits into `out` as the plan's Index entries order them;
// invalid_argument when the plan has no residual, the packet has no digits or an
// entry lies outside the packet, allocation_failure when out of memory.
potal_paired_status potal_paired_pack_residual(const potal_paired_execution_plan & plan,
                                               const residual::decoded_packet & residual, potal_paired_staging & out,
                                               std::string * why,
                                               const potal_paired_geometry & hw = potal_paired_hw_geometry());

// The tile plan and the validated execution plan of one stripe (H3-0 + H3-1), the
// planning step potal_paired_matmul_auto runs first; callers use it to decide the
// route before any command. Pure.
potal_paired_status potal_paired_plan_stripe(const potal_tile_plan & main, size_t rows, size_t n, size_t k,
                                             const residual::decoded_packet * residual, potal_paired_tile_plan & tile,
                                             potal_paired_execution_plan & plan, std::string * why,
                                             const potal_paired_geometry & hw = potal_paired_hw_geometry());

// The *_CAPACITY refusals of the planner: the stripe's geometry does not fit the paired
// resources (a refusal before any command; the serial route can still run the stripe).
bool potal_paired_capacity_refusal(potal_paired_status s);

// H3-2B. Plans (potal_make_tile_plan, potal_make_execution_plan; or validates req.plan),
// stages the residual, then issues every op through `session` (open, drained, not
// failed) and drains with a final fence. One call executes one Main I group: rows must
// not exceed the Main plan's tile_I * DIM (a stripe; refused otherwise). On ok c_main
// (rows) and c_residual (the plan's residual rows) are complete. On failure nothing
// is retried: report.stage / hw_status say what happened. A failure the hardware
// answered (an error status) is followed by potal_paired_session_cleanup (drain,
// release this call's scale entries): cleaned, not fatal, every buffer free again.
// A failed fence (or a failed cleanup) leaves the state unknown: fatal, the
// staging is abandoned, the caller must keep every buffer alive and must not
// treat the accelerator as idle.
potal_paired_status potal_paired_matmul_auto(const potal_paired_request & req, potal_paired_session * session,
                                             potal_paired_report & report);

// One paired stripe in its own session, the step every accelerator owner runs on its
// thread: potal_paired_session_open (sink: the test double; nullptr: the hardware),
// potal_paired_matmul_auto, potal_paired_session_close. Not fatal: the accelerator is
// free again. Fatal: the session is closed but the accelerator may stay owned (potal.h).
potal_paired_status potal_paired_run_session(const potal_paired_request & req, const potal_paired_sink * sink,
                                             potal_paired_report & report);

} // namespace ggml::potal
