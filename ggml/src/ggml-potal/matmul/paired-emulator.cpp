// Functional model of the paired execution (paired.hpp): runs a stripe's
// schedule op by op on a logical Scratchpad (4 banks of DIM-element rows),
// a 2-bank INT32 accumulator, one scale context per SP pair, one stationary
// weight tile and a Residual spill area. Micro-tiles read their operands from
// the SP addresses the schedule names (never from DRAM), so a wrong address,
// a bank in the wrong state or a group mix-up fails or changes the result.
// SCU and Sat32: quants/common/hp1_scu.hpp, the exact reference of PoTalSCU /
// AccumulatorMem the serialized path's goldens use.

#include "paired.hpp"

#include "quants/common/hp1_scu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ggml::potal {

namespace {

namespace hp1s = ggml::gemmini::quants::hp1;

constexpr size_t   D   = paired_dim;
constexpr uint32_t PPB = (uint32_t) (paired_block_k / D);

enum class bank_state : uint8_t { empty, loading, ready, active, drained };
enum region_id : size_t { r_main = 0, r_residual = 1, r_weight = 2, r_count = 3 };

const char * state_name(bank_state s) {
    switch (s) {
        case bank_state::empty: return "EMPTY";
        case bank_state::loading: return "LOADING";
        case bank_state::ready: return "READY";
        case bank_state::active: return "ACTIVE";
        case bank_state::drained: return "DRAINED";
    }
    return "?";
}

struct region {
    bank_state state    = bank_state::empty;
    uint32_t   instance = paired_none;
    uint32_t   group    = paired_none;
    size_t     rows     = 0; // SP rows in use
};

struct scale_context {
    bool                  valid    = false;
    uint32_t              instance = paired_none;
    std::vector<uint32_t> carriers; // [slot][D]
};

struct spill_slot {
    bool                 valid = false;
    uint32_t             epoch = 0;
    uint64_t             checksum = 0;
    std::vector<int32_t> data; // [rows][cols]
};

uint64_t checksum_of(const std::vector<int32_t> & v) {
    uint64_t h = 1469598103934665603ull; // FNV-1a over the raw INT32 bits
    for (int32_t x : v) {
        uint32_t u = (uint32_t) x;
        for (int b = 0; b < 4; ++b) {
            h = (h ^ (u & 0xffu)) * 1099511628211ull;
            u >>= 8;
        }
    }
    return h;
}

uint32_t tiles_of(size_t elements) {
    return (uint32_t) ((elements + D - 1) / D);
}

class machine {
  public:
    machine(const paired_gemm_desc & desc, const paired_tile_plan & plan, const paired_residual_request & res,
            size_t stripe, std::vector<int32_t> & main_c, std::vector<int32_t> & residual_c, paired_counts & counts,
            std::string & why) :
        desc_(desc), plan_(plan), res_(res), stripe_(stripe), main_c_(main_c), residual_c_(residual_c), n_(counts),
        why_(why), sp_(paired_sp_banks * paired_sp_bank_rows * D, 0),
        acc_(paired_acc_banks * paired_acc_bank_rows * D, 0), spill_(res.groups.size()),
        main_done_(plan.j_groups, 0), res_done_(plan.j_groups, std::vector<uint8_t>(res.groups.size(), 0)) {
        row_begin_ = stripe * plan.stripe_rows;
        rows_      = paired_stripe_rows(plan, stripe);
    }

    potal_status step(const paired_op & op);
    potal_status finish();

  private:
    potal_status bad(const std::string & what, const paired_op & op) {
        why_ = what + ": " + paired_op_text(op);
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    potal_status move(region & r, bank_state from, bank_state to, const paired_op & op) {
        if (r.state != from) {
            return bad(std::string("bank ") + state_name(r.state) + " -> " + state_name(to) + " (needs " +
                           state_name(from) + ")",
                       op);
        }
        r.state = to;
        return POTAL_OK;
    }
    uint32_t instance_of(const paired_op & op) const { return op.j_group * (uint32_t) plan_.k_windows + op.window; }
    uint32_t epoch_of(const paired_op & op) const { return op.j_group + 1; }
    size_t   col0(const paired_op & op) const { return (size_t) op.j_group * plan_.tile_J * D; }
    void     window_k(const paired_op & op, size_t & k0, size_t & k1) const {
        size_t b0 = 0, b1 = 0;
        paired_window_blocks(plan_, op.window, b0, b1);
        k0 = b0 * paired_block_k;
        k1 = b1 * paired_block_k;
    }
    void track_sp() {
        uint64_t bytes = 0;
        for (auto & pair : regions_) {
            for (const region & r : pair) {
                bytes += r.state == bank_state::empty ? 0 : r.rows * D;
            }
        }
        n_.scratchpad_peak_bytes = std::max(n_.scratchpad_peak_bytes, bytes);
    }

    potal_status load(const paired_op & op, region_id id);
    potal_status preload(const paired_op & op);
    potal_status tile(const paired_op & op);
    potal_status group(const paired_op & op);
    potal_status complete(const paired_op & op);

    const paired_gemm_desc &        desc_;
    const paired_tile_plan &        plan_;
    const paired_residual_request & res_;
    const size_t                    stripe_;
    std::vector<int32_t> &          main_c_;
    std::vector<int32_t> &          residual_c_;
    paired_counts &                 n_;
    std::string &                   why_;
    size_t                          row_begin_ = 0, rows_ = 0;

    std::vector<int8_t>  sp_;  // [row][D]
    std::vector<int32_t> acc_; // [row][D]; bank b rows [b * acc_bank_rows, (b + 1) * acc_bank_rows)
    region               regions_[paired_sp_pairs][r_count];
    scale_context        scale_[paired_sp_pairs];
    int                  active_pair_ = -1;

    int32_t     stationary_[D][D] = {};
    bool        preloaded_        = false;
    paired_op   preload_op_;
    bool        have_identity_    = false;
    paired_op   identity_;

    bool     main_valid_ = false;
    uint32_t main_epoch_ = 0;
    bool     res_valid_ = false, res_dirty_ = false;
    uint32_t res_group_ = paired_none, res_epoch_ = 0;
    size_t   spill_bytes_ = 0;
    bool     corrupted_   = false;

    std::vector<spill_slot>           spill_;
    std::vector<uint8_t>              main_done_;
    std::vector<std::vector<uint8_t>> res_done_;
};

potal_status machine::load(const paired_op & op, region_id id) {
    region & r   = regions_[op.pair][id];
    const bool reload = id == r_residual && r.state == bank_state::active && active_pair_ == (int) op.pair;
    if (reload) { // group switch inside the running window: the previous group's limbs are done
        if (move(r, bank_state::active, bank_state::drained, op) != POTAL_OK ||
            move(r, bank_state::drained, bank_state::empty, op) != POTAL_OK) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
    } else if (active_pair_ == (int) op.pair) {
        return bad("load into the ACTIVE pair", op);
    }
    if (move(r, bank_state::empty, bank_state::loading, op) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const paired_sp_layout l = paired_layout(op.pair);
    const hp1_weight &     w = *desc_.weight;
    size_t                 k0 = 0, k1 = 0;
    window_k(op, k0, k1);
    if (id == r_main) {
        r.rows = (size_t) op.i_tiles * op.sp_a_stride;
        if (op.sp_a != l.main_a_base || r.rows > paired_partition_rows || op.k_begin != k0 || op.k_count != k1 - k0) {
            return bad("Main activation load outside its partition / window", op);
        }
        const size_t passes = op.k_count / D;
        for (size_t i = 0; i < op.i_tiles; ++i) {
            for (size_t p = 0; p < passes; ++p) {
                for (size_t rr = 0; rr < D; ++rr) {
                    int8_t *     dst = sp_.data() + (op.sp_a + (i * passes + p) * D + rr) * D;
                    const size_t row = i * D + rr;
                    if (row < rows_) {
                        std::memcpy(dst, desc_.a + (row_begin_ + row) * desc_.lda + k0 + p * D, D);
                    } else {
                        std::memset(dst, 0, D);
                    }
                }
            }
        }
        n_.logical_activation_dma_bytes += (uint64_t) rows_ * op.k_count;
    } else if (id == r_residual) {
        const paired_residual_group & g = res_.groups.at(op.group);
        r.rows                          = (size_t) op.i_tiles * op.sp_a_stride;
        if (op.sp_a != l.residual_a_base || r.rows > paired_partition_rows || op.rows != g.rows) {
            return bad("Residual limbs outside their partition", op);
        }
        uint64_t k_bytes = 0;
        for (size_t x = 0; x < op.k_count; ++x) {
            const paired_fragment & fr = res_.fragments.at(op.fragment + x);
            if ((size_t) fr.block_id * paired_block_k < k0 || (size_t) fr.block_id * paired_block_k >= k1) {
                return bad("Residual fragment outside the window", op);
            }
            k_bytes += fr.count;
            for (size_t i = 0; i < op.i_tiles; ++i) {
                for (size_t rr = 0; rr < D; ++rr) {
                    int8_t *     dst = sp_.data() + (op.sp_a + (i * op.k_count + x) * D + rr) * D;
                    const size_t row = i * D + rr;
                    std::memset(dst, 0, D);
                    if (row < g.rows) {
                        std::memcpy(dst, res_.limbs.data() + (g.first_row + row) * res_.k_compact + fr.compact_begin,
                                    fr.count);
                    }
                }
            }
        }
        r.group = op.group;
        n_.logical_activation_dma_bytes += (uint64_t) g.rows * k_bytes;
    } else {
        const size_t j_run  = tiles_of(op.cols);
        const size_t passes = op.k_count / D;
        r.rows              = passes * j_run * D;
        if (op.sp_w < l.weight_bank || op.sp_w + r.rows != l.weight_end || op.k_begin != k0 ||
            op.k_count != k1 - k0) {
            return bad("weight not rear-placed in its bank", op);
        }
        for (size_t p = 0; p < passes; ++p) {
            for (size_t j = 0; j < j_run; ++j) {
                for (size_t kk = 0; kk < D; ++kk) {
                    int8_t *     dst = sp_.data() + (op.sp_w + (p * j_run + j) * D + kk) * D;
                    const size_t c0  = j * D;
                    const size_t cn  = std::min(D, op.cols - c0);
                    std::memset(dst, 0, D);
                    std::memcpy(dst, w.b + (k0 + p * D + kk) * w.ldb + col0(op) + c0, cn);
                }
            }
        }
        n_.logical_weight_loads++;
        n_.logical_weight_dma_bytes += (uint64_t) op.k_count * op.cols;
    }
    r.instance = instance_of(op);
    if (move(r, bank_state::loading, bank_state::ready, op) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (reload && move(r, bank_state::ready, bank_state::active, op) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    track_sp();
    return POTAL_OK;
}

potal_status machine::preload(const paired_op & op) {
    const region & wr = regions_[op.pair][r_weight];
    if (active_pair_ != (int) op.pair || wr.state != bank_state::active || wr.instance != instance_of(op)) {
        return bad("preload from a weight bank that is not the active window's", op);
    }
    std::memset(stationary_, 0, sizeof(stationary_));
    const size_t j_run = tiles_of(paired_group_cols(plan_, op.j_group));
    const size_t w_end = paired_layout(op.pair).weight_end;
    if (op.path == paired_path::main) {
        if (op.sp_w < w_end - wr.rows || op.sp_w + D > w_end) {
            return bad("Main weight tile outside the resident window", op);
        }
        for (size_t kk = 0; kk < D; ++kk) {
            for (size_t c = 0; c < D; ++c) {
                stationary_[kk][c] = sp_[(op.sp_w + kk) * D + c];
            }
        }
    } else {
        // Indexed preload: compact K -> original K -> resident weight row.
        size_t k0 = 0, k1 = 0;
        window_k(op, k0, k1);
        const paired_fragment & fr = res_.fragments.at(op.fragment);
        for (size_t x = 0; x < fr.count; ++x) {
            const size_t k = res_.original_k(fr, x);
            if (k < k0 || k >= k1 || k / paired_block_k != fr.block_id) {
                return bad("indexed weight read outside the resident window", op);
            }
            const size_t rel = k - k0;
            const size_t row = op.sp_w + ((rel / D) * j_run + op.j) * D + rel % D;
            if (row < w_end - wr.rows || row >= w_end) {
                return bad("indexed weight row outside the weight bank", op);
            }
            for (size_t c = 0; c < D; ++c) {
                stationary_[x][c] = sp_[row * D + c];
            }
        }
        n_.indexed_weight_reads += fr.count;
    }
    const bool same = have_identity_ && identity_.path == op.path && identity_.k_begin == op.k_begin &&
                      identity_.fragment == op.fragment && identity_.j == op.j && identity_.j_group == op.j_group &&
                      identity_.window == op.window;
    n_.weight_identity_changes += same ? 0 : 1;
    identity_      = op;
    have_identity_ = true;
    preload_op_    = op;
    preloaded_     = true;
    n_.logical_weight_preloads++;
    return POTAL_OK;
}

potal_status machine::tile(const paired_op & op) {
    const bool is_main = op.kind == paired_op_kind::main_tile;
    if (!preloaded_ || preload_op_.path != op.path || preload_op_.fragment != op.fragment ||
        preload_op_.k_begin != op.k_begin || preload_op_.j != op.j) {
        return bad("micro-tile without its stationary weight", op);
    }
    preloaded_            = false;
    const region & ar     = regions_[op.pair][is_main ? r_main : r_residual];
    const auto &   sc     = scale_[op.pair];
    const size_t   j_run  = tiles_of(paired_group_cols(plan_, op.j_group));
    if (active_pair_ != (int) op.pair || ar.state != bank_state::active || ar.instance != instance_of(op) ||
        !sc.valid || sc.instance != instance_of(op)) {
        return bad("micro-tile on a pair / partition that is not active", op);
    }
    if (op.k_count == 0) {
        n_.dummy_micro_tiles++;
        return bad("dummy micro-tile", op);
    }
    size_t acc_base = 0;
    if (is_main) {
        if (op.first) {
            main_valid_ = true;
            main_epoch_ = epoch_of(op);
        } else if (!main_valid_ || main_epoch_ != epoch_of(op)) {
            return bad("Main ACC tile not resident", op);
        }
        n_.main_micro_tiles++;
        n_.main_only_tail_micro_tiles += op.tail;
        n_.main_acc_peak_bytes =
            std::max<uint64_t>(n_.main_acc_peak_bytes, (uint64_t) op.i_tiles * j_run * D * D * sizeof(int32_t));
    } else {
        if (ar.group != op.group || !res_valid_ || res_group_ != op.group || res_epoch_ != epoch_of(op)) {
            return bad("Residual group not resident (limbs or ACC-R)", op);
        }
        if (!res_.groups[op.group].nonzero[op.fragment]) {
            n_.dummy_micro_tiles++;
            return bad("all-zero Residual micro-tile", op);
        }
        res_dirty_ = true;
        acc_base   = paired_acc_bank_rows;
        n_.residual_micro_tiles++;
        n_.residual_only_tail_micro_tiles += op.tail;
        n_.residual_acc_peak_bytes =
            std::max<uint64_t>(n_.residual_acc_peak_bytes, (uint64_t) op.i_tiles * j_run * D * D * sizeof(int32_t));
    }
    if (op.acc_addr != acc_base + op.j * D || op.scale_slot * D + D > sc.carriers.size()) {
        return bad("ACC / scale address", op);
    }
    for (size_t i = 0; i < op.i_tiles; ++i) {
        for (size_t r = 0; r < D && i * D + r < op.rows; ++r) {
            const int8_t * a   = sp_.data() + (op.sp_a + i * op.sp_a_stride + r) * D;
            int32_t *      acc = acc_.data() + (op.acc_addr + i * j_run * D + r) * D;
            for (size_t c = 0; c < op.cols; ++c) {
                int32_t partial = 0;
                for (size_t kk = 0; kk < D; ++kk) {
                    partial += (int32_t) a[kk] * stationary_[kk][c];
                }
                const int32_t q = hp1s::apply_validated(partial, sc.carriers[op.scale_slot * D + c]);
                acc[c]          = op.first ? q : hp1s::accumulate(acc[c], q);
            }
        }
    }
    return POTAL_OK;
}

potal_status machine::group(const paired_op & op) {
    if (op.group >= res_.groups.size()) {
        return bad("unknown group", op);
    }
    const size_t j_run = tiles_of(op.cols);
    int32_t *    acc   = acc_.data() + paired_acc_bank_rows * D;
    // ACC-R tile layout: row tile i, N tile j at (i * j_run + j) * D rows; (row, col) inside.
    const auto cell = [&](size_t row, size_t col) -> int32_t & {
        return acc[(((row / D) * j_run + col / D) * D + row % D) * D + col % D];
    };
    spill_slot & slot  = spill_[op.group];
    const size_t bytes = (size_t) op.rows * op.cols * sizeof(int32_t);
    switch (op.kind) {
        case paired_op_kind::group_init:
            if (res_valid_) {
                return bad("ACC-R still holds a group", op);
            }
            for (size_t row = 0; row < (size_t) op.i_tiles * D; ++row) {
                for (size_t col = 0; col < j_run * D; ++col) {
                    cell(row, col) = 0;
                }
            }
            n_.residual_group_inits++;
            break;
        case paired_op_kind::group_save:
            if (!res_valid_ || res_group_ != op.group || res_epoch_ != epoch_of(op) || slot.valid) {
                return bad("save of a group that is not resident", op);
            }
            if (!res_dirty_) {
                return bad("save of an unmodified group (unnecessary transfer)", op);
            }
            if (spill_bytes_ + bytes > desc_.spill_capacity_bytes) {
                why_ = "spill backing exhausted: " + paired_op_text(op);
                return POTAL_ERR_CAPACITY;
            }
            slot.data.resize(op.rows * op.cols);
            for (size_t row = 0; row < op.rows; ++row) {
                for (size_t col = 0; col < op.cols; ++col) {
                    slot.data[row * op.cols + col] = cell(row, col);
                }
            }
            slot.valid    = true;
            slot.epoch    = epoch_of(op);
            slot.checksum = checksum_of(slot.data);
            if (desc_.faults != nullptr && desc_.faults->corrupt_spill && !corrupted_) {
                slot.data[0] ^= 1; // the backing changes behind the executor's back
                corrupted_ = true;
            }
            spill_bytes_ += bytes;
            n_.spill_peak_bytes = std::max<uint64_t>(n_.spill_peak_bytes, spill_bytes_);
            res_valid_          = false;
            n_.residual_acc_saves++;
            n_.residual_acc_save_bytes += bytes;
            return POTAL_OK;
        case paired_op_kind::group_restore:
            if (res_valid_) {
                return bad("ACC-R still holds a group", op);
            }
            if (!slot.valid || slot.epoch != epoch_of(op) || slot.data.size() != (size_t) op.rows * op.cols) {
                return bad("restore without a saved partial sum of this epoch", op);
            }
            if (checksum_of(slot.data) != slot.checksum) {
                why_ = "spill backing corrupted: " + paired_op_text(op);
                return POTAL_ERR_BAD_RESPONSE;
            }
            for (size_t row = 0; row < op.rows; ++row) {
                for (size_t col = 0; col < op.cols; ++col) {
                    cell(row, col) = slot.data[row * op.cols + col];
                }
            }
            slot.valid = false;
            spill_bytes_ -= bytes;
            n_.residual_acc_restores++;
            n_.residual_acc_restore_bytes += bytes;
            break;
        case paired_op_kind::acc_complete: {
            if (!res_valid_ || res_group_ != op.group || res_epoch_ != epoch_of(op)) {
                return bad("completion of a group that is not resident", op);
            }
            const paired_residual_group & g = res_.groups[op.group];
            for (size_t row = 0; row < g.rows; ++row) {
                for (size_t col = 0; col < op.cols; ++col) {
                    residual_c_[(g.first_row + row) * res_.n + col0(op) + col] = cell(row, col);
                }
            }
            res_done_[op.j_group][op.group] = 1;
            res_valid_                      = false;
            n_.acc_completions++;
            return POTAL_OK;
        }
        default:
            return bad("not a group op", op);
    }
    res_valid_ = true;
    res_dirty_ = false;
    res_group_ = op.group;
    res_epoch_ = epoch_of(op);
    return POTAL_OK;
}

potal_status machine::complete(const paired_op & op) {
    if (op.path == paired_path::residual) {
        return group(op);
    }
    if (!main_valid_ || main_epoch_ != epoch_of(op)) {
        return bad("completion of a Main tile that is not resident", op);
    }
    const size_t j_run = tiles_of(op.cols);
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < op.cols; ++col) {
            const int32_t * acc = acc_.data() + (((row / D) * j_run + col / D) * D + row % D) * D + col % D;
            main_c_[row * plan_.n + col0(op) + col] = *acc;
        }
    }
    main_done_[op.j_group] = 1;
    main_valid_            = false;
    n_.acc_completions++;
    return POTAL_OK;
}

potal_status machine::step(const paired_op & op) {
    n_.ops++;
    switch (op.kind) {
        case paired_op_kind::main_a_load: return load(op, r_main);
        case paired_op_kind::residual_a_load: return load(op, r_residual);
        case paired_op_kind::weight_load: return load(op, r_weight);
        case paired_op_kind::scale_load: {
            if (active_pair_ == (int) op.pair) {
                return bad("scale context of the ACTIVE pair rewritten", op);
            }
            const size_t   j_run = tiles_of(op.cols);
            const hp1_weight & w = *desc_.weight;
            if ((size_t) op.k_count * j_run > scale_window_slots) {
                return bad("scale window exceeded", op);
            }
            scale_context & sc = scale_[op.pair];
            sc.carriers.assign((size_t) op.k_count * j_run * D, POTAL_CARRIER_ZERO);
            for (size_t b = 0; b < op.k_count; ++b) {
                for (size_t col = 0; col < op.cols; ++col) {
                    sc.carriers[(b * j_run + col / D) * D + col % D] =
                        w.carriers[(op.block_id + b) * w.cstride + col0(op) + col];
                }
            }
            sc.valid    = true;
            sc.instance = instance_of(op);
            n_.scale_loads++;
            return POTAL_OK;
        }
        case paired_op_kind::window_begin: {
            if (active_pair_ >= 0) {
                return bad("window_begin while another window is active", op);
            }
            region * rr = regions_[op.pair];
            for (size_t id = 0; id < r_count; ++id) {
                const bool optional = id == r_residual && rr[id].state == bank_state::empty;
                if (optional) {
                    continue; // no Residual work in this window
                }
                if (rr[id].instance != instance_of(op) ||
                    move(rr[id], bank_state::ready, bank_state::active, op) != POTAL_OK) {
                    return bad("window_begin on a pair that is not loaded for it", op);
                }
            }
            if (!scale_[op.pair].valid || scale_[op.pair].instance != instance_of(op)) {
                return bad("window_begin without its scale context", op);
            }
            active_pair_ = (int) op.pair;
            n_.window_instances++;
            return POTAL_OK;
        }
        case paired_op_kind::weight_preload: return preload(op);
        case paired_op_kind::main_tile:
        case paired_op_kind::residual_tile: return tile(op);
        case paired_op_kind::mode_switch: n_.mode_switches++; return POTAL_OK;
        case paired_op_kind::group_init:
        case paired_op_kind::group_save:
        case paired_op_kind::group_restore: return group(op);
        case paired_op_kind::acc_complete: return complete(op);
        case paired_op_kind::window_end: {
            if (active_pair_ != (int) op.pair) {
                return bad("window_end of a pair that is not active", op);
            }
            region * rr          = regions_[op.pair];
            bool     residual_rd = false;
            for (size_t id = 0; id < r_count; ++id) {
                if (rr[id].state == bank_state::empty && id == r_residual) {
                    continue;
                }
                residual_rd = residual_rd || id == r_residual;
                if (move(rr[id], bank_state::active, bank_state::drained, op) != POTAL_OK ||
                    move(rr[id], bank_state::drained, bank_state::empty, op) != POTAL_OK) {
                    return POTAL_ERR_INVALID_ARGUMENT;
                }
                rr[id].instance = paired_none;
                rr[id].group    = paired_none;
            }
            n_.shared_weight_windows += residual_rd ? 1 : 0;
            scale_[op.pair].valid = false;
            active_pair_          = -1;
            return POTAL_OK;
        }
    }
    return bad("unknown op", op);
}

potal_status machine::finish() {
    paired_op none;
    none.stripe = (uint32_t) stripe_;
    if (active_pair_ >= 0 || main_valid_ || res_valid_ || spill_bytes_ != 0) {
        return bad("stripe ended with live SP / ACC / spill state", none);
    }
    for (const auto & pair : regions_) {
        for (const region & r : pair) {
            if (r.state != bank_state::empty) {
                return bad("stripe ended with a loaded SP partition", none);
            }
        }
    }
    for (size_t jg = 0; jg < plan_.j_groups; ++jg) {
        if (!main_done_[jg] || std::find(res_done_[jg].begin(), res_done_[jg].end(), 0) != res_done_[jg].end()) {
            return bad("stripe ended with an output tile not completed", none);
        }
    }
    return POTAL_OK;
}

} // namespace

potal_status execute_paired_emulated(const paired_gemm_desc & desc, const paired_tile_plan & plan,
                                     const paired_residual_request & residual, const paired_schedule & schedule,
                                     std::vector<int32_t> & main_c, std::vector<int32_t> & residual_c,
                                     paired_counts & counts, uint64_t & op_index, std::string & why) {
    // Private results: the caller's buffers change only on success.
    std::vector<int32_t> staged_main(paired_stripe_rows(plan, schedule.stripe) * plan.n, 0);
    std::vector<int32_t> staged_res(residual.rows.size() * residual.n, 0);
    paired_counts        n;
    potal_status         status = POTAL_OK;
    {
        machine m(desc, plan, residual, schedule.stripe, staged_main, staged_res, n, why);
        for (const paired_op & op : schedule.ops) {
            if (desc.faults != nullptr && (int64_t) op_index == desc.faults->fail_at_op) {
                why = "injected emulator failure: " + paired_op_text(op);
                return POTAL_HW_BAD_COMMAND;
            }
            status = m.step(op);
            if (status != POTAL_OK) {
                return status;
            }
            ++op_index;
        }
        status = m.finish();
    }
    if (status != POTAL_OK) {
        return status;
    }
    main_c     = std::move(staged_main);
    residual_c = std::move(staged_res);
    counts     = n;
    return POTAL_OK;
}

} // namespace ggml::potal
