// Paired plan and the Residual descriptor over the shared weight (paired.hpp).

#include "paired.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>

namespace ggml::potal {

namespace {

constexpr size_t   D   = paired_dim;
constexpr uint32_t PPB = (uint32_t) (paired_block_k / D);

size_t tiles_of(size_t elements) {
    return (elements + D - 1) / D;
}

// The paired per-window limits (paired.hpp, paired_tile_select).
bool paired_fits(size_t I, size_t J, size_t K) {
    return I * K * D <= paired_partition_rows && K * J * D <= paired_sp_bank_rows && I * J * D <= paired_acc_bank_rows &&
           I * J <= POTAL_WORK_IDS_PER_LOOP && (K / PPB) * J <= scale_window_slots;
}

void fill(size_t m, size_t n, size_t k, size_t tile_I, size_t tile_J, size_t tile_K, paired_tile_plan & plan) {
    plan.m                = m;
    plan.n                = n;
    plan.k                = k;
    plan.tile_I           = tile_I;
    plan.tile_J           = tile_J;
    plan.tile_K           = tile_K;
    plan.passes_per_block = PPB;
    plan.stripe_rows      = tile_I * D;
    plan.stripes          = (m + plan.stripe_rows - 1) / plan.stripe_rows;
    plan.j_groups         = (tiles_of(n) + tile_J - 1) / tile_J;
    plan.window_blocks    = tile_K / PPB;
    plan.k_windows        = (k / paired_block_k + plan.window_blocks - 1) / plan.window_blocks;
    plan.main_a_rows      = tile_I * tile_K * D;
    plan.residual_a_rows  = tile_I * tile_K * D; // bound: stripe_rows rows, <= tile_K fragments
    plan.weight_rows      = tile_K * tile_J * D;
    plan.acc_rows         = tile_I * tile_J * D;
    plan.scale_slots      = (uint32_t) (plan.window_blocks * tile_J);
}

bool shape_ok(size_t m, size_t n, size_t k) {
    return m != 0 && n != 0 && k != 0 && k % paired_block_k == 0;
}

} // namespace

paired_sp_layout paired_layout(size_t pair) {
    paired_sp_layout l;
    l.main_a_base     = (2 * pair) * paired_sp_bank_rows;
    l.residual_a_base = l.main_a_base + paired_partition_rows;
    l.weight_bank     = (2 * pair + 1) * paired_sp_bank_rows;
    l.weight_end      = l.weight_bank + paired_sp_bank_rows;
    return l;
}

potal_status paired_tile_select(size_t m, size_t n, size_t k, paired_tile_plan & plan) {
    plan = {};
    if (!shape_ok(m, n, k)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const size_t i_tiles = tiles_of(m), j_tiles = tiles_of(n), k_tiles = k / D;

    // gemmini.h start: ACC side sqrt(tiles of one bank), K from the activation partition.
    const size_t max_i_j = (size_t) std::sqrt((double) (paired_acc_bank_rows / D));
    const size_t max_k   = (paired_partition_rows / D) / max_i_j;
    size_t       tile_I  = std::min(i_tiles, max_i_j);
    size_t       tile_J  = std::min(j_tiles, max_i_j);
    size_t       tile_K  = std::min(k_tiles, max_k);
    tile_K               = std::max<size_t>(PPB, tile_K - tile_K % PPB);
    while (!paired_fits(tile_I, tile_J, tile_K) && tile_K > PPB) {
        tile_K -= PPB;
    }
    while (!paired_fits(tile_I, tile_J, tile_K) && tile_J > 1) {
        tile_J--;
    }
    while (!paired_fits(tile_I, tile_J, tile_K) && tile_I > 1) {
        tile_I--;
    }
    if (!paired_fits(tile_I, tile_J, tile_K)) {
        return POTAL_ERR_CAPACITY;
    }
    // Grow in gemmini.h order: J, I, K.
    for (bool grew = true; grew;) {
        grew = false;
        if (tile_J + 1 <= j_tiles && paired_fits(tile_I, tile_J + 1, tile_K)) {
            tile_J++;
            grew = true;
        }
        if (tile_I + 1 <= i_tiles && paired_fits(tile_I + 1, tile_J, tile_K)) {
            tile_I++;
            grew = true;
        }
        if (tile_K + PPB <= k_tiles && paired_fits(tile_I, tile_J, tile_K + PPB)) {
            tile_K += PPB;
            grew = true;
        }
    }
    fill(m, n, k, tile_I, tile_J, tile_K, plan);
    plan.requested_tile_K = tile_K;
    return POTAL_OK;
}

potal_status paired_frozen_plan(size_t m, size_t n, size_t k, size_t tile_I, size_t tile_J, size_t tile_K,
                                paired_tile_plan & plan) {
    plan = {};
    if (!shape_ok(m, n, k) || tile_I == 0 || tile_J == 0 || tile_K == 0 || tile_K % PPB != 0 ||
        tile_J > tiles_of(n) || tile_K > k / D) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const size_t tile_I_run = std::min(tile_I, tiles_of(m)); // a plan selected for more rows
    // Only the common K window shrinks, one K32 block at a time.
    size_t window = tile_K;
    while (window > PPB && !paired_fits(tile_I_run, tile_J, window)) {
        window -= PPB;
    }
    if (!paired_fits(tile_I_run, tile_J, window)) {
        return POTAL_ERR_CAPACITY;
    }
    fill(m, n, k, tile_I_run, tile_J, window, plan);
    plan.requested_tile_K = tile_K;
    plan.frozen           = true;
    return POTAL_OK;
}

size_t paired_stripe_rows(const paired_tile_plan & plan, size_t stripe) {
    const size_t begin = stripe * plan.stripe_rows;
    return begin >= plan.m ? 0 : std::min(plan.stripe_rows, plan.m - begin);
}

void paired_window_blocks(const paired_tile_plan & plan, size_t window, size_t & first, size_t & end) {
    const size_t blocks = plan.k / paired_block_k;
    first               = std::min(blocks, window * plan.window_blocks);
    end                 = std::min(blocks, first + plan.window_blocks);
}

size_t paired_group_cols(const paired_tile_plan & plan, size_t j_group) {
    const size_t begin = j_group * plan.tile_J * D;
    return begin >= plan.n ? 0 : std::min(plan.tile_J * D, plan.n - begin);
}

uint32_t paired_residual_request::original_k(const paired_fragment & f, size_t x) const {
    return f.block_id * (uint32_t) paired_block_k + runs[f.run].local_k[f.local_begin + x];
}

prmd::RmdStatus build_paired_residual_request(const prmd::StripePacket & packet, const hp1_weight & w,
                                              size_t group_rows, paired_residual_request & out) try {
    out = paired_residual_request{};
    if (group_rows == 0) {
        return prmd::RmdStatus::invalid_arguments;
    }
    residual::decoded_packet decoded;
    const prmd::RmdStatus    status = residual::decode_packet(packet, w, decoded);
    if (status != prmd::RmdStatus::success) {
        return status;
    }
    paired_residual_request r;
    r.stripe_id = packet.stripe_id;
    r.row_begin = packet.row_begin;
    r.row_count = packet.row_count;
    r.n         = w.n;
    r.k_compact = decoded.k_compact;
    r.runs      = std::move(decoded.runs);
    r.rows      = std::move(decoded.rows);
    r.limbs     = std::move(decoded.digits);
    r.weight    = &w;

    // Fragments: each run's compact K in DIM pieces; no piece spans two runs.
    uint32_t compact = 0;
    for (size_t ri = 0; ri < r.runs.size(); ++ri) {
        const size_t c = r.runs[ri].local_k.size();
        for (size_t f = 0; f < c; f += D) {
            paired_fragment fr;
            fr.run           = (uint32_t) ri;
            fr.block_id      = r.runs[ri].block_id;
            fr.compact_begin = compact + (uint32_t) f;
            fr.local_begin   = (uint32_t) f;
            fr.count         = (uint32_t) std::min(D, c - f);
            r.fragments.push_back(fr);
        }
        compact += (uint32_t) c;
    }

    // Groups: consecutive rows (lane-major order), at most group_rows each.
    for (size_t first = 0; first < r.rows.size(); first += group_rows) {
        paired_residual_group g;
        g.first_row = first;
        g.rows      = std::min(group_rows, r.rows.size() - first);
        g.nonzero.assign(r.fragments.size(), 0);
        for (size_t f = 0; f < r.fragments.size(); ++f) {
            const paired_fragment & fr = r.fragments[f];
            for (size_t row = first; row < first + g.rows && !g.nonzero[f]; ++row) {
                const int8_t * d = r.limbs.data() + row * r.k_compact + fr.compact_begin;
                for (uint32_t x = 0; x < fr.count; ++x) {
                    if (d[x] != 0) {
                        g.nonzero[f] = 1;
                        break;
                    }
                }
            }
        }
        r.groups.push_back(std::move(g));
    }
    out = std::move(r);
    return prmd::RmdStatus::success;
} catch (const std::bad_alloc &) {
    return prmd::RmdStatus::allocation_failure;
} catch (const std::length_error &) {
    return prmd::RmdStatus::overflow;
}

} // namespace ggml::potal
