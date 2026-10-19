// paired_tiled_matmul_auto: validation -> plan -> Residual descriptors ->
// schedules -> emulated execution -> radix INT64 -> merge -> commit (paired.hpp).

#include "paired.hpp"

#include "quants/act/exsia/exsia-event.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace ggml::potal {

namespace {

namespace exsia = ggml::gemmini::quants::act::exsia;

// Finite tests on the bits (valid under -ffast-math), as stripe-residual.cpp.
bool finite_bits(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint64_t observed = bits;
    return (observed & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

bool finite_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint32_t observed = bits;
    return (observed & 0x7f800000u) != 0x7f800000u;
}

bool theta_ok(int16_t theta) {
    if (theta == std::numeric_limits<int16_t>::min()) {
        return false;
    }
    const float s = std::ldexp(1.0f, theta);
    return finite_bits(s) && s > 0.0f;
}

} // namespace

const char * paired_execution_backend_name(paired_execution_backend b) {
    switch (b) {
        case paired_execution_backend::none: return "NONE";
        case paired_execution_backend::paired_emulated: return "PAIRED_EMULATED";
        case paired_execution_backend::paired_hardware_unsupported: return "PAIRED_HARDWARE_UNSUPPORTED";
    }
    return "?";
}

void paired_add(paired_counts & t, const paired_counts & a) {
    t.ops += a.ops;
    t.main_micro_tiles += a.main_micro_tiles;
    t.residual_micro_tiles += a.residual_micro_tiles;
    t.mode_switches += a.mode_switches;
    t.main_only_tail_micro_tiles += a.main_only_tail_micro_tiles;
    t.residual_only_tail_micro_tiles += a.residual_only_tail_micro_tiles;
    t.dummy_micro_tiles += a.dummy_micro_tiles;
    t.skipped_zero_fragments += a.skipped_zero_fragments;
    t.window_instances += a.window_instances;
    t.shared_weight_windows += a.shared_weight_windows;
    t.logical_weight_loads += a.logical_weight_loads;
    t.logical_weight_dma_bytes += a.logical_weight_dma_bytes;
    t.logical_activation_dma_bytes += a.logical_activation_dma_bytes;
    t.logical_weight_preloads += a.logical_weight_preloads;
    t.weight_identity_changes += a.weight_identity_changes;
    t.indexed_weight_reads += a.indexed_weight_reads;
    t.scale_loads += a.scale_loads;
    t.residual_group_inits += a.residual_group_inits;
    t.residual_acc_saves += a.residual_acc_saves;
    t.residual_acc_restores += a.residual_acc_restores;
    t.residual_acc_save_bytes += a.residual_acc_save_bytes;
    t.residual_acc_restore_bytes += a.residual_acc_restore_bytes;
    t.acc_completions += a.acc_completions;
    t.main_acc_peak_bytes = std::max(t.main_acc_peak_bytes, a.main_acc_peak_bytes);
    t.residual_acc_peak_bytes = std::max(t.residual_acc_peak_bytes, a.residual_acc_peak_bytes);
    t.scratchpad_peak_bytes = std::max(t.scratchpad_peak_bytes, a.scratchpad_peak_bytes);
    t.spill_peak_bytes = std::max(t.spill_peak_bytes, a.spill_peak_bytes);
}

potal_status paired_tiled_matmul_auto(const paired_gemm_desc & desc, paired_output & out,
                                      paired_execution_report & report, std::vector<paired_schedule> * schedules) try {
    report       = paired_execution_report{};
    report.merge = desc.merge;
    const auto fail = [&](potal_status status, const char * stage, std::string detail) {
        report.status = status;
        report.stage  = stage;
        report.detail = std::move(detail);
        return status;
    };

    // 0. The paired GEMM has no RoCC implementation: refuse before any work,
    //    never fall back to the serialized GEMMs.
    if (desc.backend == paired_backend::hardware) {
        report.execution_backend = paired_execution_backend::paired_hardware_unsupported;
        return fail(POTAL_ERR_UNSUPPORTED, "backend",
                    "paired Main/Residual execution is not implemented by the current PoTal RTL");
    }
    report.execution_backend = paired_execution_backend::paired_emulated;

    // 1. Request.
    const hp1_weight * w = desc.weight;
    if (desc.m == 0 || desc.n == 0 || desc.k == 0 || desc.k % paired_block_k != 0 || desc.a == nullptr ||
        desc.lda < desc.k || w == nullptr || w->b == nullptr || w->carriers == nullptr || w->scale == nullptr ||
        w->n != desc.n || w->k != desc.k || w->ldb < desc.n || w->cstride < desc.n) {
        return fail(POTAL_ERR_INVALID_ARGUMENT, "request", "shape, A or prepared weight");
    }

    // 2. Plan: frozen Main tiles or a paired selection; t_I / t_J fixed from here on.
    paired_tile_plan plan;
    const bool       frozen = desc.tile_I != 0 || desc.tile_J != 0 || desc.tile_K != 0;
    potal_status     status = frozen ? paired_frozen_plan(desc.m, desc.n, desc.k, desc.tile_I, desc.tile_J,
                                                          desc.tile_K, plan)
                                     : paired_tile_select(desc.m, desc.n, desc.k, plan);
    if (status != POTAL_OK) {
        return fail(status, "plan", status == POTAL_ERR_CAPACITY ? "no K32 window fits the frozen tile_I / tile_J"
                                                                 : "tiles do not match the shape");
    }
    report.plan             = plan;
    report.stripe_rows      = plan.stripe_rows;
    report.common_k_windows = plan.k_windows;
    if (desc.stripe_rows != plan.stripe_rows || desc.stripes == nullptr || desc.stripe_count != plan.stripes) {
        return fail(POTAL_ERR_INVALID_ARGUMENT, "stripes",
                    "ExSIA stripe height / count differs from the paired plan (stripe_rows " +
                        std::to_string(plan.stripe_rows) + ", stripes " + std::to_string(plan.stripes) + ")");
    }

    // 3. Scales: every stripe's theta and every carrier, before any work.
    for (size_t s = 0; s < plan.stripes; ++s) {
        if (!theta_ok(desc.stripes[s].theta)) {
            return fail(POTAL_ERR_INVALID_ARGUMENT, "scales", "stripe " + std::to_string(s) + ": no valid theta");
        }
    }
    for (size_t b = 0; b < desc.k / paired_block_k; ++b) {
        for (size_t col = 0; col < desc.n; ++col) {
            if (!potal_carrier_is_valid(w->carriers[b * w->cstride + col])) {
                return fail(POTAL_ERR_INVALID_ARGUMENT, "scales",
                            "invalid carrier at block " + std::to_string(b) + " column " + std::to_string(col));
            }
        }
    }

    // 4. Residual descriptors of every stripe (indexed into w, no B copy).
    std::vector<paired_residual_request> requests(plan.stripes);
    for (size_t s = 0; s < plan.stripes; ++s) {
        const prmd::StripePacket * p = desc.stripes[s].packet;
        if (p == nullptr) {
            continue;
        }
        if (p->stripe_id != s || p->row_begin != s * plan.stripe_rows || p->row_count != paired_stripe_rows(plan, s)) {
            return fail(POTAL_ERR_INVALID_ARGUMENT, "residual_prepare",
                        "stripe " + std::to_string(s) + ": packet does not describe this stripe");
        }
        const prmd::RmdStatus built = build_paired_residual_request(*p, *w, plan.stripe_rows, requests[s]);
        if (built != prmd::RmdStatus::success) {
            return fail(POTAL_ERR_INVALID_ARGUMENT, "residual_prepare",
                        "stripe " + std::to_string(s) + ": " + prmd::rmd_status_message(built));
        }
    }

    // 5. Schedules, their static checks and the spill backing they need.
    std::vector<paired_schedule> sched(plan.stripes);
    for (size_t s = 0; s < plan.stripes; ++s) {
        sched[s]          = build_paired_schedule(plan, s, requests[s]);
        size_t      spill = 0;
        std::string why;
        status = validate_paired_schedule(plan, requests[s], sched[s], &spill, &why);
        if (status != POTAL_OK) {
            return fail(status, "schedule", "stripe " + std::to_string(s) + ": " + why);
        }
        if (spill > desc.spill_capacity_bytes) {
            return fail(POTAL_ERR_CAPACITY, "spill",
                        "stripe " + std::to_string(s) + " needs " + std::to_string(spill) +
                            " spill bytes, granted " + std::to_string(desc.spill_capacity_bytes));
        }
    }

    // 6. Emulated execution, stripe by stripe, in schedule order.
    paired_output staged;
    staged.main_c.assign(desc.m * desc.n, 0);
    staged.stripes.resize(plan.stripes);
    for (size_t s = 0; s < plan.stripes; ++s) {
        paired_stripe_report sr;
        sr.stripe_id       = s;
        sr.row_begin       = s * plan.stripe_rows;
        sr.rows            = paired_stripe_rows(plan, s);
        sr.residual_rows   = requests[s].rows.size();
        sr.residual_groups = requests[s].groups.size();
        sr.k_compact       = requests[s].k_compact;
        std::vector<int32_t> main_rows;
        std::string          why;
        status = execute_paired_emulated(desc, plan, requests[s], sched[s], main_rows,
                                         staged.stripes[s].residual_c, sr.counts, report.executed_ops, why);
        if (status != POTAL_OK) {
            return fail(status, "emulator", "stripe " + std::to_string(s) + ": " + why);
        }
        // Residual (group, fragment, N tile) not executed because every digit is zero.
        for (const paired_residual_group & g : requests[s].groups) {
            for (uint8_t nz : g.nonzero) {
                sr.counts.skipped_zero_fragments += nz ? 0 : (desc.n + paired_dim - 1) / paired_dim; // one per N tile
            }
        }
        std::memcpy(staged.main_c.data() + sr.row_begin * desc.n, main_rows.data(), main_rows.size() * sizeof(int32_t));
        staged.stripes[s].rows = requests[s].rows;
        report.residual_groups += sr.residual_groups;
        paired_add(report.totals, sr.counts);
        report.stripes.push_back(sr);
    }

    // 7. Radix INT64 reconstruction (checked __int128, INT64 range).
    for (size_t s = 0; s < plan.stripes; ++s) {
        const prmd::StripePacket * p = desc.stripes[s].packet;
        if (p == nullptr) {
            continue;
        }
        prmd::BlockScaledInt64Correction correction;
        const prmd::RmdStatus composed = residual::reconstruct_rows(
            *p, staged.stripes[s].rows, staged.stripes[s].residual_c.data(), desc.n, desc.n, correction);
        if (composed != prmd::RmdStatus::success) {
            return fail(POTAL_ERR_INVALID_ARGUMENT, "radix_reconstruct",
                        "stripe " + std::to_string(s) + ": " + prmd::rmd_status_message(composed));
        }
        staged.stripes[s].correction = std::move(correction.values);
    }

    // 8. One merge mode.
    staged.out.assign(desc.m * desc.n, 0.0f);
    for (size_t s = 0; s < plan.stripes; ++s) {
        const size_t  row_begin = s * plan.stripe_rows;
        const size_t  rows      = paired_stripe_rows(plan, s);
        const int16_t theta     = desc.stripes[s].theta;
        const auto &  corr      = staged.stripes[s].correction;
        if (desc.merge == paired_merge_mode::legacy_f32) {
            // STRIPE_PIPELINE: Main F32 reconstruction, then the checked F32 merge.
            const double act = (double) std::ldexp(1.0f, theta);
            for (size_t r = 0; r < rows; ++r) {
                reconstruct_row(staged.main_c.data() + (row_begin + r) * desc.n, desc.n, act, w->scale,
                                staged.out.data() + (row_begin + r) * desc.n, 1);
            }
            if (desc.stripes[s].packet != nullptr) {
                exsia::StripeMetadataSnapshot snap;
                snap.theta = theta;
                prmd::BlockScaledInt64Correction c;
                c.values                     = corr;
                size_t                nonzero = 0;
                const prmd::RmdStatus merged  = residual::merge_prepared_hp1(
                    *w, row_begin, row_begin + rows, snap, c, staged.out.data(), desc.m, desc.n, 1, &nonzero);
                if (merged != prmd::RmdStatus::success) {
                    return fail(POTAL_ERR_INVALID_ARGUMENT, "merge",
                                "stripe " + std::to_string(s) + ": " + prmd::rmd_status_message(merged));
                }
            }
        } else {
            // INT64: Y = check64(C_main + R), one dequantization by channel scale * 2^theta.
            const double act = (double) std::ldexp(1.0f, theta);
            for (size_t r = 0; r < rows; ++r) {
                for (size_t col = 0; col < desc.n; ++col) {
                    const int64_t main = staged.main_c[(row_begin + r) * desc.n + col];
                    const int64_t rr   = corr.empty() ? 0 : corr[r * desc.n + col];
                    int64_t       y    = 0;
                    if (__builtin_add_overflow(main, rr, &y)) {
                        return fail(POTAL_ERR_INVALID_ARGUMENT, "merge",
                                    "stripe " + std::to_string(s) + ": INT64 overflow of C_main + R");
                    }
                    const double alpha  = (double) w->scale[col] * act;
                    const double scaled = (double) y * alpha;
                    const float  value  = (float) scaled;
                    if (!finite_bits(alpha) || !finite_bits(scaled) || !finite_bits(value)) {
                        return fail(POTAL_ERR_INVALID_ARGUMENT, "merge",
                                    "stripe " + std::to_string(s) + ": non-finite INT64 dequantization");
                    }
                    staged.out[(row_begin + r) * desc.n + col] = value;
                }
            }
        }
    }

    // 9. Commit.
    out = std::move(staged);
    if (schedules != nullptr) {
        *schedules = std::move(sched);
    }
    report.status = POTAL_OK;
    return POTAL_OK;
} catch (const std::bad_alloc &) {
    report.status = POTAL_ERR_CAPACITY;
    report.stage  = "allocation";
    return POTAL_ERR_CAPACITY;
}

} // namespace ggml::potal
