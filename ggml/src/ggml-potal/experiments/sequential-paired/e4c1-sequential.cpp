#include "e4c1-sequential.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace potal_test::e4c1 {
namespace gp = ggml::potal;
namespace {

using status = gp::potal_paired_status;
constexpr uint32_t D = 16;
uint32_t tiles(size_t n) { return uint32_t(n / D + (n % D != 0)); }

struct issue_error {
    potal_status value;
    const char * stage;
};

struct batch {
    std::array<size_t, 2> fragments{};
    size_t count = 0;
    std::vector<size_t> groups;
};

struct executor {
    const gp::potal_paired_request & req;
    const options & cfg;
    gp::potal_paired_report & report;
    statistics & stats;
    potal_paired_session * session = nullptr;
    std::array<potal_resident_ws_ref, 9> main_epochs{};
    std::vector<std::array<potal_resident_ws_ref, 9>> residual_epochs;
    std::vector<std::vector<potal_resident_ws_fragment>> manifests;
    std::vector<std::vector<batch>> batches;
    std::vector<uint64_t> residual_sequence;
    std::vector<int32_t> main_result, residual_result;
    uint32_t local_generation = 0, live_scale = 0;
    int last_path = -1;

    executor(const gp::potal_paired_request & request, const options & config,
             gp::potal_paired_report & result, statistics & counters) :
        req(request), cfg(config), report(result), stats(counters) {}

    template<class Call> potal_status call(scope kind, const char * stage, Call action) {
        potal_status result;
        if (cfg.profile) {
            const auto start = std::chrono::steady_clock::now();
            result = action();
            auto & i = stats.intervals[static_cast<size_t>(kind)];
            i.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            ++i.calls;
        } else {
            result = action();
        }
        if (result != POTAL_OK) throw issue_error{result, stage};
        return result;
    }

    void * buffer(uint32_t which) {
        void * p = nullptr;
        size_t bytes = 0;
        call(scope::prepare, "buffer", [&] {
            return potal_paired_resident_buffer_view(session,
                static_cast<potal_resident_ws_buffer>(which), &p, &bytes);
        });
        if (bytes != POTAL_RESIDENT_WS_POOL_BYTES) throw issue_error{POTAL_ERR_CAPACITY, "buffer_extent"};
        return p;
    }

    uint32_t generation() {
        uint32_t & g = req.generation ? *req.generation : local_generation;
        if (g == POTAL_GENERATION_MAX) ++stats.generation_wraps;
        g = g % POTAL_GENERATION_MAX + 1;
        report.last_generation = g;
        ++stats.scale_generations;
        return g;
    }

    uint32_t original(size_t f, size_t x) const {
        return gp::potal_paired_fragment_original_k(*req.residual,
            report.tile.fragments[f], x, gp::potal_paired_hw_geometry());
    }

    void prepare_jobs() {
        const auto & t = report.tile;
        const size_t groups = (t.residual_lane_rows + D - 1) / D;
        manifests.resize(groups);
        residual_epochs.resize(groups);
        residual_sequence.resize(groups);
        std::vector<std::vector<uint8_t>> active(groups, std::vector<uint8_t>(t.fragments.size()));
        for (size_t g = 0; g < groups; ++g) {
            const size_t first = g * D, end = std::min(first + D, t.residual_lane_rows);
            for (size_t f = 0; f < t.fragments.size(); ++f) {
                const auto & fragment = t.fragments[f];
                bool nonzero = false;
                for (size_t row = first; row < end && !nonzero; ++row)
                    for (size_t x = 0; x < fragment.count; ++x)
                        nonzero = nonzero || req.residual->digits[
                            row * req.residual->k_compact + fragment.compact_begin + x] != 0;
                if (!nonzero) {
                    ++stats.skipped_zero_group_fragments;
                    continue;
                }
                active[g][f] = 1;
                potal_resident_ws_fragment declaration{};
                declaration.count = fragment.count;
                for (size_t x = 0; x < fragment.count; ++x) declaration.original_k[x] = original(f, x);
                manifests[g].push_back(declaration);
            }
        }
        batches.resize(t.panels.size());
        size_t jobs = 0;
        for (size_t p = 0; p < t.panels.size(); ++p) {
            for (const auto & window : t.panels[p].windows) {
                // A cohort shares one exact Index/Scale window across I1 lane
                // groups. A group-zero fragment is omitted, not dummy work.
                std::vector<batch> cohorts;
                for (size_t g = 0; g < groups; ++g) {
                    batch next;
                    for (size_t f = window.fragment_begin; f < window.fragment_end; ++f)
                        if (active[g][f]) {
                            if (next.count == next.fragments.size())
                                throw issue_error{POTAL_ERR_CAPACITY, "Index32_window"};
                            next.fragments[next.count++] = f;
                        }
                    if (!next.count) continue;
                    auto found = std::find_if(cohorts.begin(), cohorts.end(), [&](const batch & b) {
                        return b.count == next.count && b.fragments == next.fragments;
                    });
                    if (found == cohorts.end()) {
                        next.groups.push_back(g);
                        cohorts.push_back(std::move(next));
                    } else {
                        found->groups.push_back(g);
                    }
                    ++jobs;
                }
                for (auto & b : cohorts) batches[p].push_back(std::move(b));
            }
        }
        main_result.assign(req.rows * req.n, 0);
        residual_result.assign(t.residual_lane_rows * req.n, 0);
        if (cfg.trace) {
            size_t main_jobs = 0;
            for (const auto & p : t.panels) main_jobs += (p.k_tiles + cfg.main_k_tiles - 1) / cfg.main_k_tiles;
            stats.events.reserve(t.j_groups * (main_jobs + jobs + 2 * t.panels.size()));
        }
    }

    potal_resident_ws_ref load(size_t pair, const gp::potal_paired_k_panel & panel,
                               size_t j_begin, uint32_t columns) {
        auto * b = static_cast<int8_t *>(buffer(uint32_t(pair)));
        const uint32_t stride = tiles(columns) * D;
        std::memset(b, 0, POTAL_RESIDENT_WS_POOL_BYTES);
        for (size_t k = 0; k < panel.k_rows; ++k)
            std::memcpy(b + k * stride,
                req.weight->b + (panel.k_begin + k) * req.weight->ldb + j_begin, columns);
        const potal_resident_ws_load desc{
            uint32_t(pair), uint32_t(panel.k_rows), columns, uint32_t(panel.k_tiles),
            panel.k_begin, j_begin, 1, 1, stride
        };
        call(scope::b_issue, "b_load", [&] { return potal_paired_resident_load_issue(session, &desc); });
        ++stats.b_loads;
        stats.b_mvins += panel.k_tiles * tiles(columns);
        call(scope::b_completion, "b_completion", [&] { return potal_paired_fence(session); });
        ++stats.fences;
        potal_resident_ws_ref token{};
        call(scope::b_completion, "resident_publication", [&] {
            return potal_paired_resident_get(session, uint32_t(pair), &token);
        });
        stats.peak_b_residencies = std::max(stats.peak_b_residencies, UINT64_C(1));
        if (cfg.poison_b) std::memset(b, 93, POTAL_RESIDENT_WS_POOL_BYTES);
        if (cfg.trace) {
            event e;
            e.kind = event_kind::load; e.pair = uint32_t(pair); e.bank = uint32_t(pair * 2 + 1);
            e.base = uint32_t((pair + 1) * 8192 - panel.k_tiles * tiles(columns) * D);
            e.pitch = stride; e.original_k = panel.k_begin; e.original_j = j_begin;
            e.k_rows = uint32_t(panel.k_rows); e.columns = columns;
            stats.events.push_back(e);
        }
        return token;
    }

    void epochs(size_t j_begin, uint32_t columns) {
        std::fill(residual_sequence.begin(), residual_sequence.end(), 0);
        for (uint32_t j = 0; j < tiles(columns); ++j) {
            potal_resident_ws_epoch d{};
            d.acc_tile = cfg.acc_tile + j; d.rows = uint32_t(req.rows);
            d.columns = std::min(D, columns - j * D);
            d.output_id = 1; d.original_i = req.row_begin; d.original_j = j_begin + j * D;
            d.weight_id = 1; d.weight_generation = 1; d.logical_k_end = req.k;
            call(scope::prepare, "main_epoch", [&] {
                return potal_paired_resident_epoch_begin(session, &d, &main_epochs[j]);
            });
            for (size_t g = 0; g < manifests.size(); ++g) {
                if (manifests[g].empty()) continue;
                d.path = 1; d.acc_tile = cfg.acc_tile + uint32_t(g) * tiles(columns) + j;
                d.rows = uint32_t(std::min<size_t>(D, report.tile.residual_lane_rows - g * D));
                d.output_id = g + 2; d.original_i = g * D;
                d.fragments = manifests[g].data(); d.fragment_count = manifests[g].size();
                call(scope::prepare, "residual_epoch", [&] {
                    return potal_paired_resident_epoch_begin(session, &d, &residual_epochs[g][j]);
                });
            }
        }
    }

    potal_resident_ws_ref scales(potal_resident_ws_ref resident, uint32_t base,
                                 uint32_t partials, const uint64_t * blocks,
                                 size_t j_begin, uint32_t columns, uint32_t gen, bool residual) {
        std::array<uint32_t, 28 * 144> carriers{};
        for (uint32_t p = 0; p < partials; ++p)
            std::copy_n(req.weight->carriers + blocks[p] * req.weight->cstride + j_begin,
                        columns, carriers.data() + p * columns);
        const potal_resident_ws_scales d{
            resident, base, gen, partials, columns, j_begin, blocks, carriers.data(), columns
        };
        potal_resident_ws_ref token{};
        call(scope::scale_program, "scale_program", [&] {
            return potal_paired_resident_scale_program(session, &d, &token);
        });
        const uint32_t slots = partials * tiles(columns);
        live_scale += slots;
        stats.peak_scale_slots = std::max(stats.peak_scale_slots, uint64_t(live_scale));
        ++stats.scale_programs; stats.scale_slot_instances += slots;
        (residual ? stats.residual_scale_writes : stats.main_scale_writes) += partials * columns;
        return token;
    }

    void release(uint32_t base, uint32_t partials, uint32_t columns, uint32_t gen, bool residual) {
        for (uint32_t p = 0; p < partials; ++p)
            for (uint32_t j = 0; j < tiles(columns); ++j)
                for (uint32_t c = 0; c < std::min(D, columns - j * D); ++c)
                    call(scope::scale_release, "scale_release", [&] {
                        return potal_paired_scale_release(session, base + p * tiles(columns) + j, gen, c);
                    });
        // The first successful Release qualifies the pending consumer globally;
        // the remaining column releases retain that same allDrained contract.
        ++stats.release_boundaries;
        live_scale -= partials * tiles(columns);
        (residual ? stats.residual_scale_releases : stats.main_scale_releases) += partials * columns;
    }

    void issue(potal_resident_ws_run & d, uint32_t pair, size_t j_begin,
               const gp::potal_paired_k_panel & panel, uint32_t group, bool tail) {
        call(d.path ? scope::residual_issue : scope::main_issue,
             d.path ? "residual_issue" : "main_issue", [&] {
            return potal_paired_resident_issue(session, &d);
        });
        ++stats.views; ++stats.loops;
        if (d.path) {
            ++stats.residual_microtiles; stats.residual_partials += tiles(d.k_rows);
            if (tail) ++stats.residual_only_tail;
        } else {
            ++stats.main_microtiles; stats.main_partials += tiles(d.k_rows);
            if (tail) ++stats.main_only_tail;
        }
        if (last_path == 0 && d.path == 1) ++stats.main_to_residual;
        if (last_path == 1 && d.path == 0) ++stats.residual_to_main;
        last_path = int(d.path);
        if (cfg.trace) {
            event e;
            e.kind = d.path ? event_kind::residual : event_kind::main;
            e.pair = pair; e.bank = d.path; e.pitch = tiles(d.j_columns) * D;
            const uint32_t full_base = (pair + 1) * 8192 - uint32_t(panel.k_tiles) * tiles(d.j_columns) * D;
            e.base = d.indexed ? full_base : full_base + (d.k_offset / D) * e.pitch;
            e.acc_tile = d.acc_tile; e.k_rows = d.k_rows; e.columns = d.j_columns;
            e.original_k = panel.k_begin + d.k_offset; e.original_j = j_begin;
            e.sequence = d.sequence; e.group = group; e.indexed = d.indexed;
            e.first = d.first; e.last = d.last; e.scale_base = d.scale_base;
            e.scale_generation = d.scale_generation;
            stats.events.push_back(e);
        }
    }

    void execute_panel(potal_resident_ws_ref resident, uint32_t pair,
                       const gp::potal_paired_k_panel & panel, size_t panel_id,
                       size_t j_begin, uint32_t columns) {
        const uint32_t nj = tiles(columns);
        const uint32_t residual_base = cfg.main_k_tiles * nj;
        size_t main_k = 0, bi = 0, gi = 0;
        size_t residual_left = 0;
        for (const auto & b : batches[panel_id]) residual_left += b.groups.size();
        potal_resident_ws_ref index{}, scale{};
        uint32_t residual_generation = 0, residual_rows = 0;
        std::array<uint64_t, 32> original_k{};
        bool take_main = true;
        last_path = -1; // Alternation/tails are within one residency.
        while (main_k < panel.k_rows || residual_left) {
            const bool main_left = main_k < panel.k_rows;
            const bool main = main_left && (take_main || !residual_left);
            const bool tail = !main_left || !residual_left;
            potal_resident_ws_run d{};
            d.resident = resident; d.j_columns = columns; d.i_rows = uint32_t(req.rows);
            d.acc_tile = cfg.acc_tile; d.original_i = req.row_begin; d.output_id = 1;
            uint32_t partials = 0, gen = 0;
            size_t group = 0;
            if (main) {
                d.k_offset = uint32_t(main_k);
                d.k_rows = uint32_t(std::min<size_t>(cfg.main_k_tiles * D, panel.k_rows - main_k));
                partials = tiles(d.k_rows);
                d.a_stride = partials * D;
                auto * a = static_cast<int8_t *>(buffer(POTAL_RESIDENT_WS_MAIN_A));
                std::memset(a, 0, req.rows * d.a_stride);
                for (size_t row = 0; row < req.rows; ++row)
                    std::memcpy(a + row * d.a_stride,
                        req.a + row * req.lda + panel.k_begin + main_k, d.k_rows);
                std::array<uint64_t, 28> blocks{};
                for (uint32_t p = 0; p < partials; ++p) blocks[p] = (panel.k_begin + main_k + p * D) / 32;
                gen = generation();
                d.scale = scales(resident, 0, partials, blocks.data(), j_begin, columns, gen, false);
                d.scale_generation = gen;
                d.sequence = (panel.k_begin + main_k) / D;
                d.first = d.sequence == 0;
                d.last = panel.k_begin + main_k + d.k_rows == req.k;
                std::copy(main_epochs.begin(), main_epochs.end(), d.epochs);
                main_k += d.k_rows;
            } else {
                const batch & b = batches[panel_id][bi];
                group = b.groups[gi];
                partials = uint32_t(b.count);
                if (gi == 0) {
                    residual_rows = uint32_t((b.count - 1) * D + report.tile.fragments[b.fragments[b.count - 1]].count);
                    std::array<uint64_t, 2> blocks{};
                    for (size_t p = 0; p < b.count; ++p) {
                        const auto & f = report.tile.fragments[b.fragments[p]];
                        blocks[p] = f.block_id;
                        for (uint32_t x = 0; x < (p + 1 == b.count ? f.count : D); ++x)
                            original_k[p * D + x] = original(b.fragments[p], x < f.count ? x : 0);
                    }
                    const potal_resident_ws_indices ix{resident, residual_rows, original_k.data()};
                    call(scope::index_program, "index_program", [&] {
                        return potal_paired_resident_index_program(session, &ix, &index);
                    });
                    ++stats.index_configs; stats.index_writes += residual_rows;
                    stats.peak_index_entries = std::max(stats.peak_index_entries, uint64_t(residual_rows));
                    residual_generation = generation();
                    scale = scales(resident, residual_base, partials, blocks.data(),
                                   j_begin, columns, residual_generation, true);
                }
                d.path = 1; d.indexed = 1; d.index = index; d.scale = scale;
                d.k_offset = uint32_t(original_k[0] - panel.k_begin);
                d.k_rows = residual_rows; d.a_stride = partials * D;
                d.a_offset_tiles = uint32_t(panel.k_tiles);
                d.i_rows = uint32_t(std::min<size_t>(D, report.tile.residual_lane_rows - group * D));
                d.scale_base = residual_base; d.scale_generation = residual_generation;
                d.acc_tile += uint32_t(group) * nj; d.output_id = group + 2; d.original_i = group * D;
                d.sequence = residual_sequence[group];
                d.first = d.sequence == 0; d.last = d.sequence + partials == manifests[group].size();
                for (uint32_t j = 0; j < nj; ++j) d.epochs[j] = residual_epochs[group][j];
                auto * a = static_cast<int8_t *>(buffer(POTAL_RESIDENT_WS_RESIDUAL_A));
                std::memset(a, 0, d.i_rows * d.a_stride);
                for (size_t p = 0; p < b.count; ++p) {
                    const auto & f = report.tile.fragments[b.fragments[p]];
                    d.fragment_counts[p] = f.count;
                    for (uint32_t row = 0; row < d.i_rows; ++row)
                        std::copy_n(req.residual->digits.data() +
                            (group * D + row) * req.residual->k_compact + f.compact_begin,
                            f.count, a + row * d.a_stride + p * D);
                }
            }
            issue(d, pair, j_begin, panel, uint32_t(group), tail);
            if (main) {
                release(0, partials, columns, gen, false);
            } else {
                residual_sequence[group] += partials;
                const batch & b = batches[panel_id][bi];
                --residual_left;
                if (++gi == b.groups.size()) {
                    release(residual_base, partials, columns, residual_generation, true);
                    ++bi; gi = 0;
                } else {
                    call(scope::completion, "residual_completion", [&] { return potal_paired_fence(session); });
                    ++stats.fences;
                }
            }
            take_main = !main;
        }
    }

    void readback(uint32_t bank, size_t group, size_t j_begin, uint32_t columns) {
        potal_resident_ws_readback d{};
        d.bank = bank; d.acc_tile = cfg.acc_tile + (bank ? uint32_t(group) * tiles(columns) : 0);
        d.rows = bank ? uint32_t(std::min<size_t>(D, report.tile.residual_lane_rows - group * D)) : uint32_t(req.rows);
        d.columns = columns; d.stride = tiles(columns) * D;
        for (uint32_t j = 0; j < tiles(columns); ++j) {
            d.epochs[j] = bank ? residual_epochs[group][j] : main_epochs[j];
            d.sequences[j] = bank ? residual_sequence[group] : tiles(req.k);
        }
        potal_resident_ws_ref receipt{};
        call(scope::readback, "readback_issue", [&] {
            return potal_paired_resident_readback_issue(session, &d, &receipt);
        });
        ++stats.readbacks; stats.mvouts += tiles(columns);
        call(scope::readback, "readback_completion", [&] { return potal_paired_fence(session); });
        ++stats.fences;
        const int32_t * data = nullptr;
        size_t stride = 0;
        call(scope::readback, "result_receipt", [&] {
            return potal_paired_resident_result(session, receipt, &data, &stride);
        });
        auto & destination = bank ? residual_result : main_result;
        for (uint32_t row = 0; row < d.rows; ++row)
            std::copy_n(data + row * stride, columns,
                destination.data() + ((bank ? group * D : 0) + row) * req.n + j_begin);
    }

    void execute() {
        call(scope::prepare, "prepare", [&] { return potal_paired_resident_prepare(session); });
        ++stats.fences;
        for (size_t j = 0; j < report.tile.j_groups; ++j) {
            const size_t j_begin = j * report.tile.tile_J * D;
            const uint32_t columns = uint32_t(std::min(report.tile.tile_J * D, req.n - j_begin));
            epochs(j_begin, columns);
            for (size_t p = 0; p < report.tile.panels.size(); ++p) {
                const auto & panel = report.tile.panels[p];
                const uint32_t pair = uint32_t((j * report.tile.k_panels + p) % 2);
                const auto resident = load(pair, panel, j_begin, columns);
                execute_panel(resident, pair, panel, p, j_begin, columns);
                call(scope::retirement, "resident_retire", [&] {
                    return potal_paired_resident_retire(session, resident);
                });
                ++stats.b_retirements;
                if (cfg.trace) {
                    event e; e.kind = event_kind::retire; e.pair = pair;
                    e.original_k = panel.k_begin; e.original_j = j_begin;
                    stats.events.push_back(e);
                }
            }
            readback(0, 0, j_begin, columns);
            for (size_t g = 0; g < manifests.size(); ++g)
                if (!manifests[g].empty()) readback(1, g, j_begin, columns);
        }
        const auto q = potal_paired_resident_query(session);
        stats.commands = q.core.commands; stats.drains = q.core.drains;
        if (q.pending || q.control_pending || q.core.failed || q.core.live_scale_slots ||
            q.readable[0] || q.readable[1]) throw issue_error{POTAL_ERR_BUSY, "final_state"};
        call(scope::close, "close", [&] { return potal_paired_resident_close(&session); });
        for (size_t row = 0; row < req.rows; ++row)
            std::copy_n(main_result.data() + row * req.n, req.n, req.c_main + row * req.ldc_main);
        for (size_t row = 0; row < report.tile.residual_lane_rows; ++row)
            std::copy_n(residual_result.data() + row * req.n, req.n, req.c_residual + row * req.ldc_residual);
    }

    status fail(potal_status error, const char * stage) {
        report.status = status::execution_failed; report.hw_status = error; report.stage = stage;
        if (!session) return report.status;
        const auto q = potal_paired_resident_query(session);
        stats.commands = q.core.commands; stats.drains = q.core.drains;
        report.issued = q.core.commands != 0;
        if (q.state == POTAL_RESIDENT_WS_FAILED) {
            const auto cleanup = potal_paired_session_cleanup(session);
            if (cleanup == POTAL_OK) {
                const auto closed = potal_paired_resident_close(&session);
                report.cleaned = session == nullptr && closed == q.core.error;
            }
        } else if (q.state == POTAL_RESIDENT_WS_NEW && !q.pending && !q.control_pending) {
            (void)potal_paired_resident_close(&session);
        }
        // A quarantined or otherwise refused close retains the C11-owned DMA
        // pools and exclusion. Never retry a Fence or invent healthy ownership.
        report.fatal = session != nullptr;
        return report.status;
    }
};

} // namespace

status run(const gp::potal_paired_request & req, const potal_resident_ws_options * transport,
           const options & cfg, gp::potal_paired_report & report, statistics & stats) {
    report = {};
    stats = {};
    executor ex{req, cfg, report, stats};
    const auto start = cfg.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    try {
        const auto * w = req.weight;
        const bool residual = req.residual && !(req.residual->rows.empty() && req.residual->runs.empty());
        if (!req.main_plan || !w || !req.a || !req.c_main || !req.rows || req.rows > D ||
            !req.n || !req.k || !w->b || !w->carriers || !w->scale ||
            w->n != req.n || w->k != req.k || w->ldb < req.n || w->cstride < req.n ||
            req.lda < req.k || req.ldc_main < req.n ||
            uintptr_t(req.a) % D || uintptr_t(w->b) % D || uintptr_t(req.c_main) % (D * 4) ||
            (req.generation && *req.generation > POTAL_GENERATION_MAX) ||
            (cfg.main_k_tiles != 1 && cfg.main_k_tiles != 2 && cfg.main_k_tiles != 4) ||
            req.n > SIZE_MAX / req.rows / sizeof(int32_t) ||
            (residual && (!req.c_residual || req.ldc_residual < req.n ||
                uintptr_t(req.c_residual) % (D * 4) || req.residual->digits.empty())))
            throw issue_error{POTAL_ERR_INVALID_ARGUMENT, "request"};
        const auto hw = req.hw ? *req.hw : gp::potal_paired_hw_geometry();
        if (hw.dim != D || hw.bank_num != 4 || hw.bank_rows != 4096 || hw.acc_rows != 1024 ||
            hw.scale_entries != 256 || hw.index_entries != 32 || hw.block_k != 32)
            throw issue_error{POTAL_ERR_UNSUPPORTED, "geometry"};
        std::string why;
        status planned;
        if (req.plan) {
            const auto & t = req.plan->tile;
            if (t.rows != req.rows || t.n != req.n || t.k != req.k ||
                t.tile_J != req.main_plan->tile_J || t.tile_K_main != req.main_plan->tile_K ||
                t.tile_I != std::min<size_t>(req.main_plan->tile_I, tiles(req.rows)) ||
                t.residual != residual) throw issue_error{POTAL_ERR_INVALID_ARGUMENT, "plan_request"};
            planned = gp::potal_validate_execution_plan(*req.plan, residual ? req.residual : nullptr, &why, hw);
            report.plan = *req.plan;
            report.tile = req.plan->tile;
        } else {
            planned = gp::potal_paired_plan_stripe(*req.main_plan, req.rows, req.n, req.k,
                residual ? req.residual : nullptr, report.tile, report.plan, &why, hw);
        }
        if (planned != status::ok) {
            report.status = planned; report.stage = "plan"; report.detail = std::move(why);
            return planned;
        }
        const auto & t = report.tile;
        if (cfg.acc_tile >= 32 || t.tile_I != 1 || !t.tile_J || t.tile_J > 9 || !t.tile_K || t.tile_K > 28 ||
            cfg.acc_tile + t.tile_J > 32 ||
            (t.residual_row_tiles && (t.residual_row_tiles > (32 - cfg.acc_tile) / t.tile_J)))
            throw issue_error{POTAL_ERR_CAPACITY, "fixed_ACC_geometry"};
        if (t.residual_lane_rows && req.n > SIZE_MAX / t.residual_lane_rows / sizeof(int32_t))
            throw issue_error{POTAL_ERR_CAPACITY, "output_extent"};
        for (size_t block = 0; block < req.k / 32; ++block)
            for (size_t j = 0; j < req.n; ++j)
                if (!potal_carrier_is_valid(w->carriers[block * w->cstride + j]))
                    throw issue_error{POTAL_ERR_INVALID_ARGUMENT, "carrier"};
        ex.prepare_jobs();
        if (cfg.profile) {
            auto & i = stats.intervals[static_cast<size_t>(scope::planning)];
            i.ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            i.calls = 1;
        }
        ex.call(scope::prepare, "open", [&] { return potal_paired_resident_open(transport, &ex.session); });
        stats.software_only = potal_paired_resident_query(ex.session).software_only != 0;
        report.issued = true; // Real or explicitly software-only capability commands.
        ex.execute();
        report.counts.main_loop_ws = stats.main_microtiles;
        report.counts.residual_loop_ws = stats.residual_microtiles;
        report.counts.scale_writes = stats.main_scale_writes + stats.residual_scale_writes;
        report.counts.scale_releases = stats.main_scale_releases + stats.residual_scale_releases;
        report.counts.index_configs = stats.index_configs; report.counts.index_writes = stats.index_writes;
        report.counts.readbacks = stats.readbacks; report.counts.readback_tiles = stats.mvouts;
        report.counts.fences = stats.fences; report.counts.drains = stats.drains;
        report.counts.tlb_flushes = 1;
        report.status = status::ok;
        return status::ok;
    } catch (const issue_error & e) {
        if (!ex.session && !report.issued) {
            report.stage = e.stage; report.hw_status = e.value;
            report.status = e.value == POTAL_ERR_CAPACITY ? status::residual_acc_capacity : status::invalid_argument;
            return report.status;
        }
        return ex.fail(e.value, e.stage);
    } catch (const std::bad_alloc &) {
        if (ex.session) return ex.fail(POTAL_ERR_CAPACITY, "allocation");
        report.status = status::allocation_failure; report.stage = "allocation";
        return report.status;
    }
}

void add(statistics & total, const statistics & value) {
#define SUM(field) total.field += value.field
    SUM(main_microtiles); SUM(residual_microtiles); SUM(main_partials); SUM(residual_partials);
    SUM(main_to_residual); SUM(residual_to_main); SUM(main_only_tail); SUM(residual_only_tail);
    SUM(b_loads); SUM(b_mvins); SUM(b_retirements); SUM(scale_programs); SUM(scale_slot_instances);
    SUM(main_scale_writes); SUM(residual_scale_writes); SUM(main_scale_releases); SUM(residual_scale_releases);
    SUM(scale_generations); SUM(generation_wraps); SUM(release_boundaries); SUM(index_configs); SUM(index_writes);
    SUM(fences); SUM(readbacks); SUM(mvouts); SUM(views); SUM(loops); SUM(commands); SUM(drains);
    SUM(skipped_zero_group_fragments);
#undef SUM
    total.peak_scale_slots = std::max(total.peak_scale_slots, value.peak_scale_slots);
    total.peak_index_entries = std::max(total.peak_index_entries, value.peak_index_entries);
    total.peak_b_residencies = std::max(total.peak_b_residencies, value.peak_b_residencies);
    total.software_only = value.software_only;
    for (size_t i = 0; i < total.intervals.size(); ++i) {
        total.intervals[i].ns += value.intervals[i].ns;
        total.intervals[i].calls += value.intervals[i].calls;
    }
    total.events.insert(total.events.end(), value.events.begin(), value.events.end());
}

} // namespace potal_test::e4c1
