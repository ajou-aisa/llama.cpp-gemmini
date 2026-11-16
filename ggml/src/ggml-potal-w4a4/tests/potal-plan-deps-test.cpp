// Native W4 plan-only regression; no session, emulator or hardware commands.
// Link this same source against baseline and candidate W4 libraries. Default:
// retirement status/first-error oracle plus field-wise plan checksums.
// --manifest: also print every logical plan field (never padding or capacity).
// --bench N: 1..1000 fresh full builds per fixture, no validation-only timing.
// Timed region: frozen Main plan + tile plan + execution plan, including their
// production validation. Decoded input, checksum, printing and destruction are
// outside it. This measures native host construction, not FPGA cycles.

#include "matmul/paired-auto.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml::potal;

namespace {

static_assert(POTAL_W4A4 == 1 && POTAL_HW_DIM == 32, "requires the separate W4 DIM32 build");
static_assert(residual::digit_bits == 4, "requires radix-16 residuals");

void require(bool ok, const std::string & why) {
    if (!ok) {
        throw std::runtime_error(why);
    }
}

struct manifest {
    uint64_t hash = UINT64_C(14695981039346656037);
    bool verbose = false;

    void byte(uint8_t v) { hash = (hash ^ v) * UINT64_C(1099511628211); }
    void text(const char * text) {
        for (const char * p = text; *p; ++p) {
            byte(static_cast<uint8_t>(*p));
        }
        byte(0);
    }
    void record(const char * tag, std::initializer_list<uint64_t> values) {
        text(tag);
        if (verbose) std::printf("%s", tag);
        for (uint64_t v : values) {
            for (unsigned b = 0; b < 8; ++b) byte(static_cast<uint8_t>(v >> (8 * b)));
            if (verbose) std::printf(" %" PRIu64, v);
        }
        if (verbose) std::printf("\n");
    }
};

// Explicit members avoid padding, vector addresses and allocator capacities.
void inventory(manifest & m, const potal_paired_execution_plan & e) {
    const auto & t = e.tile;
    m.record("tile", {t.rows, t.n, t.k, t.passes_per_block, t.i_tiles, t.j_tiles, t.k_tiles,
        t.tile_I, t.tile_J, t.tile_K_main, t.tile_K_weight_limit, t.tile_K_activation_limit,
        t.tile_K_scale_limit, t.tile_K_hardware_limit, t.tile_K_residual_limit, t.tile_K,
        t.j_groups, t.k_panels, t.pad_I, t.pad_J, t.main_a_rows, t.residual_a_rows,
        t.residual_a_offset_tiles, t.weight_rows, t.main_acc_rows, t.residual_acc_rows,
        t.scale_slots_main, t.scale_slots_residual, t.residual, t.residual_lane_rows,
        t.residual_row_tiles, t.residual_pad_I, t.residual_k_compact, t.fragments_per_window,
        t.residual_windows, t.residual_filler_slots, t.residual_empty_panels, t.residual_zero_fragments});
    const auto & a = t.main;
    m.record("main", {a.tile_I, a.tile_J, a.tile_K, a.I0, a.J0, a.K0, a.pad_I, a.pad_J, a.pad_K,
        a.spad_rows, a.max_spad_rows, a.acc_rows, a.max_acc_rows, a.passes_per_block, a.scale_slots});
    m.record("sizes", {t.fragments.size(), t.panels.size(), e.residual_rows.size(), e.ops.size(),
        e.index_entries.size(), e.scale_windows.size(), e.residencies.size(), e.transitions.size()});
    for (const auto & f : t.fragments) {
        m.record("fragment", {f.run, f.block_id, f.compact_begin, f.local_begin, f.count});
    }
    for (const auto & p : t.panels) {
        m.record("panel", {p.index, p.k_begin, p.k_rows, p.k_tiles, p.block_begin, p.block_end,
            p.weight_rows, p.main_a_rows, p.fragment_begin, p.fragment_end, p.residual_k_compact,
            p.residual_a_rows, p.windows.size()});
        for (const auto & w : p.windows) {
            m.record("window", {w.index, w.fragment_begin, w.fragment_end, w.k_tiles,
                w.active_rows, w.pad_k, w.filler_slots, w.a_rows});
        }
    }
    for (const auto & r : e.residual_rows) m.record("row", {r.lane, r.source_row});
    for (const auto & o : e.ops) {
        m.record("op", {o.id, static_cast<uint64_t>(o.kind), o.j_group, o.panel, o.residency,
            o.pair, o.window, o.i_tiles, o.j_tiles, o.k_tiles, o.pad_i, o.pad_j, o.pad_k,
            o.a_offset_tiles, o.a_sp_begin, o.a_sp_rows, o.b_sp_begin, o.b_sp_rows,
            o.weight_generation, o.index_buffer, o.index_b_base, o.index_j_tiles,
            o.index_source_k_rows, o.index_active_rows, o.entry_begin, o.entry_end,
            o.same_buffer_predecessor, o.scale_window, o.acc_bank, o.acc_epoch, o.work_ids,
            o.first, o.last, o.reconstruction_required, o.deps.size()});
        for (const auto & d : o.deps) m.record("dep", {d.op, d.complete});
    }
    for (const auto & i : e.index_entries) {
        m.record("entry", {i.slot, i.fragment, i.original_k, i.resident_k, i.compact_k, i.filler});
    }
    for (const auto & w : e.scale_windows) {
        m.record("scale", {w.index, w.residual, w.residency, w.window, w.slot_base, w.slots,
            w.blocks, w.j_tiles, w.generation, w.program_op, w.consumer_op, w.release_op});
    }
    for (const auto & r : e.residencies) {
        m.record("residency", {r.index, r.j_group, r.panel, r.pair, r.j_tiles, r.cols,
            r.load_weight_op, r.main_gemm_op, r.retire_op, r.first_residual_op,
            r.last_residual_op, r.windows, r.consumers});
    }
    for (const auto & t : e.transitions) {
        m.record("transition", {t.op, t.pair, static_cast<uint64_t>(t.from),
            static_cast<uint64_t>(t.to), t.residency});
    }
    for (const auto & p : e.pairs) {
        m.record("pair", {p.pair, static_cast<uint64_t>(p.phase), p.residency, p.residencies,
            p.weight_valid, p.main_a_valid, p.residual_a_valid, p.index_window,
            p.pending_main, p.pending_residual});
    }
    m.record("counts", {e.main_gemms, e.residual_gemms, e.weight_loads, e.index_programs,
        e.index_writes, e.scale_programs, e.scale_releases, e.acc_completes, e.pair_retires, e.complete_deps});
}

struct fixture {
    const char * name;
    size_t rows, n, k, j, main_k, runs, spacing;
    bool tails;
};

const fixture fixtures[] = {
    {"windows-0", 1, 32, 896, 1, 28, 0, 1, false},
    {"windows-1", 1, 32, 896, 1, 28, 2, 1, false},
    {"windows-2", 1, 32, 896, 1, 28, 4, 1, false},
    {"windows-14", 1, 32, 896, 1, 28, 28, 1, false},
    {"multi-panel-group-tail", 33, 97, 2048, 2, 16, 64, 1, true},
    {"sparse-empty-panels", 33, 97, 2048, 2, 16, 2, 48, true},
};

residual::decoded_packet packet(const fixture & f) {
    residual::decoded_packet d;
    if (f.runs == 0) return d;
    // One lane per source row, plus a second lane to exercise radix provenance.
    for (size_t row = 0; row < f.rows; ++row) d.rows.push_back({0, static_cast<uint32_t>(row)});
    d.rows.push_back({1, 0});
    for (size_t b = 0; b < f.runs; ++b) {
        residual::run r;
        r.block_id = static_cast<uint32_t>(b * f.spacing);
        const size_t count = f.tails ? 1 + b % 32 : 1;
        for (size_t x = 0; x < count; ++x) r.local_k.push_back(static_cast<uint16_t>(x));
        d.k_compact += count;
        d.runs.push_back(std::move(r));
    }
    d.digits.resize(d.rows.size() * d.k_compact);
    for (size_t i = 0; i < d.digits.size(); ++i) d.digits[i] = i % 2 ? -7 : 7;
    return d;
}

potal_paired_status build(const fixture & f, const residual::decoded_packet & d,
                          potal_paired_execution_plan & e, std::string & why) {
    potal_tile_plan main;
    if (potal_frozen_plan(f.rows, f.n, f.k, (f.rows + 31) / 32, f.j, f.main_k, 1, &main) != POTAL_OK) {
        why = "frozen Main fixture refused";
        return potal_paired_status::invalid_argument;
    }
    potal_paired_tile_plan tile;
    const auto * residual = f.runs ? &d : nullptr;
    const auto s = potal_make_tile_plan(main, f.rows, f.n, f.k, residual, tile, &why);
    return s == potal_paired_status::ok ? potal_make_execution_plan(tile, residual, e, &why) : s;
}

bool depends(const potal_paired_op & op, size_t id) {
    return std::any_of(op.deps.begin(), op.deps.end(),
        [id](const potal_paired_dep & d) { return d.op == id && d.complete; });
}

void regression(const fixture & f, const residual::decoded_packet & d,
                const potal_paired_execution_plan & e, manifest & results, size_t & checks) {
    if (f.runs <= 28 && !f.tails) {
        require(e.residencies.size() == 1 && e.residencies[0].windows == f.runs / 2,
                std::string(f.name) + ": wrong physical W4 window count");
    } else {
        require(e.tile.k_panels > 1 && e.tile.j_groups > 1 && e.residencies.size() > 2,
                std::string(f.name) + ": missing pair reuse / K panels / J groups");
    }
    for (const auto & residency : e.residencies) {
        const size_t retire = residency.retire_op;
        std::vector<size_t> gemms;
        for (const auto & op : e.ops) {
            if (op.residency == residency.index && op.kind == potal_paired_op_kind::residual_gemm) {
                gemms.push_back(op.id);
            }
        }
        const auto original = e.ops[retire].deps;
        require(original.size() == 1 + gemms.size() && gemms.size() == residency.windows &&
            std::is_sorted(original.begin(), original.end(),
                [](const potal_paired_dep & a, const potal_paired_dep & b) { return a.op < b.op; }),
            "producer dependency count/order invariant");
        for (const auto & op : e.ops) {
            if (op.kind == potal_paired_op_kind::main_gemm) require(op.deps.size() == 3, "Main deps != 3");
            if (op.kind == potal_paired_op_kind::residual_gemm) require(op.deps.size() == 4, "Residual deps != 4");
        }
        const auto compare = [&](std::vector<potal_paired_dep> deps) {
            std::stable_sort(deps.begin(), deps.end(),
                [](const potal_paired_dep & a, const potal_paired_dep & b) { return a.op < b.op; });
            for (size_t order = 0; order < 3; ++order) {
                auto plan = e;
                auto & op = plan.ops[retire];
                op.deps = deps;
                if (order == 1) std::reverse(op.deps.begin(), op.deps.end());
                if (order == 2 && op.deps.size() > 1) {
                    std::rotate(op.deps.begin(), op.deps.begin() + 1, op.deps.end());
                }
                // Original independent existential searches, preserving first-error order.
                std::string expected;
                if (std::any_of(op.deps.begin(), op.deps.end(),
                        [retire](const potal_paired_dep & x) { return x.op >= retire; })) {
                    expected = "dependency on a later or own op";
                } else if (!depends(op, residency.main_gemm_op)) {
                    expected = "retire of a residency that is not resident or already retired";
                } else {
                    for (size_t gemm : gemms) {
                        if (!depends(op, gemm)) {
                            expected = "retire without completion of residual gemm " + std::to_string(gemm);
                            break;
                        }
                    }
                }
                const auto wanted = expected.empty() ? potal_paired_status::ok :
                    potal_paired_status::invalid_execution_plan;
                if (!expected.empty()) expected = "op " + std::to_string(retire) + " PAIR_RETIRE: " + expected;
                for (bool live : {false, true}) {
                    std::string why = "stale";
                    const auto actual = potal_validate_execution_plan_mode(
                        plan, f.runs ? &d : nullptr, &why, potal_paired_hw_geometry(), live);
                    require(actual == wanted && why == expected,
                        std::string(f.name) + ": retirement status/diagnostic mismatch: " + why + " expected " + expected);
                    results.record("status", {retire, order, live, static_cast<uint64_t>(actual)});
                    results.text(why.c_str());
                    if (results.verbose) std::printf("why %s\n", why.c_str());
                    ++checks;
                }
            }
        };
        compare(original);
        auto unrelated = original;
        for (size_t id = 0; id < retire; ++id) unrelated.push_back({id, false});
        compare(unrelated);
        auto missing_main = original;
        missing_main.erase(missing_main.begin());
        compare(missing_main);
        compare({{retire, true}}); // Direction error wins even when Main is absent.
        for (size_t gemm : gemms) {
            auto missing = original;
            missing.erase(std::remove_if(missing.begin(), missing.end(),
                [gemm](const potal_paired_dep & x) { return x.op == gemm; }), missing.end());
            compare(missing);
            for (const std::vector<bool> & bits : {std::vector<bool>{false}, {false, true}, {true, false}, {false, false}}) {
                auto duplicates = missing;
                for (bool bit : bits) duplicates.push_back({gemm, bit});
                compare(duplicates);
            }
        }
        auto all_false = original;
        for (auto & dep : all_false) dep.complete = dep.op == residency.main_gemm_op;
        compare(all_false); // Earliest missing residual must win.
        auto main_duplicates = original;
        main_duplicates.insert(main_duplicates.begin(), {residency.main_gemm_op, false});
        compare(main_duplicates);
    }
}

} // namespace

int main(int argc, char ** argv) try {
    bool verbose = false;
    size_t repetitions = 0;
    if (argc == 2 && std::strcmp(argv[1], "--manifest") == 0) {
        verbose = true;
    } else if (argc == 3 && std::strcmp(argv[1], "--bench") == 0) {
        const char * end = argv[2] + std::strlen(argv[2]);
        const auto parsed = std::from_chars(argv[2], end, repetitions);
        require(parsed.ec == std::errc{} && parsed.ptr == end && repetitions >= 1 && repetitions <= 1000,
                "--bench requires 1..1000 full builds per fixture");
    } else {
        require(argc == 1, "usage: potal-plan-deps-test [--manifest | --bench N]");
    }
    const auto hw = potal_paired_hw_geometry();
    require(hw.dim == 32 && hw.bank_num == 4 && hw.bank_rows == 2048 && hw.acc_rows == 512 &&
            hw.index_entries == 64 && hw.block_k == 32, "not the W4 DIM32/Index64 geometry");
    manifest results;
    results.verbose = verbose;
    size_t checks = 0;
    for (const auto & f : fixtures) {
        const auto d = packet(f);
        potal_paired_execution_plan reference;
        std::string why;
        require(build(f, d, reference, why) == potal_paired_status::ok, std::string(f.name) + ": " + why);
        manifest expected;
        expected.verbose = verbose;
        inventory(expected, reference);
        results.text(f.name);
        results.record("plan", {expected.hash});
        if (repetitions == 0) {
            regression(f, d, reference, results, checks);
        } else {
            int64_t elapsed_ns = 0;
            uint64_t checksum = 0;
            for (size_t i = 0; i < repetitions; ++i) {
                potal_paired_execution_plan built;
                std::string detail;
                const auto start = std::chrono::steady_clock::now();
                const auto status = build(f, d, built, detail);
                const auto stop = std::chrono::steady_clock::now();
                elapsed_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                require(status == potal_paired_status::ok && detail.empty(), std::string(f.name) + ": " + detail);
                manifest actual;
                inventory(actual, built);
                require(actual.hash == expected.hash, "fresh full plan manifest changed");
                checksum = checksum * UINT64_C(1099511628211) ^ actual.hash;
            }
            std::printf("W4_PLAN_BUILD %s builds=%zu total_build_ns=%" PRId64 " checksum=%016" PRIx64
                        " plan=%016" PRIx64 "\n", f.name, repetitions, elapsed_ns, checksum, expected.hash);
        }
        std::printf("W4_PLAN %s manifest=%016" PRIx64 "\n", f.name, expected.hash);
    }
    std::printf("W4_PLAN_DEPS PASS checks=%zu statuses=%016" PRIx64 "\n", checks, results.hash);
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "W4_PLAN_DEPS FAIL %s\n", e.what());
    return 1;
}
