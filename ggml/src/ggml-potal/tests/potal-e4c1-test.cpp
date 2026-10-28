// Link the real support/e4c1-sequential.cpp, the checked production C11 API,
// and the same ggml/PoTal objects as potal-resident-model-test. No wrappers.
// Default and only execution mode: software command model, hardware=0.
#include "support/e4c1-sequential.hpp"
#include "support/e4b2-emulator.hpp"
#include "support/paired-support.hpp"
#include "support/paired-h2-emulator.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <sys/wait.h>
#include <unistd.h>

using namespace ggml::potal;
using namespace potal_test;
namespace seq = potal_test::e4c1;

namespace {
constexpr size_t M = 17, N = 145, K = 928;
constexpr int32_t sentinel = INT32_C(0x13579bdf);
static_assert(PD == 16, "E4C1 fixtures require DIM16");
uint64_t assertions = 0, runs = 0, cells = 0, reference_runs = 0;

void require(bool value, const char *message) {
    ++assertions;
    if (!value) throw std::runtime_error(message);
}

int32_t clamp(golden_wide x) {
    return x < INT32_MIN ? INT32_MIN : x > INT32_MAX ? INT32_MAX : int32_t(x);
}

// Threshold arithmetic independent of the SCU implementation and command model.
int32_t shift(int64_t dot, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || dot == 0) return 0;
    if (carrier >= 32) return dot < 0 ? INT32_MIN : INT32_MAX;
    return clamp(golden_wide(dot) * (golden_wide(1) << carrier));
}

struct expected {
    std::vector<int32_t> main;
    std::vector<std::vector<golden_lane_row>> lanes;
    std::vector<std::vector<int64_t>> correction;
};

// Decode the sealed packet only; do not use decoded lowering, Index addresses,
// scheduler fragments, command-model arithmetic or production SCU arithmetic.
expected golden(const paired_fixture &f, const paired_inputs &in) {
    expected g;
    g.main.resize(f.M * f.N);
    g.lanes.resize(in.packets.size());
    g.correction.resize(in.packets.size());
    for (size_t i = 0; i < f.M; ++i) for (size_t j = 0; j < f.N; ++j) {
        int32_t acc = 0;
        for (size_t first = 0; first < f.K; first += PD) {
            int64_t dot = 0;
            for (size_t k = first; k < first + PD; ++k)
                dot += int64_t(f.a.get()[i * f.lda + k]) * f.code_at(k, j);
            acc = clamp(golden_wide(acc) + shift(dot, f.carrier_at(first / 32, j)));
        }
        g.main[i * f.N + j] = acc;
    }
    for (size_t s = 0; s < in.packets.size(); ++s) {
        if (!in.packets[s]) continue;
        const auto &p = *in.packets[s];
        for (uint8_t lane = 0; lane < p.lane_capacity; ++lane)
            for (size_t row = 0; row < p.row_count; ++row) {
                std::vector<std::vector<int32_t>> digits(p.blocks.size());
                bool active = false;
                for (size_t r = 0; r < p.blocks.size(); ++r) {
                    const auto &b = p.blocks[r];
                    digits[r].assign(b.compact_k_count, 0);
                    for (uint8_t pos = 0; pos < b.active_lane_count; ++pos) {
                        if (b.lane_ids[pos] != lane) continue;
                        for (size_t k = 0; k < b.compact_k_count; ++k) {
                            require(grmd::read_packet_digit(p, b, pos, row, k, digits[r][k]) ==
                                        grmd::RmdStatus::success, "sealed packet digit");
                            active = active || digits[r][k] != 0;
                        }
                    }
                }
                if (!active) continue;
                golden_lane_row result;
                result.lane = lane; result.row = uint32_t(row);
                result.c.resize(f.N);
                for (size_t j = 0; j < f.N; ++j) {
                    int32_t acc = 0;
                    for (size_t r = 0; r < p.blocks.size(); ++r) {
                        const auto &b = p.blocks[r];
                        for (size_t first = 0; first < b.compact_k_count; first += PD) {
                            int64_t dot = 0;
                            for (size_t x = first; x < std::min(first + PD, size_t(b.compact_k_count)); ++x) {
                                const size_t k = b.block_id * 32 + p.k_indices[b.k_index_offset + x];
                                dot += int64_t(digits[r][x]) * f.code_at(k, j);
                            }
                            acc = clamp(golden_wide(acc) + shift(dot, f.carrier_at(b.block_id, j)));
                        }
                    }
                    result.c[j] = acc;
                }
                g.lanes[s].push_back(std::move(result));
            }
        require(compose_golden(g.lanes[s], p.row_count, f.N, g.correction[s]),
                "independent checked wide radix composition");
    }
    return g;
}

struct observation {
    std::vector<potal_paired_cmd_record> records;
    uint64_t destroyed = 0, loops = 0, b_mvins = 0, poison_checks = 0;
    uint64_t drains = 0, peak_scale = 0, peak_index = 0;
    bool fired = false, empty = false, live_scale = false;
};

// Context is adopted by C11 on open. A quarantine child deliberately leaves it
// and all request buffers alive until _exit, just like the existing B2 tests.
struct transport {
    e4b2::machine model;
    observation &out;
    size_t fail_at = SIZE_MAX;
    uint32_t cleanup_fault = 0;
    uint64_t live_slots = 0;
    bool after = true;
    struct source {
        const int8_t *p;
        uint32_t rows, columns, address;
        uint64_t stride;
        std::vector<int8_t> original;
    };
    std::vector<source> loaded;
    explicit transport(observation &o) : out(o) {}

    static potal_status sink(void *opaque, const potal_paired_cmd_record *r) {
        auto &t = *static_cast<transport *>(opaque);
        t.out.records.push_back(*r);
        const size_t index = t.out.records.size() - 1;
        const bool fault = index == t.fail_at;
        if (fault) t.out.fired = true;
        if (fault && !t.after) return POTAL_ERR_BAD_RESPONSE;
        if (r->funct == 2 && ((uint32_t(r->rs2) / 4096) & 1)) {
            source b{reinterpret_cast<const int8_t *>(uintptr_t(r->rs1)), uint32_t(r->rs2 >> 48),
                     uint16_t(r->rs2 >> 32), uint32_t(r->rs2), t.model.load_stride[0], {}};
            for (uint32_t i = 0; i < b.rows; ++i)
                b.original.insert(b.original.end(), b.p + i * b.stride, b.p + i * b.stride + b.columns);
            t.loaded.push_back(std::move(b));
            ++t.out.b_mvins;
        }
        if (r->funct == 8) {
            for (const auto &b : t.loaded) {
                bool changed = false;
                for (uint32_t i = 0; i < b.rows; ++i)
                    for (uint32_t j = 0; j < b.columns; ++j) {
                        const auto original = b.original[i * b.columns + j];
                        changed = changed || b.p[i * b.stride + j] != original;
                        require(t.model.sp[(b.address + i) * PD + j] == original,
                                "completed resident SP retains original B after source poisoning");
                    }
                require(changed, "completed B source poisoned before consumers");
            }
            t.out.poison_checks += t.loaded.size();
            t.loaded.clear();
            ++t.out.loops;
        }
        const bool scale_command = r->funct == 23 || r->funct == 24;
        const auto &slot = t.model.scales[r->rs1 & 255];
        const bool was_live = scale_command &&
            std::any_of(slot.begin(), slot.end(), [](const auto &c) { return c.live; });
        const auto status = t.model.command(*r);
        if (scale_command) {
            const bool live = std::any_of(slot.begin(), slot.end(), [](const auto &c) { return c.live; });
            if (live && !was_live) ++t.live_slots;
            if (was_live && !live) --t.live_slots;
            t.out.peak_scale = std::max(t.out.peak_scale, t.live_slots);
        }
        if (r->funct == 27)
            for (const auto &ix : t.model.indices)
                t.out.peak_index = std::max(t.out.peak_index, uint64_t(ix.active));
        if (fault && t.cleanup_fault) {
            t.model.fault_funct = t.cleanup_fault;
            t.model.fault_occurrence = 1;
        }
        return fault ? POTAL_ERR_BAD_RESPONSE : status;
    }
    static potal_status counter(void *opaque, const potal_paired_cmd_record *r, uint64_t *value) {
        auto &t = *static_cast<transport *>(opaque);
        t.out.records.push_back(*r);
        return t.model.counter(r->rs1, r->rs2, value);
    }
    static void destroy(void *opaque) {
        auto *t = static_cast<transport *>(opaque);
        ++t->out.destroyed;
        t->out.empty = t->model.work.empty();
        t->out.drains = t->model.drains;
        for (const auto &slot : t->model.scales)
            for (const auto &c : slot) t->out.live_scale = t->out.live_scale || c.live;
        delete t;
    }
    potal_resident_ws_options options() { return {{sink, this}, destroy, counter}; }
};

struct invocation {
    residual::decoded_packet decoded;
    potal_tile_plan main;
    potal_paired_tile_plan tile;
    potal_paired_execution_plan plan;
    std::unique_ptr<int32_t, dma_deleter> mc, rc;
    size_t ldc = 0, main_cells = 0, residual_cells = 0;
    uint32_t generation = 254;
    potal_paired_request request;

    invocation(const paired_fixture &f, const paired_inputs &in, size_t stripe) {
        const size_t begin = stripe * PD, rows = std::min(PD, f.M - begin);
        require(potal_frozen_plan(f.M, f.N, f.K, 1, 9, 28, 2, &main) == POTAL_OK,
                "frozen Main I1/J9/K28");
        if (in.packets[stripe])
            require(residual::decode_packet(*in.packets[stripe], f.weight, decoded) ==
                        grmd::RmdStatus::success, "production sealed packet decode");
        ldc = align_up(f.N, POTAL_HW_MAX_BYTES / sizeof(int32_t));
        main_cells = align_up(rows, PD) * ldc;
        residual_cells = align_up(std::max(size_t(1), decoded.rows.size()), PD) * ldc;
        mc.reset(static_cast<int32_t *>(dma_alloc(main_cells * sizeof(int32_t))));
        rc.reset(static_cast<int32_t *>(dma_alloc(residual_cells * sizeof(int32_t))));
        require(mc && rc, "aligned raw output buffers");
        std::fill_n(mc.get(), main_cells, sentinel); std::fill_n(rc.get(), residual_cells, sentinel);
        request.stripe_id = stripe; request.row_begin = begin;
        request.rows = rows; request.n = f.N; request.k = f.K;
        request.main_plan = &main; request.a = f.a.get() + begin * f.lda; request.lda = f.lda;
        request.weight = &f.weight; request.residual = in.packets[stripe] ? &decoded : nullptr;
        request.c_main = mc.get(); request.ldc_main = ldc;
        request.c_residual = rc.get(); request.ldc_residual = ldc; request.generation = &generation;
    }
    void freeze() {
        std::string why;
        require(potal_paired_plan_stripe(main, request.rows, request.n, request.k, request.residual,
                                        tile, plan, &why) == potal_paired_status::ok,
                "existing active H2 fragment and tile plan");
        require(tile.tile_I == 1 && tile.tile_J == 9 && tile.tile_K == 28 &&
                tile.k_panels == 3 && tile.j_groups == 2, "active plan remains I1/J9/K28 with tails");
        request.plan = &plan;
    }
    bool unpublished() const {
        return std::all_of(mc.get(), mc.get() + main_cells, [](int32_t x) { return x == sentinel; }) &&
               std::all_of(rc.get(), rc.get() + residual_cells, [](int32_t x) { return x == sentinel; });
    }
};

enum class kind { zero, sparse, dense, ordered, reordered, compact, fragments, negative, main_tail, residual_tail };
const char *name(kind k) {
    switch (k) {
        case kind::zero: return "zero";
        case kind::sparse: return "sparse";
        case kind::dense: return "dense-three-lane";
        case kind::ordered: return "MAX-MAX-MIN";
        case kind::reordered: return "MAX-MIN-MAX";
        case kind::compact: return "compact-cancellation";
        case kind::fragments: return "carrier24-fragments";
        case kind::negative: return "negative-saturation";
        case kind::main_tail: return "main-only-tail";
        case kind::residual_tail: return "residual-only-tail";
    }
    throw std::runtime_error("unknown fixture");
}
bool witness_kind(kind k) {
    return k == kind::ordered || k == kind::reordered || k == kind::compact ||
           k == kind::fragments || k == kind::negative;
}
paired_fixture make_fixture(kind mode, size_t rows = M) {
    return paired_fixture(rows, N, K, 20261023,
        [=](size_t block, size_t j) -> int16_t {
            if (witness_kind(mode)) return mode == kind::fragments ? 24 : 31;
            // Equal carriers across block identities, adjacent different
            // carriers and explicit zero carriers, all in one fixture.
            return (block + j) % 11 == 0 ? INT16_MIN : int16_t((block / 2 + j) % 4);
        },
        [=](size_t k, size_t j, std::mt19937 &) -> int8_t {
            if (mode == kind::ordered) return k == 0 || k == 416 ? 1 : k == 448 ? -1 : 0;
            if (mode == kind::reordered) return k == 0 || k == 448 ? 1 : k == 416 ? -1 : 0;
            if (mode == kind::compact) return k == 0 ? 1 : k == 16 ? -1 : 0;
            if (mode == kind::fragments) return k < 16 ? 127 : k == 16 ? -127 : 0;
            if (mode == kind::negative) return k == 0 ? -1 : 0;
            return int8_t(int((k * 3 + j) % 7) - 3);
        },
        [=](size_t row, size_t k, std::mt19937 &) -> int8_t {
            return witness_kind(mode) ? 1 : int8_t(int((row + k) % 5) - 2);
        });
}
paired_inputs inputs(const paired_fixture &f, kind mode) {
    paired_inputs in;
    require(make_inputs(f, PD, [=, &f](size_t stripe, size_t rows) {
        std::vector<res_entry> e;
        if (mode == kind::zero) return e;
        if (mode == kind::ordered || mode == kind::reordered)
            return std::vector<res_entry>{{0, 0, 1}, {0, 416, 1}, {0, 448, 1}};
        if (mode == kind::compact) return std::vector<res_entry>{{0, 0, 1}, {0, 16, 1}};
        if (mode == kind::negative) return std::vector<res_entry>{{0, 0, 1}};
        if (mode == kind::fragments) {
            for (size_t k = 0; k < 17; ++k) e.push_back({0, k, 1});
            return e;
        }
        if (mode == kind::main_tail) return std::vector<res_entry>{{0, 0, 1}};
        const size_t counts[] = {1, 15, 16, 17, 31, 32};
        for (size_t block = 0; block < f.K / 32; ++block)
            for (size_t x = 0; x < ((mode == kind::sparse) ? counts[block % 6] : 32); ++x) {
                const int32_t sign = (block + x + stripe) % 2 ? -1 : 1;
                if (mode == kind::sparse) e.push_back({(block + x + stripe) % rows, block * 32 + x, sign * 7});
                else for (size_t row = 0; row < rows; ++row)
                    e.push_back({row, block * 32 + x, sign * 0x10101});
            }
        return e;
    }, in), "sealed residual fixture");
    return in;
}

uint64_t count(const observation &o, uint32_t funct) {
    return std::count_if(o.records.begin(), o.records.end(), [=](const auto &r) { return r.funct == funct; });
}

void check_stream(const invocation &v, const observation &o, const seq::statistics &s, unsigned grouping) {
    require(s.software_only && o.destroyed == 1 && o.empty && !o.live_scale,
            "software-only successful close drains and releases adopted context");
    require(s.peak_scale_slots <= 256 && s.peak_index_entries <= 32 &&
            o.peak_scale <= 256 && o.peak_index <= 32, "physical Scale/Index bounds");
    require(s.peak_scale_slots == o.peak_scale && s.peak_index_entries == o.peak_index,
            "reported peaks match independent command model");
    uint64_t mvins = 0;
    for (const auto &panel : v.tile.panels)
        for (size_t j = 0; j < v.request.n; j += v.tile.tile_J * PD)
            mvins += ((panel.k_rows + PD - 1) / PD) *
                     ((std::min(v.tile.tile_J * PD, v.request.n - j) + PD - 1) / PD);
    require(s.b_loads == v.tile.k_panels * v.tile.j_groups && s.b_retirements == s.b_loads &&
            s.b_mvins == mvins && o.b_mvins == mvins && count(o, 2) == mvins && o.poison_checks == mvins,
            "one poisoned B load per original panel/J residency, no reload for consumers");
    require(s.loops == count(o, 8) && s.loops == s.main_microtiles + s.residual_microtiles &&
            s.views == count(o, 28) && s.mvouts == count(o, 3) &&
            s.index_configs == count(o, 27) && s.index_writes == count(o, 26) &&
            s.main_scale_writes + s.residual_scale_writes == count(o, 23) &&
            s.main_scale_releases + s.residual_scale_releases == count(o, 24),
            "scheduler counts agree with actual software command transcript");
    uint64_t main = 0, residual = 0, main_partials = 0, residual_partials = 0;
    uint64_t mr = 0, rm = 0;
    int previous = -1;
    bool pair[2] = {false, false};
    bool loading = false;
    uint32_t k_tiles = 0;
    for (const auto &r : o.records) {
        if (r.funct == 2) {
            // Switch attribution is inside one B residency, never across a
            // replacement or J-group boundary.
            if (!loading) previous = -1;
            loading = true;
        } else if (r.funct == 8) {
            loading = false;
            const unsigned path = (r.rs1 >> 62) & 1;
            require((r.rs1 & 8) && (r.rs1 & (UINT64_C(1) << 61)) && r.xd,
                    "every LOOP uses checked resident mode, never CPU/legacy fallback");
            require(k_tiles && k_tiles <= (path ? 2u : grouping), "bounded Main K grouping / residual Index window");
            const auto selector = (r.rs1 >> 16) & 3;
            require(selector == 1 || selector == 2, "physical pair selector");
            pair[selector - 1] = true;
            if (path) { ++residual; residual_partials += k_tiles; }
            else { ++main; main_partials += k_tiles; }
            if (previous == 0 && path == 1) ++mr;
            if (previous == 1 && path == 0) ++rm;
            previous = int(path);
        } else if (r.funct == 9) k_tiles = uint16_t(r.rs2 >> 32);
    }
    require(pair[0] && pair[1] && main == s.main_microtiles && residual == s.residual_microtiles &&
            main_partials == s.main_partials && residual_partials == s.residual_partials,
            "both physical pairs and all path partials observed");
    require(s.main_to_residual == mr && s.residual_to_main == rm, "real path switch counts, not estimated alternations");
    // Immutable output epochs cross B replacements. A given path/group/J
    // stream may interleave with the other path, never with its own order.
    using key = std::tuple<unsigned, uint32_t, uint64_t>;
    std::map<key, uint64_t> next, original;
    std::map<key, bool> finished;
    uint64_t event_main = 0, event_residual = 0, event_load = 0, event_retire = 0;
    for (const auto &e : s.events) {
        if (e.kind == seq::event_kind::load) { ++event_load; continue; }
        if (e.kind == seq::event_kind::retire) { ++event_retire; continue; }
        const bool residual_path = e.kind == seq::event_kind::residual;
        const key stream{unsigned(residual_path), e.group, e.original_j};
        require(!finished[stream] && e.sequence == next[stream] && bool(e.first) == (e.sequence == 0),
                "canonical per-path contribution sequence across B residencies");
        if (e.sequence) require(e.original_k > original[stream], "original K fragments remain strictly ordered");
        require(e.k_rows > 0 && e.k_rows <= (residual_path ? 32u : grouping * PD),
                "event extent agrees with bounded issued partials");
        next[stream] += (e.k_rows + PD - 1) / PD;
        original[stream] = e.original_k;
        finished[stream] = e.last != 0;
        if (residual_path) ++event_residual; else ++event_main;
    }
    require(event_main == main && event_residual == residual &&
            event_load == s.b_loads && event_retire == s.b_retirements,
            "trace describes every actual path issue and residency transition");
    for (const auto &stream : finished) require(stream.second, "every physical output epoch has final contribution");
}

void check_results(const paired_fixture &f, const paired_inputs &in, const expected &g,
                   const invocation &v, size_t stripe) {
    const size_t begin = stripe * PD;
    for (size_t row = 0; row < v.request.rows; ++row) {
        require(std::memcmp(v.mc.get() + row * v.ldc, g.main.data() + (begin + row) * f.N,
                            f.N * sizeof(int32_t)) == 0, "independent Main INT32");
        for (size_t j = f.N; j < v.ldc; ++j)
            require(v.mc.get()[row * v.ldc + j] == sentinel, "Main padding stays untouched");
    }
    std::vector<int32_t> raw;
    for (size_t row = 0; row < v.decoded.rows.size(); ++row)
        raw.insert(raw.end(), v.rc.get() + row * v.ldc, v.rc.get() + row * v.ldc + f.N);
    require(same_residual(g.lanes[stripe], v.decoded.rows, raw, f.N),
            "independent Residual INT32, lane/source-row order");
    grmd::BlockScaledInt64Correction correction;
    if (in.packets[stripe]) {
        require(residual::reconstruct_rows(*in.packets[stripe], v.decoded.rows, v.rc.get(), v.ldc, f.N,
                                          correction) == grmd::RmdStatus::success &&
                same_bits(correction.values, g.correction[stripe]), "production checked radix INT64 equals wide golden");
    }
    std::vector<float> actual(f.M * f.N, -19.0f), want = actual;
    for (size_t row = 0; row < v.request.rows; ++row) {
        reconstruct_row(v.mc.get() + row * v.ldc, f.N, double(std::ldexp(1.0f, in.theta[stripe])),
                        f.weight.scale, actual.data() + (begin + row) * f.N, 1);
        for (size_t j = 0; j < f.N; ++j) {
            float value = main_golden(g.main[(begin + row) * f.N + j], in.theta[stripe], f.scales[j]);
            if (in.packets[stripe])
                value = merge_golden(value, g.correction[stripe][row * f.N + j], in.theta[stripe], f.scales[j]);
            want[(begin + row) * f.N + j] = value;
        }
    }
    if (in.packets[stripe]) {
        ggml::gemmini::quants::act::exsia::StripeMetadataSnapshot meta;
        meta.theta = in.theta[stripe];
        require(residual::merge_prepared_hp1(f.weight, begin, begin + v.request.rows, meta, correction,
                                            actual.data(), f.M, f.N, 1, nullptr) == grmd::RmdStatus::success,
                "existing checked legacy F32 merge");
    }
    require(same_bits(actual, want), "same-policy legacy F32 merge, including untouched neighboring stripe");
    cells += v.request.rows * f.N + raw.size();
}

void compare_existing_models(const paired_fixture &f, const paired_inputs &in, const expected &g) {
    const auto serial = serialized_of(f, PD, in);
    require(serial.main_ok && serial.legacy_ok && same_bits(serial.main_c, g.main),
            "existing automatic Serial integer result equals independent wide golden");
    ++reference_runs;
    for (size_t s = 0; s < in.packets.size(); ++s) {
        require(same_residual(g.lanes[s], serial.rows[s], serial.residual_c[s], f.N) &&
                same_bits(g.correction[s], serial.correction[s]),
                "existing Serial compact Residual and INT64 result equals independent golden");
    }
    std::vector<paired_stripe_input> stripes(in.packets.size());
    for (size_t s = 0; s < stripes.size(); ++s)
        stripes[s] = {in.theta[s], in.packets[s].get()};
    paired_gemm_desc desc;
    desc.m = f.M; desc.n = f.N; desc.k = f.K;
    desc.a = f.a.get(); desc.lda = f.lda; desc.weight = &f.weight;
    desc.stripe_rows = PD; desc.stripes = stripes.data(); desc.stripe_count = stripes.size();
    desc.tile_I = 1; desc.tile_J = 9; desc.tile_K = 28;
    paired_output output;
    paired_execution_report model_report;
    require(paired_tiled_matmul_auto(desc, output, model_report) == POTAL_OK &&
            model_report.plan.tile_I == 1 && model_report.plan.tile_J == 9 &&
            model_report.plan.tile_K == 28 && same_bits(output.main_c, g.main),
            "existing E3 alternating software schedule retains I1 J9 K28 and Main integers");
    ++reference_runs;
    require(same_bits(output.out, serial.legacy), "existing E3 and Serial same-policy merge is bitwise equal");
    for (size_t s = 0; s < stripes.size(); ++s) {
        require(same_residual(g.lanes[s], output.stripes[s].rows, output.stripes[s].residual_c, f.N) &&
                same_bits(g.correction[s], output.stripes[s].correction),
                "existing E3 compact Residual and checked INT64 equal independent golden");
        invocation v(f, in, s);
        v.freeze();
        h2_emulator model;
        auto sink = model.sink();
        potal_paired_report report;
        require(potal_paired_run_session(v.request, &sink, report) == potal_paired_status::ok &&
                !report.fatal && model.errors.empty() && !model.scale_miss,
                "existing whole-loop Paired H2 command model executes the identical packet");
        check_results(f, in, g, v, s);
        ++reference_runs;
    }
}

void numerical(kind mode) {
    auto f = make_fixture(mode);
    require(f.ok, "prepared HP1 fixture");
    const auto in = inputs(f, mode);
    const auto g = golden(f, in);
    if (mode == kind::zero)
        require(std::any_of(g.main.begin(), g.main.end(), [](int32_t x) { return x != 0; }),
                "zero Residual fixture retains nonzero Main work");
    compare_existing_models(f, in, g);
    if (witness_kind(mode)) {
        const int32_t m = mode == kind::fragments ? 16777215 : mode == kind::negative ? INT32_MIN :
                          mode == kind::reordered ? 2147483646 : -1;
        const int32_t r = mode == kind::compact ? 0 : m;
        require(std::all_of(g.main.begin(), g.main.end(), [=](int32_t x) { return x == m; }),
                "manual Main saturation/order witness");
        for (const auto &stripe : g.lanes)
            require(stripe.size() == 1 && std::all_of(stripe[0].c.begin(), stripe[0].c.end(),
                        [=](int32_t x) { return x == r; }), "manual compact Residual witness");
    }
    for (unsigned grouping : {1u, 2u, 4u}) {
        uint64_t main_tail = 0, residual_tail = 0;
        for (size_t stripe = 0; stripe < in.packets.size(); ++stripe) {
            invocation v(f, in, stripe);
            v.freeze();
            observation o;
            auto *t = new transport(o);
            const auto opt = t->options();
            seq::options config;
            config.main_k_tiles = grouping; config.acc_tile = 1; config.poison_b = true; config.trace = true;
            seq::statistics stats;
            potal_paired_report report;
            const auto status = seq::run(v.request, &opt, config, report, stats);
            if (status != potal_paired_status::ok)
                std::fprintf(stderr, "E4C1 case=%s grouping=%u stripe=%zu status=%s stage=%s detail=%s\n",
                             name(mode), grouping, stripe, potal_paired_status_name(status),
                             report.stage ? report.stage : "-", report.detail.c_str());
            require(status == potal_paired_status::ok && !report.fatal &&
                    report.hw_status == POTAL_OK, "real sequential executor succeeds without fallback");
            ++runs;
            check_results(f, in, g, v, stripe);
            check_stream(v, o, stats, grouping);
            main_tail += stats.main_only_tail; residual_tail += stats.residual_only_tail;
            std::printf("E4C1_SOFTWARE_COUNTS case=%s group=%u stripe=%zu Main=%llu Residual=%llu "
                        "ScaleWriteMain=%llu ScaleWriteResidual=%llu ScaleReleaseMain=%llu "
                        "ScaleReleaseResidual=%llu IndexConfig=%llu IndexWrite=%llu commands=%llu "
                        "peakScale=%llu peakIndex=%llu hardware=0\n", name(mode), grouping, stripe,
                        static_cast<unsigned long long>(stats.main_microtiles),
                        static_cast<unsigned long long>(stats.residual_microtiles),
                        static_cast<unsigned long long>(stats.main_scale_writes),
                        static_cast<unsigned long long>(stats.residual_scale_writes),
                        static_cast<unsigned long long>(stats.main_scale_releases),
                        static_cast<unsigned long long>(stats.residual_scale_releases),
                        static_cast<unsigned long long>(stats.index_configs),
                        static_cast<unsigned long long>(stats.index_writes),
                        static_cast<unsigned long long>(stats.commands),
                        static_cast<unsigned long long>(stats.peak_scale_slots),
                        static_cast<unsigned long long>(stats.peak_index_entries));
        }
        if (mode == kind::main_tail) require(main_tail > 0, "real Main-only remainder");
        if (mode == kind::residual_tail && grouping == 4) require(residual_tail > 0, "real Residual-only remainder");
        std::printf("E4C1_CASE name=%s grouping=%u hardware=0\n", name(mode), grouping);
    }
}

void refusals() {
    auto f = make_fixture(kind::sparse);
    require(f.ok, "refusal fixture");
    const auto in = inputs(f, kind::sparse);
    // Fixed physical ACC placement must fit BEFORE opening/issuing, rather
    // than introducing spills, changing the active plan, or serial fallback.
    for (unsigned tile : {32u, 31u}) {
        invocation v(f, in, 0);
        v.freeze();
        observation o;
        auto *t = new transport(o);
        const auto opt = t->options();
        seq::options config;
        config.acc_tile = tile;
        seq::statistics stats;
        potal_paired_report report;
        const auto status = seq::run(v.request, &opt, config, report, stats);
        require(potal_paired_capacity_refusal(status) && !report.issued && o.records.empty() &&
                v.unpublished(), "fixed ACC placement capacity refused before any command");
        require(o.destroyed == 0, "pre-open refusal leaves caller transport owned");
        delete t;
    }
    for (unsigned grouping : {0u}) {
        invocation v(f, in, 0);
        observation o;
        auto *t = new transport(o);
        const auto opt = t->options();
        seq::options config;
        config.main_k_tiles = grouping;
        seq::statistics stats;
        potal_paired_report report;
        require(seq::run(v.request, &opt, config, report, stats) == potal_paired_status::invalid_argument &&
                o.records.empty() && v.unpublished(), "zero Main grouping rejected atomically");
        require(o.destroyed == 0, "invalid options do not adopt transport");
        delete t;
    }
    invocation v(f, in, 0);
    observation o;
    auto *t = new transport(o);
    t->model.supported = false;
    const auto opt = t->options();
    seq::statistics stats;
    potal_paired_report report;
    require(seq::run(v.request, &opt, {}, report, stats) != potal_paired_status::ok &&
            count(o, 8) == 0 && count(o, 2) == 0 && v.unpublished(),
            "missing resident capability refuses rather than falling back to CPU/legacy");
    // Failed open retains the caller's context (C11 public ownership contract).
    require(o.destroyed == 0, "failed capability open does not destroy caller context");
    delete t;
}

struct fault {
    const char *label;
    size_t command;
    uint32_t funct;
};

std::vector<fault> fault_manifest() {
    auto f = make_fixture(kind::dense);
    require(f.ok, "failure manifest fixture");
    const auto in = inputs(f, kind::dense);
    invocation v(f, in, 0);
    v.freeze();
    observation o;
    auto *t = new transport(o);
    const auto opt = t->options();
    seq::options config;
    config.trace = true;
    seq::statistics stats;
    potal_paired_report report;
    require(seq::run(v.request, &opt, config, report, stats) == potal_paired_status::ok &&
            o.destroyed == 1, "real successful transcript defines deterministic injection indices");
    std::vector<fault> result;
    auto select = [&](const char *label, uint32_t funct, size_t occurrence, int path = -1) {
        for (size_t index = 0; index < o.records.size(); ++index) {
            const auto &r = o.records[index];
            if (r.funct == funct && (path < 0 || int((r.rs1 >> 62) & 1) == path) && !--occurrence) {
                result.push_back({label, index, funct});
                return;
            }
        }
        throw std::runtime_error(label);
    };
    select("first B MVIN", 2, 1);
    select("middle B MVIN", 2, 126);
    // First two full K28/J9 loads occupy 252 commands each. The third B
    // residency reuses pair0 while ACC epochs span resident generations.
    select("replacement B MVIN", 2, 505);
    select("Main issue", 8, 1, 0);
    select("Residual issue", 8, 1, 1);
    for (uint32_t funct : {9u, 10u, 11u, 12u, 13u, 28u})
        select("Main configuration/view", funct, 1);
    select("Scale first write", 23, 1);
    select("Scale middle write", 23, 8);
    select("Index configure", 27, 1);
    select("Index first write", 26, 1);
    select("Index middle write", 26, 8);
    select("readback first MVOUT", 3, 1);
    select("readback middle MVOUT", 3, 5);
    select("Scale Release", 24, 1);
    select("prepare Fence", 25, 1);
    select("B completion Fence", 25, 2);
    size_t last_fence = SIZE_MAX;
    for (size_t i = 0; i < o.records.size(); ++i)
        if (o.records[i].funct == 25) last_fence = i;
    require(last_fence != SIZE_MAX, "terminal completion/close boundary exists");
    result.push_back({"last completion before close", last_fence, 25});
    return result;
}

void failure(const fault &point, bool after, uint32_t cleanup_fault = 0) {
    auto f = make_fixture(kind::dense);
    require(f.ok, "isolated failure fixture");
    const auto in = inputs(f, kind::dense);
    invocation v(f, in, 0);
    v.freeze();
    observation o;
    auto *t = new transport(o);
    t->fail_at = point.command; t->after = after;
    // The sink arms this function-specific second error only after the
    // original issue error. No elapsed-time or wait-based injection.
    if (cleanup_fault) require(point.funct == 8 && after, "cleanup uncertainty follows an issue failure");
    t->cleanup_fault = cleanup_fault;
    const auto opt = t->options();
    seq::options config;
    config.trace = true;
    seq::statistics stats;
    potal_paired_report report;
    const auto status = seq::run(v.request, &opt, config, report, stats);
    require(o.fired && status == potal_paired_status::execution_failed &&
            report.hw_status == POTAL_ERR_BAD_RESPONSE && report.stage && v.unpublished(),
            "injected command error propagates, raw outputs remain transactionally unpublished");
    if (report.fatal) {
        require(o.destroyed == 0, "unknown completion quarantines adopted context and buffers");
        const auto before = o.records.size();
        potal_paired_session *other = nullptr;
        const potal_paired_sink sink{transport::sink, t};
        require(potal_paired_session_open(&sink, &other) == POTAL_ERR_BUSY && !other &&
                o.records.size() == before, "quarantine retains global issuer; no retry or fallback");
    } else {
        require(!cleanup_fault && report.cleaned && o.destroyed == 1 && o.empty && !o.live_scale,
                "known completion cleanup releases Scale and closes adopted context once");
        observation next;
        auto *fresh = new transport(next);
        const auto options = fresh->options();
        potal_paired_session *session = nullptr;
        require(potal_paired_resident_open(&options, &session) == POTAL_OK &&
                potal_paired_resident_prepare(session) == POTAL_OK &&
                potal_paired_resident_close(&session) == POTAL_OK && !session && next.destroyed == 1,
                "recoverable failed run leaves issuer reusable");
    }
    if (point.funct == 23 || point.funct == 24 || point.funct == 25 || cleanup_fault)
        require(report.fatal, "Scale/Release/Fence response uncertainty cannot become safe-close success");
    std::printf("E4C1_FAULT name=%s after=%u cleanup=%u fatal=%u hardware=0\n",
                point.label, unsigned(after), cleanup_fault, unsigned(report.fatal));
    // Exit here, not in the caller: returning would unwind DMA request buffers
    // while the quarantined context still owns their possible readers.
    std::fflush(nullptr);
    _exit(0);
}

template<class F> void isolated(F action) {
    std::fflush(nullptr);
    const pid_t child = fork();
    require(child >= 0, "fork isolated deterministic quarantine case");
    if (!child) {
        try { action(); std::fflush(nullptr); _exit(0); }
        catch (const std::exception &error) {
            std::fprintf(stderr, "E4C1_ISOLATED_FAIL hardware=0 %s\n", error.what());
            _exit(1);
        }
    }
    int status = 0;
    pid_t result;
    do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated failure/quarantine contract");
}

void failures() {
    const auto manifest = fault_manifest();
    for (const auto &point : manifest)
        for (bool after : {false, true})
            isolated([&] { failure(point, after); });
    const auto main = std::find_if(manifest.begin(), manifest.end(), [](const auto &f) {
        return std::strcmp(f.label, "Main issue") == 0;
    });
    require(main != manifest.end(), "Main failure point for cleanup errors");
    for (uint32_t funct : {25u, 24u}) isolated([&] { failure(*main, true, funct); });
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 1) {
        std::fprintf(stderr, "E4C1 software-only: refusing %s; no real hardware gate is installed.\n", argv[1]);
        return 2;
    }
    try {
        for (const auto mode : {kind::zero, kind::sparse, kind::dense, kind::ordered, kind::reordered,
                               kind::compact, kind::fragments, kind::negative, kind::main_tail, kind::residual_tail})
            numerical(mode);
        refusals();
        failures();
        std::printf("E4C1_SOFTWARE_PASS assertions=%llu runs=%llu cells=%llu reference_runs=%llu hardware=0 "
                    "scope=sequential_executor_and_B2_command_model\n",
                    static_cast<unsigned long long>(assertions), static_cast<unsigned long long>(runs),
                    static_cast<unsigned long long>(cells),
                    static_cast<unsigned long long>(reference_runs));
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "E4C1_SOFTWARE_FAIL hardware=0 %s\n", error.what());
        return 1;
    }
}
