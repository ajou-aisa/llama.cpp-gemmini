// E4A isolated resident-handle test. Default is software-only, including on RISC-V.
// Link tests/potal-resident.c as C11 INSTEAD OF potal.c/potal-api. That translation
// unit owns the sole included core implementation. Do not link E3 context too.
// Hardware additionally needs the existing issuer_owner backend-source closure,
// with its potal-api dependency replaced by that same resident object.
// No core-taking, legacy Scale, raw RoCC, or production runner calls here.
#include "potal-resident.h"
#include "potal_hw_params.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif
#if defined(__riscv) && defined(__linux__)
#include "matmul/execution.hpp"
#endif

namespace {
constexpr uint32_t D = 16, half = 8192, bank_rows = 4096;
static_assert(POTAL_HW_DIM == D && POTAL_HW_BANK_ROWS == bank_rows &&
              POTAL_HW_BANK_NUM == 4 && POTAL_HW_ACC_ROWS == 1024,
              "E4A requires the audited DIM16 geometry");

void check(bool value, const char *what) {
    if (!value) throw std::runtime_error(what);
}
void expect(potal_status got, potal_status wanted, const char *what) {
    if (got != wanted) {
        std::fprintf(stderr, "%s: got %s (%d), expected %s (%d)\n",
                     what, potal_status_name(got), int(got),
                     potal_status_name(wanted), int(wanted));
        throw std::runtime_error(what);
    }
}
void ok(potal_status got, const char *what) { expect(got, POTAL_OK, what); }
bool same(potal_resident_token a, potal_resident_token b) {
    return a.owner == b.owner && a.generation == b.generation;
}

enum class interval : size_t {
    load_issue, load_complete, handoff, microtile_issue, acc_complete,
    scale_setup, index_setup, readback_issue, readback_complete, retire, release, count
};
constexpr const char *interval_names[] = {
    "b_load_issue", "b_load_completion", "resident_handoff", "microtile_issue",
    "acc_completion", "scale_setup", "index_setup", "acc_readback_issue",
    "acc_readback_completion", "residency_retirement", "scale_release"
};
struct measurements {
    std::array<double, size_t(interval::count)> ms{};
    std::array<uint64_t, size_t(interval::count)> calls{};
};
thread_local measurements measured;

// This decoder does not call production encoders, validators, SCU helpers, or
// another emulator. It models command/data semantics, not timing or concurrency.
struct emulator {
    struct scale { bool live = false; uint32_t generation = 0, carrier = 0; };
    struct index {
        uint32_t base = 0, j = 0, rows = 0, count = 0;
        std::array<uint32_t, 32> source{};
        std::array<bool, 32> valid{};
    };
    std::array<int8_t, 16384 * D> sp{};
    std::array<int32_t, 1024 * D> acc{};
    scale scales[256][D]{};
    index table[2];
    uint32_t selected = 0;
    uint64_t load_stride[2]{}, store_stride = 0;
    uint64_t pads = 0, bounds = 0, a = 0, b = 0, lda = 0, ldb = 0;
    std::vector<potal_paired_cmd_record> records;
    uint64_t fences = 0, releases = 0, mvins = 0, loops = 0, mvouts = 0;
    // Fail after the command's modeled side effect, including a partial store.
    uint32_t fail_funct = UINT32_MAX, fail_occurrence = 0, seen_fail_funct = 0;
    uint64_t fail_at = 0;
    potal_status fail_status = POTAL_HW_BAD_COMMAND;
    bool fail_fence = false, fail_release = false, bad_scale = false;
    bool bad = false;
    potal_resident_handle *reentrant_handle = nullptr;
    bool reentered = false;
    static unsigned destroyed;

    static void destroy(void *p) { ++destroyed; delete static_cast<emulator *>(p); }
    static potal_status sink(void *p, const potal_paired_cmd_record *r) {
        try { return static_cast<emulator *>(p)->command(*r); }
        catch (...) {
            static_cast<emulator *>(p)->bad = true;
            return POTAL_ERR_BAD_RESPONSE; // Never unwind through the C issuer.
        }
    }
    static int32_t clamp(int64_t v) {
        return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : int32_t(v);
    }
    static int32_t scu(int64_t dot, uint32_t shift) {
        if (shift == UINT32_C(0x80000000) || dot == 0) return 0;
        if (shift >= 32) return dot < 0 ? INT32_MIN : INT32_MAX;
        return clamp(dot * (INT64_C(1) << shift));
    }
    potal_status command(const potal_paired_cmd_record &r) {
        records.push_back(r);
        if (reentrant_handle && !reentered) {
            reentered = true;
            auto *saved = reentrant_handle;
            void *p = nullptr; size_t bytes = 0;
            potal_resident_microtile t{};
            expect(potal_resident_fence(saved), POTAL_ERR_BUSY, "reentrant Fence");
            expect(potal_resident_prepare(saved), POTAL_ERR_BUSY, "reentrant prepare");
            expect(potal_resident_microtile_issue(saved, &t), POTAL_ERR_BUSY, "reentrant LOOP");
            expect(potal_resident_buffer_view(saved, POTAL_RESIDENT_B0, &p, &bytes),
                   POTAL_ERR_BUSY, "reentrant view");
            expect(potal_resident_scale_release(saved, 0, 255, 0), POTAL_ERR_BUSY, "reentrant Release");
            expect(potal_resident_close(&saved), POTAL_ERR_BUSY, "reentrant close");
            check(saved == reentrant_handle, "reentrant close retains owner");
        }
        const bool response = r.funct >= 23;
        check(r.xd == uint32_t(response), "decoder xd");
        const bool inject = records.size() == fail_at ||
                            (r.funct == fail_funct && ++seen_fail_funct == fail_occurrence);
        if (r.funct == POTAL_FUNCT_FENCE) {
            ++fences;
            check(r.rs1 == 0 && r.rs2 == 0, "Fence fields");
            return fail_fence ? POTAL_HW_BAD_COMMAND : POTAL_OK;
        }
        if (r.funct == POTAL_FUNCT_SCALE_RELEASE) {
            ++releases;
            if (fail_release) return POTAL_HW_BAD_COMMAND;
        }
        if (r.funct == POTAL_FUNCT_SCALE_WRITE && bad_scale)
            return POTAL_ERR_BAD_RESPONSE; // Possible acceptance, deliberately unknown.
        switch (r.funct) {
            case POTAL_FUNCT_FLUSH:
                check(r.rs1 == 0 && r.rs2 == 0, "FLUSH fields");
                break;
            case POTAL_FUNCT_CONFIG:
                if ((r.rs1 & 3) == 1) {
                    const uint32_t state = (r.rs1 >> 3) & 3;
                    check(state < 2 && r.rs1 == (UINT64_C(0x100101) | (state << 3)),
                          "CONFIG_LD identity, block stride, repeats, shrink");
                    load_stride[state] = r.rs2;
                } else if ((r.rs1 & 3) == 2) {
                    check(r.rs1 == 2, "CONFIG_ST full width");
                    store_stride = r.rs2;
                } else {
                    check(r.rs1 == UINT64_C(0x10004) &&
                          r.rs2 == (UINT64_C(1) << 48), "CONFIG_EX WS identity");
                }
                break;
            case POTAL_FUNCT_MVIN: {
                ++mvins;
                const uint32_t row = uint32_t(r.rs2);
                const uint32_t nr = r.rs2 >> 48, nc = (r.rs2 >> 32) & 65535;
                check(row < 16384 && nr > 0 && nr <= D && nc > 0 && nc <= D &&
                      row + nr <= 16384, "raw B MVIN shape/local address");
                const auto *src = reinterpret_cast<const int8_t *>(uintptr_t(r.rs1));
                for (uint32_t y = 0; y < (inject ? 1u : nr); ++y)
                    std::copy_n(src + y * load_stride[0], nc, &sp[(row + y) * D]);
                break;
            }
            case POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS: pads = r.rs1; bounds = r.rs2; break;
            case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB:
                a = r.rs1; b = r.rs2;
                check(b == 0, "resident consumer must never carry B DRAM");
                break;
            case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC:
            case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC:
                check(r.rs1 == 0 && r.rs2 == 0, "resident D/C disabled"); break;
            case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB: lda = r.rs1; ldb = r.rs2; break;
            case POTAL_FUNCT_SCALE_WRITE:
            case POTAL_FUNCT_SCALE_RELEASE: {
                const uint32_t slot = r.rs1 & 255, gen = (r.rs1 >> 8) & 255;
                const uint32_t col = (r.rs1 >> 16) & 63;
                check((r.rs1 >> 22) == 0 && col < D, "Scale key fields");
                auto &s = scales[slot][col];
                if (r.funct == POTAL_FUNCT_SCALE_RELEASE) {
                    check(s.live && s.generation == gen && r.rs2 == 0, "Release live key");
                    s.live = false;
                } else {
                    check(!s.live && (r.rs2 <= 32767 || r.rs2 == UINT64_C(0x80000000)),
                          "Scale write carrier/liveness");
                    s = {true, gen, uint32_t(r.rs2)};
                }
                break;
            }
            case POTAL_FUNCT_INDEX_CONFIG: {
                const uint32_t base = uint32_t(r.rs1);
                selected = base / half;
                check(selected < 2, "Index pair");
                auto &t = table[selected];
                t = {};
                t.base = base; t.j = r.rs1 >> 32;
                t.rows = uint32_t(r.rs2); t.count = r.rs2 >> 32;
                check(t.j > 0 && t.rows > 0 && t.count > 0 && t.count <= 32 &&
                      base >= selected * half + bank_rows &&
                      base + ((t.rows + 15) / 16) * t.j * D <= (selected + 1) * half,
                      "full Index descriptor");
                break;
            }
            case POTAL_FUNCT_INDEX_WRITE: {
                auto &t = table[selected];
                check(r.rs1 < t.count && r.rs2 < t.rows, "Index entry");
                t.source[size_t(r.rs1)] = uint32_t(r.rs2);
                t.valid[size_t(r.rs1)] = true;
                break;
            }
            case POTAL_FUNCT_LOOP_WS: execute(r.rs1, r.rs2); break;
            case POTAL_FUNCT_MVOUT: {
                ++mvouts;
                const uint32_t local = uint32_t(r.rs2);
                const uint32_t row = local & ((1u << 29) - 1);
                const uint32_t nr = r.rs2 >> 48, nc = (r.rs2 >> 32) & 65535;
                check((local >> 29) == 5 && nr > 0 && nr <= D &&
                      nc > 0 && nc <= D && row + nr <= 1024, "raw full ACC MVOUT");
                for (uint32_t y = 0; y < (inject ? 1u : nr); ++y) {
                    auto *dst = reinterpret_cast<void *>(uintptr_t(r.rs1 + y * store_stride));
                    std::memcpy(dst, &acc[(row + y) * D], nc * sizeof(int32_t));
                }
                break;
            }
            default: check(false, "unknown resident command");
        }
        return inject ? fail_status : POTAL_OK;
    }
    void execute(uint64_t x, uint64_t y) {
        ++loops;
        const uint32_t I = bounds & 65535, J = (bounds >> 16) & 65535;
        const uint32_t K = (bounds >> 32) & 65535;
        const uint32_t as = (x >> 18) & 3, bs = (x >> 16) & 3;
        const bool residual = (x >> 62) & 1, indexed = (x >> 63) & 1;
        const uint32_t base = (x >> 21) & 65535, gen = (x >> 37) & 255;
        const uint32_t passes = (x >> 45) & 65535;
        check(I == 1 && J == 2 && K == (residual ? 1u : 2u) && pads == 0 &&
              ((x >> 20) & 1) && ((x >> 61) & 1) && ((x >> 1) & 1) &&
              as >= 1 && as <= 2 && as == bs && indexed == residual && b == 0 &&
              passes == (residual ? 1u : 2u) && (y & 7) == 0 &&
              (y >> 3) == (residual ? 2u : 0u) && base + J <= 256, "paired LOOP fields");
        const uint32_t pair = as - 1, ab = pair * half + uint32_t(y >> 3) * D;
        const auto *host_a = reinterpret_cast<const int8_t *>(uintptr_t(a));
        check(a != 0 && lda >= K * D && load_stride[0] == lda &&
              load_stride[1] == ldb, "LOOP A/B strides");
        // Generated LdA packs each DIM tile in the pair's activation bank.
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t r = 0; r < D; ++r)
                std::copy_n(host_a + r * lda + k * D, D, &sp[(ab + k * D + r) * D]);
        const uint32_t cb = residual ? 512 : 0;
        // Native Main uses the RTL half-end formula, not the resident wrapper's base.
        const uint32_t native = (pair + 1) * half - K * J * D;
        auto &t = table[pair];
        if (indexed) check(t.j == 2 && t.rows == 448 && t.count == 16,
                           "Residual must retain full residency Index geometry");
        for (uint32_t k = 0; k < K; ++k) for (uint32_t j = 0; j < J; ++j)
            for (uint32_t c = 0; c < D; ++c) {
                const auto &s = scales[base + (k / passes) * J + j][c];
                check(s.live && s.generation == gen, "LOOP missing Scale lane");
                for (uint32_t r = 0; r < D; ++r) {
                    int64_t dot = 0;
                    for (uint32_t q = 0; q < D; ++q) {
                        uint32_t br = native + (k * J + j) * D + q;
                        if (indexed) {
                            check(t.valid[k * D + q], "unwritten Index entry");
                            const uint32_t source = t.source[k * D + q];
                            br = t.base + ((source / D) * t.j + j) * D + source % D;
                        }
                        dot += int64_t(sp[(ab + k * D + r) * D + q]) * sp[br * D + c];
                    }
                    const int32_t partial = scu(dot, s.carrier);
                    auto &out = acc[(cb + j * D + r) * D + c];
                    out = (!(x & 1) && k == 0) ? partial : clamp(int64_t(out) + partial);
                }
            }
    }
};
unsigned emulator::destroyed = 0;

struct fixture {
    potal_resident_handle *h = nullptr;
    emulator *model = nullptr; // Borrowed only; adopted heap context owns this.
    int8_t *weights[2]{}, *main_a = nullptr, *residual_a = nullptr;
    potal_resident_load layout[2]{};
    potal_resident_token resident[2]{};
    bool hardware;
    unsigned destructors;

    explicit fixture(bool hw = false, bool prepare = true) :
        hardware(hw), destructors(emulator::destroyed) {
        if (hw) ok(potal_resident_open(nullptr, nullptr, &h), "hardware open");
        else {
            model = new emulator;
            potal_paired_sink sink{emulator::sink, model};
            const auto status = potal_resident_open(&sink, emulator::destroy, &h);
            if (status != POTAL_OK) { delete model; model = nullptr; }
            ok(status, "software open");
        }
        if (prepare) {
            ok(potal_resident_prepare(h), "prepare");
            views();
        }
    }
    // Deliberately no automatic close/free: an unexpected failure may mean DMA
    // completion is unknown. The adopted context and all DMA pools remain owned.
    void views() {
        for (unsigned i = 0; i < 4; ++i) {
            void *p = nullptr; size_t bytes = 0;
            ok(potal_resident_buffer_view(h, potal_resident_buffer(i), &p, &bytes), "idle pool");
            check(p && bytes == 65536 && uintptr_t(p) % 64 == 0, "owned aligned 64KiB pool");
            if (i < 2) weights[i] = static_cast<int8_t *>(p);
            else if (i == 2) main_a = static_cast<int8_t *>(p);
            else residual_a = static_cast<int8_t *>(p);
        }
    }
    uint64_t calls() const {
        return model ? model->records.size() : potal_resident_query(h).core.commands;
    }
    template<class F> potal_status timed(interval stage, F action) {
        if (!hardware) return action();
        const auto begin = std::chrono::steady_clock::now();
        const potal_status status = action(); // Exactly once; preserve every status.
        measured.ms[size_t(stage)] += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        ++measured.calls[size_t(stage)];
        return status;
    }
    void retire(uint32_t pair) {
        ok(timed(interval::retire, [&] { return potal_resident_retire(h, resident[pair]); }),
           "safe residency retirement");
    }
    template<class F> void refusal(F action, const char *what,
                                  potal_status wanted = POTAL_ERR_INVALID_ARGUMENT) {
        const uint64_t before = calls();
        expect(action(), wanted, what);
        check(calls() == before && potal_resident_query(h).state == POTAL_RESIDENT_READY,
              "preflight refusal issued commands or became sticky");
    }
    potal_resident_load description(uint32_t pair, uint64_t origin = 0, uint64_t generation = 1) {
        potal_resident_load d{};
        d.pair = pair; d.bank = 2 * pair + 1;
        d.b_sp_base = (pair + 1) * half - 28 * 2 * D; d.b_row_stride = 32;
        d.k_rows = 448; d.j_columns = 32; d.main_a_tiles = 2;
        d.k_begin = origin; d.j_begin = 64;
        d.weight_id = 100 + pair; d.weight_generation = generation; d.source_stride = 32;
        return d;
    }
    void fill(uint32_t pair, bool saturation = false, uint64_t original_k = 0) {
        views();
        for (uint32_t k = 0; k < 448; ++k) for (uint32_t c = 0; c < 32; ++c)
            weights[pair][k * 32 + c] = saturation ? 127 : int8_t((original_k + k) / 16 + 1 + c / 16);
        for (uint32_t r = 0; r < 16; ++r) for (uint32_t k = 0; k < 32; ++k)
            main_a[r * 32 + k] = saturation ? (k < 16 ? 127 : -128) : 1;
        std::fill_n(residual_a, 16 * 16, saturation ? int8_t(-128) : int8_t(1));
    }
    void load(uint32_t pair, uint64_t origin = 0, uint64_t generation = 1,
              uint64_t weight_id = 0, uint64_t j_begin = 64) {
        layout[pair] = description(pair, origin, generation);
        if (weight_id) layout[pair].weight_id = weight_id;
        layout[pair].j_begin = j_begin;
        const size_t first = size_t(calls());
        ok(timed(interval::load_issue, [&] { return potal_resident_load_issue(h, &layout[pair]); }),
           "B load issue");
        check(potal_resident_query(h).pending != 0, "B DMA must be pending");
        if (model) {
            check(model->records.size() == first + 57, "one CONFIG and all 56 B MVINs");
            const auto &cfg = model->records[first];
            check(cfg.funct == 0 && cfg.xd == 0 && cfg.rs1 == UINT64_C(0x100101) &&
                  cfg.rs2 == 32, "B CONFIG fields");
            for (size_t t = 0; t < 56; ++t) {
                const auto &r = model->records[first + 1 + t];
                const uint64_t source = uintptr_t(weights[pair] + (t / 2) * 16 * 32 + (t % 2) * 16);
                const uint64_t dest = (UINT64_C(16) << 48) | (UINT64_C(16) << 32) |
                                      (layout[pair].b_sp_base + t * 16);
                check(r.funct == 2 && r.xd == 0 && r.rs1 == source && r.rs2 == dest,
                      "every B MVIN source/destination/rows/columns/stride");
            }
            check(potal_resident_query(h).core.outstanding_loops == 0, "B is not a core loop");
        }
        potal_resident_token unpublished{};
        check(potal_resident_get(h, pair, &unpublished) != POTAL_OK, "no token at issue return");
        ok(timed(interval::load_complete, [&] { return potal_resident_fence(h); }), "B completion Fence");
        // Tokens come from the completion-qualified query, never issue return.
        const auto s = potal_resident_query(h);
        resident[pair] = s.pair[pair].token;
        check(s.pair[pair].readable && !s.pair[pair].loading &&
              s.pair[pair].description.k_rows == 448 && resident[pair].owner != 0,
              "full resident geometry published");
        ok(timed(interval::handoff, [&] { return potal_resident_get(h, pair, &unpublished); }), "completed get");
        check(same(unpublished, resident[pair]), "get/query token identity");
        views();
        std::fill_n(weights[pair], 448 * 32, int8_t(-91)); // Only AFTER completed load.
    }
    void scales(uint32_t pair, bool residual, uint32_t generation = 255, bool saturation = false,
                uint32_t residual_begin = 96) {
        const uint64_t block = (layout[pair].k_begin + (residual ? residual_begin : 416)) / 32;
        std::array<uint32_t, 32> carriers{};
        for (uint32_t c = 0; c < 32; ++c)
            carriers[c] = saturation ? 14 : residual ? (c < 16 ? 3 : 0) : (c < 16 ? 1 : 2);
        potal_resident_scales s{};
        s.resident = resident[pair]; s.base = residual ? 2 : 0; s.generation = generation;
        s.blocks = 1; s.j_columns = 32; s.j_begin = layout[pair].j_begin;
        s.original_blocks = &block; s.carriers = carriers.data(); s.carrier_stride = 32;
        ok(timed(interval::scale_setup, [&] { return potal_resident_scale_program(h, &s); }),
           "bound Scale program");
        // Arrays die here: the interface copies them before issue.
    }
    void release(uint32_t base, uint32_t generation = 255, uint32_t first_column = 0) {
        for (uint32_t slot = base; slot < base + 2; ++slot)
            for (uint32_t c = (slot == base ? first_column : 0); c < 16; ++c)
                ok(timed(interval::release, [&] {
                    return potal_resident_scale_release(h, slot, generation, c);
                }), "checked Release");
    }
    uint64_t index(uint32_t pair, uint32_t begin = 96) {
        std::array<uint64_t, 16> indices{};
        for (uint32_t r = 0; r < 16; ++r) indices[r] = layout[pair].k_begin + begin + 2 * r;
        uint64_t generation = 0;
        const size_t start = size_t(calls());
        ok(timed(interval::index_setup, [&] {
               return potal_resident_index_program(h, resident[pair], indices.data(), 16, &generation);
           }),
           "complete retained Index program");
        if (model) {
            check(model->records.size() == start + 17, "one Index CONFIG and 16 writes");
            const auto &config = model->records[start];
            check(config.funct == 27 && config.xd == 1 &&
                  config.rs1 == ((UINT64_C(2) << 32) | layout[pair].b_sp_base) &&
                  config.rs2 == ((UINT64_C(16) << 32) | 448), "full resident Index fields");
            for (size_t r = 0; r < 16; ++r) {
                const auto &write = model->records[start + 1 + r];
                check(write.funct == 26 && write.xd == 1 && write.rs1 == r &&
                      write.rs2 == begin + 2 * r, "every ordered Index slot/source field");
            }
        }
        check(generation != 0 && potal_resident_query(h).index_generation == generation,
              "Index generation publication");
        return generation;
    }
    potal_resident_microtile tile(uint32_t pair, bool residual, uint64_t output = 10) {
        potal_resident_microtile t{};
        auto &p = t.command; auto &l = p.loop;
        p.path = residual ? POTAL_PAIRED_RESIDUAL : POTAL_PAIRED_MAIN;
        p.pair = pair; p.resident_b = residual; p.indexed = residual;
        p.a_offset_tiles = residual ? 2 : 0;
        l.a = residual ? residual_a : main_a;
        l.b = residual ? nullptr : weights[pair] + 416 * layout[pair].source_stride;
        l.lda = residual ? 16 : 32; l.ldb = residual ? 0 : layout[pair].source_stride;
        l.i_tiles = 1; l.j_tiles = 2; l.k_tiles = residual ? 1 : 2;
        l.scale_base = residual ? 2 : 0; l.scale_generation = 255;
        l.passes_per_block = residual ? 1 : 2;
        t.resident = resident[pair]; t.output_id = output; t.original_k_begin =
            layout[pair].k_begin + (residual ? 96 : 416);
        t.b_sp_base = layout[pair].b_sp_base; t.b_row_stride = 32;
        t.acc_bank = residual ? 1 : 0; t.acc_base = 0;
        t.first = 1; t.last = 1;
        return t;
    }
    void program(const potal_resident_microtile &t, size_t start) {
        if (!model) return;
        const auto &p = t.command; const auto &l = p.loop;
        // Independent field construction. The real suffix B pointer validates
        // at the public boundary, but only its encoded DRAM field becomes zero.
        const uint64_t selector = uint64_t(p.pair + 1);
        const uint64_t control = uint64_t(l.flags == POTAL_GEMM_ACCUMULATE) | 2 |
            (selector << 16) | (selector << 18) | (UINT64_C(1) << 20) |
            (uint64_t(l.scale_base) << 21) | (uint64_t(l.scale_generation) << 37) |
            (uint64_t(l.passes_per_block) << 45) | (UINT64_C(1) << 61) |
            (uint64_t(p.path == POTAL_PAIRED_RESIDUAL) << 62) | (uint64_t(p.indexed) << 63);
        const std::array<potal_paired_cmd_record, 10> expected{{
            {0, 0, UINT64_C(0x100101), l.lda},
            {0, 0, UINT64_C(0x100109), l.ldb},
            {0, 0, UINT64_C(0x10004), UINT64_C(1) << 48},
            {0, 0, 2, 0},
            {9, 0, 0, uint64_t(l.i_tiles) | (uint64_t(l.j_tiles) << 16) | (uint64_t(l.k_tiles) << 32)},
            {10, 0, uintptr_t(l.a), 0},
            {11, 0, 0, 0}, {12, 0, l.lda, l.ldb}, {13, 0, 0, 0},
            {8, 0, control, uint64_t(p.a_offset_tiles) << 3}
        }};
        check(model->records.size() == start + expected.size(), "exact ten-command resident program");
        for (size_t i = 0; i < expected.size(); ++i) {
            const auto &actual = model->records[start + i]; const auto &e = expected[i];
            check(actual.funct == e.funct && actual.xd == e.xd &&
                  actual.rs1 == e.rs1 && actual.rs2 == e.rs2, "resident program exact raw fields");
        }
    }
    potal_resident_token compute(potal_resident_microtile t, bool by_release = false) {
        const uint64_t old_mvins = model ? model->mvins : 0;
        const uint64_t old_fences = model ? model->fences : 0;
        const size_t start = size_t(calls());
        ok(timed(interval::microtile_issue, [&] { return potal_resident_microtile_issue(h, &t); }),
           "resident LOOP issue");
        program(t, start);
        check(potal_resident_query(h).pending, "LOOP remains pending at issue return");
        if (by_release) ok(timed(interval::release, [&] {
            return potal_resident_scale_release(h, 0, 255, 0);
        }), "Release completes LOOP");
        else ok(timed(interval::acc_complete, [&] { return potal_resident_fence(h); }), "LOOP completion Fence");
        const auto s = potal_resident_query(h);
        check(!s.pending && s.acc[t.acc_bank].initialized &&
              s.acc[t.acc_bank].complete == t.last,
              "completed private ACC epoch");
        check(s.pair[t.command.pair].description.k_rows == 448, "Main preserves full B K");
        if (model) check(model->mvins == old_mvins &&
                         model->fences == old_fences + (by_release ? 0 : 1) && !model->bad,
                         "consumer never reloads B; Release needs no extra Fence");
        return s.acc[t.acc_bank].token;
    }
    void result(uint32_t bank, potal_resident_token epoch, int32_t left, int32_t right,
                bool by_release = false) {
        ok(timed(interval::readback_issue, [&] {
            return potal_resident_readback_issue(h, bank, epoch);
        }), "private readback issue");
        const int32_t *p = nullptr; size_t stride = 0;
        check(potal_resident_result(h, bank, epoch, &p, &stride) != POTAL_OK,
              "no CPU view before readback drain");
        if (by_release) ok(timed(interval::release, [&] {
            return potal_resident_scale_release(h, 2, 255, 0);
        }), "Release completes readback");
        else ok(timed(interval::readback_complete, [&] { return potal_resident_fence(h); }),
                "readback completion Fence");
        ok(potal_resident_result(h, bank, epoch, &p, &stride), "completed result");
        check(p != nullptr && stride >= 32, "completed result stride");
        for (uint32_t r = 0; r < 16; ++r) for (uint32_t c = 0; c < 32; ++c)
            check(p[r * stride + c] == (c < 16 ? left : right), "independent exact output");
    }
    void close(potal_status wanted = POTAL_OK) {
        if (model) check(!model->bad, "decoder rejected command");
        expect(potal_resident_close(&h), wanted, "safe close");
        check(h == nullptr, "safe close consumes handle, including sticky status");
        if (!hardware) check(emulator::destroyed == destructors + 1, "sink destructor exactly once");
        model = nullptr;
    }
};

void positive(bool hardware, bool saturation = false) {
    fixture f(hardware);
    for (uint32_t pair = 0; pair < 2; ++pair) {
        f.fill(pair, saturation);
        f.load(pair);
        f.scales(pair, false, 255, saturation);
        f.scales(pair, true, 255, saturation);
        const uint64_t ig = f.index(pair);
        auto main = f.tile(pair, false, 10 + pair);
        // First overwrites a deliberately different completed epoch.
        auto old = f.compute(main);
        main.output_id += 100;
        const auto me = f.compute(main, true);
        check(!same(old, me), "first creates a fresh epoch");
        auto residual = f.tile(pair, true, 20 + pair);
        residual.index_generation = ig;
        const auto re = f.compute(residual);
        // Hand wide math: Sat32(16*127*127*16384)=MAX, then
        // Sat32(16*(-128)*127*16384)=MIN; Sat32(MAX+MIN)=-1.
        // Residual is one negative partial and therefore MIN.
        const __int128_t positive_partial = __int128_t(16) * 127 * 127 * 16384;
        const __int128_t negative_partial = __int128_t(16) * -128 * 127 * 16384;
        check(positive_partial > INT32_MAX && negative_partial < INT32_MIN &&
              int64_t(INT32_MAX) + INT32_MIN == -1, "hand saturation witness");
        f.result(0, me, saturation ? -1 : 1760, saturation ? -1 : 3648);
        f.result(1, re, saturation ? INT32_MIN : 960, saturation ? INT32_MIN : 136, true);
        // Reading ACC0 again after ACC1 proves bank isolation and preserved data.
        f.result(0, me, saturation ? -1 : 1760, saturation ? -1 : 3648);
        f.release(0, 255, 1); f.release(2, 255, 1);
        // Legal 8-bit wrap is distinct from nonwrapping resident tokens.
        f.scales(pair, false, 0, saturation);
        f.release(0, 0);
        f.retire(pair);
    }
    if (f.model) check(f.model->mvins == 112 && f.model->loops == 6 && f.model->mvouts == 12,
                       "two loads only; no B-bearing LOOP or later B MVIN");
    f.close();
}

void continued_epoch(bool hardware = false) {
    fixture f(hardware);
    f.fill(0); f.load(0); f.scales(0, false);
    auto first = f.tile(0, false, 900);
    first.last = 0;
    const auto epoch = f.compute(first);
    const auto s = potal_resident_query(f.h);
    check(s.acc[0].next_sequence == 1 && s.acc[0].next_original_k == 448,
          "logical continuation coordinates");
    auto unfinished_overwrite = first;
    ++unfinished_overwrite.output_id;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &unfinished_overwrite); },
              "first cannot discard an unfinished ACC epoch", POTAL_ERR_BUSY);
    f.result(0, epoch, 1760, 3648);
    const auto identity = f.layout[0];
    f.release(0);
    f.retire(0); // Completed ACC values outlive the retired B residency.
    // Each candidate has valid source geometry and its own matching Scale
    // binding. Only the epoch's matrix/generation/J identity conflicts.
    for (unsigned conflict = 0; conflict < 3; ++conflict) {
        f.fill(1, false, 32);
        f.load(1, 32, identity.weight_generation + (conflict == 1),
               identity.weight_id + (conflict == 0),
               identity.j_begin + (conflict == 2 ? 16 : 0));
        f.scales(1, false);
        auto conflicting = f.tile(1, false, 900);
        conflicting.first = 0; conflicting.epoch = epoch; conflicting.sequence = 1;
        conflicting.command.loop.flags = POTAL_GEMM_ACCUMULATE;
        f.refusal([&] { return potal_resident_microtile_issue(f.h, &conflicting); },
                  "continuation rejects conflicting weight identity/generation/J origin");
        const auto unchanged = potal_resident_query(f.h).acc[0];
        check(same(unchanged.token, epoch) && unchanged.next_sequence == 1 &&
              unchanged.next_original_k == 448 && unchanged.initialized && !unchanged.complete,
              "identity refusal preserves unfinished output epoch");
        f.release(0);
        f.retire(1);
    }
    f.fill(1, false, 32);
    f.load(1, 32, identity.weight_generation, identity.weight_id, identity.j_begin);
    f.scales(1, false);
    auto next = f.tile(1, false, 900);
    next.first = 0; next.epoch = epoch; next.sequence = 1;
    next.command.loop.flags = POTAL_GEMM_ACCUMULATE;
    auto bad = next; bad.output_id++;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &bad); }, "wrong continuation output");
    bad = next; ++bad.sequence;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &bad); }, "wrong continuation sequence");
    bad = next; bad.original_k_begin += 32;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &bad); }, "noncontiguous Main original K");
    const auto completed = f.compute(next);
    check(same(epoch, completed), "continuation retains ACC epoch");
    // Global K448..479: (16*29 + 16*30)*2 = 1888;
    // J1: (16*30 + 16*31)*4 = 3904. Add the first raw 1760/3648.
    f.result(0, completed, 3648, 7552);
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &next); }, "closed logical epoch");
    f.release(0);
    // Residual fragments use sequence, not contiguous source K.
    f.scales(1, true);
    auto r = f.tile(1, true, 901);
    r.index_generation = f.index(1); r.last = 0;
    const auto re = f.compute(r);
    f.release(2);
    f.scales(1, true, 255, false, 128);
    r.index_generation = f.index(1, 128);
    r.original_k_begin = f.layout[1].k_begin + 128;
    r.first = 0; r.last = 1; r.sequence = 1; r.epoch = re;
    r.command.loop.flags = POTAL_GEMM_ACCUMULATE;
    f.compute(r);
    // Global K128..158 gives 1216/168; K160..190 gives 1472/200.
    f.result(1, re, 2688, 368);
    f.release(2);
    f.retire(1);
    f.close();
}

// Exercise the same global exclusion rule for B load, Main, Residual and
// readback. Queries and healthy drain/Release are the only issuing exceptions.
void pending_refusals(fixture &f, potal_resident_microtile t) {
    check(potal_resident_query(f.h).pending, "pending guard test setup");
    const auto epoch = potal_resident_query(f.h).acc[t.acc_bank].token;
    void *p = nullptr; size_t bytes = 0;
    const int32_t *out = nullptr; size_t stride = 0;
    uint64_t indices[16]{}, ig = 0;
    uint64_t block = 13; uint32_t carriers[32]{};
    potal_resident_scales scale{};
    scale.resident = f.resident[0]; scale.blocks = 1; scale.j_columns = 32;
    scale.j_begin = 64; scale.original_blocks = &block;
    scale.carriers = carriers; scale.carrier_stride = 32; scale.generation = 255;
    f.refusal([&] { return potal_resident_prepare(f.h); }, "pending prepare", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_buffer_view(f.h, POTAL_RESIDENT_B0, &p, &bytes); },
              "pending mutable CPU view", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_load_issue(f.h, &f.layout[0]); }, "pending overwrite", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &t); }, "pending second consumer", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_index_program(f.h, f.resident[0], indices, 16, &ig); },
              "pending Index", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_scale_program(f.h, &scale); }, "pending Scale", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_retire(f.h, f.resident[0]); }, "pending retire", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_readback_issue(f.h, t.acc_bank, epoch); },
              "pending readback", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_result(f.h, t.acc_bank, epoch, &out, &stride); },
              "pending result", POTAL_ERR_BUSY);
    auto *saved = f.h;
    f.refusal([&] { return potal_resident_close(&f.h); }, "pending close", POTAL_ERR_BUSY);
    check(f.h == saved && emulator::destroyed == f.destructors, "pending retains ownership");
    // A preflight-rejected Release is not a drain.
    f.refusal([&] { return potal_resident_scale_release(f.h, 255, 256, 0); },
              "invalid Release cannot complete pending work");
    check(potal_resident_query(f.h).pending, "rejected Release preserves pending");
}

void refusals() {
    fixture f;
    f.fill(0);
    for (unsigned hidden = 4; hidden < 6; ++hidden) {
        void *data = nullptr; size_t bytes = 0;
        f.refusal([&] { return potal_resident_buffer_view(f.h, potal_resident_buffer(hidden), &data, &bytes); },
                  "private result pool is not a mutable buffer view");
    }
    f.layout[0] = f.description(0);
    const auto original = f.layout[0];
    for (unsigned field = 0; field < 10; ++field) {
        auto d = original;
        switch (field) {
            case 0: d.bank = 0; break;
            case 1: ++d.b_sp_base; break;
            case 2: d.b_row_stride = 16; break;
            case 3: d.pair = 2; break;
            case 4: d.k_rows = 449; break;
            case 5: d.j_columns = 0; break;
            case 6: d.source_stride = 31; break;
            case 7: d.source_stride = SIZE_MAX; break;
            case 8: d.k_begin = UINT64_MAX - 15; break;
            case 9: d.j_columns = 160; d.source_stride = 160; break; // K28/J10 exceeds one bank.
        }
        f.refusal([&] { return potal_resident_load_issue(f.h, &d); }, "invalid resident layout",
                  field == 9 ? POTAL_ERR_CAPACITY : POTAL_ERR_INVALID_ARGUMENT);
    }
    ok(potal_resident_load_issue(f.h, &original), "pending B fixture");
    pending_refusals(f, f.tile(0, false));
    ok(potal_resident_fence(f.h), "pending B Fence");
    f.resident[0] = potal_resident_query(f.h).pair[0].token;
    // No Scale at all: the core's geometric checker alone is insufficient.
    auto main = f.tile(0, false);
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &main); }, "missing Scale window");
    uint64_t block = 13; std::array<uint32_t, 32> carrier{};
    potal_resident_scales s{};
    s.resident = f.resident[0]; s.generation = 255; s.blocks = 1;
    s.j_columns = 32; s.j_begin = 64; s.original_blocks = &block;
    s.carriers = carrier.data(); s.carrier_stride = 32;
    // 126 + 18 + 126 = 270 live slot demand cannot fit the 256-entry cache.
    // Materialize the three disjoint windows rather than allocating fake arrays.
    std::array<uint64_t, 63> blocks{};
    std::array<uint32_t, 63 * 32> many{};
    for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = i % 14;
    auto large = s; large.blocks = 63; large.original_blocks = blocks.data(); large.carriers = many.data();
    ok(potal_resident_scale_program(f.h, &large), "Scale first 126 slots");
    large.base = 126; large.blocks = 9;
    ok(potal_resident_scale_program(f.h, &large), "Scale next 18 slots");
    large.base = 144; large.blocks = 63;
    const uint64_t before = f.calls();
    expect(potal_resident_scale_program(f.h, &large), POTAL_ERR_CAPACITY, "Scale270 > 256");
    check(f.calls() == before, "Scale capacity refusal emits nothing");
    for (uint32_t slot = 0; slot < 144; ++slot) for (uint32_t c = 0; c < D; ++c)
        ok(potal_resident_scale_release(f.h, slot, 255, c), "release capacity fixture");
    auto wrong_scale = s; wrong_scale.j_begin++;
    f.refusal([&] { return potal_resident_scale_program(f.h, &wrong_scale); }, "wrong Scale J origin");
    wrong_scale = s; wrong_scale.generation = 256;
    f.refusal([&] { return potal_resident_scale_program(f.h, &wrong_scale); }, "Scale generation width");
    wrong_scale = s; ++wrong_scale.resident.owner;
    f.refusal([&] { return potal_resident_scale_program(f.h, &wrong_scale); }, "Scale retained identity");
    wrong_scale = s; carrier[0] = 32768;
    f.refusal([&] { return potal_resident_scale_program(f.h, &wrong_scale); }, "invalid SCU carrier");
    carrier[0] = 0;
    // A wrong original block may be stored, but cannot bind the suffix consumer.
    block = 12;
    f.model->reentrant_handle = f.h;
    ok(potal_resident_scale_program(f.h, &s), "well-formed wrong-block Scale");
    check(f.model->reentered && !f.model->bad, "reentrant preflight emits no nested command");
    f.refusal([&] { return potal_resident_retire(f.h, f.resident[0]); },
              "retire refuses associated live Scale", POTAL_ERR_BUSY);
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &main); }, "wrong Scale original block");
    f.release(0);
    f.scales(0, false);
    for (unsigned field = 0; field < 25; ++field) {
        auto t = main;
        switch (field) {
            case 0: ++t.resident.owner; break;
            case 1: ++t.resident.generation; break;
            case 2: t.command.pair = 1; break;
            case 3: ++t.b_sp_base; break;
            case 4: t.b_row_stride = 16; break;
            case 5: t.acc_bank = 1; break;
            case 6: t.acc_base = 16; break;
            case 7: t.command.loop.b = f.weights[0]; break; // Not the native suffix source.
            case 8: t.command.loop.ldb = 48; break;
            case 9: t.command.loop.k_tiles = 0; break;
            case 10: t.command.loop.j_tiles = 0; break;
            case 11: t.command.loop.pad_k = 16; break;
            case 12: t.command.loop.pad_j = 1; break;
            case 13: t.command.loop.pad_i = 1; break;
            case 14: t.command.loop.passes_per_block = 0; break;
            case 15: t.command.loop.scale_generation = 254; break;
            case 16: t.original_k_begin = 400; break;
            case 17: t.command.a_offset_tiles = 1; break;
            case 18: t.command.loop.a = f.weights[0]; break; // Full pool alias, not suffix alias.
            case 19: t.command.loop.a = f.main_a + 1; break;
            case 20: t.command.loop.a = f.main_a + 65520; break; // Aligned, outside retained extent.
            case 21: t.command.loop.lda = SIZE_MAX; break;
            case 22: t.first = 0; t.command.loop.flags = POTAL_GEMM_ACCUMULATE; break;
            case 23: t.sequence = 1; break;
            case 24: t.command.resident_b = 1; break;
        }
        f.refusal([&] { return potal_resident_microtile_issue(f.h, &t); }, "consumer descriptor refusal",
                  field == 12 || field == 13 ? POTAL_ERR_UNSUPPORTED : POTAL_ERR_INVALID_ARGUMENT);
    }
    // A geometrically valid I2 shape is outside this harness, not impossible RTL.
    auto unsupported = main; unsupported.command.loop.i_tiles = 2;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &unsupported); },
              "valid I2 unsupported", POTAL_ERR_UNSUPPORTED);
    unsupported = main; unsupported.command.loop.k_tiles = 1;
    unsupported.command.loop.passes_per_block = 1;
    unsupported.command.loop.b += 16 * 32; unsupported.original_k_begin += 16;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &unsupported); },
              "valid K1 Main unsupported", POTAL_ERR_UNSUPPORTED);
    const int32_t *out = nullptr; size_t stride = 0;
    f.refusal([&] { return potal_resident_readback_issue(f.h, 0, {}); }, "readback before initialized epoch");
    ok(potal_resident_microtile_issue(f.h, &main), "Main pending guards");
    pending_refusals(f, main);
    ok(potal_resident_fence(f.h), "Main guard drain");
    const auto old_epoch = potal_resident_query(f.h).acc[0].token;
    auto overwrite = main; overwrite.output_id++;
    const auto new_epoch = f.compute(overwrite);
    check(!same(old_epoch, new_epoch), "overwrite invalidates epoch");
    f.refusal([&] { return potal_resident_readback_issue(f.h, 0, old_epoch); }, "stale ACC epoch");
    f.refusal([&] { return potal_resident_result(f.h, 0, old_epoch, &out, &stride); }, "stale result epoch");
    // Missing one column must refuse, not execute a partial Scale window.
    ok(potal_resident_scale_release(f.h, 0, 255, 7), "release one Scale column");
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &main); }, "missing Scale column");
    for (uint32_t slot = 0; slot < 2; ++slot) for (uint32_t c = 0; c < D; ++c)
        if (slot != 0 || c != 7) ok(potal_resident_scale_release(f.h, slot, 255, c), "remaining Main Scale");
    f.scales(0, true);
    uint64_t indices[16]{};
    for (uint32_t r = 0; r < 16; ++r) indices[r] = 96 + 2 * r;
    uint64_t ig = 0;
    indices[15] = 448;
    f.refusal([&] { return potal_resident_index_program(f.h, f.resident[0], indices, 16, &ig); },
              "out-of-range original Index");
    indices[15] = 126;
    auto residual = f.tile(0, true);
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &residual); }, "Index not installed");
    const auto old_ig = f.index(0);
    const auto new_ig = f.index(0);
    check(old_ig != new_ig, "Index replacement changes generation");
    residual.index_generation = old_ig;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &residual); }, "stale Index incarnation");
    residual.index_generation = new_ig;
    auto wrong = residual; wrong.original_k_begin++;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); }, "Residual first original Index mismatch");
    wrong = residual; wrong.command.a_offset_tiles = 1;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); }, "Residual overwrites Main A reservation");
    wrong = residual; wrong.command.loop.b = f.weights[0];
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); }, "Residual must not carry B DRAM");
    wrong = residual; wrong.command.indexed = 0;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); }, "Residual must use Index");
    wrong = residual; wrong.command.resident_b = 0;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); }, "Residual must use retained residency");
    wrong = residual; wrong.command.loop.k_tiles = 2;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); },
              "Residual K exceeds its A row stride");
    wrong.command.loop.lda = 32;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); },
              "Residual K versus Index count", POTAL_ERR_UNSUPPORTED);
    // Strict ordering and full descriptor validation happen before CONFIG.
    indices[8] = indices[7];
    f.refusal([&] { return potal_resident_index_program(f.h, f.resident[0], indices, 16, &ig); },
              "duplicate Index entries");
    indices[8] = 109;
    f.refusal([&] { return potal_resident_index_program(f.h, f.resident[0], indices, 16, &ig); },
              "reordered Index entries");
    for (uint32_t r = 0; r < 16; ++r) indices[r] = 96 + 3 * r;
    ok(potal_resident_index_program(f.h, f.resident[0], indices, 16, &ig), "cross-block Index descriptor");
    wrong = residual; wrong.index_generation = ig;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); },
              "cross-block partial carrier unsupported", POTAL_ERR_UNSUPPORTED);
    std::array<uint64_t, 32> full_index{};
    for (uint32_t r = 0; r < 32; ++r) full_index[r] = 96 + r;
    ok(potal_resident_index_program(f.h, f.resident[0], full_index.data(), 32, &ig),
       "full 32-entry Index descriptor can be programmed");
    wrong = residual; wrong.index_generation = ig;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &wrong); },
              "execution requires exactly 16 Index entries");
    residual.index_generation = f.index(0);
    ok(potal_resident_microtile_issue(f.h, &residual), "Residual pending guards");
    pending_refusals(f, residual);
    ok(potal_resident_fence(f.h), "Residual guard drain");
    const auto re = potal_resident_query(f.h).acc[1].token;
    ok(potal_resident_readback_issue(f.h, 1, re), "readback pending guards");
    pending_refusals(f, residual);
    ok(potal_resident_fence(f.h), "readback guard drain");
    f.release(2);
    const auto stale = f.resident[0];
    ok(potal_resident_retire(f.h, stale), "retire before reload");
    f.fill(0); f.load(0, 0, 2); f.scales(0, false);
    main = f.tile(0, false); main.resident = stale;
    f.refusal([&] { return potal_resident_microtile_issue(f.h, &main); }, "stale resident token after reload");
    f.release(0);
    ok(potal_resident_retire(f.h, f.resident[0]), "retire reloaded B");
    f.close();
    fixture reopened;
    uint64_t next = 0;
    ok(potal_resident_generation_next(UINT64_MAX - 1, &next), "last nonwrapping generation");
    check(next == UINT64_MAX, "last software generation exact");
    expect(potal_resident_generation_next(UINT64_MAX, &next), POTAL_ERR_CAPACITY, "software generation overflow");
    reopened.refusal([&] { return potal_resident_retire(reopened.h, stale); }, "cross-handle resident token");
    reopened.close();
}

void general_loads() {
    // Loading is more general than execution. Neither case is a claimed GEMM.
    fixture f;
    for (const auto shape : {std::array<uint32_t, 2>{16, 16}, std::array<uint32_t, 2>{448, 144}}) {
        f.views(); std::fill_n(f.weights[0], 65536, int8_t(1));
        auto d = f.description(0);
        d.k_rows = shape[0]; d.j_columns = shape[1]; d.source_stride = shape[1];
        d.b_row_stride = shape[1]; d.b_sp_base = half - (shape[0] / 16) * shape[1];
        ok(potal_resident_load_issue(f.h, &d), "general valid B load");
        ok(potal_resident_fence(f.h), "general B load Fence");
        const auto token = potal_resident_query(f.h).pair[0].token;
        check(potal_resident_query(f.h).pair[0].description.j_columns == shape[1],
              "general load preserves J extent");
        ok(potal_resident_retire(f.h, token), "retire general load");
    }
    f.close();
}

void release_completes_b() {
    fixture f;
    f.fill(0); f.load(0); f.scales(0, false);
    f.fill(1); f.layout[1] = f.description(1);
    const uint64_t fences = f.model->fences;
    ok(potal_resident_load_issue(f.h, &f.layout[1]), "second pair B load");
    ok(potal_resident_scale_release(f.h, 0, 255, 0), "global Release completes B-only DMA");
    const auto q = potal_resident_query(f.h);
    check(!q.pending && q.pair[1].readable && f.model->fences == fences,
          "Release publishes successful B load without Fence");
    f.resident[1] = q.pair[1].token;
    f.views(); std::fill_n(f.weights[1], 448 * 32, int8_t(-91));
    f.release(0, 255, 1);
    ok(potal_resident_retire(f.h, f.resident[0]), "retire first Release fixture");
    ok(potal_resident_retire(f.h, f.resident[1]), "retire Release-completed B");
    f.close();
}

// All fault cases run in fresh forked software-only processes. No fault hook
// exists on the hardware path. Successful cleanup also checks its destructor.
void blocked_after_failure(fixture &f, bool terminal) {
    const uint64_t before = f.calls();
    const unsigned destroyed = emulator::destroyed;
    auto *saved = f.h;
    auto t = f.weights[0] ? f.tile(0, false) : potal_resident_microtile{};
    const int32_t *data = nullptr; size_t stride = 0;
    check(potal_resident_microtile_issue(f.h, &t) != POTAL_OK, "failed LOOP forbidden");
    check(potal_resident_load_issue(f.h, &f.layout[0]) != POTAL_OK, "failed load forbidden");
    check(potal_resident_scale_release(f.h, 0, 255, 0) != POTAL_OK, "failed Release forbidden");
    check(potal_resident_readback_issue(f.h, 0, {}) != POTAL_OK, "failed readback forbidden");
    check(potal_resident_result(f.h, 0, {}, &data, &stride) != POTAL_OK, "failed result unpublished");
    check(potal_resident_close(&f.h) != POTAL_OK && f.h == saved, "failed close retains handle");
    if (terminal) {
        check(potal_resident_fence(f.h) != POTAL_OK, "unknown Fence never retried");
        check(potal_resident_cleanup(f.h) != POTAL_OK, "quarantine cleanup never retried");
        check(potal_resident_prepare(f.h) != POTAL_OK, "quarantine never prepares");
        check(potal_resident_query(f.h).owned_bytes >= 4 * 65536, "quarantine retains DMA pools");
        auto *context = new emulator;
        potal_paired_sink sink{emulator::sink, context};
        potal_resident_handle *other = nullptr;
        expect(potal_resident_open(&sink, emulator::destroy, &other), POTAL_ERR_BUSY,
               "quarantine retains sole accelerator owner");
        check(!other && emulator::destroyed == destroyed, "failed open does not adopt context");
        delete context; // Open failed: caller still owns this unused context.
    }
    check(f.calls() == before && emulator::destroyed == destroyed, "no command, retry, or free after failure");
}

enum class fault { b, main, residual, readback, hidden_fence, empty_fence, scale_response,
                   scale_conflict, cleanup_release, index, release_response };

void failure_case(fault kind, uint32_t position) {
    fixture f(false, kind != fault::empty_fence);
    if (kind == fault::empty_fence) {
        f.model->fail_fence = true;
        check(potal_resident_prepare(f.h) != POTAL_OK, "empty prepare Fence fails");
        const auto q = potal_resident_query(f.h);
        check(q.state == POTAL_RESIDENT_QUARANTINED && q.unknown_fence &&
              q.core.outstanding_loops == 0 && q.core.live_scale_slots == 0 &&
              f.model->fences == 1, "empty-core failure quarantines");
        blocked_after_failure(f, true);
        return;
    }
    f.fill(0); f.layout[0] = f.description(0);
    potal_status error = POTAL_HW_BAD_COMMAND;
    if (kind == fault::b || kind == fault::hidden_fence) {
        f.model->fail_funct = POTAL_FUNCT_MVIN; f.model->fail_occurrence = position;
        expect(potal_resident_load_issue(f.h, &f.layout[0]), error, "partial B MVIN failure");
        const auto q = potal_resident_query(f.h);
        check(q.pending && q.attempted_b_mvins == position && !q.pair[0].readable &&
              q.core.outstanding_loops == 0 && q.core.outstanding_readbacks == 0 &&
              q.core.outstanding_acc_updates == 0, "private B pending despite zero core counters");
    } else {
        f.load(0);
        if (kind == fault::scale_response || kind == fault::scale_conflict) {
            if (kind == fault::scale_response) f.model->bad_scale = true;
            else {
                f.model->fail_funct = POTAL_FUNCT_SCALE_WRITE; f.model->fail_occurrence = 1;
                f.model->fail_status = POTAL_HW_SCALE_CONFLICT;
            }
            uint64_t block = 13; uint32_t carriers[32]{};
            potal_resident_scales s{};
            s.resident = f.resident[0]; s.generation = 255; s.blocks = 1;
            s.j_columns = 32; s.j_begin = 64; s.original_blocks = &block;
            s.carriers = carriers; s.carrier_stride = 32;
            check(potal_resident_scale_program(f.h, &s) != POTAL_OK, "uncertain Scale response");
            check(potal_resident_query(f.h).state == POTAL_RESIDENT_QUARANTINED,
                  "untracked possible Scale acceptance cannot be cleaned");
            blocked_after_failure(f, true);
            return;
        }
        f.scales(0, false);
        if (kind == fault::release_response) {
            auto t = f.tile(0, false);
            ok(potal_resident_microtile_issue(f.h, &t), "pending LOOP before failed Release");
            f.model->fail_release = true;
            expect(potal_resident_scale_release(f.h, 0, 255, 0), POTAL_HW_BAD_COMMAND,
                   "post-dispatch Release failure");
            const auto q = potal_resident_query(f.h);
            check(q.state == POTAL_RESIDENT_QUARANTINED && q.pending,
                  "failed Release supplies no completion proof");
            blocked_after_failure(f, true);
            return;
        }
        if (kind == fault::residual || kind == fault::index) f.scales(0, true);
        auto t = f.tile(0, kind == fault::residual);
        if (kind == fault::residual) t.index_generation = f.index(0);
        if (kind == fault::readback) {
            const auto epoch = f.compute(t);
            f.model->fail_funct = POTAL_FUNCT_MVOUT; f.model->fail_occurrence = position;
            expect(potal_resident_readback_issue(f.h, 0, epoch), error, "partial readback failure");
            const int32_t *p = nullptr; size_t stride = 0;
            check(potal_resident_result(f.h, 0, epoch, &p, &stride) != POTAL_OK,
                  "partial destination never published");
            check(potal_resident_query(f.h).core.outstanding_readbacks == 0,
                  "partial readback success-only counter witness");
        } else if (kind == fault::index) {
            uint64_t indices[16]{}, generation = 0;
            for (uint32_t r = 0; r < 16; ++r) indices[r] = 96 + 2 * r;
            f.model->fail_funct = POTAL_FUNCT_INDEX_WRITE; f.model->fail_occurrence = position;
            expect(potal_resident_index_program(f.h, f.resident[0], indices, 16, &generation),
                   error, "partial Index failure");
        } else {
            // Ten-command program: four CONFIGs, bounds, AB, DC, strides AB/DC,
            // LOOP. The final LOOP computes before returning the injected error.
            f.model->fail_at = f.calls() + position;
            if (kind == fault::main) {
                error = POTAL_ERR_INVALID_ARGUMENT; // Dispatched error is not a preflight refusal.
                f.model->fail_status = error;
            }
            expect(potal_resident_microtile_issue(f.h, &t), error, "partial LOOP program failure");
            check(potal_resident_query(f.h).pending &&
                  potal_resident_query(f.h).core.outstanding_loops == 0,
                  "premarked consumer survives zero success-only counter");
        }
    }
    const auto failed = potal_resident_query(f.h);
    check(failed.state == POTAL_RESIDENT_FAILED && failed.error == error, "first sticky issue error");
    const std::string stage = failed.core.error_stage ? failed.core.error_stage : "";
    blocked_after_failure(f, false);
    const uint64_t before = f.calls(), fences = f.model->fences;
    if (kind == fault::hidden_fence) f.model->fail_fence = true;
    if (kind == fault::cleanup_release) f.model->fail_release = true;
    const auto cleanup = potal_resident_cleanup(f.h);
    const auto q = potal_resident_query(f.h);
    check(f.model->fences == fences + 1 && f.model->records[before].funct == POTAL_FUNCT_FENCE,
          "cleanup explicit Fence exactly once, even with zero core counters");
    check(q.cleanup_attempted && q.error == error &&
          stage == (q.core.error_stage ? q.core.error_stage : ""), "cleanup preserves first error/stage");
    if (kind == fault::hidden_fence || kind == fault::cleanup_release) {
        check(cleanup != POTAL_OK && q.state == POTAL_RESIDENT_QUARANTINED,
              "failed cleanup permanently quarantines");
        if (kind == fault::hidden_fence)
            check(q.unknown_fence && f.calls() == before + 1, "hidden Fence failure emits no Release");
        blocked_after_failure(f, true);
        return;
    }
    ok(cleanup, "successful recovery cleanup");
    check(q.state == POTAL_RESIDENT_CLEANED && !q.pending && !q.pair[0].readable &&
          !q.core.live_scale_slots && !q.core.index_transaction_open, "cleaned state never publishes failed work");
    for (size_t i = size_t(before + 1); i < f.model->records.size(); ++i)
        check(f.model->records[i].funct == POTAL_FUNCT_SCALE_RELEASE, "only Releases after cleanup Fence");
    const uint64_t cleaned_calls = f.calls();
    ok(potal_resident_cleanup(f.h), "idempotent cleaned cleanup");
    check(f.calls() == cleaned_calls, "cleaned cleanup emits nothing");
    f.close(error);
    fixture reopened;
    reopened.close(); // Safe error close released the sole owner.
}

void isolated(fault kind, uint32_t position) {
#if defined(__unix__)
    std::fflush(nullptr);
    const pid_t pid = fork();
    check(pid >= 0, "fork software-only failure case");
    if (pid == 0) {
        try { failure_case(kind, position); std::fflush(nullptr); _exit(0); }
        catch (const std::exception &e) {
            std::fprintf(stderr, "software-only fault=%u position=%u FAIL: %s\n",
                         unsigned(kind), position, e.what());
            std::fflush(nullptr); _exit(1);
        }
    }
    int result = 0;
    pid_t waited;
    do { waited = waitpid(pid, &result, 0); } while (waited < 0 && errno == EINTR);
    check(waited == pid && WIFEXITED(result) && WEXITSTATUS(result) == 0,
          "isolated software-only fault child failed");
#else
    (void)kind; (void)position;
    throw std::runtime_error("software-only terminal cases require fork/waitpid; not silently skipped");
#endif
}

void no_hardware() {
#if !defined(__riscv)
    potal_resident_handle *h = nullptr;
    // Whether absence is rejected by open or prepare, it must never masquerade
    // as an actual dispatched Fence failure or transfer unknown ownership.
    const auto opened = potal_resident_open(nullptr, nullptr, &h);
    if (opened == POTAL_OK) {
        expect(potal_resident_prepare(h), POTAL_ERR_NO_HARDWARE, "absent hardware prepare");
        const auto q = potal_resident_query(h);
        check(!q.pending && !q.unknown_fence && q.state != POTAL_RESIDENT_QUARANTINED,
              "no-hardware preflight has no unknown completion");
        ok(potal_resident_close(&h), "absent hardware safe close");
        check(h == nullptr, "absent hardware handle consumed");
    } else {
        expect(opened, POTAL_ERR_NO_HARDWARE, "absent hardware open");
        check(h == nullptr, "absent hardware open leaves no handle");
    }
#endif
}

int hardware_run(unsigned repeat) {
#if defined(__riscv) && defined(__linux__)
    ggml::potal::issuer_owner owner;
    std::string why;
    potal_status started_status = POTAL_OK;
    bool passed = false, fatal = false;
    double cpu_observed_ms = 0;
    const bool started = owner.start([&](const ggml::potal::owner_runners &) {
        const auto info = ggml::potal::get_issuer_info();
        if (!info.ready || info.cpu != 1)
            return ggml::potal::consume_outcome{true, POTAL_ERR_INVALID_ARGUMENT, "resident_affinity"};
        try {
            const auto begin = std::chrono::steady_clock::now();
            for (unsigned n = 0; n < repeat; ++n) {
                measured = {};
                const auto sample_begin = std::chrono::steady_clock::now();
                positive(true);
                positive(true, true);
                continued_epoch(true);
                const double sample_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - sample_begin).count();
                std::printf("POTAL_RESIDENT_SAMPLE sample=%u cpu_observed_test_ms=%.6f "
                            "exact=1 negative_preflight=1 hardware_fault_injection=0\n", n, sample_ms);
                for (size_t s = 0; s < size_t(interval::count); ++s)
                    std::printf("POTAL_RESIDENT_INTERVAL sample=%u scope=%s cpu_ms=%.6f calls=%llu\n",
                                n, interval_names[s], measured.ms[s],
                                (unsigned long long)measured.calls[s]);
            }
            cpu_observed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            passed = true;
        } catch (const std::exception &e) {
            // Unexpected hardware failure: retained handles/context/pools must
            // outlive this callback. No guessed drain and no fallback.
            fatal = true;
            std::fprintf(stderr, "POTAL_RESIDENT actual RoCC FAIL: %s\n", e.what());
        }
        return ggml::potal::consume_outcome{fatal, passed ? POTAL_OK : POTAL_ERR_BAD_RESPONSE, "resident"};
    }, why, started_status);
    owner.join();
    if (!started || !passed) {
        std::fprintf(stderr, "POTAL_RESIDENT actual RoCC FAIL status=%s detail=%s\n",
                     potal_status_name(started_status), why.c_str());
        return 1;
    }
    std::printf("POTAL_RESIDENT actual RoCC PASS repeats=%u CPU1/hart0 "
                "cpu_observed_test_ms=%.6f includes_validation_and_host_comparison=1\n",
                repeat, cpu_observed_ms);
    std::puts("DMA/bus/FPGA time and overlap not measured; hardware fault injection excluded.");
    return 0;
#else
    (void)repeat;
    std::fputs("--hardware requires RISC-V Linux issuer_owner on CPU1/hart0; no CPU fallback.\n", stderr);
    return 2;
#endif
}

} // namespace

int main(int argc, char **argv) {
    bool hardware = false, repeat_given = false;
    unsigned repeat = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hardware") == 0 && !hardware) hardware = true;
        else if (std::strcmp(argv[i], "--repeat") == 0 && !repeat_given && i + 1 < argc) {
            const char *text = argv[++i]; char *end = nullptr;
            errno = 0;
            const unsigned long value = std::strtoul(text, &end, 10);
            if (errno || *text < '0' || *text > '9' || !end || *end || value == 0 || value > UINT_MAX)
                return 2;
            repeat = unsigned(value); repeat_given = true;
        } else {
            std::fprintf(stderr, "Usage: %s [--hardware [--repeat N]]\n", argv[0]); return 2;
        }
    }
    if (repeat_given && !hardware) {
        std::fputs("--repeat requires --hardware.\n", stderr); return 2;
    }
    if (hardware) return hardware_run(repeat);
    try {
        no_hardware();
        positive(false);
        positive(false, true);
        continued_epoch();
        refusals();
        general_loads();
        release_completes_b();
        for (uint32_t n : {1u, 28u, 56u}) isolated(fault::b, n);
        for (uint32_t n = 1; n <= 10; ++n) {
            isolated(fault::main, n);
            isolated(fault::residual, n);
        }
        for (uint32_t n : {1u, 2u}) isolated(fault::readback, n);
        for (uint32_t n : {1u, 8u, 16u}) isolated(fault::index, n);
        isolated(fault::hidden_fence, 28);
        isolated(fault::empty_fence, 0);
        isolated(fault::scale_response, 0);
        isolated(fault::scale_conflict, 0);
        isolated(fault::cleanup_release, 10);
        isolated(fault::release_response, 0);
        std::puts("POTAL_RESIDENT software-only PASS: packed-SP decoder, exact arithmetic, "
                  "resident lifecycle, refusals, isolated failures; no hardware or timing evidence.");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "POTAL_RESIDENT software-only FAIL: %s\n", e.what());
        return 1;
    }
}
