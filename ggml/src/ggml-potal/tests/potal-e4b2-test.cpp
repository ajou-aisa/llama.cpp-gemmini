#include "potal.h"
#include "support/e4b2-emulator.hpp"
#include "support/e4b2-fixtures.h"

#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

extern "C" int potal_e4b2_c11_header_test(void);

namespace {
uint64_t assertions = 0;
void require(bool value, const char *message) {
    ++assertions;
    if (!value) throw std::runtime_error(message);
}
void ok(potal_status value, const char *message) {
    if (value != POTAL_OK) {
        std::fprintf(stderr, "%s: status=%d\n", message, int(value));
        require(false, message);
    }
}
uint64_t dma_shape(unsigned rows, unsigned columns, uint32_t address) {
    return uint64_t(rows) << 48 | uint64_t(columns) << 32 | address;
}
void command(e4b2::machine &m, uint32_t funct, uint64_t a, uint64_t b, bool xd = false) {
    ok(m.command({funct, uint32_t(xd), a, b}), "software command interpreter");
}
void configure_loop(e4b2::machine &m, const int8_t *a) {
    command(m, 9, 0, UINT64_C(1) << 32 | UINT64_C(1) << 16 | 1);
    command(m, 10, uintptr_t(a), 0);
    command(m, 11, 0, 0);
    command(m, 12, 16, 0);
    command(m, 13, 0, 0);
}
uint64_t run_word(bool indexed, unsigned path) {
    return UINT64_C(2) | UINT64_C(8) | UINT64_C(1) << 16 | UINT64_C(1) << 18 |
        UINT64_C(1) << 20 | UINT64_C(9) << 37 | UINT64_C(1) << 45 |
        UINT64_C(1) << 61 | uint64_t(path) << 62 | uint64_t(indexed) << 63;
}

void interpreter_self_test() {
    e4b2::machine m;
    alignas(64) std::array<int8_t, 256> source{}, a{};
    alignas(64) std::array<int32_t, 256> result{};
    std::array<int8_t, 256> original{};
    for (unsigned k = 0; k < 16; ++k)
        for (unsigned j = 0; j < 16; ++j) {
            source[k * 16 + j] = int8_t(int((k * 3 + j) % 9) - 4);
            a[k * 16 + j] = int8_t(int((k + j * 2) % 7) - 3);
        }
    original = source;
    command(m, 0, UINT64_C(16) << 16 | UINT64_C(0x101), 16);
    command(m, 2, uintptr_t(source.data()), dma_shape(16, 16, 4672));
    require(m.work.size() == 1 && m.sp[4672 * 16] == 0,
            "DMA acknowledgement is not a model completion");
    command(m, 25, 0, 0, true);
    std::fill(source.begin(), source.end(), int8_t(93));
    for (unsigned x = 0; x < 256; ++x)
        require(m.sp[4672 * 16 + x] == original[x], "modeled B source poison retains SP bytes");

    for (unsigned c = 0; c < 16; ++c) {
        const uint32_t carrier = c % 3 == 0 ? UINT32_C(0x80000000) : c % 3 == 1 ? 0 : 31;
        command(m, 23, uint64_t(c) << 16 | UINT64_C(9) << 8, carrier, true);
    }
    configure_loop(m, a.data());
    command(m, 28, UINT64_C(80) << 48 |
            UINT64_C(144) << 16 | UINT64_C(4672), UINT64_C(0x11), true);
    command(m, 8, run_word(false, 0), 0, true);
    require(m.acc[80 * 16] == INT32_C(0x12345678) && m.work.size() == 1,
            "accepted generated LOOP is not a completed ACC write");
    command(m, 25, 0, 0, true);
    require(m.preload_rows.size() == 1 && m.preload_rows[0] == 4672,
            "native model uses effective interior base rather than legacy suffix");

    command(m, 0, 2, 16 * sizeof(int32_t));
    command(m, 3, uintptr_t(result.data()), dma_shape(16, 16, UINT32_C(0xa0000050)));
    require(result[1] == 0, "MVOUT result stays unpublished until model drain");
    command(m, 25, 0, 0, true);
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned j = 0; j < 16; ++j) {
            int64_t dot = 0;
            for (unsigned k = 0; k < 16; ++k)
                dot += int64_t(a[i * 16 + k]) * original[k * 16 + j];
            const uint32_t carrier = j % 3 == 0 ? UINT32_C(0x80000000) : j % 3 == 1 ? 0 : 31;
            require(result[i * 16 + j] == e4b2_shift(dot, carrier),
                    "independent signed INT32 SCU golden");
        }

    command(m, 27, UINT64_C(9) << 32 | UINT64_C(4160),
            UINT64_C(16) << 32 | UINT64_C(448), true);
    for (unsigned k = 0; k < 16; ++k) command(m, 26, k, 48 + k, true);
    for (unsigned path = 0; path < 2; ++path) {
        configure_loop(m, a.data());
        command(m, 28, UINT64_C(80) << 48 | UINT64_C(5) << 32 |
                UINT64_C(144) << 16 | UINT64_C(4160), UINT64_C(0x15) | path * 8, true);
        command(m, 8, run_word(true, path), path ? UINT64_C(28) << 3 : 0, true);
        require(m.command({27, 1, UINT64_C(9) << 32 | UINT64_C(4160),
                          UINT64_C(16) << 32 | UINT64_C(448)}) == POTAL_ERR_BUSY,
                "outstanding indexed reader blocks Index replacement");
        command(m, 25, 0, 0, true);
        for (unsigned i = 0; i < 16; ++i)
            for (unsigned j = 0; j < 16; ++j)
                require(m.acc[(path * 512 + 80 + i) * 16 + j] == result[i * 16 + j],
                        "indexed J5 Main ACC0 and Residual ACC1 match native interior result");
    }
    const auto before = m.acc;
    require(m.command({8, 1, run_word(false, 0), 0}) != POTAL_OK &&
            m.work.empty() && before == m.acc,
            "consumed slot cannot reuse stale geometry or resident descriptor");
    command(m, 28, 0, 0, true);
    configure_loop(m, a.data());
    require(m.command({8, 1, run_word(false, 0), 0}) != POTAL_OK && m.work.empty(),
            "disabled resident view never falls back to legacy arithmetic");
    command(m, 7, 0, 0);
    require(m.staged.mask == 0 && !m.view.valid, "accepted model flush clears staging only");
    for (unsigned c = 0; c < 16; ++c)
        command(m, 24, uint64_t(c) << 16 | UINT64_C(9) << 8, 0, true);
    require(m.work.empty(), "software interpreter finishes drained");
}

uint64_t destroyed = 0, numerical_runs = 0, numerical_cells = 0;
struct fixture {
    e4b2::machine *device = new e4b2::machine;
    potal_paired_session *session = nullptr;
    std::array<potal_resident_ws_load, 2> loads{};
    std::array<potal_resident_ws_ref, 2> residents{};
    std::array<std::array<potal_resident_ws_ref, 32>, 2> epochs{};
    std::array<std::array<uint64_t, 32>, 2> progress{};
    std::array<unsigned, 2> weight_modes{};
    std::vector<int32_t> golden = std::vector<int32_t>(1024 * 16, INT32_C(0x12345678));
    uint32_t generation = 1;

    static potal_status sink(void *context, const potal_paired_cmd_record *record) {
        return static_cast<e4b2::machine *>(context)->command(*record);
    }
    static potal_status counter(void *context, const potal_paired_cmd_record *record, uint64_t *value) {
        return static_cast<e4b2::machine *>(context)->counter(record->rs1, record->rs2, value);
    }
    static void destroy(void *context) {
        ++destroyed;
        delete static_cast<e4b2::machine *>(context);
    }
    explicit fixture(bool prepare = true) {
        const potal_resident_ws_options options{{sink, device}, destroy, counter};
        ok(potal_paired_resident_open(&options, &session), "open production resident mode");
        auto q = potal_paired_resident_query(session);
        require(q.software_only && q.capability_matched && q.core.commands == 2 &&
                q.owned_bytes == 6 * UINT64_C(65536), "software mode and exact counter gate are explicit");
        if (prepare) ok(potal_paired_resident_prepare(session), "prepare production resident mode");
    }
    int8_t *buffer(unsigned pool) {
        void *data = nullptr;
        size_t bytes = 0;
        ok(potal_paired_resident_buffer_view(session, static_cast<potal_resident_ws_buffer>(pool),
                                            &data, &bytes), "owned input view");
        require(bytes == 65536 && uintptr_t(data) % 64 == 0, "owned bounded aligned allocation");
        return static_cast<int8_t *>(data);
    }
    void load(unsigned pair, unsigned mode = 0, unsigned k = 448, unsigned j = 144,
              uint64_t k_begin = 0, uint64_t j_begin = 0) {
        auto &d = loads[pair];
        d = {pair, k, j, (k + 15) / 16, k_begin, j_begin,
             mode ? 71 : 71 + pair, 9, 144};
        auto *b = buffer(pair);
        for (unsigned row = 0; row < k; ++row)
            for (unsigned col = 0; col < j; ++col)
                b[row * 144 + col] = e4b2_weight(pair, mode, unsigned(k_begin + row), unsigned(j_begin + col));
        ok(potal_paired_resident_load_issue(session, &d), "B-only issue");
        potal_resident_ws_ref unavailable{};
        require(potal_paired_resident_get(session, pair, &unavailable) == POTAL_ERR_BUSY &&
                potal_paired_resident_query(session).loading[pair], "issue never publishes resident");
        ok(potal_paired_fence(session), "completion-qualified B publication");
        ok(potal_paired_resident_get(session, pair, &residents[pair]), "published resident token");
        std::fill_n(b, 65536, int8_t(93));
        weight_modes[pair] = mode;
    }
    void release() {
        for (unsigned slot = 0; slot < 256; ++slot)
            for (unsigned c = 0; c < 16; ++c) if (device->scales[slot][c].live) {
                ok(potal_paired_scale_release(session, slot, device->scales[slot][c].generation, c),
                   "checked global Scale release");
            }
    }
    using fragments = std::vector<std::vector<uint64_t>>;
    fragments canonical(unsigned pair, unsigned offset, unsigned count) const {
        fragments result;
        for (unsigned p = 0; p < count; p += 16) {
            result.emplace_back();
            for (unsigned x = p; x < std::min(p + 16, count); ++x)
                result.back().push_back(loads[pair].k_begin + offset + x);
        }
        return result;
    }
    void declare(potal_resident_ws_run &d, unsigned pair, uint64_t begin, uint64_t end,
                 const fragments &manifest) {
        std::vector<potal_resident_ws_fragment> copied;
        for (const auto &part : manifest) {
            potal_resident_ws_fragment f{};
            f.count = uint32_t(part.size());
            std::copy(part.begin(), part.end(), f.original_k);
            copied.push_back(f);
        }
        const auto &l = loads[pair];
        for (unsigned j = 0; j < (d.j_columns + 15) / 16; ++j) {
            potal_resident_ws_epoch e{};
            e.path = d.path; e.acc_tile = d.acc_tile + j; e.rows = d.i_rows;
            e.columns = std::min(16u, d.j_columns - j * 16);
            e.output_id = d.output_id; e.original_i = d.original_i;
            e.original_j = l.j_begin + d.j_offset + j * 16;
            e.weight_id = l.weight_id; e.weight_generation = l.weight_generation;
            e.logical_k_begin = begin; e.logical_k_end = end;
            if (d.path) { e.fragment_count = copied.size(); e.fragments = copied.data(); }
            ok(potal_paired_resident_epoch_begin(session, &e, &epochs[d.path][d.acc_tile + j]),
               "reserve immutable physical epoch");
            d.epochs[j] = epochs[d.path][d.acc_tile + j];
            progress[d.path][d.acc_tile + j] = 0;
        }
        for (auto &f : copied) { f.count = 0; std::fill_n(f.original_k, 16, UINT64_MAX); }
    }
    potal_resident_ws_run prepare(unsigned pair, unsigned path, bool indexed,
        unsigned offset, unsigned count, unsigned j_origin, unsigned columns, unsigned tile,
        uint64_t begin, uint64_t end, const fragments &parts, const fragments &manifest = {},
        bool first = true, bool last = true, unsigned rows = 16, unsigned arithmetic = 0) {
        release();
        potal_resident_ws_run d{};
        d.resident = residents[pair]; d.path = path; d.indexed = indexed;
        d.k_offset = offset; d.k_rows = count; d.j_offset = j_origin; d.j_columns = columns;
        d.i_rows = rows; d.a_offset_tiles = path ? loads[pair].main_a_tiles : 0;
        d.acc_tile = tile; d.a_stride = ((count + 15) / 16) * 16;
        d.output_id = 100 + tile; d.first = first; d.last = last;
        d.sequence = first ? 0 : progress[path][tile];
        if (first) declare(d, pair, begin, end, manifest);
        else for (unsigned j = 0; j < (columns + 15) / 16; ++j) d.epochs[j] = epochs[path][tile + j];
        std::vector<uint64_t> originals;
        for (unsigned p = 0; p < parts.size(); ++p) {
            if (indexed) d.fragment_counts[p] = uint32_t(parts[p].size());
            originals.insert(originals.end(), parts[p].begin(), parts[p].end());
            if (p + 1 < parts.size()) while (originals.size() % 16) originals.push_back(parts[p][0]);
        }
        if (indexed) {
            d.k_rows = uint32_t(originals.size());
            d.a_stride = ((d.k_rows + 15) / 16) * 16;
            const potal_resident_ws_indices ix{residents[pair], uint32_t(originals.size()), originals.data()};
            ok(potal_paired_resident_index_program(session, &ix, &d.index), "full-resident Index program");
        }
        auto *a = buffer(2 + path);
        std::fill_n(a, 65536, int8_t(0));
        for (unsigned i = 0; i < rows; ++i)
            for (unsigned p = 0; p < parts.size(); ++p)
                for (unsigned x = 0; x < parts[p].size(); ++x)
                    a[i * d.a_stride + p * 16 + x] = e4b2_activation(arithmetic, i, unsigned(parts[p][x]));
        std::vector<uint64_t> blocks;
        std::vector<uint32_t> carriers;
        for (const auto &part : parts) {
            blocks.push_back(part[0] / 32);
            for (unsigned j = 0; j < columns; ++j)
                carriers.push_back(e4b2_carrier(arithmetic, unsigned(part[0] / 32),
                    unsigned(loads[pair].j_begin + j_origin + j)));
        }
        d.scale_generation = generation++ % 256;
        const potal_resident_ws_scales scales{residents[pair], 0, d.scale_generation,
            uint32_t(parts.size()), columns, loads[pair].j_begin + j_origin,
            blocks.data(), carriers.data(), columns};
        ok(potal_paired_resident_scale_program(session, &scales, &d.scale), "rebind complete P1 Scale window");
        return d;
    }
    void execute(const potal_resident_ws_run &d, unsigned pair, const fragments &parts,
                 unsigned arithmetic = 0, bool use_release = false) {
        const auto before = device->records.size();
        const auto b_loads = device->b_loads;
        ok(potal_paired_resident_run_check(session, &d), "checked session descriptor");
        ok(potal_paired_resident_issue(session, &d), "actual production C11 issue to software interpreter");
        require(device->records.size() == before + 11 && device->b_loads == b_loads,
                "eleven commands and no B reload per extended invocation");
        const auto &view = device->records[before + 9];
        const auto &run = device->records[before + 10];
        const uint32_t base = (pair + 1) * 8192 - ((loads[pair].k_rows + 15) / 16) *
            ((loads[pair].j_columns + 15) / 16) * 16;
        const uint32_t pitch = ((loads[pair].j_columns + 15) / 16) * 16;
        require(view.funct == 28 && view.xd == 1 &&
                view.rs1 == (uint64_t(d.acc_tile * 16) << 48 |
                    uint64_t(d.indexed ? d.j_offset / 16 : 0) << 32 | uint64_t(pitch) << 16 |
                    uint64_t(base + (d.indexed ? 0 : d.k_offset / 16 * pitch + d.j_offset))) &&
                view.rs2 == (UINT64_C(0x11) | uint64_t(pair) << 1 |
                    uint64_t(d.indexed) << 2 | uint64_t(d.path) << 3),
                "exact frozen resident configuration words");
        require(run.funct == 8 && run.xd == 1 && (run.rs1 & 8) &&
                ((run.rs1 >> 45) & 65535) == 1 && device->records[before + 5].rs2 == 0,
                "extended run returns status, uses P1 and B DRAM zero");
        require(potal_paired_resident_query(session).pending, "C11 pending survives issue acknowledgement");
        const auto pending_commands = device->records.size();
        void *unavailable = nullptr;
        size_t unavailable_bytes = 0;
        potal_resident_ws_ref unavailable_ref{};
        require(potal_paired_resident_buffer_view(session, POTAL_RESIDENT_WS_MAIN_A,
                                                  &unavailable, &unavailable_bytes) == POTAL_ERR_BUSY &&
                potal_paired_resident_load_issue(session, &loads[pair]) == POTAL_ERR_BUSY &&
                potal_paired_resident_index_program(session, nullptr, &unavailable_ref) == POTAL_ERR_BUSY &&
                potal_paired_resident_scale_program(session, nullptr, &unavailable_ref) == POTAL_ERR_BUSY &&
                potal_paired_resident_issue(session, &d) == POTAL_ERR_BUSY &&
                potal_paired_resident_readback_issue(session, nullptr, &unavailable_ref) == POTAL_ERR_BUSY &&
                potal_paired_resident_retire(session, residents[pair]) == POTAL_ERR_BUSY &&
                potal_paired_resident_close(&session) == POTAL_ERR_BUSY &&
                device->records.size() == pending_commands,
                "external pending readers block overwrite control readback retirement and close");
        require(potal_paired_scale_release(session, d.scale_base, d.scale_generation ^ 1, 0) ==
                    POTAL_ERR_INVALID_ARGUMENT && device->records.size() == pending_commands &&
                potal_paired_resident_query(session).pending,
                "invalid Release does not complete outstanding work");
        ok(potal_paired_fence(session), "publish completed saturated contribution");
        for (unsigned i = 0; i < d.i_rows; ++i)
            for (unsigned j = 0; j < d.j_columns; ++j) {
                const unsigned cell = (d.path * 512 + (d.acc_tile + j / 16) * 16 + i) * 16 + j % 16;
                int32_t value = d.first ? 0 : golden[cell];
                for (const auto &part : parts) {
                    std::array<uint32_t, 16> original{};
                    std::transform(part.begin(), part.end(), original.begin(),
                                   [](uint64_t k) { return uint32_t(k); });
                    const auto partial = e4b2_oracle_partial(pair, weight_modes[pair], arithmetic, i,
                        unsigned(loads[pair].j_begin + d.j_offset + j), original.data(), unsigned(part.size()));
                    value = e4b2_clamp(int64_t(value) + partial);
                }
                golden[cell] = value;
            }
        for (unsigned cell = 0; cell < golden.size(); ++cell)
            require(device->acc[cell] == golden[cell], "all outputs and untouched physical ACC cells");
        potal_resident_ws_readback rb{};
        rb.bank = d.path; rb.acc_tile = d.acc_tile; rb.rows = d.i_rows; rb.columns = d.j_columns;
        rb.stride = ((d.j_columns + 15) / 16) * 16;
        for (unsigned j = 0; j < (d.j_columns + 15) / 16; ++j) {
            rb.epochs[j] = d.epochs[j];
            rb.sequences[j] = d.sequence + parts.size();
            progress[d.path][d.acc_tile + j] = rb.sequences[j];
        }
        potal_resident_ws_ref receipt{};
        ok(potal_paired_resident_readback_issue(session, &rb, &receipt), "checked addressed readback");
        const int32_t *result = nullptr;
        size_t stride = 0;
        require(potal_paired_resident_result(session, receipt, &result, &stride) == POTAL_ERR_BUSY,
                "readback receipt not readable before completion");
        if (use_release) ok(potal_paired_scale_release(session, d.scale_base, d.scale_generation, 0),
                            "Release qualifies readback completion");
        else ok(potal_paired_fence(session), "readback completion");
        ok(potal_paired_resident_result(session, receipt, &result, &stride), "completed const readback");
        for (unsigned i = 0; i < d.i_rows; ++i)
            for (unsigned j = 0; j < d.j_columns; ++j) {
                ++numerical_cells;
                const unsigned cell = (d.path * 512 + (d.acc_tile + j / 16) * 16 + i) * 16 + j % 16;
                require(result[i * stride + j] == golden[cell], "every exact readback INT32");
            }
        ++numerical_runs;
    }
    void close() {
        release();
        auto q = potal_paired_resident_query(session);
        for (unsigned p = 0; p < 2; ++p)
            if (q.readable[p]) ok(potal_paired_resident_retire(session, residents[p]), "retire after completed consumers");
        ok(potal_paired_resident_close(&session), "safe resident close");
        require(session == nullptr, "safe close consumes session");
        device = nullptr;
    }
};

void production_numerics() {
    for (unsigned pair = 0; pair < 2; ++pair) {
        fixture f;
        f.load(pair);
        for (const auto &cut : std::array<std::array<unsigned, 4>, 5>{{
                 {{0, 16, 0, 16}}, {{48, 16, 80, 16}}, {{112, 48, 48, 32}},
                 {{432, 16, 128, 16}}, {{0, 448, 0, 144}}}}) {
            const auto parts = f.canonical(pair, cut[0], cut[1]);
            auto d = f.prepare(pair, 0, false, cut[0], cut[1], cut[2], cut[3], cut[2] / 16,
                               cut[0], cut[0] + cut[1], parts);
            f.execute(d, pair, parts);
        }
        auto parts = f.canonical(pair, 112, 32);
        auto d = f.prepare(pair, 0, true, 112, 32, 48, 32, 20, 112, 144, parts);
        const auto old_index = d.index;
        std::array<uint64_t, 32> repeated_index{};
        for (unsigned k = 0; k < 32; ++k) repeated_index[k] = 112 + k;
        const potal_resident_ws_indices replacement{f.residents[pair], 32, repeated_index.data()};
        ok(potal_paired_resident_index_program(f.session, &replacement, &d.index), "same-address fresh Index incarnation");
        auto stale = d;
        stale.index = old_index;
        auto commands = f.device->records.size();
        require(potal_paired_resident_issue(f.session, &stale) == POTAL_ERR_INVALID_ARGUMENT &&
                f.device->records.size() == commands, "reprogrammed identical Index data cannot revive old token");
        f.execute(d, pair, parts, 0, true);
        parts = {{96, 98, 126}, {128, 135, 143}};
        d = f.prepare(pair, 1, true, 96, 19, 48, 32, 5, 96, 160, parts, parts);
        auto *a = f.buffer(POTAL_RESIDENT_WS_RESIDUAL_A);
        a[3] = 1;
        commands = f.device->records.size();
        require(potal_paired_resident_issue(f.session, &d) == POTAL_ERR_INVALID_ARGUMENT &&
                f.device->records.size() == commands, "nonzero interior compact filler refuses before issue");
        a[3] = 0;
        f.execute(d, pair, parts);
        f.close();
        fixture tail;
        tail.load(pair, 0, 443, 139, 16, 32);
        parts = tail.canonical(pair, 432, 11);
        d = tail.prepare(pair, 0, false, 432, 11, 128, 11, 25, 448, 459, parts, {}, true, true, 7);
        tail.execute(d, pair, parts);
        d = tail.prepare(pair, 0, true, 432, 11, 128, 11, 27, 448, 459, parts, {}, true, true, 7);
        tail.execute(d, pair, parts);
        tail.close();
    }
}

void saturation_and_epochs() {
    fixture f;
    f.load(0, 1);
    auto parts = f.canonical(0, 0, 16);
    auto d = f.prepare(0, 0, false, 0, 16, 0, 16, 5, 0, 48, parts, {}, true, false, 16, 1);
    f.execute(d, 0, parts, 1);
    f.release();
    ok(potal_paired_resident_retire(f.session, f.residents[0]), "drained B retires with independent OPEN epoch");
    const auto count = f.device->records.size();
    require(potal_paired_resident_close(&f.session) == POTAL_ERR_BUSY && f.session &&
            f.device->records.size() == count, "close cannot silently discard an initialized unfinished epoch");
    f.load(1, 1);
    for (unsigned k : {16u, 32u}) {
        parts = f.canonical(1, k, 16);
        d = f.prepare(1, 0, false, k, 16, 0, 16, 5, 0, 48, parts, {}, false, k == 32, 16, 1);
        f.execute(d, 1, parts, 1);
    }
    require(f.golden[80 * 16] == -1, "MAX MAX MIN ordered recurrence");
    f.close();
    fixture compact;
    compact.load(0, 2);
    parts = {{96, 112}};
    d = compact.prepare(0, 1, true, 96, 2, 80, 16, 15, 96, 128, parts, parts, true, true, 16, 2);
    compact.execute(d, 0, parts, 2);
    require(compact.golden[(512 + 240) * 16] == 0, "compact cancellation occurs before SCU");
    parts = compact.canonical(0, 96, 32);
    d = compact.prepare(0, 0, false, 96, 32, 80, 16, 15, 96, 128, parts, {}, true, true, 16, 2);
    compact.execute(d, 0, parts, 2);
    require(compact.golden[240 * 16] == -1, "dense DIM partial cancellation follows saturation");
    compact.close();
    fixture bias;
    bias.load(1, 3);
    parts = bias.canonical(1, 0, 16);
    parts.push_back({16});
    d = bias.prepare(1, 1, true, 0, 17, 0, 16, 22, 0, 32, parts, parts, true, true, 16, 3);
    bias.execute(d, 1, parts, 3);
    require(bias.golden[(512 + 352) * 16] == 16777215, "carrier24 partial saturation stays unmerged");
    bias.close();
}

void capability_and_atomic_refusals() {
    auto *old = new e4b2::machine;
    old->supported = false;
    potal_resident_ws_options options{{fixture::sink, old}, fixture::destroy, fixture::counter};
    potal_paired_session *session = nullptr;
    const auto before_destroy = destroyed;
    require(potal_paired_resident_open(&options, &session) == POTAL_ERR_UNSUPPORTED &&
            !session && old->records.size() == 2 && destroyed == before_destroy,
            "old image refuses before any extension and retains caller context");
    require(old->records[0].funct == 126 && old->records[0].rs1 == 0xa &&
            old->records[1].funct == 126 &&
            old->records[1].rs1 == UINT64_C(0x504f544100000000),
            "hardware gate uses only the two known base-counter commands");
    delete old;
#if !defined(__riscv)
    require(potal_paired_resident_open(nullptr, &session) == POTAL_ERR_NO_HARDWARE && !session,
            "native no-sink cannot fabricate hardware capability");
#endif
    fixture f;
    f.load(0);
    const auto parts = f.canonical(0, 112, 32);
    auto d = f.prepare(0, 0, false, 112, 32, 48, 32, 20, 112, 144, parts);
    using mutation = std::function<void(potal_resident_ws_run &)>;
    for (const auto &mutate : std::vector<mutation>{
             [](auto &x) { ++x.resident.generation; },
             [](auto &x) { ++x.resident.owner; },
             [](auto &x) { ++x.scale.generation; },
             [](auto &x) { ++x.scale_generation; },
             [](auto &x) { ++x.epochs[1].generation; },
             [](auto &x) { ++x.output_id; },
             [](auto &x) { ++x.original_i; },
             [](auto &x) { ++x.sequence; },
             [](auto &x) { x.first = 0; },
             [](auto &x) { x.last = 0; },
             [](auto &x) { x.k_offset = 113; },
             [](auto &x) { x.k_rows = 17; },
             [](auto &x) { x.j_offset = 49; },
             [](auto &x) { x.i_rows = 17; },
             [](auto &x) { x.acc_tile = 31; },
             [](auto &x) { x.scale_base = 255; },
             [](auto &x) { x.a_offset = 65536; },
             [](auto &x) { x.a_stride = SIZE_MAX; },
             [](auto &x) { x.a_offset_tiles = 255; },
             [](auto &x) { x.index = {1, 1}; }}) {
        auto bad = d;
        mutate(bad);
        const auto commands = f.device->records.size();
        const auto q = potal_paired_resident_query(f.session);
        require(potal_paired_resident_issue(f.session, &bad) != POTAL_OK &&
                commands == f.device->records.size(), "complete invalid descriptor issues zero commands");
        const auto after = potal_paired_resident_query(f.session);
        require(!after.core.failed && !after.pending &&
                after.core.commands == q.core.commands && after.readable[0],
                "preflight refusal does not poison healthy ownership");
    }
    auto *a = f.buffer(2);
    potal_resident_ws_program encoded{};
    ok(potal_resident_ws_run_encode(&f.loads[0], &d, a, 65536, &encoded), "pure run encoder");
    const auto saved = encoded;
    auto bad = d;
    bad.acc_tile = 31;
    require(potal_resident_ws_run_encode(&f.loads[0], &bad, a, 65536, &encoded) == POTAL_ERR_CAPACITY &&
            !std::memcmp(&encoded, &saved, sizeof(saved)), "pure rejection preserves output program");
    f.device->on_command = [&] {
        const auto n = f.device->records.size();
        void *pointer = nullptr;
        size_t bytes = 0;
        potal_resident_ws_ref reference{};
        require(potal_paired_resident_prepare(f.session) == POTAL_ERR_BUSY &&
                potal_paired_resident_buffer_view(f.session, POTAL_RESIDENT_WS_MAIN_A,
                                                 &pointer, &bytes) == POTAL_ERR_BUSY &&
                potal_paired_resident_load_issue(f.session, &f.loads[0]) == POTAL_ERR_BUSY &&
                potal_paired_resident_get(f.session, 0, &reference) == POTAL_ERR_BUSY &&
                potal_paired_resident_scale_program(f.session, nullptr, &reference) == POTAL_ERR_BUSY &&
                potal_paired_resident_index_program(f.session, nullptr, &reference) == POTAL_ERR_BUSY &&
                potal_paired_resident_epoch_begin(f.session, nullptr, &reference) == POTAL_ERR_BUSY &&
                potal_paired_resident_issue(f.session, &d) == POTAL_ERR_BUSY &&
                potal_paired_resident_readback_issue(f.session, nullptr, &reference) == POTAL_ERR_BUSY &&
                potal_paired_resident_retire(f.session, f.residents[0]) == POTAL_ERR_BUSY &&
                potal_paired_resident_close(&f.session) == POTAL_ERR_BUSY &&
                potal_paired_fence(f.session) == POTAL_ERR_BUSY &&
                potal_paired_session_cleanup(f.session) == POTAL_ERR_BUSY &&
                potal_paired_scale_release(f.session, 0, d.scale_generation, 0) == POTAL_ERR_BUSY,
                "every resident mutation refuses sink callback reentry");
        require(potal_paired_scale_write(f.session, 0, 0, 0, 0) == POTAL_ERR_BUSY &&
                potal_paired_index_config(f.session, nullptr) == POTAL_ERR_BUSY &&
                potal_paired_index_write(f.session, nullptr) == POTAL_ERR_BUSY &&
                potal_paired_loop_ws_issue(f.session, nullptr) == POTAL_ERR_BUSY &&
                potal_paired_acc_bias_issue(f.session, nullptr) == POTAL_ERR_BUSY &&
                potal_paired_acc_readback_issue(f.session, nullptr) == POTAL_ERR_BUSY &&
                potal_paired_tlb_flush(f.session) == POTAL_ERR_BUSY &&
                potal_paired_session_close(f.session) == POTAL_ERR_BUSY &&
                potal_scale_write(0, 0, 0, 0) == POTAL_ERR_BUSY &&
                potal_scale_release(0, 0, 0) == POTAL_ERR_BUSY &&
                potal_fence() == POTAL_ERR_BUSY && potal_tlb_flush() == POTAL_ERR_BUSY &&
                f.device->records.size() == n,
                "legacy issuing aliases cannot bypass the owning resident mode");
    };
    f.execute(d, 0, parts);
    f.close();

    fixture residual;
    residual.load(1);
    const fixture::fragments manifest{{96, 98, 126}, {192, 200, 222}};
    fixture::fragments first{manifest[0]};
    d = residual.prepare(1, 1, true, 96, 3, 48, 32, 3, 96, 224, first, manifest, true, false);
    residual.execute(d, 1, first);
    const fixture::fragments wrong{{128, 130, 158}};
    d = residual.prepare(1, 1, true, 128, 3, 48, 32, 3, 96, 224, wrong, {}, false, true);
    const auto n = residual.device->records.size();
    require(potal_paired_resident_issue(residual.session, &d) != POTAL_OK &&
            residual.device->records.size() == n, "undeclared Residual gap cannot replace immutable next fragment");
    fixture::fragments second{manifest[1]};
    d = residual.prepare(1, 1, true, 192, 3, 48, 32, 3, 96, 224, second, {}, false, true);
    residual.execute(d, 1, second);
    residual.close();
}

void capacity_and_stale_bindings() {
    fixture f;
    f.load(0);
    auto parts = f.canonical(0, 0, 448);
    auto d = f.prepare(0, 0, false, 0, 448, 0, 144, 0, 0, 448, parts);
    require(potal_paired_resident_query(f.session).core.live_scale_slots == 252, "full P1 Main owns252 slots");
    const uint64_t blocks[2] = {0, 1};
    std::array<uint32_t, 288> carriers{};
    const potal_resident_ws_scales extra{f.residents[0], 252, 9, 2, 144, 0, blocks, carriers.data(), 144};
    potal_resident_ws_ref output{123, 456};
    auto n = f.device->records.size();
    require(potal_paired_resident_scale_program(f.session, &extra, &output) == POTAL_ERR_CAPACITY &&
            output.owner == 123 && output.generation == 456 &&
            f.device->records.size() == n &&
            potal_paired_resident_query(f.session).core.live_scale_slots == 252,
            "252 plus18 live slots refuses270 atomically");
    require(potal_paired_resident_retire(f.session, f.residents[0]) == POTAL_ERR_BUSY &&
            potal_paired_resident_load_issue(f.session, &f.loads[0]) == POTAL_ERR_BUSY,
            "live Scale and residency reject retirement and overwrite");
    f.execute(d, 0, parts);
    f.release();
    const auto old_scale = d.scale;
    auto fresh = f.prepare(0, 0, false, 0, 448, 0, 144, 0, 0, 448, parts);
    auto stale = fresh;
    stale.scale = old_scale;
    n = f.device->records.size();
    require(potal_paired_resident_issue(f.session, &stale) != POTAL_OK &&
            f.device->records.size() == n, "same Scale addresses cannot resurrect a released incarnation");
    f.execute(fresh, 0, parts);
    f.release();
    uint64_t block = 0;
    std::array<uint32_t, 16> negative{};
    negative[15] = UINT32_MAX;
    potal_resident_ws_scales invalid{f.residents[0], 0, 0, 1, 16, 0, &block, negative.data(), 16};
    n = f.device->records.size();
    require(potal_paired_resident_scale_program(f.session, &invalid, &output) == POTAL_ERR_INVALID_ARGUMENT &&
            f.device->records.size() == n, "negative mathematical carrier rejects before any Scale write");
    std::array<uint64_t, 33> too_many{};
    potal_resident_ws_indices index{f.residents[0], 33, too_many.data()};
    require(potal_paired_resident_index_program(f.session, &index, &output) == POTAL_ERR_CAPACITY &&
            f.device->records.size() == n, "Index32 cannot admit33 entries");
    const auto old_resident = f.residents[0];
    ok(potal_paired_resident_retire(f.session, old_resident), "retire before fresh residency");
    f.load(0);
    require(potal_paired_resident_retire(f.session, old_resident) == POTAL_ERR_INVALID_ARGUMENT,
            "same packed layout cannot resurrect stale resident token");
    f.close();
}

void recover_after_issue(fixture &f, potal_status error) {
    const auto q = potal_paired_resident_query(f.session);
    require(q.core.failed && q.core.error == error && (q.pending || q.control_pending),
            "partial issue keeps pending ownership and first error");
    const auto before = f.device->records.size();
    const auto fences = std::count_if(f.device->records.begin(), f.device->records.end(),
        [](const auto &r) { return r.funct == 25; });
    ok(potal_paired_session_cleanup(f.session), "single recovery drain");
    const auto after = potal_paired_resident_query(f.session);
    require(after.state == POTAL_RESIDENT_WS_CLEANED && after.cleanup_attempted &&
            !after.pending && !after.control_pending && after.core.error == error &&
            std::count_if(f.device->records.begin(), f.device->records.end(),
                [](const auto &r) { return r.funct == 25; }) == fences + 1,
            "known cleanup drains once without publishing failed work");
    require(f.device->records.size() > before, "cleanup actually issued completion");
    const auto clean_commands = f.device->records.size();
    ok(potal_paired_session_cleanup(f.session), "cleanup saved outcome");
    require(f.device->records.size() == clean_commands, "second cleanup issues nothing");
    require(potal_paired_resident_close(&f.session) == error && !f.session,
            "known-safe failed close consumes session and returns original status");
    f.device = nullptr;
}

void recoverable_failures() {
    for (unsigned step : {1u, 126u, 252u}) {
        fixture f;
        auto *b = f.buffer(0);
        std::fill_n(b, 65536, int8_t(3));
        const potal_resident_ws_load load{0, 448, 144, 28, 0, 0, 71, 9, 144};
        f.device->fault_funct = 2; f.device->fault_occurrence = step;
        require(potal_paired_resident_load_issue(f.session, &load) == POTAL_ERR_BAD_RESPONSE,
                "first middle final B DMA injected failure");
        require(!potal_paired_resident_query(f.session).readable[0], "failed B load never publishes");
        recover_after_issue(f, POTAL_ERR_BAD_RESPONSE);
    }
    for (const auto [funct, occurrence] : std::array<std::array<unsigned, 2>, 11>{{
             {{0, 1}}, {{0, 2}}, {{0, 3}}, {{0, 4}}, {{9, 1}}, {{10, 1}},
             {{11, 1}}, {{12, 1}}, {{13, 1}}, {{28, 1}}, {{8, 1}}}}) {
        fixture f;
        f.load(0);
        const auto parts = f.canonical(0, 0, 16);
        const auto d = f.prepare(0, 0, false, 0, 16, 0, 16, 0, 0, 16, parts);
        f.device->fault_funct = funct; f.device->fault_occurrence = occurrence;
        require(potal_paired_resident_issue(f.session, &d) == POTAL_ERR_BAD_RESPONSE,
                "each of11 baseline view run commands can partially fail");
        recover_after_issue(f, POTAL_ERR_BAD_RESPONSE);
    }
    for (const auto [funct, occurrence] : std::array<std::array<unsigned, 2>, 4>{{
             {{27, 1}}, {{26, 1}}, {{26, 8}}, {{26, 16}}}}) {
        fixture f;
        f.load(0);
        std::array<uint64_t, 16> original{};
        for (unsigned x = 0; x < 16; ++x) original[x] = 112 + x;
        const potal_resident_ws_indices ix{f.residents[0], 16, original.data()};
        potal_resident_ws_ref reference{};
        f.device->fault_funct = funct; f.device->fault_occurrence = occurrence;
        require(potal_paired_resident_index_program(f.session, &ix, &reference) == POTAL_ERR_BAD_RESPONSE,
                "partial Index configuration and writes stay unpublished");
        recover_after_issue(f, POTAL_ERR_BAD_RESPONSE);
    }
    for (const auto [funct, occurrence] : std::array<std::array<unsigned, 2>, 3>{{
             {{0, 1}}, {{3, 1}}, {{3, 2}}}}) {
        fixture f;
        f.load(0);
        const auto parts = f.canonical(0, 0, 16);
        const auto d = f.prepare(0, 0, false, 0, 16, 0, 32, 3, 0, 16, parts);
        f.execute(d, 0, parts);
        potal_resident_ws_readback rb{};
        rb.bank = 0; rb.acc_tile = 3; rb.rows = 16; rb.columns = 32; rb.stride = 32;
        for (unsigned j = 0; j < 2; ++j) { rb.epochs[j] = d.epochs[j]; rb.sequences[j] = 1; }
        potal_resident_ws_ref receipt{};
        f.device->fault_funct = funct; f.device->fault_occurrence = occurrence;
        require(potal_paired_resident_readback_issue(f.session, &rb, &receipt) == POTAL_ERR_BAD_RESPONSE &&
                !receipt.owner, "partial addressed readback has no successful receipt");
        recover_after_issue(f, POTAL_ERR_BAD_RESPONSE);
    }
}

void terminal_case(unsigned kind) {
    fixture f(kind != 5);
    potal_resident_ws_run d{};
    if (kind == 5) {
        f.device->fault_funct = 25; f.device->fault_occurrence = 1;
        require(potal_paired_resident_prepare(f.session) == POTAL_ERR_BAD_RESPONSE,
                "prepare Fence failure is terminal");
    } else {
        f.load(0);
        auto parts = f.canonical(0, 0, 16);
        d = f.prepare(0, 0, false, 0, 16, 0, 16, 0, 0, 16, parts);
        if (kind == 2) {
            f.release();
            const uint64_t block = 0;
            const std::array<uint32_t, 16> carriers{};
            const potal_resident_ws_scales scales{f.residents[0], 0, 10, 1, 16, 0, &block, carriers.data(), 16};
            potal_resident_ws_ref token{};
            f.device->fault_funct = 23; f.device->fault_occurrence = 8;
            require(potal_paired_resident_scale_program(f.session, &scales, &token) == POTAL_ERR_BAD_RESPONSE,
                    "accepted-side-effect Scale response uncertainty quarantines");
        } else if (kind == 1 || kind == 4) {
            f.device->fault_funct = 8; f.device->fault_occurrence = 1;
            require(potal_paired_resident_issue(f.session, &d) == POTAL_ERR_BAD_RESPONSE,
                    "first issue error precedes terminal cleanup uncertainty");
            f.device->fault_funct = kind == 1 ? 25 : 24; f.device->fault_occurrence = 1;
            require(potal_paired_session_cleanup(f.session) == POTAL_ERR_BAD_RESPONSE,
                    "cleanup Fence or Release can become unknown");
        } else {
            ok(potal_paired_resident_issue(f.session, &d), "pending contribution before uncertainty");
            f.device->fault_funct = kind == 3 ? 24 : 25; f.device->fault_occurrence = 1;
            const auto status = kind == 3 ?
                potal_paired_scale_release(f.session, 0, d.scale_generation, 0) :
                potal_paired_fence(f.session);
            require(status == POTAL_ERR_BAD_RESPONSE, "direct completion uncertainty quarantines");
        }
    }
    const auto q = potal_paired_resident_query(f.session);
    require(q.state == POTAL_RESIDENT_WS_QUARANTINED && q.core.failed &&
            q.owned_bytes == 6 * UINT64_C(65536), "terminal state retains all owned DMA storage");
    const auto commands = f.device->records.size(), before_destroy = destroyed;
    auto *retained = f.session;
    require(potal_paired_fence(f.session) == q.core.error &&
            potal_paired_session_cleanup(f.session) == q.core.error &&
            potal_paired_resident_close(&f.session) == q.core.error && f.session == retained &&
            f.device->records.size() == commands && destroyed == before_destroy,
            "quarantine has no retry cleanup close destructor or owner-release path");
    potal_paired_session *other = nullptr;
    const potal_paired_sink sink{fixture::sink, f.device};
    require(potal_paired_session_open(&sink, &other) == POTAL_ERR_BUSY && !other,
            "unknown accelerator cannot be reacquired by a legacy issuer");
    if (kind == 1) require(q.unknown_fence && q.cleanup_attempted &&
        std::strcmp(q.core.error_stage, "resident_loop_ws") == 0, "later Fence failure preserves earlier sticky stage");
    if (kind == 4) require(q.last_drain == POTAL_OK && q.cleanup_status == POTAL_ERR_BAD_RESPONSE,
        "failed cleanup Release retains ownership after known DMA drain");
    /* Session, context and pending source buffers intentionally survive until _exit. */
}

void isolated_terminal_failures() {
    std::fflush(nullptr);
    for (unsigned kind = 0; kind < 6; ++kind) {
        const pid_t child = fork();
        require(child >= 0, "fork deterministic terminal case");
        if (child == 0) {
            try { terminal_case(kind); _exit(0); }
            catch (const std::exception &error) {
                std::fprintf(stderr, "E4B2_TERMINAL_FAIL kind=%u %s\n", kind, error.what());
                _exit(1);
            }
        }
        int status = 0;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "isolated terminal failure proof");
    }
}

void legacy_mode_switching() {
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        std::vector<potal_paired_cmd_record> records;
        const potal_paired_sink sink{
            [](void *context, const potal_paired_cmd_record *record) {
                static_cast<std::vector<potal_paired_cmd_record> *>(context)->push_back(*record);
                return POTAL_OK;
            }, &records};
        potal_paired_session *legacy = nullptr;
        ok(potal_paired_session_open(&sink, &legacy), "legacy owner after resident close");
        alignas(64) std::array<int8_t, 512> a{};
        alignas(64) std::array<int8_t, 1024> b{};
        potal_paired_loop_ws loop{};
        loop.loop.a = a.data(); loop.loop.b = b.data();
        loop.loop.lda = 32; loop.loop.ldb = 32;
        loop.loop.i_tiles = 1; loop.loop.j_tiles = 2; loop.loop.k_tiles = 2;
        loop.loop.scale_generation = 7; loop.loop.passes_per_block = 2;
        loop.path = POTAL_PAIRED_MAIN;
        for (unsigned slot = 0; slot < 2; ++slot)
            for (unsigned col = 0; col < 16; ++col)
                ok(potal_paired_scale_write(legacy, slot, 7, col, 0), "legacy checked Scale write");
        const auto before = records.size();
        ok(potal_paired_loop_ws_issue(legacy, &loop), "legacy paired LOOP issue");
        require(records.size() == before + 10 && records.back().funct == 8 &&
                records.back().xd == 0 && !(records.back().rs1 & 8) &&
                records[before + 5].rs2 == uintptr_t(b.data()),
                "legacy keeps ten commands, no resident mode and real B DRAM");
        ok(potal_paired_fence(legacy), "legacy drain after mode switch");
        for (unsigned slot = 0; slot < 2; ++slot)
            for (unsigned col = 0; col < 16; ++col)
                ok(potal_paired_scale_release(legacy, slot, 7, col), "legacy checked Scale release");
        ok(potal_paired_session_close(legacy), "legacy close after mode switch");
        fixture resident;
        resident.close();
    }
}

void validation_cost() {
    fixture f;
    f.load(0);
    const auto parts = f.canonical(0, 112, 48);
    const auto run = f.prepare(0, 0, false, 112, 48, 48, 32, 3, 112, 160, parts);
    auto *a = f.buffer(POTAL_RESIDENT_WS_MAIN_A);
    potal_resident_ws_program program{};
    uint64_t checksum = 0;
    for (unsigned sample = 0; sample < 7; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned call = 0; call < 512; ++call) {
            ok(potal_resident_ws_run_encode(&f.loads[0], &run, a, 65536, &program),
               "timed checked physical encoder");
            checksum += (program.view.rs1 & 65535) + program.view.rs2;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("E4B2_CPU_ENCODER sample=%u calls=512 cpu_ms=%.6f "
                    "scope=physical_check_encode_output_checksum hardware=0\n", sample, ms);
    }
    const auto commands = f.device->records.size();
    for (unsigned sample = 0; sample < 7; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned call = 0; call < 512; ++call) {
            ok(potal_paired_resident_run_check(f.session, &run), "timed live binding validation");
            checksum += run.output_id;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("E4B2_CPU_BINDING sample=%u calls=512 cpu_ms=%.6f "
                    "scope=session_check_status_checksum hardware=0\n", sample, ms);
    }
    require(f.device->records.size() == commands, "timed checks issue no accelerator command");
    std::printf("E4B2_CPU_ENCODER_CHECKSUM value=%llu\n", static_cast<unsigned long long>(checksum));
    f.close(); // This epoch reservation was never initialized by a GEMM.
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 1) {
        std::fprintf(stderr, "Software-only conformance rejects argument: %s\n", argv[1]);
        return 2;
    }
    try {
        interpreter_self_test();
        require(potal_e4b2_c11_header_test() == 0, "C11 public header and pure transport caller");
        production_numerics();
        saturation_and_epochs();
        capability_and_atomic_refusals();
        capacity_and_stale_bindings();
        recoverable_failures();
        isolated_terminal_failures();
        legacy_mode_switching();
        validation_cost();
        std::printf("E4B2_SOFTWARE_PASS assertions=%llu runs=%llu cells=%llu destroyed=%llu "
                    "scope=production_C11_and_independent_command_model hardware=0\n",
                    static_cast<unsigned long long>(assertions),
                    static_cast<unsigned long long>(numerical_runs),
                    static_cast<unsigned long long>(numerical_cells),
                    static_cast<unsigned long long>(destroyed));
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "E4B2_SOFTWARE_FAIL %s\n", error.what());
        return 1;
    }
}
