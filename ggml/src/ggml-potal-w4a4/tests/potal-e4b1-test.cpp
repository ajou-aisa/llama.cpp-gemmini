// E4B1 proposed interface conformance. Every completion and byte image here is
// software-only. This executable has no issuer, sink, or accelerator handle.
#include "potal-e4b1.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace potal_e4b1;
using wide = __int128_t;
constexpr int32_t untouched = 0x12345678;
constexpr potal_status invalid = POTAL_ERR_INVALID_ARGUMENT;
constexpr potal_status busy = POTAL_ERR_BUSY;
struct totals {
    uint64_t microtiles = 0, partials = 0, scale_records = 0;
    uint64_t index_records = 0, snapshot_bytes = 0, restore_bytes = 0, assertions = 0;
} counts;

void require(bool value, const char *message) {
    ++counts.assertions;
    if (!value) throw std::runtime_error(message);
}
void ok(potal_status status, const char *message) {
    if (status != POTAL_OK) {
        std::fprintf(stderr, "%s: status=%d\n", message, int(status));
        require(false, message);
    }
}
uint32_t ceil16(uint32_t n) { return (n + 15) / 16; }
uint32_t mask(uint32_t n) { return (uint32_t(1) << n) - 1; }

// The model uses iterative saturating doubling, never a host overflowing shift.
int32_t model_scale(int64_t dot, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || dot == 0) return 0;
    int64_t value = dot;
    for (uint32_t n = 0; n < carrier; ++n) {
        value *= 2;
        if (value > INT32_MAX) return INT32_MAX;
        if (value < INT32_MIN) return INT32_MIN;
    }
    return int32_t(value);
}
int32_t model_add(int32_t a, int32_t b) {
    const int64_t value = int64_t(a) + b;
    return value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : int32_t(value);
}

// Independent wide oracle, with no translator addresses, slots, or SCU helpers.
int32_t golden_clamp(wide x) {
    return x > INT32_MAX ? INT32_MAX : x < INT32_MIN ? INT32_MIN : int32_t(x);
}
int32_t golden_scale(wide x, uint32_t carrier) {
    if (!x || carrier == POTAL_CARRIER_ZERO) return 0;
    if (carrier >= 32) return x > 0 ? INT32_MAX : INT32_MIN;
    return golden_clamp(x * (wide(1) << carrier));
}

bool equal_translation(const translation &a, const translation &b) {
    if (a.requires_rtl != b.requires_rtl || a.partial_count != b.partial_count ||
        a.j_tiles != b.j_tiles || a.passes_per_block != b.passes_per_block ||
        a.b_effective_base != b.b_effective_base || a.b_row_stride != b.b_row_stride ||
        a.indexed_j_origin != b.indexed_j_origin ||
        a.acc_effective_base != b.acc_effective_base || a.a_sp_base != b.a_sp_base ||
        a.next_sequence != b.next_sequence || a.next_original_k != b.next_original_k ||
        a.acc_rows != b.acc_rows || a.valid_columns != b.valid_columns) return false;
    for (uint32_t p = 0; p < max_k; ++p) {
        const auto &x = a.partials[p];
        const auto &y = b.partials[p];
        if (x.valid_k != y.valid_k || x.phase != y.phase ||
            x.original_block != y.original_block || x.original_k != y.original_k ||
            x.b_rows != y.b_rows || x.scale_slots != y.scale_slots) return false;
    }
    return true;
}

struct snapshot {
    token identity{};
    uint32_t bank = 0, tiles = 0;
    std::array<epoch, 32> epochs{};
    std::vector<int32_t> bytes = std::vector<int32_t>(512 * 16);
    bool valid = false, sole_open_copy = false;
};

// One state owns the view, sources, packed SP/ACC, and two fixed save slots.
// No copy is allowed because view.activation points into its owned allocation.
struct model {
    view state{};
    std::vector<int8_t> activation = std::vector<int8_t>(65536);
    std::array<std::vector<int8_t>, 2> source{
        std::vector<int8_t>(65536), std::vector<int8_t>(65536)};
    std::vector<int8_t> sp = std::vector<int8_t>(16384 * 16);
    std::vector<int32_t> acc = std::vector<int32_t>(1024 * 16, untouched);
    std::array<snapshot, 2> saves{};
    token clock{};
    enum class operation { none, load, compute, readback, restore, control } pending = operation::none;
    operation first_failure = operation::none;
    resident loading{};
    tile request{};
    translation translated{};
    uint32_t pending_pair = 0, pending_save = 0, pending_bank = 0, pending_tiles = 0;
    bool issued = false, cleanup_attempted = false, cleaned = false, closed = false;
    bool unknown_completion = false;
    potal_status last_drain = POTAL_OK, cleanup_status = POTAL_OK;
    uint32_t drains = 0, cleanup_drains = 0, effects = 0;
    static uint64_t next_domain;

    model() {
        require(next_domain != UINT64_MAX, "model domain cannot wrap");
        clock.domain = ++next_domain;
        state.activation = activation.data();
        state.activation_bytes = activation.size();
    }
    model(const model &) = delete;
    model &operator=(const model &) = delete;
    token fresh() {
        token result;
        ok(next_token(clock, result), "model token cannot wrap");
        clock = result;
        return result;
    }
    potal_status ready() const {
        if (state.sticky != POTAL_OK) return state.sticky;
        if (closed || cleaned || state.quarantined) return busy;
        return state.pending ? busy : POTAL_OK;
    }
    void reserve(operation kind) {
        require(!state.pending, "pending is reserved before every effect");
        state.pending = true;
        pending = kind;
        issued = false;
        effects = 0;
    }
    potal_status fail(potal_status error) {
        if (state.sticky == POTAL_OK) {
            state.sticky = error;
            first_failure = pending;
        }
        return error;
    }
    potal_status load(resident r, int fail_after = -1) {
        if (ready() != POTAL_OK) return ready();
        const auto status = check_resident(r);
        if (status != POTAL_OK) return status;
        const auto &g = r.geometry;
        if (state.residents[g.pair].readable) return busy;
        if (clock.generation == UINT64_MAX) return POTAL_ERR_CAPACITY;
        r.identity = fresh();
        r.readable = false;
        loading = r;
        pending_pair = g.pair;
        reserve(operation::load);
        state.residents[g.pair] = r;
        state.indices[g.pair] = {};
        // Old Scale bindings become unusable through resident identity, but
        // their live columns still require completion-qualified Release.
        // Explicit packed B layout; the later compute consumes translation rows.
        for (uint32_t k = 0; k < g.k_rows; ++k) {
            for (uint32_t j = 0; j < g.j_columns; ++j) {
                const uint32_t row = g.b_sp_base +
                    (k / 16 * ceil16(g.j_columns) + j / 16) * 16 + k % 16;
                sp[row * 16 + j % 16] = source[g.pair][k * g.source_stride + j];
            }
            ++effects;
            if (int(effects) == fail_after) return fail(POTAL_ERR_BAD_RESPONSE);
        }
        issued = true;
        return POTAL_OK;
    }
    potal_status retire(uint32_t pair) {
        if (ready() != POTAL_OK) return ready();
        if (pair > 1 || !state.residents[pair].readable) return invalid;
        for (const auto &entry : state.scales)
            if (entry.live_columns && same(entry.resident_identity, state.residents[pair].identity))
                return busy;
        state.residents[pair].readable = false;
        state.indices[pair] = {};
        return POTAL_OK;
    }
    potal_status program_index(uint32_t pair, const std::vector<uint64_t> &rows,
                               int fail_after = -1) {
        if (ready() != POTAL_OK) return ready();
        if (pair > 1 || rows.empty() || rows.size() > 32 || !state.residents[pair].readable)
            return invalid;
        index_window window;
        const auto next = next_token(clock, window.incarnation);
        if (next != POTAL_OK) return next;
        window.resident_identity = state.residents[pair].identity;
        window.count = uint32_t(rows.size());
        std::copy(rows.begin(), rows.end(), window.original_k.begin());
        window.published = true;
        std::array<potal_paired_cmd_record, 33> wire{};
        uint32_t n = 0;
        const auto status = encode_index(state.residents[pair], window, wire, n);
        if (status != POTAL_OK) return status;
        require(n == rows.size() + 1, "Index config plus one write per entry");
        const auto &g = state.residents[pair].geometry;
        require(wire[0].funct == 27 && wire[0].xd == 1 &&
                wire[0].rs1 == (uint64_t(ceil16(g.j_columns)) << 32 | g.b_sp_base) &&
                wire[0].rs2 == (uint64_t(rows.size()) << 32 | g.k_rows),
                "exact Index config fields");
        for (uint32_t x = 0; x < rows.size(); ++x)
            require(wire[x + 1].funct == 26 && wire[x + 1].xd == 1 &&
                    wire[x + 1].rs1 == x && wire[x + 1].rs2 == rows[x] - g.k_begin,
                    "exact panel-relative Index write");
        pending_pair = pair;
        reserve(operation::control);
        state.indices[pair] = {}; // Invalidate before the first modeled CONFIG.
        for (uint32_t step = 0; step < n; ++step) {
            ++effects;
            if (int(step) == fail_after) return fail(POTAL_ERR_BAD_RESPONSE);
        }
        counts.index_records += n;
        // Control publication is atomic, not a data drain.
        clock = window.incarnation;
        state.indices[pair] = window;
        state.pending = false;
        pending = operation::none;
        return POTAL_OK;
    }
    potal_status program_scales(const std::array<scale_entry, 256> &entries,
                                uint32_t base, uint32_t count, int unknown_after = -1) {
        if (ready() != POTAL_OK) return ready();
        if (!count || base >= 256 || count > 256 - base) return POTAL_ERR_CAPACITY;
        uint64_t writes = 0;
        for (uint32_t n = 0; n < count; ++n) {
            const auto &s = entries[base + n];
            if (state.scales[base + n].live_columns) return POTAL_ERR_CAPACITY;
            if (!same(s.incarnation, s.incarnation) || !s.live_columns ||
                (s.live_columns >> 16) || s.generation > 255) return invalid;
            if (s.incarnation.domain != clock.domain ||
                s.incarnation.generation > clock.generation ||
                same(s.incarnation, state.scales[base + n].incarnation)) return invalid;
            const resident *r = nullptr;
            for (const auto &candidate : state.residents)
                if (candidate.readable && same(candidate.identity, s.resident_identity)) r = &candidate;
            if (!r || s.original_block < r->geometry.k_begin / 32 ||
                s.original_block > (r->geometry.k_begin + r->geometry.k_rows - 1) / 32 ||
                s.original_j < r->geometry.j_begin ||
                s.original_j - r->geometry.j_begin >= r->geometry.j_columns ||
                (s.original_j - r->geometry.j_begin) % 16) return invalid;
            const uint32_t columns = std::min(uint64_t(16),
                r->geometry.j_columns - (s.original_j - r->geometry.j_begin));
            if (s.live_columns & ~mask(columns)) return invalid;
            for (uint32_t c = 0; c < 16; ++c) if (s.live_columns & (1u << c)) {
                potal_paired_cmd_record wire;
                if (encode_scale(base + n, s.generation, c, s.carriers[c], wire) != POTAL_OK)
                    return invalid;
                ++writes;
            }
        }
        reserve(operation::control);
        for (uint32_t n = 0; n < count; ++n) {
            state.scales[base + n] = entries[base + n];
            ++effects;
            if (int(n) == unknown_after) {
                state.quarantined = true;
                unknown_completion = true;
                return fail(POTAL_ERR_BAD_RESPONSE);
            }
        }
        counts.scale_records += writes;
        state.pending = false; // Complete ordered model acknowledgements, not a DMA drain.
        pending = operation::none;
        return POTAL_OK;
    }
    potal_status release(uint32_t slot, uint32_t generation, uint32_t column, bool known = true) {
        if (state.quarantined || cleaned || closed || state.sticky != POTAL_OK) return busy;
        if (slot >= 256 || column >= 16 || generation != state.scales[slot].generation ||
            !(state.scales[slot].live_columns & (1u << column))) return invalid;
        // A rejected Release above must not qualify older work.
        const auto status = drain(known);
        if (status != POTAL_OK) return status;
        state.scales[slot].live_columns &= ~(1u << column);
        return POTAL_OK;
    }
    potal_status clear_scales() {
        if (ready() != POTAL_OK) return ready();
        for (uint32_t s = 0; s < 256; ++s)
            for (uint32_t c = 0; c < 16; ++c)
                if (state.scales[s].live_columns & (1u << c)) {
                    const auto status = release(s, state.scales[s].generation, c);
                    if (status != POTAL_OK) return status;
                }
        return POTAL_OK;
    }
    potal_status accept(const tile &t, const translation &record, int fail_partial = -1) {
        if (ready() != POTAL_OK) return ready();
        if (state.activation != activation.data() || state.activation_bytes != activation.size())
            return invalid;
        translation current;
        const auto status = translate(state, t, current);
        if (status != POTAL_OK) return status;
        if (!equal_translation(current, record)) return invalid;
        if (t.first && clock.generation > UINT64_MAX - current.j_tiles)
            return POTAL_ERR_CAPACITY;
        request = t;
        translated = current;
        pending_bank = t.acc_bank;
        pending_tiles = 0;
        for (uint32_t j = 0; j < current.j_tiles; ++j)
            pending_tiles |= 1u << ((current.acc_rows[j] % 512) / 16);
        reserve(operation::compute);
        for (uint32_t p = 0; p < current.partial_count; ++p) {
            // A is a real owned vector and is packed as I rows by K lanes.
            for (uint32_t i = 0; i < t.i_rows; ++i)
                for (uint32_t x = 0; x < 16; ++x)
                    sp[(current.a_sp_base + p * 16 + i) * 16 + x] =
                        x < current.partials[p].valid_k ?
                        activation[t.a_offset + i * t.a_stride + p * 16 + x] : 0;
            const auto &part = current.partials[p];
            for (uint32_t j = 0; j < current.j_tiles; ++j)
                for (uint32_t i = 0; i < t.i_rows; ++i)
                    for (uint32_t c = 0; c < current.valid_columns[j]; ++c) {
                        int64_t dot = 0;
                        for (uint32_t x = 0; x < part.valid_k; ++x)
                            dot += int64_t(sp[(current.a_sp_base + p * 16 + i) * 16 + x]) *
                                   sp[part.b_rows[j][x] * 16 + c];
                        const int32_t value = model_scale(dot,
                            state.scales[part.scale_slots[j]].carriers[c]);
                        auto &target = acc[(current.acc_rows[j] + i) * 16 + c];
                        target = t.first && p == 0 ? value : model_add(target, value);
                    }
            ++effects;
            if (int(p) == fail_partial) return fail(POTAL_ERR_BAD_RESPONSE);
        }
        ++counts.microtiles;
        counts.partials += current.partial_count * current.j_tiles;
        issued = true;
        return POTAL_OK;
    }
    potal_status readback(uint32_t bank, uint32_t tiles, uint32_t save, int fail_tile = -1) {
        if (ready() != POTAL_OK) return ready();
        if (bank > 1 || save > 1 || !tiles || saves[save].sole_open_copy) return invalid;
        if (clock.generation == UINT64_MAX) return POTAL_ERR_CAPACITY;
        for (uint32_t x = 0; x < 32; ++x)
            if ((tiles & (1u << x)) && !state.epochs[bank][x].initialized) return invalid;
        pending_bank = bank;
        pending_tiles = tiles;
        pending_save = save;
        reserve(operation::readback);
        auto &s = saves[save];
        s.valid = false;
        s.identity = fresh();
        s.bank = bank;
        s.tiles = tiles;
        s.epochs = state.epochs[bank];
        for (uint32_t x = 0; x < 32; ++x) if (tiles & (1u << x)) {
            std::copy_n(acc.begin() + (bank * 512 + x * 16) * 16, 256,
                        s.bytes.begin() + x * 256);
            ++effects;
            if (int(effects) == fail_tile) return fail(POTAL_ERR_BAD_RESPONSE);
        }
        issued = true;
        return POTAL_OK;
    }
    potal_status relinquish(uint32_t save, token identity) {
        if (ready() != POTAL_OK) return ready();
        if (save > 1 || !saves[save].valid || !same(saves[save].identity, identity)) return invalid;
        auto &s = saves[save];
        for (uint32_t x = 0; x < 32; ++x) if (s.tiles & (1u << x))
            if (!same(s.epochs[x].identity, state.epochs[s.bank][x].identity)) return invalid;
        s.sole_open_copy = false;
        for (uint32_t x = 0; x < 32; ++x) if (s.tiles & (1u << x)) {
            s.sole_open_copy |= !s.epochs[x].complete;
            state.epochs[s.bank][x] = {};
        }
        return POTAL_OK;
    }
    potal_status restore(uint32_t save, token identity, int fail_tile = -1) {
        if (ready() != POTAL_OK) return ready();
        if (save > 1 || !saves[save].valid || !same(saves[save].identity, identity)) return invalid;
        const auto &s = saves[save];
        uint32_t restored_tiles = 0;
        for (uint32_t x = 0; x < 32; ++x) restored_tiles += (s.tiles >> x) & 1;
        if (clock.generation > UINT64_MAX - restored_tiles) return POTAL_ERR_CAPACITY;
        for (uint32_t x = 0; x < 32; ++x)
            if ((s.tiles & (1u << x)) && state.epochs[s.bank][x].initialized) return busy;
        pending_save = save;
        pending_bank = s.bank;
        pending_tiles = s.tiles;
        reserve(operation::restore);
        for (uint32_t x = 0; x < 32; ++x) if (s.tiles & (1u << x)) {
            state.epochs[s.bank][x] = {};
            std::copy_n(s.bytes.begin() + x * 256, 256,
                        acc.begin() + (s.bank * 512 + x * 16) * 16);
            ++effects;
            if (int(effects) == fail_tile) return fail(POTAL_ERR_BAD_RESPONSE);
        }
        issued = true;
        return POTAL_OK;
    }
    potal_status drain(bool known = true) {
        if (state.quarantined || closed || cleaned) return busy;
        if (state.sticky != POTAL_OK && !cleanup_attempted) return cleanup(known);
        ++drains;
        if (!known) {
            last_drain = POTAL_ERR_BUSY;
            unknown_completion = true;
            state.quarantined = true;
            return fail(last_drain);
        }
        last_drain = POTAL_OK;
        if (!state.pending) return POTAL_OK;
        if (state.sticky == POTAL_OK && issued) {
            if (pending == operation::load) {
                loading.readable = true;
                state.residents[pending_pair] = loading;
            } else if (pending == operation::compute) {
                const auto &g = state.residents[request.pair].geometry;
                for (uint32_t j = 0; j < translated.j_tiles; ++j) {
                    auto &e = state.epochs[request.acc_bank][translated.acc_rows[j] % 512 / 16];
                    if (request.first) {
                        e = {};
                        e.identity = fresh();
                    }
                    e.output_id = request.output_id;
                    e.weight_id = g.weight_id;
                    e.weight_generation = g.weight_generation;
                    e.original_i = request.original_i;
                    e.original_j = g.j_begin + request.j_offset + j * 16;
                    e.logical_k_begin = request.logical_k_begin;
                    e.logical_k_end = request.logical_k_end;
                    e.next_sequence = translated.next_sequence;
                    e.next_original_k = translated.next_original_k;
                    e.total_partials = request.total_partials;
                    e.rows = request.i_rows;
                    e.columns = translated.valid_columns[j];
                    e.initialized = true;
                    e.complete = request.last;
                }
            } else if (pending == operation::readback) {
                saves[pending_save].valid = true;
                counts.snapshot_bytes += uint64_t(effects) * 256 * sizeof(int32_t);
            } else if (pending == operation::restore) {
                auto &s = saves[pending_save];
                for (uint32_t x = 0; x < 32; ++x) if (pending_tiles & (1u << x)) {
                    state.epochs[pending_bank][x] = s.epochs[x];
                    state.epochs[pending_bank][x].identity = fresh();
                }
                s.sole_open_copy = false;
                counts.restore_bytes += uint64_t(effects) * 256 * sizeof(int32_t);
            }
        } else {
            if (pending == operation::load) state.residents[pending_pair].readable = false;
            if (pending == operation::compute || pending == operation::restore)
                for (uint32_t x = 0; x < 32; ++x)
                    if (pending_tiles & (1u << x)) state.epochs[pending_bank][x] = {};
            if (pending == operation::readback) saves[pending_save].valid = false;
        }
        state.pending = false;
        pending = operation::none;
        return POTAL_OK;
    }
    potal_status cleanup(bool known = true, bool release_known = true) {
        if (cleanup_attempted) return cleanup_status;
        if (state.quarantined || closed) return busy;
        if (state.sticky == POTAL_OK) return busy;
        cleanup_attempted = true;
        ++cleanup_drains;
        cleanup_status = drain(known);
        if (cleanup_status != POTAL_OK) return cleanup_status;
        if (!release_known) {
            cleanup_status = POTAL_ERR_BAD_RESPONSE;
            unknown_completion = true;
            state.quarantined = true;
            return fail(cleanup_status);
        }
        for (auto &r : state.residents) r.readable = false;
        for (auto &s : state.scales) s.live_columns = 0;
        state.indices = {};
        state.epochs = {};
        cleaned = true;
        return POTAL_OK;
    }
    potal_status close() {
        if (!cleaned && state.sticky != POTAL_OK) return state.sticky;
        if (state.pending || state.quarantined) return busy;
        if (!cleaned) {
            if (state.sticky != POTAL_OK) return state.sticky;
            for (const auto &s : state.scales) if (s.live_columns) return busy;
            for (const auto &r : state.residents) if (r.readable) return busy;
            for (const auto &bank : state.epochs)
                for (const auto &e : bank) if (e.initialized && !e.complete) return busy;
            for (const auto &s : saves) if (s.sole_open_copy) return busy;
        }
        closed = true;
        return state.sticky;
    }
};
uint64_t model::next_domain = 0;

using weight_function = std::function<int8_t(uint64_t, uint64_t)>;
using activation_function = std::function<int8_t(uint32_t, uint64_t)>;
using carrier_function = std::function<uint32_t(uint64_t, uint64_t)>;
const weight_function varied_b = [](uint64_t k, uint64_t j) {
    // Marker lanes uniquely identify every one of the 28*9 resident tiles.
    if (j % 4 == 0) return int8_t(int((k / 16 % 28) * 9 + j / 16 % 9) - 126);
    return int8_t(int((k * 3 + j * 5) % 15) - 7);
};
const activation_function varied_a = [](uint32_t i, uint64_t k) {
    return int8_t(int((i * 7 + k * 2) % 11) - 5);
};
const carrier_function varied_c = [](uint64_t block, uint64_t j) {
    return (block + j) % 13 == 0 ? POTAL_CARRIER_ZERO : uint32_t((block + j) % 4);
};

resident geometry(uint32_t pair, uint32_t k = 448, uint32_t j = 144,
                  uint64_t k_begin = 0, uint64_t j_begin = 0) {
    resident r;
    auto &g = r.geometry;
    g.pair = pair;
    g.bank = pair * 2 + 1;
    g.b_sp_base = (pair * 2 + 2) * 4096 - ceil16(k) * ceil16(j) * 16;
    g.b_row_stride = ceil16(j) * 16;
    g.k_rows = k;
    g.j_columns = j;
    g.main_a_tiles = ceil16(k);
    g.k_begin = k_begin;
    g.j_begin = j_begin;
    g.weight_id = 71;
    g.weight_generation = 9;
    g.source_stride = j;
    r.identity = {1, 1};
    return r;
}
void load_fixture(model &m, resident r, const weight_function &b = varied_b) {
    const auto &g = r.geometry;
    for (uint32_t k = 0; k < g.k_rows; ++k)
        for (uint32_t j = 0; j < g.j_columns; ++j)
            m.source[g.pair][k * g.source_stride + j] = b(g.k_begin + k, g.j_begin + j);
    ok(m.load(r), "load fixture");
    require(m.state.pending && !m.state.residents[g.pair].readable, "issue is not B publication");
    ok(m.drain(), "load completion");
    require(m.state.residents[g.pair].readable, "drain publishes B");
    std::fill(m.source[g.pair].begin(), m.source[g.pair].end(), int8_t(93));
}

tile main_tile(model &m, uint32_t pair, uint32_t k, uint32_t rows,
               uint32_t j, uint32_t columns, uint32_t output_j = 0) {
    tile t;
    const auto &r = m.state.residents[pair];
    t.resident_identity = r.identity;
    t.pair = pair;
    t.k_offset = k;
    t.k_rows = rows;
    t.j_offset = j;
    t.j_columns = columns;
    t.i_rows = 16;
    t.a_stride = ceil16(rows) * 16;
    t.output_j = output_j;
    t.output_id = 100 + output_j;
    t.logical_k_begin = r.geometry.k_begin + k;
    t.logical_k_end = t.logical_k_begin + rows;
    t.total_partials = ceil16(rows);
    return t;
}
void continue_tile(model &m, tile &t, uint32_t k, uint32_t rows, bool last) {
    t.first = false;
    t.accumulate = true;
    t.last = last;
    t.k_offset = k;
    t.k_rows = rows;
    t.a_stride = ceil16(rows) * 16;
    for (uint32_t j = 0; j < ceil16(t.j_columns); ++j) {
        const uint32_t slot = t.acc_base / 16 + t.output_i * t.output_j_stride + t.output_j + j;
        t.epochs[j] = m.state.epochs[t.acc_bank][slot].identity;
    }
    const auto &e = m.state.epochs[t.acc_bank][t.acc_base / 16 +
        t.output_i * t.output_j_stride + t.output_j];
    t.sequence = e.next_sequence;
}

// The fixture supplies canonical source lists independently of translate().
// Short interior compact fragments occupy a full DIM slot with zero A fillers.
using fragments = std::vector<std::vector<uint64_t>>;
fragments canonical(const resident &r, const tile &t) {
    fragments rows;
    for (uint32_t k = 0; k < t.k_rows; k += 16) {
        rows.emplace_back();
        for (uint32_t x = k; x < std::min(k + 16, t.k_rows); ++x)
            rows.back().push_back(r.geometry.k_begin + t.k_offset + x);
    }
    return rows;
}
void prepare(model &m, tile &t, const fragments &rows,
             const activation_function &a = varied_a, const carrier_function &c = varied_c) {
    ok(m.clear_scales(), "retire old Scale window");
    const auto &r = m.state.residents[t.pair];
    t.resident_identity = r.identity;
    if (t.indexed) {
        std::vector<uint64_t> index;
        for (uint32_t p = 0; p < rows.size(); ++p) {
            t.fragment_counts[p] = uint32_t(rows[p].size());
            index.insert(index.end(), rows[p].begin(), rows[p].end());
            if (p + 1 < rows.size())
                while (index.size() % 16) index.push_back(rows[p].back());
        }
        t.k_rows = uint32_t(index.size());
        t.a_stride = ceil16(t.k_rows) * 16;
        ok(m.program_index(t.pair, index), "install complete Index window");
        t.index_incarnation = m.state.indices[t.pair].incarnation;
    }
    std::fill(m.activation.begin(), m.activation.end(), int8_t(0));
    for (uint32_t i = 0; i < t.i_rows; ++i)
        for (uint32_t p = 0; p < rows.size(); ++p)
            for (uint32_t x = 0; x < rows[p].size(); ++x)
                m.activation[t.a_offset + i * t.a_stride + p * 16 + x] = a(i, rows[p][x]);
    t.scale_incarnation = m.fresh();
    t.scale_generation = 255;
    const uint32_t jt = ceil16(t.j_columns);
    require(t.scale_base + rows.size() * jt <= 256, "bounded P1 window");
    auto programmed = m.state.scales;
    for (uint32_t p = 0; p < rows.size(); ++p) for (uint32_t j = 0; j < jt; ++j) {
        auto &s = programmed[t.scale_base + p * jt + j];
        s = {};
        s.incarnation = t.scale_incarnation;
        s.resident_identity = r.identity;
        s.original_block = rows[p][0] / 32;
        s.original_j = r.geometry.j_begin + t.j_offset + j * 16;
        s.generation = t.scale_generation;
        const uint32_t columns = std::min(16u, t.j_columns - j * 16);
        s.live_columns = mask(columns);
        for (uint32_t x = 0; x < columns; ++x) {
            s.carriers[x] = c(s.original_block, s.original_j + x);
            potal_paired_cmd_record wire{};
            ok(encode_scale(t.scale_base + p * jt + j, s.generation, x, s.carriers[x], wire),
               "pure Scale encoder");
            require(wire.funct == 23 && wire.xd == 1 &&
                    wire.rs1 == (uint64_t(t.scale_base + p * jt + j) |
                                 uint64_t(s.generation) << 8 | uint64_t(x) << 16) &&
                    wire.rs2 == s.carriers[x], "exact Scale wire fields");
        }
    }
    ok(m.program_scales(programmed, t.scale_base, uint32_t(rows.size()) * jt),
       "publish validated Scale control transaction");
}

translation checked(model &m, const tile &t) {
    translation out;
    ok(translate(m.state, t, out), "valid pure translation");
    require(out.requires_rtl && out.passes_per_block == 1, "proposed P1 metadata is not an opcode");
    return out;
}

void execute(model &m, const tile &t, const fragments &rows,
             const weight_function &b = varied_b, const activation_function &a = varied_a,
             const carrier_function &c = varied_c, int32_t constant = untouched) {
    const auto before = m.acc;
    auto expected = before;
    std::vector<bool> selected(expected.size(), false);
    const auto &g = m.state.residents[t.pair].geometry;
    // Enumerate logical output coordinates directly, without translated ACC rows.
    for (uint32_t i = 0; i < t.i_rows; ++i) for (uint32_t j = 0; j < t.j_columns; ++j) {
        const uint32_t tile_slot = t.acc_base / 16 + t.output_i * t.output_j_stride + t.output_j + j / 16;
        const uint32_t cell = (t.acc_bank * 512 + tile_slot * 16 + i) * 16 + j % 16;
        int32_t value = t.first ? 0 : before[cell];
        for (const auto &fragment : rows) {
            wide dot = 0;
            for (uint64_t k : fragment) dot += wide(a(i, k)) * b(k, g.j_begin + t.j_offset + j);
            value = golden_clamp(wide(value) + golden_scale(dot, c(fragment[0] / 32,
                g.j_begin + t.j_offset + j)));
        }
        expected[cell] = value;
        selected[cell] = true;
        if (constant != untouched) require(value == constant, "independent manually calculated witness");
    }
    const auto tr = checked(m, t);
    require(tr.partial_count == rows.size() && tr.j_tiles == ceil16(t.j_columns),
            "exact canonical contribution count");
    for (uint32_t p = 0; p < rows.size(); ++p) {
        const auto &part = tr.partials[p];
        const uint32_t valid = uint32_t(rows[p].size());
        require(part.valid_k == valid && part.original_block == rows[p][0] / 32 &&
                part.phase == (rows[p][0] / 16) % 2, "original block and first-source phase");
        for (uint32_t x = 0; x < rows[p].size(); ++x)
            require(part.original_k[x] == rows[p][x], "ordered source contribution identity");
        for (uint32_t j = 0; j < tr.j_tiles; ++j) {
            require(part.scale_slots[j] == t.scale_base + p * tr.j_tiles + j, "P1 per-partial binding");
            for (uint32_t x = 0; x < rows[p].size(); ++x) {
                const uint32_t q = uint32_t(rows[p][x] - g.k_begin);
                const uint32_t expected_row = g.b_sp_base +
                    (q / 16 * ceil16(g.j_columns) + t.j_offset / 16 + j) * 16 + q % 16;
                require(part.b_rows[j][x] == expected_row, "full resident pitch and interior B origin");
            }
        }
    }
    ok(m.accept(t, tr), "accept revalidated translation");
    require(m.state.pending, "compute issue return is not publication");
    ok(m.drain(), "software compute drain");
    for (size_t x = 0; x < expected.size(); ++x)
        require(m.acc[x] == expected[x] && (selected[x] || m.acc[x] == before[x]),
                "every raw INT32 and every untouched ACC cell");
}

void numerical_geometry() {
    for (uint32_t pair = 0; pair < 2; ++pair) {
        model m;
        load_fixture(m, geometry(pair));
        tile full = main_tile(m, pair, 0, 448, 0, 144);
        auto rows = canonical(m.state.residents[pair], full);
        prepare(m, full, rows);
        execute(m, full, rows);
        tile extra = main_tile(m, pair, 0, 32, 0, 144);
        extra.indexed = true;
        extra.path = POTAL_PAIRED_RESIDUAL;
        extra.acc_bank = 1;
        extra.a_offset_tiles = 28;
        extra.scale_base = 252;
        translation excess;
        require(translate(m.state, extra, excess) == POTAL_ERR_CAPACITY &&
                m.state.scales[251].live_columns == 0xffff,
                "P1 Main252 plus Residual18 rejects270 before programming");
        // Independent first, interior, last K/J and a joint odd-phase interior.
        for (const auto &cut : std::array<std::array<uint32_t, 4>, 4>{{
                 {{0, 16, 0, 16}}, {{112, 48, 48, 32}},
                 {{432, 16, 128, 16}}, {{48, 32, 80, 16}}}}) {
            tile t = main_tile(m, pair, cut[0], cut[1], cut[2], cut[3], cut[2] / 16);
            t.acc_base = 160;
            t.output_i = 1;
            t.output_j_stride = 9;
            rows = canonical(m.state.residents[pair], t);
            prepare(m, t, rows);
            execute(m, t, rows);
        }
        tile indexed = main_tile(m, pair, 112, 32, 48, 32, 3);
        indexed.indexed = true; // Indexed Main remains Main and writes ACC0.
        indexed.acc_base = 320;
        rows = canonical(m.state.residents[pair], indexed);
        prepare(m, indexed, rows);
        execute(m, indexed, rows);
        tile residual = main_tile(m, pair, 96, 19, 48, 32, 3);
        residual.path = POTAL_PAIRED_RESIDUAL;
        residual.acc_bank = 1;
        residual.indexed = true;
        residual.a_offset_tiles = m.state.residents[pair].geometry.main_a_tiles;
        residual.logical_k_begin = 96;
        residual.logical_k_end = 160;
        residual.total_partials = 2;
        rows = {{96, 98, 126}, {128, 135, 143}};
        prepare(m, residual, rows);
        execute(m, residual, rows);
        require(m.activation[3] == 0 && m.activation[15] == 0, "short interior fragment has zero filler");
        ok(m.clear_scales(), "release P1 window");
        ok(m.retire(pair), "retire before healthy close");
        ok(m.close(), "close healthy model");
    }
    for (uint32_t pair = 0; pair < 2; ++pair) {
        model m;
        load_fixture(m, geometry(pair, 443, 139, 16, 32));
        tile tail = main_tile(m, pair, 0, 443, 0, 139);
        tail.i_rows = 7;
        auto rows = canonical(m.state.residents[pair], tail);
        prepare(m, tail, rows);
        execute(m, tail, rows);
        tail = main_tile(m, pair, 432, 11, 128, 11, 8);
        tail.i_rows = 1;
        tail.acc_base = 256;
        tail.indexed = true;
        rows = canonical(m.state.residents[pair], tail);
        prepare(m, tail, rows);
        execute(m, tail, rows);
    }
}

void numerical_witnesses() {
    const activation_function one = [](uint32_t, uint64_t) { return int8_t(1); };
    const carrier_function c31 = [](uint64_t, uint64_t) { return 31u; };
    const carrier_function c24 = [](uint64_t, uint64_t) { return 24u; };
    const weight_function cancel = [](uint64_t k, uint64_t) {
        return int8_t(k == 0 ? 1 : k == 16 ? -1 : 0);
    };
    for (bool compact : {false, true}) {
        model m;
        load_fixture(m, geometry(0), cancel);
        tile t = main_tile(m, 0, 0, 32, 0, 144);
        fragments rows = canonical(m.state.residents[0], t);
        if (compact) {
            t.path = POTAL_PAIRED_RESIDUAL;
            t.acc_bank = 1;
            t.indexed = true;
            t.a_offset_tiles = 28;
            t.total_partials = 1;
            rows = {{0, 16}};
        }
        prepare(m, t, rows, one, c31);
        execute(m, t, rows, cancel, one, c31, compact ? 0 : -1);
    }
    {
        model m;
        const weight_function b = [](uint64_t k, uint64_t) {
            return int8_t(k < 16 ? 127 : k == 16 ? -127 : 0);
        };
        load_fixture(m, geometry(0), b);
        tile t = main_tile(m, 0, 0, 17, 0, 144);
        t.path = POTAL_PAIRED_RESIDUAL;
        t.acc_bank = 1;
        t.indexed = true;
        t.a_offset_tiles = 28;
        t.logical_k_end = 32;
        t.total_partials = 2;
        fragments rows(2);
        for (uint64_t k = 0; k < 16; ++k) rows[0].push_back(k);
        rows[1] = {16};
        prepare(m, t, rows, one, c24);
        execute(m, t, rows, b, one, c24, 16777215);
        require(golden_scale(2032 - 127, 24) == INT32_MAX, "merged dot is a distinct wrong result");
    }
    {
        model m;
        const weight_function b = [](uint64_t k, uint64_t) {
            return int8_t(k == 0 || k == 416 ? 1 : k == 448 ? -1 : 0);
        };
        load_fixture(m, geometry(0), b);
        tile t = main_tile(m, 0, 0, 448, 0, 144);
        t.logical_k_end = 464;
        t.total_partials = 29;
        t.last = false;
        auto rows = canonical(m.state.residents[0], t);
        prepare(m, t, rows, one, c31);
        execute(m, t, rows, b, one, c31);
        ok(m.clear_scales(), "release old residency carriers");
        ok(m.retire(0), "B retirement preserves OPEN output");
        load_fixture(m, geometry(1, 16, 144, 448), b);
        t.pair = 1;
        t.resident_identity = m.state.residents[1].identity;
        continue_tile(m, t, 0, 16, true);
        rows = canonical(m.state.residents[1], t);
        prepare(m, t, rows, one, c31);
        execute(m, t, rows, b, one, c31, -1);
        require(golden_clamp(wide(golden_clamp(wide(INT32_MAX) + INT32_MIN)) + INT32_MAX)
                == 2147483646, "reordered MAX MIN MAX is not original MAX MAX MIN");
    }
    {
        model m;
        const weight_function negative = [](uint64_t, uint64_t) { return int8_t(-127); };
        const activation_function positive = [](uint32_t, uint64_t) { return int8_t(127); };
        const carrier_function large = [](uint64_t, uint64_t j) {
            return j % 3 == 0 ? POTAL_CARRIER_ZERO : j % 3 == 1 ? 0u : 32767u;
        };
        load_fixture(m, geometry(0), negative);
        tile t = main_tile(m, 0, 0, 32, 0, 144);
        auto rows = canonical(m.state.residents[0], t);
        prepare(m, t, rows, positive, large);
        execute(m, t, rows, negative, positive, large);
        require(m.acc[0] == 0 && m.acc[1] == -516128 && m.acc[2] == INT32_MIN,
                "zero sentinel, identity and negative partial/ACC saturation");
    }
}

void rejected(model &m, tile t, const std::function<void(view &, tile &)> &mutate) {
    view copy = m.state;
    const auto before = m.acc;
    mutate(copy, t);
    translation out;
    require(translate(copy, t, out) != POTAL_OK, "invalid semantic descriptor rejected");
    require(before == m.acc && !m.state.pending, "pure rejection has no effects");
}

void descriptor_negatives() {
    model m;
    load_fixture(m, geometry(0));
    tile t = main_tile(m, 0, 112, 32, 48, 32, 3);
    auto rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    const auto valid = checked(m, t);
    using mutation = std::function<void(view &, tile &)>;
    const std::vector<mutation> mutations = {
        [](view &, tile &x) { x.resident_identity.generation++; },
        [](view &, tile &x) { x.scale_incarnation.generation++; },
        [](view &, tile &x) { x.scale_generation = 0; },
        [](view &, tile &x) { x.k_offset = 113; },
        [](view &, tile &x) { x.k_rows = 17; x.logical_k_end = 129; },
        [](view &, tile &x) { x.j_offset = 49; },
        [](view &, tile &x) { x.j_columns = 145; },
        [](view &, tile &x) { x.i_rows = 17; },
        [](view &, tile &x) { x.acc_bank = 1; },
        [](view &, tile &x) { x.acc_base = 1; },
        [](view &, tile &x) { x.output_i = 4; },
        [](view &, tile &x) { x.a_offset_tiles = 1; },
        [](view &, tile &x) { x.a_offset = 65536; },
        [](view &, tile &x) { x.a_stride = std::numeric_limits<size_t>::max(); },
        [](view &, tile &x) { x.sequence = 1; },
        [](view &, tile &x) { x.total_partials = 3; },
        [](view &, tile &x) { x.scale_base = 253; },
        [](view &, tile &x) { x.logical_k_end += 16; },
        [](view &, tile &x) { x.accumulate = true; },
        [](view &, tile &x) { x.epochs[0] = {1, 1}; },
        [](view &v, tile &) { v.pending = true; },
        [](view &v, tile &) { v.quarantined = true; },
        [](view &v, tile &) { v.sticky = POTAL_ERR_BAD_RESPONSE; },
        [](view &v, tile &) { v.scales[0].original_block++; },
        [](view &v, tile &) { v.scales[0].original_j += 16; },
        [](view &v, tile &) { v.scales[0].resident_identity.generation++; },
        [](view &v, tile &) { v.scales[0].live_columns &= ~1u; },
        [](view &v, tile &) { v.scales[0].carriers[0] = 0xffffffffu; },
        [](view &v, tile &) { v.residents[0].readable = false; }
    };
    for (const auto &change : mutations) rejected(m, t, change);
    translation corrupt = valid;
    corrupt.partials[0].b_rows[0][0]++;
    require(m.accept(t, corrupt) != POTAL_OK && !m.state.pending, "tampered translated record rejected");
    // Rebinding the identical physical slots cannot resurrect a captured record.
    prepare(m, t, rows);
    tile stale = t;
    stale.scale_incarnation.generation--;
    require(m.accept(stale, valid) != POTAL_OK, "stale translated Scale incarnation rejected");
    // Wire generation may wrap only after Release; software identity never wraps.
    const token old_scale = t.scale_incarnation;
    ok(m.clear_scales(), "release before wire generation wrap");
    t.scale_generation = 0;
    t.scale_incarnation = m.fresh();
    auto wrapped_entries = m.state.scales;
    for (uint32_t p = 0; p < rows.size(); ++p) for (uint32_t j = 0; j < 2; ++j) {
        auto &s = wrapped_entries[p * 2 + j];
        s.incarnation = t.scale_incarnation;
        s.generation = 0;
        s.live_columns = 0xffff;
    }
    ok(m.program_scales(wrapped_entries, 0, uint32_t(rows.size()) * 2),
       "wire generation wraps only with a fresh software incarnation");
    require(!same(old_scale, t.scale_incarnation), "wire wrap retains independent software identity");
    checked(m, t);
    rejected(m, t, [old_scale](view &, tile &x) { x.scale_incarnation = old_scale; });
    const auto fresh_record = checked(m, t);
    require(m.retire(0) == busy, "retirement retains associated live Scale");
    require(m.load(geometry(0)) == busy, "replacement requires explicit retirement");
    require(m.cleanup() == busy, "healthy cleanup cannot discard an output");
    ok(m.clear_scales(), "release before residency replacement");
    ok(m.retire(0), "retire stale resident");
    require(m.program_index(0, {112}) != POTAL_OK, "retired B cannot publish an Index window");
    load_fixture(m, geometry(0));
    require(m.accept(t, fresh_record) != POTAL_OK, "replacement rejects old resident record");
    t = main_tile(m, 0, 112, 32, 48, 32, 3);
    t.indexed = true;
    rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    const auto indexed = checked(m, t);
    rejected(m, t, [](view &v, tile &) { v.indices[0].published = false; });
    rejected(m, t, [](view &v, tile &) { v.indices[0].count--; });
    rejected(m, t, [](view &v, tile &) { std::swap(v.indices[0].original_k[0], v.indices[0].original_k[1]); });
    rejected(m, t, [](view &v, tile &) { v.indices[0].original_k[0] = 448; });
    rejected(m, t, [](view &v, tile &) { v.indices[0].resident_identity.generation++; });
    rejected(m, t, [](view &, tile &x) { x.fragment_counts[0] = 15; });
    std::vector<uint64_t> identity;
    for (uint64_t k = 112; k < 144; ++k) identity.push_back(k);
    ok(m.program_index(0, identity), "replace Index with identical physical values");
    require(m.accept(t, indexed) != POTAL_OK, "Index software incarnation is not an address");
    t = main_tile(m, 0, 96, 19, 48, 32, 3);
    t.indexed = true;
    t.path = POTAL_PAIRED_RESIDUAL;
    t.acc_bank = 1;
    t.a_offset_tiles = 28;
    t.logical_k_end = 160;
    t.total_partials = 2;
    rows = {{96, 98, 126}, {128, 135, 143}};
    prepare(m, t, rows);
    checked(m, t);
    rejected(m, t, [](view &, tile &x) { x.a_offset_tiles = 27; });
    rejected(m, t, [](view &v, tile &) { v.indices[0].original_k[2] = 128; });
    rejected(m, t, [](view &v, tile &) { v.indices[0].original_k[1] = 96; });
    rejected(m, t, [](view &v, tile &) { v.indices[0].original_k[16] = 96; });
    m.activation[3] = 1;
    translation out;
    require(translate(m.state, t, out) != POTAL_OK, "nonzero interior Index filler rejected");
    m.activation[3] = 0;
    uint32_t carrier = 123;
    require(shift_carrier(-1, carrier) != POTAL_OK &&
            shift_carrier(INT64_MIN, carrier) != POTAL_OK, "negative signed carriers reject before narrowing");
    ok(shift_carrier(0, carrier), "shift zero is legal");
    require(carrier == 0, "shift zero is identity, not zero sentinel");
    ok(shift_carrier(32767, carrier), "maximum shift legal");
    require(shift_carrier(32768, carrier) != POTAL_OK, "shift overflow rejected");
    potal_paired_cmd_record wire{};
    require(encode_scale(0, 0, 0, 0xffffffffu, wire) != POTAL_OK &&
            encode_scale(0, 0, 0, 0x80000001u, wire) != POTAL_OK &&
            encode_scale(256, 0, 0, 0, wire) != POTAL_OK &&
            encode_scale(0, 256, 0, 0, wire) != POTAL_OK &&
            encode_scale(0, 0, 16, 0, wire) != POTAL_OK, "invalid Scale fields rejected");
    token next;
    require(next_token({1, UINT64_MAX}, next) != POTAL_OK, "software token wrap rejected");
    require(next_token({0, 1}, next) != POTAL_OK, "empty token domain rejected");
    resident r = geometry(0);
    for (int field = 0; field < 9; ++field) {
        resident bad = r;
        switch (field) {
            case 0: bad.geometry.b_sp_base++; break;
            case 1: bad.geometry.b_row_stride = 16; break;
            case 2: bad.geometry.k_rows = 449; break;
            case 3: bad.geometry.j_columns = 145; break;
            case 4: bad.geometry.source_stride = 65536; break;
            case 5: bad.geometry.k_begin = 1; break;
            case 6: bad.geometry.j_begin = 1; break;
            case 7: bad.geometry.weight_id = 0; break;
            case 8: bad.geometry.weight_generation = 0; break;
        }
        require(check_resident(bad) != POTAL_OK, "invalid full resident geometry rejected");
    }
}

void pending_conflicts(model &m, const tile &t, const translation &tr) {
    require(m.state.pending, "conflicts require outstanding model transaction");
    const auto bytes = m.acc;
    const uint32_t drains = m.drains;
    require(m.load(geometry(1)) != POTAL_OK && m.accept(t, tr) != POTAL_OK &&
            m.program_index(t.pair, {0}) != POTAL_OK && m.clear_scales() != POTAL_OK &&
            m.retire(t.pair) != POTAL_OK && m.readback(0, 1, 0) != POTAL_OK &&
            m.restore(0, {}) != POTAL_OK && m.relinquish(0, {}) != POTAL_OK &&
            m.close() != POTAL_OK, "all mutators and free refuse pending work");
    require(m.release(255, 17, 0) != POTAL_OK && m.drains == drains &&
            m.state.pending && m.acc == bytes, "invalid Release neither drains nor changes bytes");
}

void epochs_and_snapshots() {
    model m;
    load_fixture(m, geometry(0));
    tile t = main_tile(m, 0, 0, 16, 48, 16, 3);
    t.logical_k_end = 48;
    t.total_partials = 3;
    t.last = false;
    auto rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    auto tr = checked(m, t);
    ok(m.accept(t, tr), "reserve output writer");
    pending_conflicts(m, t, tr);
    require(!m.state.epochs[0][3].initialized, "output token unpublished until drain");
    ok(m.release(0, 255, 0), "valid Release drains compute");
    require(!m.state.pending && m.state.epochs[0][3].initialized, "valid Release publishes complete older work");
    const token old = m.state.epochs[0][3].identity;
    tile other = main_tile(m, 0, 16, 16, 64, 16, 4);
    auto other_rows = canonical(m.state.residents[0], other);
    prepare(m, other, other_rows);
    execute(m, other, other_rows);
    require(same(old, m.state.epochs[0][3].identity), "independent J subview epoch survives");
    prepare(m, t, rows);
    rejected(m, t, [](view &, tile &) {}); // First cannot overwrite an OPEN output.
    continue_tile(m, t, 16, 16, false);
    rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    rejected(m, t, [](view &v, tile &x) { x.epochs[0] = v.epochs[0][4].identity; });
    rejected(m, t, [](view &, tile &x) { x.output_id++; });
    rejected(m, t, [](view &, tile &x) { x.original_i++; });
    rejected(m, t, [](view &, tile &x) { x.k_offset = 32; });
    rejected(m, t, [](view &, tile &x) { x.sequence++; });
    rejected(m, t, [](view &, tile &x) { x.total_partials++; });
    rejected(m, t, [](view &, tile &x) { x.j_columns = 15; });
    rejected(m, t, [](view &, tile &x) { x.last = true; });
    rejected(m, t, [](view &v, tile &) { v.residents[0].geometry.weight_generation++; });
    rejected(m, t, [](view &v, tile &) { v.residents[0].geometry.weight_id++; });
    ok(m.readback(0, (1u << 3) | (1u << 4), 0), "snapshot drained OPEN and FINAL tiles");
    require(!m.saves[0].valid, "readback issue does not publish snapshot");
    pending_conflicts(m, t, tr);
    ok(m.drain(), "publish INT32 snapshot");
    const auto saved = m.saves[0].bytes;
    const token saved_token = m.saves[0].identity;
    ok(m.relinquish(0, saved_token), "snapshot permits explicit OPEN relinquish");
    require(m.readback(0, 1u << 4, 0) != POTAL_OK, "cannot overwrite sole OPEN snapshot");
    require(m.restore(0, {saved_token.domain, saved_token.generation + 1}) != POTAL_OK,
            "stale snapshot generation rejected");
    ok(m.restore(0, saved_token), "restore raw INT32 without SCU");
    require(!m.state.epochs[0][3].initialized, "restore token unpublished");
    pending_conflicts(m, t, tr);
    ok(m.drain(), "restore completion");
    require(!same(old, m.state.epochs[0][3].identity), "restore creates fresh physical epoch");
    for (uint32_t slot : {3u, 4u}) for (uint32_t x = 0; x < 256; ++x)
        require(m.acc[slot * 256 + x] == saved[slot * 256 + x], "all restored raw INT32 bits");
    rejected(m, t, [](view &, tile &) {}); // The pre-restore physical token is stale.
    continue_tile(m, t, 16, 16, false);
    prepare(m, t, rows);
    execute(m, t, rows);
    continue_tile(m, t, 32, 16, true);
    rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    execute(m, t, rows);
    require(m.state.epochs[0][3].complete, "resume finishes exact canonical sequence");
    for (uint32_t i = 0; i < 16; ++i) for (uint32_t j = 0; j < 16; ++j) {
        int32_t expected = 0;
        for (uint64_t first = 0; first < 48; first += 16) {
            wide dot = 0;
            for (uint64_t k = first; k < first + 16; ++k)
                dot += wide(varied_a(i, k)) * varied_b(k, 48 + j);
            expected = golden_clamp(wide(expected) + golden_scale(dot, varied_c(first / 32, 48 + j)));
        }
        require(m.acc[(48 + i) * 16 + j] == expected, "snapshot resume equals full independent three-partial golden");
    }
    ok(m.readback(0, 1u << 3, 0), "replace snapshot after restore");
    ok(m.drain(), "publish replacement snapshot");
    require(!same(saved_token, m.saves[0].identity) &&
            m.restore(0, saved_token) != POTAL_OK, "snapshot replacement invalidates old token");
    continue_tile(m, t, 32, 16, true);
    rejected(m, t, [](view &, tile &) {}); // FINAL cannot continue.
    t = main_tile(m, 0, 0, 16, 48, 16, 3);
    rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    execute(m, t, rows); // Explicit first replaces FINAL, never OPEN.
    require(!same(old, m.state.epochs[0][3].identity), "replacement cannot resurrect old epoch");
}

void residual_continuation() {
    model m;
    load_fixture(m, geometry(1));
    tile t = main_tile(m, 1, 96, 3, 48, 32, 3);
    t.path = POTAL_PAIRED_RESIDUAL;
    t.acc_bank = 1;
    t.indexed = true;
    t.a_offset_tiles = 28;
    t.logical_k_end = 224;
    t.total_partials = 3;
    t.last = false;
    fragments rows = {{96, 98, 126}};
    prepare(m, t, rows);
    execute(m, t, rows);
    continue_tile(m, t, 128, 3, false);
    rows = {{128, 130, 158}};
    prepare(m, t, rows);
    rejected(m, t, [](view &v, tile &x) {
        x.k_offset = 120;
        v.indices[1].original_k[0] = 120;
        v.indices[1].original_k[1] = 122;
        v.indices[1].original_k[2] = 124;
        for (uint32_t j = 0; j < 2; ++j) v.scales[j].original_block = 3;
    });
    execute(m, t, rows);
    // Compact fragments may skip original K, but never move backward.
    continue_tile(m, t, 192, 3, true);
    rows = {{192, 200, 222}};
    prepare(m, t, rows);
    execute(m, t, rows);
    require(m.state.epochs[1][3].complete && m.state.epochs[1][3].next_sequence == 3 &&
            m.state.epochs[1][3].next_original_k == 223,
            "Residual continuation preserves compact order and legal gaps");
}

void recoverable_failures() {
    for (int partial = 0; partial < 3; ++partial) {
        model m;
        load_fixture(m, geometry(0));
        tile t = main_tile(m, 0, 0, 48, 0, 16);
        auto rows = canonical(m.state.residents[0], t);
        prepare(m, t, rows);
        auto tr = checked(m, t);
        require(m.accept(t, tr, partial) == POTAL_ERR_BAD_RESPONSE && m.state.pending &&
                m.effects == uint32_t(partial + 1), "first middle last partial failure retains pending");
        pending_conflicts(m, t, tr);
        const auto first = m.state.sticky;
        const auto first_stage = m.first_failure;
        if (partial == 1) ok(m.drain(), "failed-state Fence routes through one cleanup");
        else ok(m.cleanup(), "one explicit cleanup drain");
        require(m.cleanup_drains == 1 && !m.state.pending && !m.state.epochs[0][0].initialized &&
                m.state.sticky == first && m.first_failure == first_stage,
                "cleanup discards failed outputs and retains first status and stage");
        ok(m.cleanup(), "cleanup idempotently returns saved outcome");
        require(m.cleanup_drains == 1 && m.close() == first, "safe failed close returns sticky status");
    }
    for (bool restore_failure : {false, true}) for (int step = 1; step <= 3; ++step) {
        model m;
        load_fixture(m, geometry(0));
        tile t = main_tile(m, 0, 0, 16, 0, 48);
        auto rows = canonical(m.state.residents[0], t);
        prepare(m, t, rows);
        execute(m, t, rows);
        if (restore_failure) {
            ok(m.readback(0, 7, 0), "save before partial restore");
            ok(m.drain(), "publish save");
            const token id = m.saves[0].identity;
            ok(m.relinquish(0, id), "relinquish restore target");
            require(m.restore(0, id, step) != POTAL_OK && m.saves[0].valid,
                    "partial restore retains complete immutable source");
        } else {
            require(m.readback(0, 7, 0, step) != POTAL_OK && !m.saves[0].valid,
                    "partial readback cannot publish");
        }
        ok(m.cleanup(), "partial snapshot operation cleanup");
        require(!m.state.pending && m.cleanup_drains == 1, "snapshot cleanup drains once");
    }
    model load;
    require(load.load(geometry(0), 1) != POTAL_OK && load.state.pending &&
            !load.state.residents[0].readable, "partial load cannot publish");
    ok(load.cleanup(), "partial load cleanup");
    for (int step : {0, 8, 16}) {
        model index;
        load_fixture(index, geometry(0));
        std::vector<uint64_t> rows;
        for (uint64_t k = 0; k < 16; ++k) rows.push_back(k);
        require(index.program_index(0, rows, step) != POTAL_OK &&
                !index.state.indices[0].published && index.state.pending,
                "partial Index transaction is never published");
        ok(index.cleanup(), "partial Index cleanup");
        require(index.cleanup_drains == 1 && index.close() == POTAL_ERR_BAD_RESPONSE,
                "partial Index requires one drain before safe close");
    }
}

void scale_capacity_and_generation() {
    model m;
    load_fixture(m, geometry(0));
    load_fixture(m, geometry(1, 448, 144, 448));
    std::array<scale_entry, 256> entries{};
    const token first = m.fresh();
    for (uint32_t s = 0; s < 144; ++s) {
        entries[s].incarnation = first;
        entries[s].resident_identity = m.state.residents[0].identity;
        entries[s].original_block = (s / 9) % 14;
        entries[s].original_j = (s % 9) * 16;
        entries[s].generation = 255;
        entries[s].live_columns = 0xffff;
    }
    ok(m.program_scales(entries, 0, 126), "current P2 Main126");
    ok(m.program_scales(entries, 126, 18), "current Residual18");
    const auto before = counts.scale_records;
    require(m.program_scales(entries, 144, 126) == POTAL_ERR_CAPACITY &&
            counts.scale_records == before && m.state.scales[143].live_columns == 0xffff &&
            !m.state.scales[144].live_columns,
            "current126 plus Residual18 plus next126 cannot allocate270");
    require(m.program_scales(entries, 0, 1) == POTAL_ERR_CAPACITY,
            "live slot cannot be overwritten at the same generation");
    require(m.close() == busy && m.retire(0) == busy, "live carrier bindings retain residency");
    ok(m.clear_scales(), "drained Scale release");
    require(m.program_scales(entries, 0, 1) == invalid,
            "released slot cannot resurrect its old software incarnation");
    ok(m.retire(0), "retire first pair after release");
    ok(m.retire(1), "B-only next pair never required Scale setup");
    ok(m.close(), "healthy fully retired close");

    model wrapped;
    load_fixture(wrapped, geometry(0));
    tile t = main_tile(wrapped, 0, 0, 16, 0, 16);
    const auto rows = canonical(wrapped.state.residents[0], t);
    prepare(wrapped, t, rows);
    const auto tr = checked(wrapped, t);
    wrapped.clock.generation = UINT64_MAX;
    const auto acc = wrapped.acc;
    require(wrapped.accept(t, tr) == POTAL_ERR_CAPACITY && !wrapped.state.pending &&
            wrapped.acc == acc, "epoch generation wrap rejects before compute effects");
    require(wrapped.program_index(0, {0}) == POTAL_ERR_CAPACITY &&
            !wrapped.state.indices[0].published, "Index generation wrap rejects before CONFIG");
}

void validation_cost() {
    model m;
    load_fixture(m, geometry(0));
    tile t = main_tile(m, 0, 112, 48, 48, 32, 3);
    const auto rows = canonical(m.state.residents[0], t);
    prepare(m, t, rows);
    translation record;
    constexpr unsigned repetitions = 512;
    uint64_t checksum = 0;
    for (unsigned sample = 0; sample < 7; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned n = 0; n < repetitions; ++n) {
            ok(translate(m.state, t, record), "timed pure validation");
            checksum += record.partials[n % 3].b_rows[0][0] + record.next_sequence;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("E4B1_CPU_VALIDATION sample=%u calls=%u cpu_ms=%.6f "
                    "scope=translate_output_and_checksum software_only=1\n",
                    sample, repetitions, ms);
    }
    std::printf("E4B1_CPU_CHECKSUM value=%llu\n", static_cast<unsigned long long>(checksum));
}

// Unknown-completion cases intentionally leak their model until child exit.
// No destructor, close, free, or second cleanup may run on quarantined state.
void terminal_case(int kind) {
    auto *m = new model;
    load_fixture(*m, geometry(0));
    tile t = main_tile(*m, 0, 0, 48, 0, 16);
    auto rows = canonical(m->state.residents[0], t);
    prepare(*m, t, rows);
    const auto tr = checked(*m, t);
    if (kind == 4) {
        require(m->accept(t, tr, 1) == POTAL_ERR_BAD_RESPONSE, "issue failure before cleanup Release");
        const auto stage = m->first_failure;
        require(m->cleanup(true, false) != POTAL_OK && m->last_drain == POTAL_OK &&
                m->first_failure == stage && m->state.scales[0].live_columns,
                "uncertain cleanup Release retains owner after known DMA drain");
    } else if (kind == 3) {
        auto entries = m->state.scales;
        ok(m->clear_scales(), "release before uncertain control response");
        const auto incarnation = m->fresh();
        for (uint32_t i = 0; i < 3; ++i) entries[i].incarnation = incarnation;
        require(m->program_scales(entries, 0, 3, 1) != POTAL_OK,
                "uncertain Scale acceptance quarantines");
    } else if (kind == 1) {
        require(m->accept(t, tr, 1) == POTAL_ERR_BAD_RESPONSE, "earlier issue error");
        require(m->cleanup(false) != POTAL_OK, "unknown cleanup completion");
        require(m->state.sticky == POTAL_ERR_BAD_RESPONSE, "unknown drain cannot replace first error");
    } else {
        ok(m->accept(t, tr), "issue before unknown drain");
        if (kind == 2)
            require(m->release(0, 255, 0, false) != POTAL_OK, "unknown Release completion");
        else
            require(m->drain(false) != POTAL_OK, "unknown direct completion");
    }
    const auto *retained_a = m->activation.data();
    const auto *retained_b = m->source[0].data();
    const uint32_t drains = m->drains;
    require(m->state.quarantined && m->unknown_completion && m->state.pending == (kind != 4),
            "independent unknown bit retains pending ownership");
    require(m->cleanup() != POTAL_OK && m->drain() != POTAL_OK &&
            m->close() != POTAL_OK && m->accept(t, tr) != POTAL_OK &&
            m->load(geometry(1)) != POTAL_OK && m->readback(0, 1, 0) != POTAL_OK &&
            m->restore(0, {}) != POTAL_OK && m->release(0, 255, 0) != POTAL_OK,
            "quarantine has no issue drain free path");
    require(m->drains == drains && m->activation.data() == retained_a &&
            m->source[0].data() == retained_b && m->acc.size() == 16384,
            "quarantine retains all owned images without retry");
}
void isolated_terminal_cases() {
    std::fflush(nullptr);
    for (int kind = 0; kind < 5; ++kind) {
        const pid_t child = fork();
        require(child >= 0, "fork terminal model case");
        if (child == 0) {
            try {
                terminal_case(kind);
                _exit(0);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "terminal software-only case: %s\n", e.what());
                _exit(1);
            }
        }
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
        require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "isolated terminal conformance case");
    }
}
} // namespace

int main(int argc, char **argv) {
    for (int x = 1; x < argc; ++x) {
        if (std::strcmp(argv[x], "--hardware") == 0) {
            std::fputs("E4B1 is software-only; --hardware is rejected without fallback.\n", stderr);
            return 2;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[x]);
        return 2;
    }
    try {
        std::puts("E4B1 software-only conformance; logical counts, no hardware timing evidence.");
        std::puts("Logical counts below exclude the isolated terminal child processes.");
        numerical_geometry();
        numerical_witnesses();
        descriptor_negatives();
        epochs_and_snapshots();
        residual_continuation();
        recoverable_failures();
        scale_capacity_and_generation();
        isolated_terminal_cases();
        validation_cost();
        std::printf("software-only PASS assertions=%llu microtiles=%llu partial_J_tiles=%llu "
                    "Scale_records=%llu Index_records=%llu snapshot_bytes=%llu restore_bytes=%llu\n",
                    static_cast<unsigned long long>(counts.assertions),
                    static_cast<unsigned long long>(counts.microtiles),
                    static_cast<unsigned long long>(counts.partials),
                    static_cast<unsigned long long>(counts.scale_records),
                    static_cast<unsigned long long>(counts.index_records),
                    static_cast<unsigned long long>(counts.snapshot_bytes),
                    static_cast<unsigned long long>(counts.restore_bytes));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "E4B1 software-only FAIL: %s\n", e.what());
        return 1;
    }
}
