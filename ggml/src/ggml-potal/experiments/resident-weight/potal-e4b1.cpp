// Pure E4B1 validation and semantic translation. No issuing/session calls.
#include "potal-e4b1.hpp"
#include <algorithm>
#include <limits>

namespace potal_e4b1 {
namespace {
constexpr auto invalid = POTAL_ERR_INVALID_ARGUMENT;
uint32_t tiles(uint32_t n) { return n / dim + (n % dim != 0); }
bool empty(token t) { return t.domain == 0 && t.generation == 0; }
}

bool same(token a, token b) {
    return a.domain && a.generation && a.domain == b.domain && a.generation == b.generation;
}
potal_status next_token(token current, token &next) {
    if (!current.domain) return invalid;
    if (current.generation == UINT64_MAX) return POTAL_ERR_CAPACITY;
    next = {current.domain, current.generation + 1};
    return POTAL_OK;
}
potal_status shift_carrier(int64_t shift, uint32_t &carrier) {
    if (shift < 0 || shift > POTAL_CARRIER_MAX_SHIFT) return invalid;
    carrier = uint32_t(shift);
    return POTAL_OK;
}

potal_status check_resident(const resident &r) {
    const auto &g = r.geometry;
    if (!same(r.identity, r.identity) || g.pair >= 2 || g.bank != 2 * g.pair + 1 ||
        !g.k_rows || g.k_rows > max_k * dim || !g.j_columns ||
        g.j_columns > max_j * dim || !g.weight_id || !g.weight_generation ||
        g.k_begin % dim || g.j_begin % dim ||
        g.k_begin > UINT64_MAX - g.k_rows || g.j_begin > UINT64_MAX - g.j_columns ||
        g.source_stride < g.j_columns || g.main_a_tiles >= 256)
        return invalid;
    const uint64_t rows = uint64_t(tiles(g.k_rows)) * tiles(g.j_columns) * dim;
    if (rows > 4096) return POTAL_ERR_CAPACITY;
    if (g.b_sp_base != (g.pair + 1) * 8192 - rows ||
        g.b_row_stride != tiles(g.j_columns) * dim) return invalid;
    if (g.k_rows > 1 && g.source_stride > (65536 - g.j_columns) / (g.k_rows - 1))
        return invalid;
    return POTAL_OK;
}

potal_status encode_scale(uint32_t slot, uint32_t generation, uint32_t column,
                          uint32_t carrier, potal_paired_cmd_record &out) {
    uint64_t key;
    const auto status = potal_scale_key_encode(slot, generation, column, &key);
    if (status != POTAL_OK) return status;
    if (!potal_carrier_is_valid(carrier)) return invalid;
    out = {POTAL_FUNCT_SCALE_WRITE, 1, key, carrier};
    return POTAL_OK;
}

potal_status encode_index(const resident &r, const index_window &w,
                          std::array<potal_paired_cmd_record, 33> &out,
                          uint32_t &count) {
    const auto checked = check_resident(r);
    if (checked != POTAL_OK) return checked;
    if (!same(w.resident_identity, r.identity) || !same(w.incarnation, w.incarnation) ||
        !w.count) return invalid;
    if (w.count > 32) return POTAL_ERR_CAPACITY;
    const auto &g = r.geometry;
    const potal_index_config config{g.pair, g.b_sp_base, tiles(g.j_columns), g.k_rows, w.count};
    std::array<potal_paired_cmd_record, 33> result{};
    potal_rocc_cmd raw{};
    auto status = potal_index_config_encode(&config, &raw);
    if (status != POTAL_OK) return status;
    result[0] = {POTAL_FUNCT_INDEX_CONFIG, 1, raw.rs1, raw.rs2};
    for (uint32_t i = 0; i < w.count; ++i) {
        if (w.original_k[i] < g.k_begin || w.original_k[i] - g.k_begin >= g.k_rows)
            return invalid;
        const potal_index_write entry{i, uint32_t(w.original_k[i] - g.k_begin)};
        status = potal_index_write_encode(&config, &entry, &raw);
        if (status != POTAL_OK) return status;
        result[i + 1] = {POTAL_FUNCT_INDEX_WRITE, 1, raw.rs1, raw.rs2};
    }
    out = result;
    count = w.count + 1;
    return POTAL_OK;
}

potal_status translate(const view &s, const tile &d, translation &out) {
    if (s.sticky != POTAL_OK) return s.sticky;
    if (s.quarantined) return POTAL_ERR_BAD_RESPONSE;
    if (s.pending) return POTAL_ERR_BUSY;
    if (d.pair >= 2 || d.path > POTAL_PAIRED_RESIDUAL || d.acc_bank != d.path ||
        !d.output_id || !d.i_rows || d.i_rows > dim ||
        !d.k_rows || d.k_rows > max_k * dim ||
        !d.j_columns || d.j_columns > max_j * dim ||
        d.j_offset % dim || d.acc_base % dim || !d.output_j_stride ||
        d.output_j_stride > 32 || d.output_j >= d.output_j_stride ||
        d.scale_generation > 255 || d.first == d.accumulate ||
        !d.total_partials || d.logical_k_begin >= d.logical_k_end ||
        (d.path == POTAL_PAIRED_RESIDUAL && !d.indexed))
        return invalid;
    const auto &r = s.residents[d.pair];
    auto checked = check_resident(r);
    if (checked != POTAL_OK) return checked;
    if (r.geometry.pair != d.pair || !r.readable ||
        !same(r.identity, d.resident_identity)) return invalid;
    const auto &g = r.geometry;
    if (d.j_columns > g.j_columns || d.j_offset > g.j_columns - d.j_columns)
        return invalid;
    const uint32_t np = tiles(d.k_rows), nj = tiles(d.j_columns);
    if (nj > d.output_j_stride - d.output_j ||
        uint64_t(d.acc_base) + (uint64_t(d.output_i) * d.output_j_stride +
            d.output_j + nj) * dim > 512 ||
        uint64_t(d.a_offset_tiles) + np > 256)
        return POTAL_ERR_CAPACITY;
    if ((!d.path && (d.a_offset_tiles || np > g.main_a_tiles)) ||
        (d.path && d.a_offset_tiles < g.main_a_tiles)) return invalid;
    if (d.scale_base >= 256 || uint64_t(np) * nj > 256 - d.scale_base)
        return POTAL_ERR_CAPACITY;
    if (!same(d.scale_incarnation, d.scale_incarnation) ||
        d.sequence > UINT64_MAX - np || d.sequence + np > d.total_partials ||
        d.last != (d.sequence + np == d.total_partials) ||
        (d.first && d.sequence != 0)) return invalid;
    if (!s.activation || d.a_stride < d.k_rows || d.a_offset > s.activation_bytes ||
        d.k_rows > s.activation_bytes - d.a_offset ||
        (d.i_rows > 1 && d.a_stride >
            (s.activation_bytes - d.a_offset - d.k_rows) / (d.i_rows - 1)) ||
        uintptr_t(s.activation) > UINTPTR_MAX - s.activation_bytes ||
        (uintptr_t(s.activation) + d.a_offset) % dim) return invalid;

    if (!d.indexed) {
        if (!empty(d.index_incarnation) || d.k_offset % dim ||
            d.k_rows > g.k_rows || d.k_offset > g.k_rows - d.k_rows ||
            (d.k_rows % dim && d.k_offset + d.k_rows != g.k_rows))
            return invalid;
    } else {
        const auto &ix = s.indices[d.pair];
        if (d.k_rows > 32 || !ix.published || ix.count != d.k_rows ||
            !same(ix.incarnation, d.index_incarnation) ||
            !same(ix.resident_identity, r.identity) ||
            ix.original_k[0] < g.k_begin ||
            ix.original_k[0] - g.k_begin != d.k_offset) return invalid;
        for (uint32_t p = 0; p < np; ++p) {
            const auto n = d.fragment_counts[p];
            if (!n || n > dim || (p + 1 == np && n != d.k_rows - p * dim))
                return invalid;
        }
    }
    if (!d.path) {
        const uint64_t length = d.logical_k_end - d.logical_k_begin;
        if (d.logical_k_begin % dim ||
            d.total_partials != length / dim + (length % dim != 0) ||
            d.sequence > (UINT64_MAX - d.logical_k_begin) / dim ||
            g.k_begin + d.k_offset != d.logical_k_begin + d.sequence * dim)
            return invalid;
    }

    translation result{};
    result.partial_count = np;
    result.j_tiles = nj;
    result.b_row_stride = g.b_row_stride;
    result.indexed_j_origin = d.j_offset / dim;
    result.b_effective_base = g.b_sp_base + d.j_offset / dim * dim +
        (d.indexed ? 0 : d.k_offset / dim * g.b_row_stride);
    result.acc_effective_base = d.acc_bank * 512 + d.acc_base +
        (d.output_i * d.output_j_stride + d.output_j) * dim;
    result.a_sp_base = d.pair * 8192 + d.a_offset_tiles * dim;
    result.next_sequence = d.sequence + np;
    uint64_t previous_original = 0;
    bool have_previous = false;
    for (uint32_t p = 0; p < np; ++p) {
        auto &part = result.partials[p];
        part.valid_k = d.indexed ? d.fragment_counts[p] : std::min(dim, d.k_rows - p * dim);
        for (uint32_t q = 0; q < part.valid_k; ++q) {
            const uint64_t original = d.indexed ? s.indices[d.pair].original_k[p * dim + q] :
                g.k_begin + d.k_offset + p * dim + q;
            if (original < g.k_begin || original - g.k_begin >= g.k_rows ||
                original < d.logical_k_begin || original >= d.logical_k_end ||
                (have_previous && original <= previous_original)) return invalid;
            if (!q) {
                part.original_block = original / 32;
                part.phase = uint32_t(original / dim % 2);
                if (!d.path && original != d.logical_k_begin + (d.sequence + p) * dim)
                    return invalid;
            }
            if (original / 32 != part.original_block ||
                (!d.path && q != 0 && original != part.original_k[0] + q)) return invalid;
            part.original_k[q] = original;
            previous_original = original;
            have_previous = true;
            const uint32_t local = uint32_t(original - g.k_begin);
            for (uint32_t j = 0; j < nj; ++j)
                part.b_rows[j][q] = g.b_sp_base + (local / dim) * g.b_row_stride +
                    (d.j_offset / dim + j) * dim + local % dim;
        }
        if (!d.path && part.valid_k != dim &&
            (p + 1 != np || previous_original + 1 != g.k_begin + g.k_rows))
            return invalid;
        if (d.indexed && p + 1 < np) {
            for (uint32_t q = part.valid_k; q < dim; ++q) {
                const uint64_t filler = s.indices[d.pair].original_k[p * dim + q];
                if (filler < g.k_begin || filler - g.k_begin >= g.k_rows) return invalid;
                for (uint32_t row = 0; row < d.i_rows; ++row)
                    if (s.activation[d.a_offset + row * d.a_stride + p * dim + q] != 0)
                        return invalid;
            }
        }
        for (uint32_t j = 0; j < nj; ++j) {
            const uint32_t slot = d.scale_base + p * nj + j;
            part.scale_slots[j] = slot;
            const uint32_t columns = std::min(dim, d.j_columns - j * dim);
            const uint32_t mask = (uint32_t(1) << columns) - 1;
            const auto &sc = s.scales[slot];
            if (!same(sc.incarnation, d.scale_incarnation) ||
                !same(sc.resident_identity, r.identity) ||
                sc.original_block != part.original_block ||
                sc.original_j != g.j_begin + d.j_offset + j * dim ||
                sc.generation != d.scale_generation ||
                (sc.live_columns & mask) != mask || (sc.live_columns >> dim)) return invalid;
            for (uint32_t c = 0; c < columns; ++c)
                if (!potal_carrier_is_valid(sc.carriers[c])) return invalid;
        }
    }
    result.next_original_k = previous_original + 1;
    if (!d.path && d.last && result.next_original_k != d.logical_k_end) return invalid;
    for (uint32_t j = 0; j < nj; ++j) {
        result.acc_rows[j] = result.acc_effective_base + j * dim;
        result.valid_columns[j] = std::min(dim, d.j_columns - j * dim);
        const uint32_t slot = (result.acc_rows[j] - d.acc_bank * 512) / dim;
        const auto &ep = s.epochs[d.acc_bank][slot];
        if (d.first) {
            if (ep.initialized && !ep.complete) return POTAL_ERR_BUSY;
            if (!empty(d.epochs[j])) return invalid;
        } else {
            if (!ep.initialized || ep.complete || !same(ep.identity, d.epochs[j]) ||
                ep.output_id != d.output_id || ep.weight_id != g.weight_id ||
                ep.weight_generation != g.weight_generation ||
                ep.original_i != d.original_i ||
                ep.original_j != g.j_begin + d.j_offset + j * dim ||
                ep.rows != d.i_rows || ep.columns != result.valid_columns[j] ||
                ep.logical_k_begin != d.logical_k_begin || ep.logical_k_end != d.logical_k_end ||
                ep.total_partials != d.total_partials || ep.next_sequence != d.sequence ||
                (!d.path && ep.next_original_k != result.partials[0].original_k[0]) ||
                (d.path && ep.next_original_k > result.partials[0].original_k[0]))
                return invalid;
        }
    }
    out = result;
    return POTAL_OK;
}
} // namespace potal_e4b1
