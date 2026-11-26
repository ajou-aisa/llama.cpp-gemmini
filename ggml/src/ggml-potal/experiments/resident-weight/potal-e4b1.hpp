// Test-only proposed resident interface. These records have no hardware issuer.
#ifndef POTAL_E4B1_HPP
#define POTAL_E4B1_HPP

#include "potal-resident.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace potal_e4b1 {
constexpr uint32_t dim = 16, max_j = 9, max_k = 28;

// Deliberately not convertible to potal_resident_token.
struct token {
    uint64_t domain = 0, generation = 0;
};
bool same(token a, token b);
potal_status next_token(token current, token &next);
potal_status shift_carrier(int64_t shift, uint32_t &carrier);

struct resident {
    potal_resident_load geometry{};
    token identity{};
    bool readable = false;
};
struct scale_entry {
    token incarnation{}, resident_identity{};
    uint64_t original_block = 0, original_j = 0;
    uint32_t generation = 0, live_columns = 0;
    std::array<uint32_t, dim> carriers{};
};
struct index_window {
    token incarnation{}, resident_identity{};
    uint32_t count = 0;
    std::array<uint64_t, 32> original_k{};
    bool published = false;
};
struct epoch {
    token identity{};
    uint64_t output_id = 0, weight_id = 0, weight_generation = 0;
    uint64_t original_i = 0, original_j = 0;
    uint64_t logical_k_begin = 0, logical_k_end = 0;
    uint64_t next_sequence = 0, next_original_k = 0, total_partials = 0;
    uint32_t rows = 0, columns = 0;
    bool initialized = false, complete = false;
};

// Immutable borrowed view during one pure validation call. The software model
// owns every allocation and revalidates before acceptance. No real DMA lease.
struct view {
    std::array<resident, 2> residents{};
    std::array<scale_entry, 256> scales{};
    std::array<index_window, 2> indices{};
    std::array<std::array<epoch, 32>, 2> epochs{};
    const int8_t *activation = nullptr;
    std::size_t activation_bytes = 0;
    bool pending = false, quarantined = false;
    potal_status sticky = POTAL_OK;
};

struct tile {
    token resident_identity{}, scale_incarnation{}, index_incarnation{};
    std::array<token, max_j> epochs{};
    uint32_t pair = 0, path = POTAL_PAIRED_MAIN;
    bool indexed = false, first = true, accumulate = false, last = true;
    uint32_t k_offset = 0, j_offset = 0; // Elements relative to full residency.
    uint32_t k_rows = 0, j_columns = 0, i_rows = dim;
    std::size_t a_offset = 0, a_stride = 0;
    uint32_t a_offset_tiles = 0;
    uint32_t scale_base = 0, scale_generation = 0;
    uint32_t acc_bank = 0, acc_base = 0; // Bank-local SP-style row units.
    uint32_t output_i = 0, output_j = 0, output_j_stride = max_j;
    uint64_t output_id = 0, original_i = 0;
    uint64_t logical_k_begin = 0, logical_k_end = 0;
    uint64_t sequence = 0, total_partials = 0;
    // Indexed fragments occupy DIM-wide slots. Interior filler A must be zero.
    std::array<uint32_t, 2> fragment_counts{};
};

// Proposed metadata, NOT an opcode or a potal_paired_cmd_record. P1 Scale
// windows rebind each canonical partial to its original block/J carriers.
struct partial {
    uint32_t valid_k = 0, phase = 0;
    uint64_t original_block = 0;
    std::array<uint64_t, dim> original_k{};
    std::array<std::array<uint32_t, dim>, max_j> b_rows{};
    std::array<uint32_t, max_j> scale_slots{};
};
struct translation {
    bool requires_rtl = true;
    uint32_t partial_count = 0, j_tiles = 0, passes_per_block = 1;
    uint32_t b_effective_base = 0, b_row_stride = 0;
    uint32_t indexed_j_origin = 0, acc_effective_base = 0, a_sp_base = 0;
    uint64_t next_sequence = 0, next_original_k = 0;
    std::array<uint32_t, max_j> acc_rows{}, valid_columns{};
    std::array<partial, max_k> partials{};
};

potal_status check_resident(const resident &r);
potal_status translate(const view &state, const tile &request, translation &out);
potal_status encode_scale(uint32_t slot, uint32_t generation, uint32_t column,
                          uint32_t carrier, potal_paired_cmd_record &out);
potal_status encode_index(const resident &r, const index_window &window,
                          std::array<potal_paired_cmd_record, 33> &out,
                          uint32_t &count);
} // namespace potal_e4b1
#endif
