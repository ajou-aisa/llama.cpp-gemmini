#include "dec_h1_jpack.hpp"

#include <chrono>
#include <limits>
#include <new>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ggml::gemmini::quants::dec
{
namespace
{
bool multiply_size(size_t lhs, size_t rhs, size_t &result) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

bool scalar_entries_valid(
    const DecH1JPack &pack, const DecH1JPackEntry *entries, size_t entry_count) noexcept
{
    if (!pack.valid() || !entries || entry_count == 0)
        return false;
    for (size_t index = 0; index < entry_count; ++index)
        if (entries[index].k >= pack.depth() || entries[index].residual == 0)
            return false;
    return true;
}

bool source_entries_valid(
    const block_q8_h1 *source, size_t columns, size_t depth,
    const DecH1JPackEntry *entries, size_t entry_count) noexcept
{
    if (!source || columns == 0 || depth == 0 || !entries || entry_count == 0)
        return false;
    for (size_t index = 0; index < entry_count; ++index)
        if (entries[index].k >= depth || entries[index].residual == 0)
            return false;
    return true;
}

uint64_t residual_magnitude(int32_t residual) noexcept
{
    return residual < 0 ? static_cast<uint64_t>(-
        static_cast<int64_t>(residual)) : static_cast<uint64_t>(residual);
}
}

DecH1JPack::DecH1JPack(const block_q8_h1 *source, size_t columns, size_t depth)
{
    const auto start = std::chrono::steady_clock::now();
    if (!source || columns == 0 || depth == 0)
        return;

    const size_t panels = columns / kDecH1JPackNr + (columns % kDecH1JPackNr != 0);
    const size_t groups = depth / kDecH1JPackGroupSize +
        (depth % kDecH1JPackGroupSize != 0);
    size_t source_blocks = 0;
    size_t panel_groups = 0;
    size_t packed_bytes = 0;
    if (!multiply_size(columns, groups, source_blocks) ||
        !multiply_size(groups, panels, panel_groups) ||
        !multiply_size(panel_groups, kDecH1JPackGroupSize * kDecH1JPackNr, packed_bytes))
        return;

    data_ = static_cast<int8_t *>(::operator new(packed_bytes, std::align_val_t {64}));
    columns_ = columns;
    depth_ = depth;
    accounting_.packed_bytes = packed_bytes;
    accounting_.allocated_bytes = packed_bytes;
    accounting_.panel_count = panels;
    accounting_.group_count = groups;
    accounting_.tail_panel_count = columns % kDecH1JPackNr != 0 ? 1 : 0;

    for (size_t k_group = 0; k_group < groups; ++k_group)
        for (size_t j_panel = 0; j_panel < panels; ++j_panel)
            for (size_t k_offset = 0; k_offset < kDecH1JPackGroupSize; ++k_offset)
                for (size_t j_lane = 0; j_lane < kDecH1JPackNr; ++j_lane)
                {
                    const size_t j = j_panel * kDecH1JPackNr + j_lane;
                    const size_t k = k_group * kDecH1JPackGroupSize + k_offset;
                    int8_t code = 0;
                    if (j < columns && k < depth)
                    {
                        code = source[j * groups + k_group].qs[k_offset];
                        ++accounting_.source_code_bytes_read;
                    }
                    data_[offset(k_group, j_panel, k_offset, j_lane)] = code;
                    ++accounting_.packed_bytes_written;
                }
    const auto end = std::chrono::steady_clock::now();
    accounting_.pack_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

DecH1JPack::~DecH1JPack()
{
    reset();
}

DecH1JPack::DecH1JPack(DecH1JPack &&other) noexcept :
    data_(other.data_),
    columns_(other.columns_),
    depth_(other.depth_),
    accounting_(other.accounting_)
{
    other.data_ = nullptr;
    other.columns_ = 0;
    other.depth_ = 0;
    other.accounting_ = {};
}

DecH1JPack & DecH1JPack::operator=(DecH1JPack &&other) noexcept
{
    if (this == &other)
        return *this;
    reset();
    data_ = other.data_;
    columns_ = other.columns_;
    depth_ = other.depth_;
    accounting_ = other.accounting_;
    other.data_ = nullptr;
    other.columns_ = 0;
    other.depth_ = 0;
    other.accounting_ = {};
    return *this;
}

bool DecH1JPack::valid() const noexcept
{
    return data_ != nullptr;
}

size_t DecH1JPack::columns() const noexcept
{
    return columns_;
}

size_t DecH1JPack::depth() const noexcept
{
    return depth_;
}

size_t DecH1JPack::panel_count() const noexcept
{
    return accounting_.panel_count;
}

size_t DecH1JPack::group_count() const noexcept
{
    return accounting_.group_count;
}

const DecH1JPackAccounting & DecH1JPack::accounting() const noexcept
{
    return accounting_;
}

const int8_t *DecH1JPack::code_pointer(size_t k, size_t panel) const noexcept
{
    return data_ + offset(k / kDecH1JPackGroupSize, panel,
                          k % kDecH1JPackGroupSize, 0);
}

int8_t DecH1JPack::packed_code(
    size_t k_group, size_t j_panel, size_t k_offset, size_t j_lane) const noexcept
{
    return data_[offset(k_group, j_panel, k_offset, j_lane)];
}

void DecH1JPack::reset() noexcept
{
    if (data_)
        ::operator delete(data_, std::align_val_t {64});
    data_ = nullptr;
}

size_t DecH1JPack::offset(
    size_t k_group, size_t j_panel, size_t k_offset, size_t j_lane) const noexcept
{
    return (((k_group * accounting_.panel_count + j_panel) * kDecH1JPackGroupSize +
             k_offset) * kDecH1JPackNr + j_lane);
}

DecH1JPackSafety dec_h1_jpack_classify(
    const DecH1JPack &pack, const DecH1JPackEntry *entries, size_t entry_count) noexcept
{
    if (!scalar_entries_valid(pack, entries, entry_count) || entry_count > 4)
        return DecH1JPackSafety::INVALID;

    uint64_t magnitude_sum = 0;
    for (size_t index = 0; index < entry_count; ++index)
    {
        const uint64_t magnitude = residual_magnitude(entries[index].residual);
        if (magnitude > std::numeric_limits<uint64_t>::max() - magnitude_sum)
            return DecH1JPackSafety::INVALID;
        magnitude_sum += magnitude;
    }
    return magnitude_sum <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / 128 ?
        DecH1JPackSafety::INT32_SAFE : DecH1JPackSafety::INT64_REQUIRED;
}

bool dec_h1_jpack_execute_source_scalar(
    const block_q8_h1 *source, size_t columns, size_t depth,
    const DecH1JPackEntry *entries, size_t entry_count, int64_t *output) noexcept
{
    if (!output || !source_entries_valid(source, columns, depth, entries, entry_count))
        return false;
    const size_t groups = depth / kDecH1JPackGroupSize +
        (depth % kDecH1JPackGroupSize != 0);
    for (size_t j = 0; j < columns; ++j)
    {
        int64_t accumulator = 0;
        for (size_t entry = 0; entry < entry_count; ++entry)
        {
            const size_t k = entries[entry].k;
            accumulator += static_cast<int64_t>(entries[entry].residual) *
                source[j * groups + k / kDecH1JPackGroupSize].qs[k % kDecH1JPackGroupSize];
        }
        output[j] = accumulator;
    }
    return true;
}

bool dec_h1_jpack_execute_packed(
    const DecH1JPack &pack, const DecH1JPackEntry *entries, size_t entry_count,
    int64_t *output, DecH1JPackCounters &counters) noexcept
{
    if (!output || !scalar_entries_valid(pack, entries, entry_count))
        return false;

    const DecH1JPackSafety safety = dec_h1_jpack_classify(pack, entries, entry_count);
    const bool use_vector = safety == DecH1JPackSafety::INT32_SAFE;
    const size_t full_panels = pack.columns() / kDecH1JPackNr;
    const bool use_small_entry_bases = entry_count <= 4;
    const int8_t *entry_bases[4] = {};
    if (use_small_entry_bases)
        for (size_t entry = 0; entry < entry_count; ++entry)
            entry_bases[entry] = pack.code_pointer(entries[entry].k, 0);
    if (use_vector && full_panels != 0)
    {
        int32_t residuals[4];
        for (size_t entry = 0; entry < entry_count; ++entry)
            residuals[entry] = entries[entry].residual;
        dec_h1_jpack_neon_nr8(entry_bases, residuals, entry_count, full_panels, output,
                              &counters.vector_kernel_hits, &counters.vector_loads,
                              &counters.scalar_fallback_loads, &counters.processed_macs);
    }

    const size_t scalar_panel_begin = use_vector ? full_panels : 0;
    for (size_t panel = scalar_panel_begin; panel < pack.panel_count(); ++panel)
    {
        const size_t remaining_lanes = pack.columns() - panel * kDecH1JPackNr;
        const size_t lane_count = remaining_lanes < kDecH1JPackNr ?
            remaining_lanes : kDecH1JPackNr;
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            int64_t accumulator = 0;
            for (size_t entry = 0; entry < entry_count; ++entry)
            {
                accumulator += static_cast<int64_t>(entries[entry].residual) *
                    (use_small_entry_bases ?
                         entry_bases[entry][panel * kDecH1JPackGroupSize * kDecH1JPackNr + lane] :
                         pack.code_pointer(entries[entry].k, panel)[lane]);
                ++counters.scalar_fallback_loads;
                ++counters.processed_macs;
            }
            output[panel * kDecH1JPackNr + lane] = accumulator;
        }
    }
    return true;
}

bool dec_h1_jpack_has_neon() noexcept
{
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}
}

#if defined(__aarch64__)
extern "C" __attribute__((noinline)) void dec_h1_jpack_neon_nr8(
    const int8_t *const *codes, const int32_t *residuals, size_t entry_count,
    size_t panel_count, int64_t *output, uint64_t *vector_kernel_hits,
    uint64_t *vector_loads, uint64_t *scalar_fallback_loads, uint64_t *processed_macs)
{
    (void) scalar_fallback_loads;
    uint64_t vector_kernel_hit_count = *vector_kernel_hits;
    uint64_t vector_load_count = *vector_loads;
    uint64_t mac_count = *processed_macs;
    for (size_t panel = 0; panel < panel_count; ++panel)
    {
    int32x4_t accumulator_low = vdupq_n_s32(0);
    int32x4_t accumulator_high = vdupq_n_s32(0);
#define DEC_H1_JPACK_ACCUMULATE(ENTRY) \
    do { \
        const int8x8_t codes_i8 = vld1_s8(codes[ENTRY] + panel * \
            ggml::gemmini::quants::dec::kDecH1JPackGroupSize * \
            ggml::gemmini::quants::dec::kDecH1JPackNr); \
        const int16x8_t codes_i16 = vmovl_s8(codes_i8); \
        const int32x4_t residual = vdupq_n_s32(residuals[ENTRY]); \
        accumulator_low = vmlaq_s32( \
            accumulator_low, vmovl_s16(vget_low_s16(codes_i16)), residual); \
        accumulator_high = vmlaq_s32( \
            accumulator_high, vmovl_s16(vget_high_s16(codes_i16)), residual); \
        ++vector_load_count; \
        mac_count += ggml::gemmini::quants::dec::kDecH1JPackNr; \
    } while (false)
    switch (entry_count)
    {
    case 1:
        DEC_H1_JPACK_ACCUMULATE(0);
        break;
    case 2:
        DEC_H1_JPACK_ACCUMULATE(0);
        DEC_H1_JPACK_ACCUMULATE(1);
        break;
    case 3:
        DEC_H1_JPACK_ACCUMULATE(0);
        DEC_H1_JPACK_ACCUMULATE(1);
        DEC_H1_JPACK_ACCUMULATE(2);
        break;
    case 4:
        DEC_H1_JPACK_ACCUMULATE(0);
        DEC_H1_JPACK_ACCUMULATE(1);
        DEC_H1_JPACK_ACCUMULATE(2);
        DEC_H1_JPACK_ACCUMULATE(3);
        break;
    default:
        break;
    }
#undef DEC_H1_JPACK_ACCUMULATE
    int32_t accumulator[8];
    vst1q_s32(accumulator, accumulator_low);
    vst1q_s32(accumulator + 4, accumulator_high);
    for (size_t lane = 0; lane < 8; ++lane)
        output[panel * 8 + lane] = accumulator[lane];
    ++vector_kernel_hit_count;
    }
    *vector_kernel_hits = vector_kernel_hit_count;
    *vector_loads = vector_load_count;
    *processed_macs = mac_count;
}
#else
extern "C" __attribute__((noinline)) void dec_h1_jpack_neon_nr8(
    const int8_t *const *codes, const int32_t *residuals, size_t entry_count,
    size_t panel_count, int64_t *output, uint64_t *vector_kernel_hits,
    uint64_t *vector_loads, uint64_t *scalar_fallback_loads, uint64_t *processed_macs)
{
    (void) vector_loads;
    (void) vector_kernel_hits;
    uint64_t fallback_load_count = *scalar_fallback_loads;
    uint64_t mac_count = *processed_macs;
    for (size_t panel = 0; panel < panel_count; ++panel)
    {
        int64_t accumulator[8] = {};
        for (size_t entry = 0; entry < entry_count; ++entry)
            for (size_t lane = 0; lane < 8; ++lane)
            {
            accumulator[lane] += static_cast<int64_t>(residuals[entry]) *
                codes[entry][panel * ggml::gemmini::quants::dec::kDecH1JPackGroupSize *
                    ggml::gemmini::quants::dec::kDecH1JPackNr + lane];
            ++fallback_load_count;
            ++mac_count;
            }
        for (size_t lane = 0; lane < 8; ++lane)
            output[panel * 8 + lane] = accumulator[lane];
    }
    *scalar_fallback_loads = fallback_load_count;
    *processed_macs = mac_count;
}
#endif
