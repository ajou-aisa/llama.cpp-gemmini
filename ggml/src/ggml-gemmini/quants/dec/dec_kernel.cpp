#include "dec_kernel.hpp"
#include "dec_internal.hpp"
#include "../../ggml-gemmini-args.h"

#if LOG_CYCLE
#include <gemmini/cycle_reader.hpp>
#include <atomic>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <vector>

#ifndef DEC_VALIDATION
#define DEC_VALIDATION 0
#endif

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

namespace ggml::gemmini::quants::dec
{
int resolve_dec_threads(size_t task_count, int omp_max_threads)
{
    const int task_limit = task_count > static_cast<size_t>(std::numeric_limits<int>::max()) ?
        std::numeric_limits<int>::max() : std::max(1, static_cast<int>(task_count));
    const int worker_limit = std::max(1, omp_max_threads);
    int dec_threads = std::min(task_limit, worker_limit);

    if (const char *env = std::getenv("DEC_THREADS"))
    {
        char *end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(env, &end, 10);
        if (errno == 0 && end != env && end != nullptr && *end == '\0' &&
            parsed > 0 && parsed <= std::numeric_limits<int>::max())
            dec_threads = static_cast<int>(parsed);
    }

    return std::min(std::max(1, dec_threads), std::min(task_limit, worker_limit));
}

int resolve_dec_threads(size_t task_count)
{
#if defined(GGML_GEMMINI_HAS_OPENMP)
    return resolve_dec_threads(task_count, omp_get_max_threads());
#else
    return resolve_dec_threads(task_count, 1);
#endif
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math")))
#endif
float apply_h1_scale_ordered(
    int64_t accumulator,
    uint64_t c_eff,
    float s_rf,
    float activation_scale)
{
#if defined(__clang__)
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
#endif
    double value = static_cast<double>(accumulator);
    value = value * static_cast<double>(c_eff);
    value = value * static_cast<double>(s_rf);
    value = value * static_cast<double>(activation_scale);
    return static_cast<float>(value);
}

H1ScaleParams h1_scale_params(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t j,
    size_t block)
{
    const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
    const uint64_t offset = native ? native->R :
        (args.stripe_J > 1 ? args.R_stripe[j / args.stripe_J] : args.R[j]);
    const uint64_t c_eff = static_cast<uint64_t>(
        native ? native->c_b : static_cast<uint16_t>(args.c_b[j * args.blocks_per_row + block])) +
        offset;
    const float s_rf = native ? native->s_rf :
        (args.stripe_J > 1 ? args.s_rf_stripe[j / args.stripe_J] : args.s_rf[j]);
    return {c_eff, s_rf};
}

namespace
{
#if LOG_CYCLE
    static inline uint64_t dec_group_k_csc_stage_cycle_read()
    {
        return ggml::gemmini::cycle::read();
    }

    static inline uint64_t dec_group_k_csc_stage_cycle_span(uint64_t start, uint64_t end)
    {
        return end >= start ? end - start : 0;
    }
#endif

    struct H1ScaleTile
    {
        alignas(64) std::array<uint64_t, kDecInt64JTileWidth> c_eff{};
        alignas(64) std::array<float, kDecInt64JTileWidth> s_rf{};
    };

    size_t resolve_out_stride_row(const ggml_gemmini_args_t &args)
    {
        return args.stride_f_out ? args.stride_f_out : args.J;
    }

    size_t resolve_out_stride_col(const ggml_gemmini_args_t &args)
    {
        return args.col_stride_f_out ? args.col_stride_f_out : 1;
    }

    template <typename TileWork>
    void for_each_grouped_j_tile(size_t scratch_rows, size_t J, TileWork tile_work)
    {
        if (scratch_rows == 0 || J == 0)
            return;

        const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
        const auto run_range = [&](size_t j_begin, size_t j_end)
        {
            std::vector<int64_t> accumulator(scratch_rows * tile_capacity);
            for (size_t jb = j_begin; jb < j_end; jb += kDecInt64JTileWidth)
                tile_work(jb, std::min(kDecInt64JTileWidth, j_end - jb), accumulator);
        };

#if defined(GGML_GEMMINI_HAS_OPENMP)
        const size_t tile_count = dec_int64_j_tile_count(J);
        const int dec_threads = resolve_dec_threads(tile_count);
        if (dec_threads == 1 || omp_in_parallel() != 0)
        {
            run_range(0, J);
            return;
        }
#pragma omp parallel num_threads(dec_threads)
        {
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            const size_t threads = static_cast<size_t>(omp_get_num_threads());
            const size_t first_tile = tile_count * tid / threads;
            const size_t last_tile = tile_count * (tid + 1) / threads;
            run_range(first_tile * kDecInt64JTileWidth,
                      std::min(J, last_tile * kDecInt64JTileWidth));
        }
#else
        run_range(0, J);
#endif
    }

    template <typename TileWork>
    void for_each_grouped_j_tile_int32(size_t scratch_rows, size_t J, TileWork tile_work)
    {
        if (scratch_rows == 0 || J == 0)
            return;

        const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
        const auto run_range = [&](size_t j_begin, size_t j_end)
        {
            std::vector<int32_t> accumulator(scratch_rows * tile_capacity);
            for (size_t jb = j_begin; jb < j_end; jb += kDecInt64JTileWidth)
                tile_work(jb, std::min(kDecInt64JTileWidth, j_end - jb), accumulator);
        };

#if defined(GGML_GEMMINI_HAS_OPENMP)
        const size_t tile_count = dec_int64_j_tile_count(J);
        const int dec_threads = resolve_dec_threads(tile_count);
        if (dec_threads == 1 || omp_in_parallel() != 0)
        {
            run_range(0, J);
            return;
        }
#pragma omp parallel num_threads(dec_threads)
        {
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            const size_t threads = static_cast<size_t>(omp_get_num_threads());
            const size_t first_tile = tile_count * tid / threads;
            const size_t last_tile = tile_count * (tid + 1) / threads;
            run_range(first_tile * kDecInt64JTileWidth,
                      std::min(J, last_tile * kDecInt64JTileWidth));
        }
#else
        run_range(0, J);
#endif
    }

    template <typename TileWork>
    void for_each_grouped_j_tile_mixed(
        size_t I,
        size_t fallback_rows,
        size_t J,
        TileWork tile_work)
    {
        if (I == 0 || J == 0)
            return;

        const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
        const auto run_range = [&](size_t j_begin, size_t j_end)
        {
            std::vector<int32_t> int32_accumulator(I * tile_capacity);
            std::vector<int64_t> int64_accumulator(fallback_rows * tile_capacity);
            for (size_t jb = j_begin; jb < j_end; jb += kDecInt64JTileWidth)
                tile_work(jb, std::min(kDecInt64JTileWidth, j_end - jb),
                          int32_accumulator, int64_accumulator);
        };

#if defined(GGML_GEMMINI_HAS_OPENMP)
        const size_t tile_count = dec_int64_j_tile_count(J);
        const int dec_threads = resolve_dec_threads(tile_count);
        if (dec_threads == 1 || omp_in_parallel() != 0)
        {
            run_range(0, J);
            return;
        }
#pragma omp parallel num_threads(dec_threads)
        {
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            const size_t threads = static_cast<size_t>(omp_get_num_threads());
            const size_t first_tile = tile_count * tid / threads;
            const size_t last_tile = tile_count * (tid + 1) / threads;
            run_range(first_tile * kDecInt64JTileWidth,
                      std::min(J, last_tile * kDecInt64JTileWidth));
        }
#else
        run_range(0, J);
#endif
    }

    template <typename CodeFor>
    void add_group_dot(
        const std::vector<ResidualGroupEntry> &entries,
        size_t begin,
        size_t end,
        size_t jb,
        size_t width,
        CodeFor code_for,
        int64_t *accumulator)
    {
        for (size_t p = begin; p < end; ++p)
        {
            const ResidualGroupEntry &entry = entries[p];
            const int64_t residual = entry.residual;
            for (size_t t = 0; t < width; ++t)
            {
                const int64_t term = residual * code_for(jb + t, entry.k);
#if DEC_VALIDATION
                const __int128 checked_sum = static_cast<__int128>(accumulator[t]) + term;
                if (checked_sum < std::numeric_limits<int64_t>::min() ||
                    checked_sum > std::numeric_limits<int64_t>::max())
                    std::abort();
#endif
                accumulator[t] += term;
            }
        }
    }

    size_t saturating_mul_size(size_t lhs, size_t rhs)
    {
        if (lhs == 0 || rhs == 0)
            return 0;
        return lhs > std::numeric_limits<size_t>::max() / rhs ?
            std::numeric_limits<size_t>::max() : lhs * rhs;
    }

    size_t saturating_add_size(size_t lhs, size_t rhs)
    {
        return rhs > std::numeric_limits<size_t>::max() - lhs ?
            std::numeric_limits<size_t>::max() : lhs + rhs;
    }

    bool valid_group_k_csc_scalar_plan(
        const GroupKCSCPlan &group_k_csc_plan,
        size_t I,
        size_t K,
        const std::vector<ResidualGroupEntry> &entries)
    {
        constexpr size_t offsets_per_group = kDecGroupSizeK + 1;
        const size_t group_count = K / kDecGroupSizeK + (K % kDecGroupSizeK != 0);
        if (group_count != group_k_csc_plan.num_groups ||
            group_count > std::numeric_limits<size_t>::max() / offsets_per_group ||
            group_k_csc_plan.column_offsets.size() != group_count * offsets_per_group ||
            group_k_csc_plan.entry_order.size() != entries.size() ||
            group_k_csc_plan.active_row_offsets.size() != group_count + 1 ||
            group_k_csc_plan.active_row_offsets.empty() ||
            group_k_csc_plan.active_row_offsets.front() != 0)
            return false;

        size_t previous_active_row_offset = 0;
        for (const uint32_t offset : group_k_csc_plan.active_row_offsets)
        {
            if (offset < previous_active_row_offset || offset > group_k_csc_plan.active_rows.size())
                return false;
            previous_active_row_offset = offset;
        }
        if (previous_active_row_offset != group_k_csc_plan.active_rows.size())
            return false;
        for (const uint32_t row : group_k_csc_plan.active_rows)
            if (row >= I)
                return false;

        size_t previous_entry_end = 0;
        for (size_t group = 0; group < group_count; ++group)
        {
            const size_t group_offset = group * offsets_per_group;
            if (group_k_csc_plan.column_offsets[group_offset] != previous_entry_end)
                return false;

            for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            {
                const size_t entry_begin = group_k_csc_plan.column_offsets[group_offset + k_offset];
                const size_t entry_end = group_k_csc_plan.column_offsets[group_offset + k_offset + 1];
                if (entry_begin > entry_end || entry_end > group_k_csc_plan.entry_order.size())
                    return false;

                size_t previous_entry_index = 0;
                for (size_t position = entry_begin; position < entry_end; ++position)
                {
                    const size_t entry_index = group_k_csc_plan.entry_order[position];
                    if (entry_index >= entries.size() ||
                        (position > entry_begin && entry_index <= previous_entry_index))
                        return false;
                    previous_entry_index = entry_index;
                    const ResidualGroupEntry &entry = entries[entry_index];
                    if (entry.row >= I || entry.k >= K || entry.k / kDecGroupSizeK != group ||
                        entry.k % kDecGroupSizeK != k_offset)
                        return false;
                }
            }
            previous_entry_end = group_k_csc_plan.column_offsets[group_offset + kDecGroupSizeK];
        }

        return previous_entry_end == group_k_csc_plan.entry_order.size();
    }

    bool checked_add_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
            return false;
        out = lhs + rhs;
        return true;
    }

    bool checked_mul_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if (lhs == 0 || rhs == 0)
        {
            out = 0;
            return true;
        }
        if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
            (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()))
            return false;
        if ((lhs > 0 && rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (lhs > 0 && rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs) ||
            (lhs < 0 && rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
            (lhs < 0 && rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
            return false;
        out = lhs * rhs;
        return true;
    }

    bool coefficient_envelope(int64_t coefficient, int64_t &lower, int64_t &upper)
    {
        return coefficient >= 0 ?
            checked_mul_i64(-128, coefficient, lower) && checked_mul_i64(127, coefficient, upper) :
            checked_mul_i64(127, coefficient, lower) && checked_mul_i64(-128, coefficient, upper);
    }

    enum class RowAccumulationWidth
    {
        Int32,
        Int64,
        Invalid,
    };

    struct MixedGroupKCSCEntryPlan
    {
        std::vector<uint32_t> int32_column_offsets;
        std::vector<uint32_t> int32_entry_order;
        std::vector<uint32_t> int64_column_offsets;
        std::vector<uint32_t> int64_entry_order;
    };

    RowAccumulationWidth classify_row_accumulation_width(
        const std::vector<ResidualGroupEntry> &entries,
        size_t entry_begin,
        size_t entry_end)
    {
        int64_t completed_lower = 0;
        int64_t completed_upper = 0;
        bool requires_int64 = false;
        for (size_t position = entry_begin; position < entry_end;)
        {
            const uint32_t k = entries[position].k;
            int64_t partial_coefficient = 0;
            int64_t partial_lower = 0;
            int64_t partial_upper = 0;
            do
            {
                int64_t next_coefficient = 0;
                if (!checked_add_i64(
                        partial_coefficient,
                        static_cast<int64_t>(entries[position].residual),
                        next_coefficient))
                    return RowAccumulationWidth::Invalid;
                partial_coefficient = next_coefficient;
                if (!coefficient_envelope(partial_coefficient, partial_lower, partial_upper))
                    return RowAccumulationWidth::Invalid;

                int64_t prefix_lower = 0;
                int64_t prefix_upper = 0;
                if (!checked_add_i64(completed_lower, partial_lower, prefix_lower) ||
                    !checked_add_i64(completed_upper, partial_upper, prefix_upper))
                    return RowAccumulationWidth::Invalid;
                requires_int64 = requires_int64 ||
                    prefix_lower < std::numeric_limits<int32_t>::min() ||
                    prefix_upper > std::numeric_limits<int32_t>::max();
                ++position;
            } while (position < entry_end && entries[position].k == k);

            int64_t next_completed_lower = 0;
            int64_t next_completed_upper = 0;
            if (!checked_add_i64(completed_lower, partial_lower, next_completed_lower) ||
                !checked_add_i64(completed_upper, partial_upper, next_completed_upper))
                return RowAccumulationWidth::Invalid;
            completed_lower = next_completed_lower;
            completed_upper = next_completed_upper;
        }
        return requires_int64 ? RowAccumulationWidth::Int64 : RowAccumulationWidth::Int32;
    }

    size_t count_active_rows(const std::vector<ResidualGroupEntry> &entries)
    {
        size_t active_rows = 0;
        uint32_t previous_row = std::numeric_limits<uint32_t>::max();
        for (const ResidualGroupEntry &entry : entries)
        {
            if (entry.row == previous_row)
                continue;
            previous_row = entry.row;
            ++active_rows;
        }
        return active_rows;
    }

    void build_mixed_group_k_csc_entry_plan(
        const GroupKCSCPlan &group_k_csc_plan,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<size_t> &fallback_slots,
        size_t no_fallback_slot,
        MixedGroupKCSCEntryPlan &mixed_plan)
    {
        mixed_plan.int32_column_offsets.resize(group_k_csc_plan.column_offsets.size());
        mixed_plan.int32_entry_order.clear();
        mixed_plan.int32_entry_order.reserve(entries.size());
        mixed_plan.int64_column_offsets.resize(group_k_csc_plan.column_offsets.size());
        mixed_plan.int64_entry_order.clear();
        mixed_plan.int64_entry_order.reserve(entries.size());

        constexpr size_t offsets_per_group = kDecGroupSizeK + 1;
        for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
        {
            const size_t group_offset = group * offsets_per_group;
            mixed_plan.int32_column_offsets[group_offset] =
                static_cast<uint32_t>(mixed_plan.int32_entry_order.size());
            mixed_plan.int64_column_offsets[group_offset] =
                static_cast<uint32_t>(mixed_plan.int64_entry_order.size());

            for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            {
                const size_t column_begin =
                    group_k_csc_plan.column_offsets[group_offset + k_offset];
                const size_t column_end =
                    group_k_csc_plan.column_offsets[group_offset + k_offset + 1];
                for (size_t position = column_begin; position < column_end; ++position)
                {
                    const uint32_t entry_index = group_k_csc_plan.entry_order[position];
                    const ResidualGroupEntry &entry = entries[entry_index];
                    if (fallback_slots[entry.row] == no_fallback_slot)
                        mixed_plan.int32_entry_order.push_back(entry_index);
                    else
                        mixed_plan.int64_entry_order.push_back(entry_index);
                }

                mixed_plan.int32_column_offsets[group_offset + k_offset + 1] =
                    static_cast<uint32_t>(mixed_plan.int32_entry_order.size());
                mixed_plan.int64_column_offsets[group_offset + k_offset + 1] =
                    static_cast<uint32_t>(mixed_plan.int64_entry_order.size());
            }
        }
    }

    template <typename CodeFor, typename ScaleForColumn>
    void run_whole_k_grouped_dec(
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        if (!Y_com || entries.empty() || groups.empty() || group_offsets.size() < 2 ||
            group_row_group_indices.size() != groups.size() || I == 0 || J == 0)
            return;

        for_each_grouped_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
            for (size_t k_group = 0; k_group + 1 < group_offsets.size(); ++k_group)
                for (size_t position = group_offsets[k_group]; position < group_offsets[k_group + 1]; ++position)
                {
                    const ActiveRowGroup &group = groups[group_row_group_indices[position]];
                    add_group_dot(entries, group.entry_begin, group.entry_end, jb, width, code_for,
                                  accumulator.data() + static_cast<size_t>(group.row) * width);
                }

            uint32_t previous_row = std::numeric_limits<uint32_t>::max();
            for (const ActiveRowGroup &group : groups)
            {
                if (group.row == previous_row)
                    continue;
                previous_row = group.row;
                const size_t row_index = group.row;
                const float activation_scale = activation_scales ? activation_scales[row_index] : 1.0f;
                const int64_t *row = accumulator.data() + row_index * width;
                float *output = Y_com + row_index * J + jb;
                for (size_t t = 0; t < width; ++t)
                    output[t] += static_cast<float>(
                        static_cast<double>(row[t]) * activation_scale * scale_for_column(jb + t));
            }
        });
    }

    template <typename CodeFor, typename ScaleFor>
    void run_scaled_grouped_dec(
        size_t I,
        size_t J,
        size_t scale_group_size,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        ScaleFor scale_for,
        float *Y_com)
    {
        if (!Y_com || entries.empty() || groups.empty() || group_offsets.size() < 2 ||
            group_row_group_indices.size() != groups.size() || I == 0 || J == 0 || scale_group_size == 0)
            return;

        for_each_grouped_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            alignas(64) std::array<float, kDecInt64JTileWidth> scale_tile{};
            for (size_t k_group = 0; k_group + 1 < group_offsets.size(); ++k_group)
            {
                if (group_offsets[k_group] == group_offsets[k_group + 1])
                    continue;

                if (scale_group_size == kDecGroupSizeK)
                    for (size_t t = 0; t < width; ++t)
                        scale_tile[t] = scale_for(jb + t, k_group);

                for (size_t position = group_offsets[k_group]; position < group_offsets[k_group + 1]; ++position)
                {
                    const ActiveRowGroup &group = groups[group_row_group_indices[position]];
                    for (size_t begin = group.entry_begin; begin < group.entry_end;)
                    {
                        const size_t scale_block = entries[begin].k / scale_group_size;
                        size_t end = begin + 1;
                        while (end < group.entry_end && entries[end].k / scale_group_size == scale_block)
                            ++end;

                        std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
                        add_group_dot(entries, begin, end, jb, width, code_for, accumulator.data());
                        const size_t row = group.row;
                        const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
                        float *output = Y_com + row * J + jb;
                        for (size_t t = 0; t < width; ++t)
                        {
                            const float weight_scale = scale_group_size == kDecGroupSizeK ?
                                scale_tile[t] : scale_for(jb + t, scale_block);
                            output[t] += static_cast<float>(
                                static_cast<double>(accumulator[t]) * activation_scale * weight_scale);
                        }
                        begin = end;
                    }
                }
            }
        });
    }

    template <typename CodeFor, typename ScaleParamsFor>
    void run_h1_grouped_dec(
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        ScaleParamsFor scale_params_for,
        float *Y_com)
    {
        static_assert(kDecGroupSizeK == QK8_0, "H1 scale blocks must match DEC compute groups");
        if (!Y_com || entries.empty() || groups.empty() || group_offsets.size() < 2 ||
            group_row_group_indices.size() != groups.size() || I == 0 || J == 0)
            return;

        for_each_grouped_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            H1ScaleTile scale_tile;
            for (size_t k_group = 0; k_group + 1 < group_offsets.size(); ++k_group)
            {
                if (group_offsets[k_group] == group_offsets[k_group + 1])
                    continue;

                for (size_t t = 0; t < width; ++t)
                {
                    const H1ScaleParams params = scale_params_for(jb + t, k_group);
                    scale_tile.c_eff[t] = params.c_eff;
                    scale_tile.s_rf[t] = params.s_rf;
                }

                for (size_t position = group_offsets[k_group]; position < group_offsets[k_group + 1]; ++position)
                {
                    const ActiveRowGroup &group = groups[group_row_group_indices[position]];
                    std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
                    add_group_dot(entries, group.entry_begin, group.entry_end, jb, width, code_for, accumulator.data());
                    const size_t row = group.row;
                    const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
                    float *output = Y_com + row * J + jb;
                    for (size_t t = 0; t < width; ++t)
                        output[t] += apply_h1_scale_ordered(
                            accumulator[t], scale_tile.c_eff[t], scale_tile.s_rf[t], activation_scale);
                }
            }
        });
    }

    template <typename CodeFor>
    void run_block_route(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        float *Y_com)
    {
        run_scaled_grouped_dec(I, J, plan.scales.block_size, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            code_for,
            [&args, &plan](size_t j, size_t block) {
                return dec_route_weight_scale(plan, args, j, block);
            }, Y_com);
    }
}

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

namespace
{
template <size_t NR>
size_t group_k_csc_vector_load_count(size_t active_k_count, size_t J)
{
    if constexpr (NR == 1)
        return 0;

    size_t vector_loads_per_k = 0;
    for (size_t jb = 0; jb < J; jb += kDecInt64JTileWidth)
    {
        const size_t width = std::min(kDecInt64JTileWidth, J - jb);
        vector_loads_per_k = saturating_add_size(
            vector_loads_per_k, width / NR + (width % NR != 0));
    }
    return saturating_mul_size(active_k_count, vector_loads_per_k);
}

template <size_t NR>
bool accumulate_to_ycom_int64_scalar_group_k_csc_impl(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    static_assert(NR > 0, "NR must be positive");
    stats = {};
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
    const size_t minimum_weight_stride = plan.layout == WeightLayout::KxJ_RowMajor ? J : args.K;
    if (!weights || !Y_com || !plan.valid || plan.route != DecWeightRoute::Dense ||
        !plan.scales.scalar_mode || I == 0 || J == 0 || args.K == 0 || I != args.I || J != args.J ||
        plan.weight_stride < minimum_weight_stride ||
        I > std::numeric_limits<size_t>::max() / tile_capacity ||
        !valid_group_k_csc_scalar_plan(group_k_csc_plan, I, args.K, entries))
        return false;

    if (entries.empty())
        return true;

    size_t active_k_count = 0;
    for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
    {
        const size_t group_offset = group * (kDecGroupSizeK + 1);
        for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            active_k_count += group_k_csc_plan.column_offsets[group_offset + k_offset] !=
                group_k_csc_plan.column_offsets[group_offset + k_offset + 1];
    }
    stats.logical_weight_reference_count = saturating_mul_size(entries.size(), J);
    stats.weight_scalar_load_count = saturating_mul_size(active_k_count, J);
    stats.weight_vector_load_count = group_k_csc_vector_load_count<NR>(active_k_count, J);
    stats.scratch_init_count = saturating_mul_size(I, J);
    stats.sparse_update_count = stats.logical_weight_reference_count;
    stats.fallback_update_count = stats.logical_weight_reference_count;
    stats.merge_count = saturating_mul_size(count_active_rows(entries), J);
    stats.thread_scratch_bytes = saturating_mul_size(
        saturating_mul_size(I, tile_capacity), sizeof(int64_t));
#if LOG_CYCLE
    std::atomic<uint64_t> scratch_init_cycles {0};
    std::atomic<uint64_t> sparse_update_cycles {0};
    std::atomic<uint64_t> merge_cycles {0};
#endif

    for_each_grouped_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
    {
#if LOG_CYCLE
        const uint64_t scratch_init_start = dec_group_k_csc_stage_cycle_read();
#endif
        std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
#if LOG_CYCLE
        scratch_init_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(
                scratch_init_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
        const uint64_t sparse_update_start = dec_group_k_csc_stage_cycle_read();
#endif
        for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
        {
            const size_t group_offset = group * (kDecGroupSizeK + 1);
            for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            {
                const size_t entry_begin = group_k_csc_plan.column_offsets[group_offset + k_offset];
                const size_t entry_end = group_k_csc_plan.column_offsets[group_offset + k_offset + 1];
                if (entry_begin == entry_end)
                    continue;

                const size_t k = group * kDecGroupSizeK + k_offset;
                for (size_t j_offset = 0; j_offset < width; j_offset += NR)
                {
                    const size_t lane_count = std::min(NR, width - j_offset);
                    std::array<int64_t, NR> weight_lanes;
                    for (size_t lane = 0; lane < lane_count; ++lane)
                        weight_lanes[lane] = plan.layout == WeightLayout::KxJ_RowMajor ?
                            weights[k * plan.weight_stride + jb + j_offset + lane] :
                            weights[(jb + j_offset + lane) * plan.weight_stride + k];
                    for (size_t position = entry_begin; position < entry_end; ++position)
                    {
                        const ResidualGroupEntry &entry = entries[group_k_csc_plan.entry_order[position]];
                        int64_t *entry_accumulator = accumulator.data() +
                            static_cast<size_t>(entry.row) * width + j_offset;
                        const int64_t residual = entry.residual;
                        for (size_t lane = 0; lane < lane_count; ++lane)
                        {
                            const int64_t term = residual * weight_lanes[lane];
#if DEC_VALIDATION
                            const __int128 checked_sum =
                                static_cast<__int128>(entry_accumulator[lane]) + term;
                            if (checked_sum < std::numeric_limits<int64_t>::min() ||
                                checked_sum > std::numeric_limits<int64_t>::max())
                                std::abort();
#endif
                            entry_accumulator[lane] += term;
                        }
                    }
                }
            }
        }

#if LOG_CYCLE
        sparse_update_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(
                sparse_update_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
        const uint64_t merge_start = dec_group_k_csc_stage_cycle_read();
#endif
        uint32_t previous_row = std::numeric_limits<uint32_t>::max();
        for (const ResidualGroupEntry &entry : entries)
        {
            if (entry.row == previous_row)
                continue;
            previous_row = entry.row;

            const size_t row = entry.row;
            const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
            const int64_t *row_accumulator = accumulator.data() + row * width;
            float *output = Y_com + row * J + jb;
            for (size_t t = 0; t < width; ++t)
                output[t] += static_cast<float>(
                    static_cast<double>(row_accumulator[t]) * activation_scale * plan.scales.scalar);
        }
#if LOG_CYCLE
        merge_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(merge_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
#endif
    });

#if LOG_CYCLE
    stats.scratch_init_cycles = scratch_init_cycles.load(std::memory_order_relaxed);
    stats.sparse_update_cycles = sparse_update_cycles.load(std::memory_order_relaxed);
    stats.merge_cycles = merge_cycles.load(std::memory_order_relaxed);
#endif
    return true;
}
}

bool accumulate_to_ycom_int64_scalar_group_k_csc(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    return accumulate_to_ycom_int64_scalar_group_k_csc_impl<1>(
        args, plan, I, J, activation_scales, entries, group_k_csc_plan, Y_com, stats);
}

bool accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    return accumulate_to_ycom_int64_scalar_group_k_csc_impl<8>(
        args, plan, I, J, activation_scales, entries, group_k_csc_plan, Y_com, stats);
}

bool accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    return accumulate_to_ycom_int64_scalar_group_k_csc_impl<4>(
        args, plan, I, J, activation_scales, entries, group_k_csc_plan, Y_com, stats);
}

namespace
{
template <size_t NR>
bool accumulate_to_ycom_int32_mixed_group_k_csc_impl(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    static_assert(NR > 0, "NR must be positive");
    stats = {};
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
    const size_t minimum_weight_stride = plan.layout == WeightLayout::KxJ_RowMajor ? J : args.K;
    if (!weights || !Y_com || !plan.valid || plan.route != DecWeightRoute::Dense ||
        !plan.scales.scalar_mode || I == 0 || J == 0 || args.K == 0 || I != args.I || J != args.J ||
        plan.weight_stride < minimum_weight_stride ||
        I > std::numeric_limits<size_t>::max() / tile_capacity ||
        !valid_group_k_csc_scalar_plan(group_k_csc_plan, I, args.K, entries))
        return false;
    if (entries.empty())
        return true;

    const size_t no_fallback_slot = std::numeric_limits<size_t>::max();
    std::vector<size_t> fallback_slots(I, no_fallback_slot);
    std::vector<size_t> int32_rows;
    std::vector<size_t> int64_rows;
#if LOG_CYCLE
    const uint64_t classification_start = dec_group_k_csc_stage_cycle_read();
#endif
    size_t entry_begin = 0;
    while (entry_begin < entries.size())
    {
        const uint32_t row = entries[entry_begin].row;
        size_t entry_end = entry_begin + 1;
        while (entry_end < entries.size() && entries[entry_end].row == row)
            ++entry_end;
        const RowAccumulationWidth width =
            classify_row_accumulation_width(entries, entry_begin, entry_end);
        if (width == RowAccumulationWidth::Invalid)
            return false;
        if (width == RowAccumulationWidth::Int32)
        {
            ++stats.int32_row_count;
            int32_rows.push_back(row);
        }
        else
        {
            if (stats.int64_fallback_row_count == no_fallback_slot)
                return false;
            fallback_slots[row] = stats.int64_fallback_row_count;
            ++stats.int64_fallback_row_count;
            int64_rows.push_back(row);
        }
        entry_begin = entry_end;
    }
    stats.classification_work_count = entries.size();

    size_t active_k_count = 0;
    for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
    {
        const size_t group_offset = group * (kDecGroupSizeK + 1);
        for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            active_k_count += group_k_csc_plan.column_offsets[group_offset + k_offset] !=
                group_k_csc_plan.column_offsets[group_offset + k_offset + 1];
    }
    stats.logical_weight_reference_count = saturating_mul_size(entries.size(), J);
    stats.weight_scalar_load_count = saturating_mul_size(active_k_count, J);
    stats.weight_vector_load_count = group_k_csc_vector_load_count<NR>(active_k_count, J);

    const bool all_int32 = stats.int64_fallback_row_count == 0;
    const bool all_int64 = stats.int32_row_count == 0;
    stats.width_path = all_int32 ? GroupKCSCWidthPath::AllInt32 :
        all_int64 ? GroupKCSCWidthPath::AllInt64 : GroupKCSCWidthPath::Mixed;
    MixedGroupKCSCEntryPlan mixed_entry_plan;
    if (!all_int32 && !all_int64)
    {
        build_mixed_group_k_csc_entry_plan(
            group_k_csc_plan, entries, fallback_slots, no_fallback_slot, mixed_entry_plan);
        stats.branch_entry_classification_count = entries.size();
    }
#if LOG_CYCLE
    stats.classification_cycles = dec_group_k_csc_stage_cycle_span(
        classification_start, dec_group_k_csc_stage_cycle_read());
#endif
    const size_t int32_scratch_rows = all_int64 ? 0 : I;
    const size_t int64_scratch_rows = all_int32 ? 0 : stats.int64_fallback_row_count;
    stats.scratch_init_count = saturating_mul_size(
        saturating_add_size(int32_scratch_rows, int64_scratch_rows), J);
    if (all_int32)
        stats.safe_update_count = stats.logical_weight_reference_count;
    else if (all_int64)
        stats.fallback_update_count = stats.logical_weight_reference_count;
    else
    {
        stats.safe_update_count = saturating_mul_size(mixed_entry_plan.int32_entry_order.size(), J);
        stats.fallback_update_count = saturating_mul_size(mixed_entry_plan.int64_entry_order.size(), J);
    }
    stats.sparse_update_count = saturating_add_size(
        stats.safe_update_count, stats.fallback_update_count);
    stats.merge_count = saturating_mul_size(
        saturating_add_size(int32_rows.size(), int64_rows.size()), J);
    const size_t int32_scratch_bytes = saturating_mul_size(
        saturating_mul_size(int32_scratch_rows, tile_capacity), sizeof(int32_t));
    const size_t int64_scratch_bytes = saturating_mul_size(
        saturating_mul_size(int64_scratch_rows, tile_capacity), sizeof(int64_t));
    stats.thread_scratch_bytes = saturating_add_size(int32_scratch_bytes, int64_scratch_bytes);
#if LOG_CYCLE
    std::atomic<uint64_t> scratch_init_cycles {0};
    std::atomic<uint64_t> sparse_update_cycles {0};
    std::atomic<uint64_t> merge_cycles {0};
#endif

    const auto accumulate_group = [&](size_t group, size_t jb, size_t width,
        const std::vector<uint32_t> &column_offsets, const auto &entry_order,
        auto update_entry)
    {
        const size_t group_offset = group * (kDecGroupSizeK + 1);
        for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
        {
            const size_t column_begin = column_offsets[group_offset + k_offset];
            const size_t column_end = column_offsets[group_offset + k_offset + 1];
            if (column_begin == column_end)
                continue;

            const size_t k = group * kDecGroupSizeK + k_offset;
            for (size_t j_offset = 0; j_offset < width; j_offset += NR)
            {
                const size_t lane_count = std::min(NR, width - j_offset);
                std::array<int64_t, NR> weight_lanes;
                for (size_t lane = 0; lane < lane_count; ++lane)
                    weight_lanes[lane] = plan.layout == WeightLayout::KxJ_RowMajor ?
                        weights[k * plan.weight_stride + jb + j_offset + lane] :
                        weights[(jb + j_offset + lane) * plan.weight_stride + k];
                for (size_t position = column_begin; position < column_end; ++position)
                {
                    const ResidualGroupEntry &entry = entries[entry_order[position]];
                    update_entry(entry, j_offset, lane_count, weight_lanes);
                }
            }
        }
    };

    const auto merge_int32 = [&](size_t jb, size_t width, const std::vector<int32_t> &accumulator)
    {
        for (size_t row : int32_rows)
        {
            const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
            float *output = Y_com + row * J + jb;
            const int32_t *row_accumulator = accumulator.data() + row * width;
            for (size_t t = 0; t < width; ++t)
                output[t] += static_cast<float>(
                    static_cast<double>(row_accumulator[t]) * activation_scale * plan.scales.scalar);
        }
    };

    const auto merge_int64 = [&](size_t jb, size_t width, const std::vector<int64_t> &accumulator)
    {
        for (size_t row : int64_rows)
        {
            const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
            float *output = Y_com + row * J + jb;
            const int64_t *row_accumulator = accumulator.data() + fallback_slots[row] * width;
            for (size_t t = 0; t < width; ++t)
                output[t] += static_cast<float>(
                    static_cast<double>(row_accumulator[t]) * activation_scale * plan.scales.scalar);
        }
    };

    if (all_int32)
    {
        for_each_grouped_j_tile_int32(I, J,
            [&](size_t jb, size_t width, std::vector<int32_t> &accumulator)
        {
#if LOG_CYCLE
            const uint64_t scratch_init_start = dec_group_k_csc_stage_cycle_read();
#endif
            std::fill(accumulator.begin(), accumulator.begin() + I * width, int32_t {0});
#if LOG_CYCLE
            scratch_init_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(
                    scratch_init_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
            const uint64_t sparse_update_start = dec_group_k_csc_stage_cycle_read();
#endif
            const auto update_int32 = [&](const ResidualGroupEntry &entry, size_t j_offset,
                size_t lane_count, const std::array<int64_t, NR> &weight_lanes)
            {
                int32_t *entry_accumulator = accumulator.data() +
                    static_cast<size_t>(entry.row) * width + j_offset;
                const int64_t residual = static_cast<int64_t>(entry.residual);
                for (size_t lane = 0; lane < lane_count; ++lane)
                {
                    const int64_t updated = static_cast<int64_t>(entry_accumulator[lane]) +
                        residual * weight_lanes[lane];
#if DEC_VALIDATION
                    if (updated < std::numeric_limits<int32_t>::min() ||
                        updated > std::numeric_limits<int32_t>::max())
                        std::abort();
#endif
                    entry_accumulator[lane] = static_cast<int32_t>(updated);
                }
            };
            for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
                accumulate_group(group, jb, width, group_k_csc_plan.column_offsets,
                    group_k_csc_plan.entry_order, update_int32);
#if LOG_CYCLE
            sparse_update_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(
                    sparse_update_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
            const uint64_t merge_start = dec_group_k_csc_stage_cycle_read();
#endif
            merge_int32(jb, width, accumulator);
#if LOG_CYCLE
            merge_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(merge_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
#endif
        });
#if LOG_CYCLE
        stats.scratch_init_cycles = scratch_init_cycles.load(std::memory_order_relaxed);
        stats.sparse_update_cycles = sparse_update_cycles.load(std::memory_order_relaxed);
        stats.merge_cycles = merge_cycles.load(std::memory_order_relaxed);
#endif
        return true;
    }

    const auto update_int64_for = [&](std::vector<int64_t> &accumulator, size_t width)
    {
        return [&, width](const ResidualGroupEntry &entry, size_t j_offset, size_t lane_count,
            const std::array<int64_t, NR> &weight_lanes)
        {
            int64_t *entry_accumulator = accumulator.data() +
                fallback_slots[entry.row] * width + j_offset;
            const int64_t residual = static_cast<int64_t>(entry.residual);
            for (size_t lane = 0; lane < lane_count; ++lane)
            {
                const int64_t term = residual * weight_lanes[lane];
                int64_t updated = 0;
                if (!checked_add_i64(entry_accumulator[lane], term, updated))
                    std::abort();
#if DEC_VALIDATION
                const __int128 checked_sum =
                    static_cast<__int128>(entry_accumulator[lane]) + term;
                if (checked_sum < std::numeric_limits<int64_t>::min() ||
                    checked_sum > std::numeric_limits<int64_t>::max())
                    std::abort();
#endif
                entry_accumulator[lane] = updated;
            }
        };
    };

    if (all_int64)
    {
        for_each_grouped_j_tile(stats.int64_fallback_row_count, J,
            [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
#if LOG_CYCLE
            const uint64_t scratch_init_start = dec_group_k_csc_stage_cycle_read();
#endif
            std::fill(accumulator.begin(),
                accumulator.begin() + stats.int64_fallback_row_count * width, int64_t {0});
#if LOG_CYCLE
            scratch_init_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(
                    scratch_init_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
            const uint64_t sparse_update_start = dec_group_k_csc_stage_cycle_read();
#endif
            const auto update_int64 = update_int64_for(accumulator, width);
            for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
                accumulate_group(group, jb, width, group_k_csc_plan.column_offsets,
                    group_k_csc_plan.entry_order, update_int64);
#if LOG_CYCLE
            sparse_update_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(
                    sparse_update_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
            const uint64_t merge_start = dec_group_k_csc_stage_cycle_read();
#endif
            merge_int64(jb, width, accumulator);
#if LOG_CYCLE
            merge_cycles.fetch_add(
                dec_group_k_csc_stage_cycle_span(merge_start, dec_group_k_csc_stage_cycle_read()),
                std::memory_order_relaxed);
#endif
        });
#if LOG_CYCLE
        stats.scratch_init_cycles = scratch_init_cycles.load(std::memory_order_relaxed);
        stats.sparse_update_cycles = sparse_update_cycles.load(std::memory_order_relaxed);
        stats.merge_cycles = merge_cycles.load(std::memory_order_relaxed);
#endif
        return true;
    }

    for_each_grouped_j_tile_mixed(
        I, stats.int64_fallback_row_count, J,
        [&](size_t jb, size_t width, std::vector<int32_t> &int32_accumulator,
            std::vector<int64_t> &int64_accumulator)
    {
#if LOG_CYCLE
        const uint64_t scratch_init_start = dec_group_k_csc_stage_cycle_read();
#endif
        std::fill(int32_accumulator.begin(), int32_accumulator.begin() + I * width, int32_t {0});
        std::fill(int64_accumulator.begin(),
                  int64_accumulator.begin() + stats.int64_fallback_row_count * width,
                  int64_t {0});
#if LOG_CYCLE
        scratch_init_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(
                scratch_init_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
        const uint64_t sparse_update_start = dec_group_k_csc_stage_cycle_read();
#endif
        const auto update_int32 = [&](const ResidualGroupEntry &entry, size_t j_offset,
            size_t lane_count, const std::array<int64_t, NR> &weight_lanes)
        {
            int32_t *entry_accumulator = int32_accumulator.data() +
                static_cast<size_t>(entry.row) * width + j_offset;
            const int64_t residual = static_cast<int64_t>(entry.residual);
            for (size_t lane = 0; lane < lane_count; ++lane)
            {
                const int64_t updated = static_cast<int64_t>(entry_accumulator[lane]) +
                    residual * weight_lanes[lane];
#if DEC_VALIDATION
                if (updated < std::numeric_limits<int32_t>::min() ||
                    updated > std::numeric_limits<int32_t>::max())
                    std::abort();
#endif
                entry_accumulator[lane] = static_cast<int32_t>(updated);
            }
        };
        const auto update_int64 = update_int64_for(int64_accumulator, width);
        for (size_t group = 0; group < group_k_csc_plan.num_groups; ++group)
        {
            const size_t group_offset = group * (kDecGroupSizeK + 1);
            for (size_t k_offset = 0; k_offset < kDecGroupSizeK; ++k_offset)
            {
                const size_t int32_begin =
                    mixed_entry_plan.int32_column_offsets[group_offset + k_offset];
                const size_t int32_end =
                    mixed_entry_plan.int32_column_offsets[group_offset + k_offset + 1];
                const size_t int64_begin =
                    mixed_entry_plan.int64_column_offsets[group_offset + k_offset];
                const size_t int64_end =
                    mixed_entry_plan.int64_column_offsets[group_offset + k_offset + 1];
                if (int32_begin == int32_end && int64_begin == int64_end)
                    continue;

                const size_t k = group * kDecGroupSizeK + k_offset;
                for (size_t j_offset = 0; j_offset < width; j_offset += NR)
                {
                    const size_t lane_count = std::min(NR, width - j_offset);
                    std::array<int64_t, NR> weight_lanes;
                    for (size_t lane = 0; lane < lane_count; ++lane)
                        weight_lanes[lane] = plan.layout == WeightLayout::KxJ_RowMajor ?
                            weights[k * plan.weight_stride + jb + j_offset + lane] :
                            weights[(jb + j_offset + lane) * plan.weight_stride + k];
                    for (size_t position = int32_begin; position < int32_end; ++position)
                        update_int32(
                            entries[mixed_entry_plan.int32_entry_order[position]],
                            j_offset, lane_count, weight_lanes);
                    for (size_t position = int64_begin; position < int64_end; ++position)
                        update_int64(
                            entries[mixed_entry_plan.int64_entry_order[position]],
                            j_offset, lane_count, weight_lanes);
                }
            }
        }

#if LOG_CYCLE
        sparse_update_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(
                sparse_update_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
        const uint64_t merge_start = dec_group_k_csc_stage_cycle_read();
#endif
        merge_int32(jb, width, int32_accumulator);
        merge_int64(jb, width, int64_accumulator);
#if LOG_CYCLE
        merge_cycles.fetch_add(
            dec_group_k_csc_stage_cycle_span(merge_start, dec_group_k_csc_stage_cycle_read()),
            std::memory_order_relaxed);
#endif
    });

#if LOG_CYCLE
    stats.scratch_init_cycles = scratch_init_cycles.load(std::memory_order_relaxed);
    stats.sparse_update_cycles = sparse_update_cycles.load(std::memory_order_relaxed);
    stats.merge_cycles = merge_cycles.load(std::memory_order_relaxed);
#endif
    return true;
}
}

bool accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    return accumulate_to_ycom_int32_mixed_group_k_csc_impl<8>(
        args, plan, I, J, activation_scales, entries, group_k_csc_plan, Y_com, stats);
}

bool accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan, float *Y_com, GroupKCSCScalarStats &stats)
{
    return accumulate_to_ycom_int32_mixed_group_k_csc_impl<4>(
        args, plan, I, J, activation_scales, entries, group_k_csc_plan, Y_com, stats);
}

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    if (plan.route == DecWeightRoute::Q8HP1)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_block_route(args, plan, I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    run_h1_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [&args, &plan, weights](size_t j, size_t k) {
            const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, k / QK8_0) : nullptr;
            return native ? native->qs[k % QK8_0] : weights[j * plan.weight_stride + k];
        },
        [&args, &plan](size_t j, size_t block) {
            return h1_scale_params(args, plan, j, block);
        }, Y_com);
}

void apply_ycom_to_output(
    const float *Y_com,
    size_t I,
    size_t J,
    const ggml_gemmini_args_t &args)
{
    float *out_data = args.f_out;
    if (!Y_com || !out_data || I == 0 || J == 0)
        return;

    const size_t stride_row = resolve_out_stride_row(args);
    const size_t stride_col = resolve_out_stride_col(args);
    for (size_t r = 0; r < I; ++r)
    {
        const float *src = Y_com + r * J;
        float *dst = out_data + r * stride_row;
        if (stride_col == 1)
            for (size_t j = 0; j < J; ++j)
                dst[j] += src[j];
        else
            for (size_t j = 0; j < J; ++j)
                dst[j * stride_col] += src[j];
    }
}
} // namespace ggml::gemmini::quants::dec
