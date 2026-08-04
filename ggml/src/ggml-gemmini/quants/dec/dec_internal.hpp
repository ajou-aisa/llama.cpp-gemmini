#pragma once

#include "../../ggml-gemmini-args.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants::dec
{
    inline constexpr size_t kDecGroupSizeK = 32;

    struct ResidualGroupEntry
    {
        uint32_t row;
        uint32_t k;
        int32_t residual;
    };

    struct ActiveRowGroup
    {
        uint32_t row;
        uint32_t k_group;
        size_t entry_begin;
        size_t entry_end;
    };

    inline void build_active_row_groups(
        std::vector<ResidualGroupEntry> &entries,
        std::vector<ActiveRowGroup> &groups)
    {
        std::sort(entries.begin(), entries.end(), [](const ResidualGroupEntry &lhs, const ResidualGroupEntry &rhs)
        {
            if (lhs.row != rhs.row)
                return lhs.row < rhs.row;
            const uint32_t lhs_group = lhs.k / kDecGroupSizeK;
            const uint32_t rhs_group = rhs.k / kDecGroupSizeK;
            if (lhs_group != rhs_group)
                return lhs_group < rhs_group;
            if (lhs.k != rhs.k)
                return lhs.k < rhs.k;
            return lhs.residual < rhs.residual;
        });

        groups.clear();
        for (size_t begin = 0; begin < entries.size();)
        {
            const uint32_t row = entries[begin].row;
            const uint32_t k_group = entries[begin].k / kDecGroupSizeK;
            size_t end = begin + 1;
            while (end < entries.size() && entries[end].row == row &&
                   entries[end].k / kDecGroupSizeK == k_group)
                ++end;
            groups.push_back({row, k_group, begin, end});
            begin = end;
        }
    }

    inline void build_group_major_index(
        const std::vector<ActiveRowGroup> &groups,
        size_t group_count,
        std::vector<size_t> &group_offsets,
        std::vector<size_t> &group_row_group_indices)
    {
        group_offsets.assign(group_count + 1, 0);
        group_row_group_indices.resize(groups.size());

        for (const ActiveRowGroup &group : groups)
            ++group_offsets[static_cast<size_t>(group.k_group) + 1];
        for (size_t group = 1; group < group_offsets.size(); ++group)
            group_offsets[group] += group_offsets[group - 1];
        for (size_t index = 0; index < groups.size(); ++index)
        {
            const size_t group = groups[index].k_group;
            group_row_group_indices[group_offsets[group]++] = index;
        }
        for (size_t group = group_count; group > 0; --group)
            group_offsets[group] = group_offsets[group - 1];
        group_offsets[0] = 0;
    }

    struct GroupKCSCPlan
    {
        inline static constexpr uint32_t group_size_k = static_cast<uint32_t>(kDecGroupSizeK);

        uint32_t num_groups = 0;
        std::vector<uint32_t> column_offsets;
        std::vector<uint32_t> entry_order;
        std::vector<uint32_t> active_row_offsets;
        std::vector<uint32_t> active_rows;
        std::vector<uint32_t> fill_cursors;

        void clear()
        {
            num_groups = 0;
            column_offsets.clear();
            entry_order.clear();
            active_row_offsets.clear();
            active_rows.clear();
            fill_cursors.clear();
        }
    };

    inline size_t group_k_csc_plan_logical_bytes(const GroupKCSCPlan &plan)
    {
        size_t bytes = 0;
        const auto add_bytes = [&bytes](size_t count, size_t element_size)
        {
            if (bytes == std::numeric_limits<size_t>::max() ||
                count > std::numeric_limits<size_t>::max() / element_size)
            {
                bytes = std::numeric_limits<size_t>::max();
                return;
            }

            const size_t item_bytes = count * element_size;
            bytes = item_bytes > std::numeric_limits<size_t>::max() - bytes ?
                std::numeric_limits<size_t>::max() : bytes + item_bytes;
        };

        add_bytes(plan.column_offsets.size(), sizeof(uint32_t));
        add_bytes(plan.entry_order.size(), sizeof(uint32_t));
        add_bytes(plan.active_row_offsets.size(), sizeof(uint32_t));
        add_bytes(plan.active_rows.size(), sizeof(uint32_t));
        return bytes;
    }

    inline bool build_group_k_csc_plan(
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &active_row_groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        size_t group_count,
        GroupKCSCPlan &plan)
    {
        constexpr size_t offsets_per_group = kDecGroupSizeK + 1;
        const size_t max_u32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max());

        plan.clear();
        if (group_count > max_u32 || entries.size() > max_u32 ||
            active_row_groups.size() > max_u32 ||
            group_count == std::numeric_limits<size_t>::max() ||
            group_count > std::numeric_limits<size_t>::max() / offsets_per_group ||
            group_offsets.size() != group_count + 1 ||
            group_row_group_indices.size() != active_row_groups.size())
            return false;

        for (size_t group = 0; group < group_count; ++group)
        {
            const size_t row_group_begin = group_offsets[group];
            const size_t row_group_end = group_offsets[group + 1];
            if (row_group_begin > row_group_end || row_group_end > group_row_group_indices.size())
                return false;

            for (size_t position = row_group_begin; position < row_group_end; ++position)
            {
                const size_t row_group_index = group_row_group_indices[position];
                if (row_group_index >= active_row_groups.size())
                    return false;

                const ActiveRowGroup &row_group = active_row_groups[row_group_index];
                if (row_group.k_group != group || row_group.entry_begin >= row_group.entry_end ||
                    row_group.entry_end > entries.size())
                    return false;
            }
        }

        for (const ResidualGroupEntry &entry : entries)
        {
            if (static_cast<size_t>(entry.k / kDecGroupSizeK) >= group_count)
                return false;
        }

        plan.num_groups = static_cast<uint32_t>(group_count);
        plan.column_offsets.resize(group_count * offsets_per_group);
        std::fill(plan.column_offsets.begin(), plan.column_offsets.end(), 0);
        plan.active_row_offsets.resize(group_count + 1);
        plan.active_rows.clear();
        plan.entry_order.resize(entries.size());

        for (const ResidualGroupEntry &entry : entries)
        {
            const size_t group = entry.k / kDecGroupSizeK;
            const size_t column = entry.k % kDecGroupSizeK;
            ++plan.column_offsets[group * offsets_per_group + column + 1];
        }

        uint32_t next_entry = 0;
        for (size_t group = 0; group < group_count; ++group)
        {
            const size_t offset = group * offsets_per_group;
            plan.column_offsets[offset] = next_entry;
            for (size_t column = 0; column < kDecGroupSizeK; ++column)
            {
                const uint32_t count = plan.column_offsets[offset + column + 1];
                if (count > std::numeric_limits<uint32_t>::max() -
                        plan.column_offsets[offset + column])
                {
                    plan.clear();
                    return false;
                }
                plan.column_offsets[offset + column + 1] =
                    plan.column_offsets[offset + column] + count;
            }
            next_entry = plan.column_offsets[offset + kDecGroupSizeK];
        }

        if (static_cast<size_t>(next_entry) != entries.size())
        {
            plan.clear();
            return false;
        }

        plan.fill_cursors.resize(plan.column_offsets.size());
        std::copy(plan.column_offsets.begin(), plan.column_offsets.end(), plan.fill_cursors.begin());
        plan.active_row_offsets[0] = 0;
        for (size_t group = 0; group < group_count; ++group)
        {
            const size_t row_group_begin = group_offsets[group];
            const size_t row_group_end = group_offsets[group + 1];
            const size_t offset = group * offsets_per_group;
            for (size_t position = row_group_begin; position < row_group_end; ++position)
            {
                const ActiveRowGroup &row_group = active_row_groups[group_row_group_indices[position]];
                plan.active_rows.push_back(row_group.row);
                for (size_t entry_index = row_group.entry_begin;
                     entry_index < row_group.entry_end;
                     ++entry_index)
                {
                    const ResidualGroupEntry &entry = entries[entry_index];
                    if (entry.k / kDecGroupSizeK != group)
                    {
                        plan.clear();
                        return false;
                    }

                    const size_t cursor = offset + entry.k % kDecGroupSizeK;
                    const uint32_t destination = plan.fill_cursors[cursor]++;
                    if (destination >= plan.entry_order.size())
                    {
                        plan.clear();
                        return false;
                    }
                    plan.entry_order[destination] = static_cast<uint32_t>(entry_index);
                }
            }
            plan.active_row_offsets[group + 1] = static_cast<uint32_t>(plan.active_rows.size());
        }

        for (size_t group = 0; group < group_count; ++group)
        {
            const size_t offset = group * offsets_per_group;
            for (size_t column = 0; column < kDecGroupSizeK; ++column)
                if (plan.fill_cursors[offset + column] != plan.column_offsets[offset + column + 1])
                {
                    plan.clear();
                    return false;
                }
        }

        return true;
    }

    struct WeightScaleInfo
    {
        const float *data = nullptr;
        size_t rows = 0;
        size_t cols = 0;
        size_t block_size = 0;
        float scalar = 1.0f;
        bool scalar_mode = false;
        bool row_header_mode = false;
        bool channel_mode = false;
        bool on_demand_mode = false;
        bool supported = true;
    };

    enum class WeightScaleInfoMode
    {
        CommonOutput,
        Dec,
    };

    WeightScaleInfo build_weight_scale_info(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    enum class WeightLayout
    {
        KxJ_RowMajor,
        JxK_ColMajor,
    };

    enum class DecWeightRoute
    {
        Unsupported,
        Dense,
        Q8ChannelDirect,
        Q8ChannelSidecar,
        Q8H1,
        Q8H2,
        Q8HP1,
        Q8HP2,
    };

    struct DecRoutePlan
    {
        DecWeightRoute route = DecWeightRoute::Unsupported;
        WeightLayout layout = WeightLayout::KxJ_RowMajor;
        WeightScaleInfo scales{};
        size_t weight_stride = 0;
        const char *reject_reason = "unsupported weight format";
        bool native_weight_blocks = false;
        bool valid = false;
    };

    DecRoutePlan resolve_dec_route_plan(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    const char *dec_route_name(const DecRoutePlan &plan);

    const char *dec_scale_mode_name(const DecRoutePlan &plan);

    bool dec_route_covers_k(const DecRoutePlan &plan, size_t k_count);

    bool dec_route_block_for_range(
        const DecRoutePlan &plan,
        size_t k_offset,
        size_t block_k,
        size_t &block_index);

    float dec_route_weight_scale(
        const DecRoutePlan &plan,
        const ggml_gemmini_args_t &args,
        size_t j,
        size_t block_index);

    inline bool is_q8_h1_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 &&
               args.q8_h1_blocks != nullptr;
    }

    inline bool is_q8_h2_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 &&
               args.q8_h2_blocks != nullptr &&
               args.q8_h2_blocks_per_row > 0;
    }

    inline bool is_q8_hp1_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 &&
               args.q8_hp1_blocks != nullptr &&
               args.q8_hp1_block_count > 0 &&
               args.q8_hp1_blocks_per_row > 0;
    }

    inline bool is_q8_hp2_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 &&
               args.q8_hp2_blocks != nullptr &&
               args.q8_hp2_block_count > 0 &&
               args.q8_hp2_blocks_per_row > 0;
    }

    inline bool is_q8_channel_direct_read_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
               args.has_q8_channel_row_metadata();
    }

    inline bool is_q8_channel_dense_sidecar_args(const ggml_gemmini_args_t &args)
    {
        return args.has_q8_channel_dense_sidecar_contract();
    }

    inline bool has_q8_hp1_native_dec_contract(const ggml_gemmini_args_t &args)
    {
        return args.B == nullptr &&
               !args.weight_i8_scale_active &&
               is_q8_hp1_args(args) &&
               args.has_q8_hp1_im2p_contract();
    }

    inline bool has_q8_hp2_native_dec_contract(const ggml_gemmini_args_t &args)
    {
        return args.B == nullptr &&
               !args.weight_i8_scale_active &&
               is_q8_hp2_args(args) &&
               args.has_q8_hp2_im2p_contract();
    }

    inline bool is_q8_h1_weight_args(const ggml_gemmini_args_t &args)
    {
        return is_q8_h1_args(args) ||
               (args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0);
    }

    inline WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q8_channel_direct_read_args(args) || is_q8_channel_dense_sidecar_args(args))
            return WeightLayout::JxK_ColMajor;

        if (is_q8_hp1_args(args) || is_q8_hp2_args(args) ||
            is_q8_h1_weight_args(args) || args.transpose_B)
            return WeightLayout::JxK_ColMajor;

        return WeightLayout::KxJ_RowMajor;
    }

    inline size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q8_hp1_args(args) || is_q8_hp2_args(args))
            return args.K;

        if (is_q8_h1_weight_args(args))
            return args.K;

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }
}
