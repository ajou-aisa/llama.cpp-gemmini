#include "weight_reader.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace ggml::gemmini::quants::wreader
{

namespace
{

using Format = ggml_gemmini_args_t::im2p_weight_format_t;
using Route = wroute::WeightRouteKind;
using ScaleDomain = wroute::WeightScaleDomain;

constexpr size_t kBlockSize = 32;

bool checked_mul_size(size_t lhs, size_t rhs, size_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

bool checked_block_offset(
    size_t row,
    size_t blocks_per_row,
    size_t block,
    size_t block_count,
    size_t &offset)
{
    size_t row_offset = 0;
    if (!checked_mul_size(row, blocks_per_row, row_offset) ||
        block > std::numeric_limits<size_t>::max() - row_offset) {
        return false;
    }
    offset = row_offset + block;
    return offset < block_count;
}

bool finite_float(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

bool zero_bytes(const uint8_t *data, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != 0)
            return false;
    }
    return true;
}

bool native_extent_covers(
    const ggml_gemmini_args_t &args,
    size_t block_count,
    size_t block_bytes)
{
    if (args.native_weight_bytes == 0)
        return false;

    size_t required = 0;
    return checked_mul_size(block_count, block_bytes, required) &&
        args.native_weight_bytes >= required;
}

bool describe_format(Format format, Route &route, uint8_t &bits)
{
    switch (format) {
        case Format::q4_h0:  route = Route::H0;  bits = 4;  return true;
        case Format::q4_h1:  route = Route::H1;  bits = 4;  return true;
        case Format::q4_hp1: route = Route::HP1; bits = 4;  return true;
        case Format::q8_h0:  route = Route::H0;  bits = 8;  return true;
        case Format::q8_h1:
        case Format::q8_0_unpacked_to_h1:
            route = Route::H1;
            bits = 8;
            return true;
        case Format::q8_hp1: route = Route::HP1; bits = 8;  return true;
        case Format::q16_h0:  route = Route::H0;  bits = 16; return true;
        case Format::q16_h1:  route = Route::H1;  bits = 16; return true;
        case Format::q16_hp1: route = Route::HP1; bits = 16; return true;
        default:
            return false;
    }
}

bool dense_extent_is_addressable(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan)
{
    const bool column_major = plan.layout == wroute::WeightLayout::JxK_ColMajor;
    const size_t major_count = column_major ? args.J : args.K;
    const size_t minor_count = column_major ? args.K : args.J;
    if (major_count == 0 || minor_count == 0 || plan.weight_stride < minor_count)
        return false;

    size_t major_offset = 0;
    return checked_mul_size(major_count - 1, plan.weight_stride, major_offset) &&
        minor_count - 1 <= std::numeric_limits<size_t>::max() - major_offset;
}

WeightReaderStatus validate_dense_q8_h1(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan)
{
    size_t scale_count = 0;
    if (args.B == nullptr || args.J == 0 || args.K == 0 ||
        args.blocks_per_row == 0 || args.c_b == nullptr ||
        args.K > std::numeric_limits<size_t>::max() - (kBlockSize - 1) ||
        args.blocks_per_row != (args.K + kBlockSize - 1) / kBlockSize ||
        !checked_mul_size(args.J, args.blocks_per_row, scale_count) ||
        !dense_extent_is_addressable(args, plan)) {
        return WeightReaderStatus::InvalidMetadata;
    }

    if (args.stripe_J > 1) {
        if (args.s_rf_stripe == nullptr || args.R_stripe == nullptr)
            return WeightReaderStatus::InvalidMetadata;
    } else if (args.s_rf == nullptr || args.R == nullptr) {
        return WeightReaderStatus::InvalidMetadata;
    }
    return WeightReaderStatus::Success;
}

WeightReaderStatus validate_storage(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan)
{
    if (args.J == 0 || args.K == 0)
        return WeightReaderStatus::InvalidMetadata;

    if (plan.route == Route::Dense ||
        plan.route == Route::Q8ChannelDirect ||
        plan.route == Route::Q8ChannelSidecar) {
        if (plan.weight_bits != 8)
            return WeightReaderStatus::UnsupportedFormat;
        return args.B != nullptr && dense_extent_is_addressable(args, plan) ?
            WeightReaderStatus::Success : WeightReaderStatus::InvalidMetadata;
    }

    Route expected_route = Route::Unsupported;
    uint8_t expected_bits = 0;
    if (!describe_format(args.weight_format, expected_route, expected_bits) ||
        plan.route != expected_route || plan.weight_bits != expected_bits) {
        return WeightReaderStatus::UnsupportedFormat;
    }

    size_t expected_blocks = 0;
    switch (args.weight_format) {
        case Format::q4_h0:
        case Format::q4_h1:
        case Format::q4_hp1:
        case Format::q16_h0:
        case Format::q16_h1:
        case Format::q16_hp1:
            if (!args.has_native_matched_width_contract() ||
                !native_extent_covers(args, args.native_block_count,
                    args.weight_format == Format::q4_h0 ? sizeof(block_q4_h0) :
                    args.weight_format == Format::q4_h1 ? sizeof(block_q4_h1) :
                    args.weight_format == Format::q4_hp1 ? sizeof(block_q4_hp1) :
                    args.weight_format == Format::q16_h0 ? sizeof(block_q16_h0) :
                    args.weight_format == Format::q16_h1 ? sizeof(block_q16_h1) :
                                                          sizeof(block_q16_hp1))) {
                return WeightReaderStatus::InvalidMetadata;
            }
            return WeightReaderStatus::Success;

        case Format::q8_h0:
            if (args.K % kBlockSize != 0 || args.blocks_K != args.K / kBlockSize ||
                args.blocks_J != args.J ||
                !checked_mul_size(args.J, args.blocks_K, expected_blocks)) {
                return WeightReaderStatus::InvalidMetadata;
            }
            if (args.B_blocks != nullptr) {
                return reinterpret_cast<uintptr_t>(args.B_blocks) % alignof(block_q8_0) == 0 &&
                    native_extent_covers(args, expected_blocks, sizeof(block_q8_0)) ?
                    WeightReaderStatus::Success : WeightReaderStatus::InvalidMetadata;
            }
            return args.B != nullptr && args.B_scales != nullptr &&
                dense_extent_is_addressable(args, plan) ?
                WeightReaderStatus::Success : WeightReaderStatus::InvalidMetadata;

        case Format::q8_h1:
            if (!args.has_q8_h1_im2p_contract() ||
                !checked_mul_size(args.J, args.blocks_per_row, expected_blocks) ||
                !native_extent_covers(args, expected_blocks, sizeof(block_q8_h1))) {
                return WeightReaderStatus::InvalidMetadata;
            }
            return WeightReaderStatus::Success;

        case Format::q8_hp1:
            if (!wroute::has_q8_hp1_native_contract(args) ||
                !native_extent_covers(args, args.q8_hp1_block_count, sizeof(block_q8_hp1))) {
                return WeightReaderStatus::InvalidMetadata;
            }
            return WeightReaderStatus::Success;

        case Format::q8_0_unpacked_to_h1:
            return validate_dense_q8_h1(args, plan);

        default:
            return WeightReaderStatus::UnsupportedFormat;
    }
}

template <typename Block>
const Block *native_block(
    const Block *blocks,
    size_t block_count,
    size_t blocks_per_row,
    size_t row,
    size_t block)
{
    size_t offset = 0;
    return blocks != nullptr &&
        checked_block_offset(row, blocks_per_row, block, block_count, offset) ?
        blocks + offset : nullptr;
}

int32_t decode_q4(const uint8_t *quants, size_t index)
{
    const size_t packed_index = index % (kBlockSize / 2);
    const uint8_t packed = quants[packed_index];
    const uint8_t nibble = index < kBlockSize / 2 ?
        packed & 0x0fu : packed >> 4;
    return static_cast<int32_t>(nibble) - 8;
}

WeightScaleResult invalid_scale(WeightReaderStatus status)
{
    WeightScaleResult result{};
    result.status = status;
    return result;
}

WeightScaleResult make_h0_scale(float scale)
{
    if (!finite_float(scale))
        return invalid_scale(WeightReaderStatus::InvalidMetadata);
    WeightScaleResult result{};
    result.status = WeightReaderStatus::Success;
    result.domain = ScaleDomain::FloatingBlock;
    result.floating_block_scale = scale;
    return result;
}

WeightScaleResult make_h1_scale(uint64_t block_scale, float column_scale)
{
    if (!finite_float(column_scale))
        return invalid_scale(WeightReaderStatus::InvalidMetadata);
    WeightScaleResult result{};
    result.status = WeightReaderStatus::Success;
    result.domain = ScaleDomain::IntegerBlockTimesColumn;
    result.integer_block_scale = block_scale;
    result.column_scale = column_scale;
    return result;
}

WeightScaleResult make_hp1_scale(
    int16_t exponent,
    float column_scale,
    ScaleDomain domain)
{
    if (!finite_float(column_scale))
        return invalid_scale(WeightReaderStatus::InvalidMetadata);
    if (domain == ScaleDomain::FloatingBlock) {
        if (exponent == INT16_MIN)
            return make_h0_scale(0.0f);
        return make_h0_scale(std::ldexp(column_scale, static_cast<int>(exponent)));
    }
    if (domain != ScaleDomain::IntegerBlockTimesColumn)
        return invalid_scale(WeightReaderStatus::InvalidMetadata);
    if (exponent == INT16_MIN)
        return make_h1_scale(0, column_scale);
    if (exponent < 0)
        return invalid_scale(WeightReaderStatus::InvalidMetadata);
    if (exponent >= 63)
        return invalid_scale(WeightReaderStatus::ScaleOverflow);
    return make_h1_scale(uint64_t{1} << static_cast<unsigned>(exponent), column_scale);
}

WeightScaleResult read_scale_unchecked(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan,
    size_t j,
    size_t block_index)
{
    if (j >= args.J || plan.scales.block_size == 0 ||
        block_index >= plan.scales.cols) {
        return invalid_scale(WeightReaderStatus::InvalidArguments);
    }

    switch (args.weight_format) {
        case Format::q4_h0: {
            const block_q4_h0 *block = native_block(
                args.q4_h0_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            return block == nullptr ? invalid_scale(WeightReaderStatus::InvalidMetadata) :
                make_h0_scale(ggml_fp16_to_fp32(block->d));
        }
        case Format::q8_h0:
            if (args.B_blocks != nullptr) {
                size_t offset = 0;
                size_t block_count = 0;
                if (!checked_mul_size(args.blocks_J, args.blocks_K, block_count) ||
                    !checked_block_offset(j, args.blocks_K, block_index, block_count, offset)) {
                    return invalid_scale(WeightReaderStatus::InvalidMetadata);
                }
                return make_h0_scale(ggml_fp16_to_fp32(args.B_blocks[offset].d));
            }
            return args.B_scales == nullptr ? invalid_scale(WeightReaderStatus::InvalidMetadata) :
                make_h0_scale(args.B_scales[j * args.blocks_K + block_index]);
        case Format::q16_h0: {
            const block_q16_h0 *block = native_block(
                args.q16_h0_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            return block == nullptr ? invalid_scale(WeightReaderStatus::InvalidMetadata) :
                make_h0_scale(ggml_fp16_to_fp32(block->d));
        }
        case Format::q4_h1: {
            const block_q4_h1 *block = native_block(
                args.q4_h1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)) ||
                !zero_bytes(block->tail_padding, sizeof(block->tail_padding))) {
                return invalid_scale(WeightReaderStatus::InvalidMetadata);
            }
            return make_h1_scale(static_cast<uint64_t>(block->c_b) + block->R, block->s_rf);
        }
        case Format::q8_h1: {
            const block_q8_h1 *block = args.q8_h1_block(j, block_index);
            return block == nullptr ? invalid_scale(WeightReaderStatus::InvalidMetadata) :
                make_h1_scale(static_cast<uint64_t>(block->c_b) + block->R, block->s_rf);
        }
        case Format::q8_0_unpacked_to_h1: {
            const size_t index = j * args.blocks_per_row + block_index;
            const size_t stripe = args.stripe_J > 1 ? j / args.stripe_J : 0;
            const uint64_t offset = args.stripe_J > 1 ? args.R_stripe[stripe] : args.R[j];
            const float column = args.stripe_J > 1 ? args.s_rf_stripe[stripe] : args.s_rf[j];
            return make_h1_scale(static_cast<uint64_t>(args.c_b[index]) + offset, column);
        }
        case Format::q16_h1: {
            const block_q16_h1 *block = native_block(
                args.q16_h1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)) ||
                !zero_bytes(block->tail_padding, sizeof(block->tail_padding))) {
                return invalid_scale(WeightReaderStatus::InvalidMetadata);
            }
            return make_h1_scale(static_cast<uint64_t>(block->c_b) + block->R, block->s_rf);
        }
        case Format::q4_hp1: {
            const block_q4_hp1 *block = native_block(
                args.q4_hp1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)))
                return invalid_scale(WeightReaderStatus::InvalidMetadata);
            return make_hp1_scale(block->m, block->channel_scale, plan.scale_domain);
        }
        case Format::q8_hp1: {
            const block_q8_hp1 *block = args.q8_hp1_block(j, block_index);
            if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)))
                return invalid_scale(WeightReaderStatus::InvalidMetadata);
            return make_hp1_scale(block->m, block->channel_scale, plan.scale_domain);
        }
        case Format::q16_hp1: {
            const block_q16_hp1 *block = native_block(
                args.q16_hp1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)))
                return invalid_scale(WeightReaderStatus::InvalidMetadata);
            return make_hp1_scale(block->m, block->channel_scale, plan.scale_domain);
        }
        default:
            return invalid_scale(WeightReaderStatus::UnsupportedFormat);
    }
}

}

WeightReaderStatus validate(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan)
{
    const WeightReaderStatus storage = validate_storage(args, plan);
    if (storage != WeightReaderStatus::Success)
        return storage;
    if (plan.scales.cols == 0 || plan.scales.block_size != kBlockSize)
        return WeightReaderStatus::InvalidMetadata;

    for (size_t j = 0; j < args.J; ++j) {
        for (size_t block = 0; block < plan.scales.cols; ++block) {
            const WeightScaleResult scale = read_scale_unchecked(args, plan, j, block);
            if (!scale.ok())
                return scale.status;
        }
    }
    return WeightReaderStatus::Success;
}

WeightCodeResult read_code(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan,
    size_t j,
    size_t k)
{
    WeightCodeResult result{};
    const WeightReaderStatus storage = validate_storage(args, plan);
    if (storage != WeightReaderStatus::Success) {
        result.status = storage;
        return result;
    }
    if (j >= args.J || k >= args.K) {
        result.status = WeightReaderStatus::InvalidArguments;
        return result;
    }
    result.status = WeightReaderStatus::Success;

    if (plan.route == Route::Dense ||
        plan.route == Route::Q8ChannelDirect ||
        plan.route == Route::Q8ChannelSidecar) {
        const size_t offset = plan.layout == wroute::WeightLayout::JxK_ColMajor ?
            j * plan.weight_stride + k : k * plan.weight_stride + j;
        result.value = reinterpret_cast<const int8_t *>(args.B)[offset];
        return result;
    }

    const size_t block_index = k / kBlockSize;
    const size_t local_index = k % kBlockSize;
    switch (args.weight_format) {
        case Format::q4_h0: {
            const block_q4_h0 *block = native_block(
                args.q4_h0_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = decode_q4(block->qs, local_index);
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q4_h1: {
            const block_q4_h1 *block = native_block(
                args.q4_h1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = decode_q4(block->qs, local_index);
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q4_hp1: {
            const block_q4_hp1 *block = native_block(
                args.q4_hp1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = decode_q4(block->qs, local_index);
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q8_h0:
            if (args.B_blocks != nullptr) {
                size_t offset = 0;
                size_t count = 0;
                if (checked_mul_size(args.blocks_J, args.blocks_K, count) &&
                    checked_block_offset(j, args.blocks_K, block_index, count, offset)) {
                    result.value = args.B_blocks[offset].qs[local_index];
                } else {
                    result.status = WeightReaderStatus::InvalidMetadata;
                }
            } else {
                const size_t offset = plan.layout == wroute::WeightLayout::JxK_ColMajor ?
                    j * plan.weight_stride + k : k * plan.weight_stride + j;
                result.value = reinterpret_cast<const int8_t *>(args.B)[offset];
            }
            break;
        case Format::q8_h1: {
            const block_q8_h1 *block = args.q8_h1_block(j, block_index);
            if (block != nullptr) result.value = block->qs[local_index];
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q8_0_unpacked_to_h1: {
            const size_t offset = plan.layout == wroute::WeightLayout::JxK_ColMajor ?
                j * plan.weight_stride + k : k * plan.weight_stride + j;
            result.value = reinterpret_cast<const int8_t *>(args.B)[offset];
            break;
        }
        case Format::q8_hp1: {
            const block_q8_hp1 *block = args.q8_hp1_block(j, block_index);
            if (block != nullptr) result.value = block->qs[local_index];
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q16_h0: {
            const block_q16_h0 *block = native_block(
                args.q16_h0_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = block->qs[local_index];
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q16_h1: {
            const block_q16_h1 *block = native_block(
                args.q16_h1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = block->qs[local_index];
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        case Format::q16_hp1: {
            const block_q16_hp1 *block = native_block(
                args.q16_hp1_blocks, args.native_block_count,
                args.native_blocks_per_row, j, block_index);
            if (block != nullptr) result.value = block->qs[local_index];
            else result.status = WeightReaderStatus::InvalidMetadata;
            break;
        }
        default:
            result.status = WeightReaderStatus::UnsupportedFormat;
            return result;
    }

    return result;
}

WeightScaleResult read_scale(
    const ggml_gemmini_args_t &args,
    const wroute::WeightRoutePlan &plan,
    size_t j,
    size_t block_index)
{
    const WeightReaderStatus storage = validate_storage(args, plan);
    return storage == WeightReaderStatus::Success ?
        read_scale_unchecked(args, plan, j, block_index) : invalid_scale(storage);
}

const char *weight_reader_status_name(WeightReaderStatus status)
{
    switch (status) {
        case WeightReaderStatus::Success:           return "success";
        case WeightReaderStatus::InvalidArguments:  return "invalid-arguments";
        case WeightReaderStatus::InvalidMetadata:   return "invalid-metadata";
        case WeightReaderStatus::ScaleOverflow:     return "scale-overflow";
        case WeightReaderStatus::UnsupportedFormat: return "unsupported-format";
    }
    return "unsupported-format";
}

}
