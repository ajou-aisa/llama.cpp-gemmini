// Copied from ggml-gemmini/quants/common/weight_reader.cpp, Q8_HP1 branches only.

#include "weight_reader.hpp"
#include "hp1_scu.hpp"

#include "../../ggml-potal-args.h"

#include <cstring>

namespace ggml::potal::quants::wreader {

namespace {

constexpr size_t kBlockSize = QK8_HP;

bool finite_float(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint32_t observed = bits;
    return (observed & 0x7f800000u) != 0x7f800000u;
}

bool zero_bytes(const uint8_t * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != 0)
            return false;
    }
    return true;
}

} // namespace

Hp1CarrierResult read_hp1_carrier_validated(const ggml_potal_args_t & args, size_t column, size_t original_block) {
    Hp1CarrierResult result;
    if (args.weight_format != ggml_potal_args_t::weight_format_t::q8_hp1)
        return result;
    result.status = WeightReaderStatus::InvalidMetadata;
    if (column >= args.J || args.block_size_k != kBlockSize || original_block >= args.q8_hp1_blocks_per_row)
        return result;
    const block_q8_hp1 * block = args.q8_hp1_block(column, original_block);
    if (block == nullptr || !zero_bytes(block->padding, sizeof(block->padding)) ||
        !finite_float(block->channel_scale) || !hp1::encode_carrier(block->m, result.carrier))
        return result;
    result.column_scale = block->channel_scale;
    result.status       = WeightReaderStatus::Success;
    return result;
}

ColumnScaleResult read_column_scale_validated(const ggml_potal_args_t & args, size_t column, size_t original_block) {
    const auto value = read_hp1_carrier_validated(args, column, original_block);
    return { value.status, value.column_scale };
}

WeightReaderStatus validate(const ggml_potal_args_t & args) {
    if (args.weight_format != ggml_potal_args_t::weight_format_t::q8_hp1)
        return WeightReaderStatus::UnsupportedFormat;
    if (!args.has_q8_hp1_contract() || args.block_size_k != kBlockSize)
        return WeightReaderStatus::InvalidMetadata;

    for (size_t j = 0; j < args.J; ++j) {
        for (size_t block = 0; block < args.q8_hp1_blocks_per_row; ++block) {
            const auto carrier = read_hp1_carrier_validated(args, j, block);
            if (!carrier.ok())
                return carrier.status;
        }
    }
    return WeightReaderStatus::Success;
}

WeightCodeResult read_code_validated(const ggml_potal_args_t & args, size_t j, size_t k) {
    WeightCodeResult result{};
    if (j >= args.J || k >= args.K) {
        result.status = WeightReaderStatus::InvalidArguments;
        return result;
    }
    result.status = WeightReaderStatus::Success;

    const size_t         block_index = k / kBlockSize;
    const size_t         local_index = k % kBlockSize;
    const block_q8_hp1 * block       = args.q8_hp1_block(j, block_index);
    if (block != nullptr)
        result.value = block->qs[local_index];
    else
        result.status = WeightReaderStatus::InvalidMetadata;
    return result;
}

const char * weight_reader_status_name(WeightReaderStatus status) {
    switch (status) {
    case WeightReaderStatus::Success:
        return "success";
    case WeightReaderStatus::InvalidArguments:
        return "invalid_arguments";
    case WeightReaderStatus::InvalidMetadata:
        return "invalid_metadata";
    case WeightReaderStatus::ScaleOverflow:
        return "scale_overflow";
    case WeightReaderStatus::UnsupportedFormat:
        return "unsupported_format";
    }
    return "unknown";
}

} // namespace ggml::potal::quants::wreader
