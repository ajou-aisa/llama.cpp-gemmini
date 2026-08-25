#pragma once

#include "weight_route.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::quants::wreader
{
    enum class WeightReaderStatus
    {
        Success,
        InvalidArguments,
        InvalidMetadata,
        ScaleOverflow,
        UnsupportedFormat,
    };

    struct WeightCodeResult
    {
        WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
        int32_t value = 0;

        bool ok() const { return status == WeightReaderStatus::Success; }
    };

    struct WeightScaleResult
    {
        WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
        wroute::WeightScaleDomain domain = wroute::WeightScaleDomain::None;
        uint64_t integer_block_scale = 1;
        float column_scale = 1.0f;
        float floating_block_scale = 1.0f;

        bool ok() const { return status == WeightReaderStatus::Success; }
    };

    // Compact native transport consumes adjacent logical values from each
    // byte: low nibble first, then high nibble, as signed two's-complement
    // INT4. This is not the frontend's GGUF split-half model layout.
    bool native_mvin_q4_position(
        size_t logical_count,
        size_t index,
        size_t &byte_index,
        uint8_t &shift) noexcept;

    bool decode_native_mvin_q4(
        const uint8_t *packed,
        size_t packed_size,
        size_t logical_count,
        size_t index,
        int8_t &value) noexcept;

    WeightReaderStatus validate(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan);

    WeightCodeResult read_code(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t k);

    // Requires a plan returned by resolve_weight_route_plan. The plan already
    // validated immutable weight storage, so hot loops must not repeat it.
    WeightCodeResult read_code_validated(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t k);

    WeightScaleResult read_scale(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t block_index);

    WeightScaleResult read_scale_validated(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t block_index);

    const char *weight_reader_status_name(WeightReaderStatus status);

#if defined(GGML_GEMMINI_TESTING)
    void test_reset_weight_reader_counters();
    size_t test_weight_reader_storage_validations();
#endif
}
