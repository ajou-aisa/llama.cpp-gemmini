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

    WeightReaderStatus validate(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan);

    WeightCodeResult read_code(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t k);

    WeightScaleResult read_scale(
        const ggml_gemmini_args_t &args,
        const wroute::WeightRoutePlan &plan,
        size_t j,
        size_t block_index);

    const char *weight_reader_status_name(WeightReaderStatus status);
}
