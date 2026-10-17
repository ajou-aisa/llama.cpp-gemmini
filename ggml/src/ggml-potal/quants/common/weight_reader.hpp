#pragma once

// Copied from ggml-gemmini/quants/common/weight_reader.hpp for the one format
// PoTal executes, Q8_HP1. Gemmini resolves a WeightRoutePlan per format; here
// the route is fixed (HP1, SCU carriers, JxK row-major blocks), so the plan
// argument is gone and validate() checks the storage contract itself.

#include <cstddef>
#include <cstdint>

struct ggml_potal_args_t;

namespace ggml::potal::quants::wreader {
enum class WeightReaderStatus {
    Success,
    InvalidArguments,
    InvalidMetadata,
    ScaleOverflow,
    UnsupportedFormat,
};

struct WeightCodeResult {
    WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
    int32_t            value  = 0;

    bool ok() const {
        return status == WeightReaderStatus::Success;
    }
};

struct Hp1CarrierResult {
    WeightReaderStatus status       = WeightReaderStatus::UnsupportedFormat;
    uint32_t           carrier      = 0;
    float              column_scale = 1.0f;
    bool               ok() const {
        return status == WeightReaderStatus::Success;
    }
};

// Original Q8_HP1 block metadata, never integer-factor -> log2.
// Requires validate() == Success on the same args.
Hp1CarrierResult read_hp1_carrier_validated(const ggml_potal_args_t & args, size_t column, size_t original_block);

// Column metadata for final reconstruction; this API cannot return an
// integer factor, so carrier-domain consumers cannot accidentally rescale.
struct ColumnScaleResult {
    WeightReaderStatus status       = WeightReaderStatus::UnsupportedFormat;
    float              column_scale = 1.0f;
    bool               ok() const {
        return status == WeightReaderStatus::Success;
    }
};
ColumnScaleResult read_column_scale_validated(const ggml_potal_args_t & args, size_t column, size_t original_block);

// Storage contract (ggml_potal_args_t::has_q8_hp1_contract), then every
// block's carrier metadata (padding, finite channel_scale, m -> carrier).
WeightReaderStatus validate(const ggml_potal_args_t & args);

// Requires validate() == Success on the same args. Raw int8 code, no shift applied.
WeightCodeResult read_code_validated(const ggml_potal_args_t & args, size_t j, size_t k);

const char * weight_reader_status_name(WeightReaderStatus status);
} // namespace ggml::potal::quants::wreader
