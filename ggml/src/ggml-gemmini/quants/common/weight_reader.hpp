#pragma once

#include "weight_route.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::quants::wreader {
enum class WeightReaderStatus {
  Success,
  InvalidArguments,
  InvalidMetadata,
  ScaleOverflow,
  UnsupportedFormat,
};

struct WeightCodeResult {
  WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
  int32_t value = 0;

  bool ok() const { return status == WeightReaderStatus::Success; }
};

struct WeightScaleResult {
  WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
  wroute::WeightScaleDomain domain = wroute::WeightScaleDomain::None;
  uint64_t integer_block_scale = 1;
  float column_scale = 1.0f;
  float floating_block_scale = 1.0f;

  bool ok() const { return status == WeightReaderStatus::Success; }
};

struct Hp1CarrierResult {
  WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
  uint32_t carrier = 0;
  float column_scale = 1.0f;
  bool ok() const { return status == WeightReaderStatus::Success; }
};

// Original q4/q8/q16 HP1 block metadata, never integer-factor -> log2.
// The plan must have validated immutable storage; metadata is checked here.
Hp1CarrierResult read_hp1_carrier_validated(const ggml_gemmini_args_t &,
                                            const wroute::WeightRoutePlan &,
                                            size_t column,
                                            size_t original_block);

// Column metadata for final reconstruction; this API cannot return an
// integer factor, so carrier-domain consumers cannot accidentally rescale.
struct ColumnScaleResult {
  WeightReaderStatus status = WeightReaderStatus::UnsupportedFormat;
  float column_scale = 1.0f;
  bool ok() const { return status == WeightReaderStatus::Success; }
};
ColumnScaleResult read_column_scale_validated(const ggml_gemmini_args_t &,
                                              const wroute::WeightRoutePlan &,
                                              size_t column,
                                              size_t original_block);

// Compact native transport consumes adjacent logical values from each
// byte: low nibble first, then high nibble, as signed two's-complement
// INT4. This is not the frontend's GGUF split-half model layout.
bool native_mvin_q4_position(size_t logical_count, size_t index,
                             size_t &byte_index, uint8_t &shift) noexcept;

bool decode_native_mvin_q4(const uint8_t *packed, size_t packed_size,
                           size_t logical_count, size_t index,
                           int8_t &value) noexcept;

WeightReaderStatus validate(const ggml_gemmini_args_t &args,
                            const wroute::WeightRoutePlan &plan);

WeightCodeResult read_code(const ggml_gemmini_args_t &args,
                           const wroute::WeightRoutePlan &plan, size_t j,
                           size_t k);

// Requires a plan returned by resolve_weight_route_plan. The plan already
// validated immutable weight storage, so hot loops must not repeat it.
WeightCodeResult read_code_validated(const ggml_gemmini_args_t &args,
                                     const wroute::WeightRoutePlan &plan,
                                     size_t j, size_t k);

// Same plan lifetime as read_code_validated. Values are DIM-by-DIM caller
// scratch; native H1/HP1 resolves one block per column, others use checked
// reads.
WeightReaderStatus read_code_tile_validated(const ggml_gemmini_args_t &args,
                                            const wroute::WeightRoutePlan &plan,
                                            size_t block_index,
                                            const uint16_t *local_k,
                                            size_t valid_k, size_t col_base,
                                            size_t valid_cols, int32_t *values,
                                            size_t &address_resolutions);

WeightScaleResult read_scale(const ggml_gemmini_args_t &args,
                             const wroute::WeightRoutePlan &plan, size_t j,
                             size_t block_index);

WeightScaleResult read_scale_validated(const ggml_gemmini_args_t &args,
                                       const wroute::WeightRoutePlan &plan,
                                       size_t j, size_t block_index);

const char *weight_reader_status_name(WeightReaderStatus status);

#if defined(GGML_GEMMINI_TESTING)
void test_reset_weight_reader_counters();
size_t test_weight_reader_storage_validations();
size_t test_weight_reader_code_address_resolutions();
#endif
} // namespace ggml::gemmini::quants::wreader
