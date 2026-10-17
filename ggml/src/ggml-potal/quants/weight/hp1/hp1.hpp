#pragma once

// Q8_HP1 weights -> PoTal operands, on the weight reader copied from Gemmini
// (quants/common/weight_reader, hp1_scu).
//
//   block_q8_hp1 (ggml-common.h, QK8_HP = 32): qs[32] int8, m int16, padding[2], channel_scale f32
//   value = qs * channel_scale * 2^m          (dequantize_row_q8_hp1; m = INT16_MIN -> 0)
//
//   qs             -> B[k][n] int8 payload           (wreader::read_code_validated)
//   m              -> SCU carrier                     (wreader::read_hp1_carrier_validated)
//   channel_scale  -> F32 reconstruction, per column  (wreader::read_column_scale_validated)
//
// Beyond the reader's block checks (padding, finite channel_scale, m >= 0 or
// INT16_MIN), PoTal reconstructs with one scale per output column, so every
// block of a row must carry the same channel_scale bits (the rule Gemmini's
// rmd-compose enforces), the scale must not be negative, and a row with scale
// 0 may only hold zero blocks. Nothing is dequantized and requantized.

#include "../../common/weight_reader.hpp"

#include <cstddef>
#include <cstdint>

struct ggml_potal_args_t;

namespace ggml::potal::quants::weight::hp1 {

constexpr uint32_t block_k = 32; // QK8_HP, the K extent of one scale block

enum class status {
    ok,
    invalid,        // null output, stride too small
    metadata,       // wreader::validate() failed (see the reader status)
    channel_scale,  // negative, or 0 in a row with a non-zero block
    scale_mismatch, // blocks of one row carry different channel_scale bits
};

const char * status_name(status s);

struct stats {
    uint64_t blocks       = 0;
    uint64_t zero_blocks  = 0; // m = INT16_MIN (zero carrier)
    uint64_t zero_rows    = 0; // channel_scale == 0
    int32_t  min_m        = 0; // over non-zero blocks
    int32_t  max_m        = 0;
    double   sum_m        = 0; // for the mean over non-zero blocks
    float    min_channel_scale = 0.0f; // over rows with a non-zero scale
    float    max_channel_scale = 0.0f;
};

// args: J = N rows (output columns), K, Q8_HP1 view set. Writes
//   B[k * ldb + n]                         = qs
//   carriers[(k / block_k) * cstride + n]  = carrier(m)
//   channel_scale[n]                       = the row's channel_scale
// Outputs are partially written when the status is not ok.
status unpack(const ggml_potal_args_t & args, int8_t * B, size_t ldb, uint32_t * carriers, size_t cstride,
              float * channel_scale, stats * st, wreader::WeightReaderStatus * reader = nullptr);

} // namespace ggml::potal::quants::weight::hp1
