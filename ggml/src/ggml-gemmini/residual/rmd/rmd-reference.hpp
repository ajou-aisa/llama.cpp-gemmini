#pragma once

#include "rmd-types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {

struct ReferenceResidual {
  uint32_t local_row;
  uint32_t k; // original K
  int32_t residual;
};

// Independent HP1 numerical oracle over validated packet grouping/order. It
// computes exact raw products per hardware fragment, applies SCU then Sat32
// add, and only then recomposes radix lanes using checked __int128.
RmdStatus reference_hp1_packet_correction(const ggml_gemmini_args_t &,
                                          const StripePacket &,
                                          std::vector<OutputValue> &);

// Explicit CPU-direct legacy oracle, not the HP1 NPU saturation oracle.
// Direct wide residual matmul in the block-scaled INT64 domain, without any
// radix decomposition. It intentionally remains separate from HP1 NPU
// semantics.
RmdStatus
reference_direct_correction(const ggml_gemmini_args_t &args, size_t row_count,
                            const std::vector<ReferenceResidual> &residuals,
                            std::vector<OutputValue> &correction);

// HP1 A4/A8 uses the hardware-equivalent fragment oracle. Other explicit
// reference routes retain their historical decomposition-only semantics.
RmdStatus
reference_rmd_correction(const ggml_gemmini_args_t &args, size_t row_count,
                         const std::vector<ReferenceResidual> &residuals,
                         std::vector<OutputValue> &correction);

} // namespace ggml::gemmini::rmd
