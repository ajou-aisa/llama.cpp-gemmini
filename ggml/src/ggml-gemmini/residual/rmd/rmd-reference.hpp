#pragma once

#include "rmd-types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {

struct ReferenceResidual {
    uint32_t local_row;
    uint32_t k;      // original K
    int32_t residual;
};

// Direct wide residual matmul in the block-scaled INT64 domain, without any radix
// decomposition. This is the parity oracle for the packet/executor/composer chain.
RmdStatus reference_direct_correction(const ggml_gemmini_args_t & args,
                                      size_t row_count,
                                      const std::vector<ReferenceResidual> & residuals,
                                      std::vector<OutputValue> & correction);

// Same result, reconstructed through the route's balanced radix 16, 256, or 65536
// digits. Used to prove decomposition is exact independently of packing and tiling.
RmdStatus reference_rmd_correction(const ggml_gemmini_args_t & args,
                                   size_t row_count,
                                   const std::vector<ReferenceResidual> & residuals,
                                   std::vector<OutputValue> & correction);

}
