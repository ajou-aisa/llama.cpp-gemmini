#pragma once

#include "direct-types.hpp"
#include "../rmd/rmd-types.hpp"

#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::residual {

struct DirectExecutionMetrics {
    size_t event_count = 0;
    size_t call_count = 0;
};

// Computes one immutable direct payload into a staged row-major [row_count, J]
// correction. The caller's correction is replaced only after complete success.
rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     std::vector<rmd::OutputValue> & correction,
                                     DirectExecutionMetrics * metrics = nullptr);

}
