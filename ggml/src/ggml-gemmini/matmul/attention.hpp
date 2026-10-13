#pragma once

#include "ggml-gemmini.h"
#include "../residual/rmd/rmd-executor.hpp"

namespace ggml::gemmini {

struct AttentionMetrics {
    uint64_t qk_nodes = 0, pv_nodes = 0;
    uint64_t main_dots = 0, residual_dots = 0;
    uint64_t main_fragments = 0, residual_fragments = 0, add_ops = 0;
    rmd::RmdProviderStats provider{};
};

bool attention_supported(const ggml_tensor * op);
// Signed INT4 GEMMs use the post-array SCU and INT32 accumulation across K;
// the host applies only radix weights and the final stripe/column scale.
// Both output and metrics are published only after the whole node succeeds.
rmd::RmdStatus attention_execute(ggml_tensor * op, AttentionMetrics & metrics);

} // namespace ggml::gemmini
