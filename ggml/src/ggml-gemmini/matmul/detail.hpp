#pragma once

#include "types.hpp"

namespace ggml::gemmini::detail {

struct MatmulCycleDrain {
    ~MatmulCycleDrain();
};

void record_matmul_cpu_wall(const MatmulCpuSample & start, const MatmulCpuSample & end);
void emit_matmul_cpu_interval(const char *              layer,
                              const char *              op,
                              const MatmulCpuSample &   start,
                              const MatmulCpuSample &   end,
                              bool                      operation_success,
                              const MatmulJobMetrics *  profile           = nullptr,
                              const MatmulCpuInterval * explicit_interval = nullptr,
                              std::optional<uint64_t>   invocation_run_id = {}) noexcept;
void emit_rmd_stripe_metrics(const std::string &              layer,
                             const MatmulJobMetrics &         profile,
                             RmdBackend                       backend,
                             bool                             success,
                             const char *                     reason,
                             const rmd::RmdExecutionMetrics * metrics) noexcept;

} // namespace ggml::gemmini::detail

namespace ggml::gemmini::test_detail {

void observe_execution_construction();
void observe_allocation_attempt();
void observe_residual_dispatch();
void observe_backend_dispatch(bool fallback);

} // namespace ggml::gemmini::test_detail
