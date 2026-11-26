#pragma once
#include "../../ggml-npu-common/profiling/matmul-detail.hpp"

namespace ggml::gemmini::detail { using namespace ggml::npu::detail; }

namespace ggml::gemmini::test_detail {

void observe_execution_construction();
void observe_allocation_attempt();
void observe_residual_dispatch();
void observe_backend_dispatch(bool fallback);

} // namespace ggml::gemmini::test_detail
