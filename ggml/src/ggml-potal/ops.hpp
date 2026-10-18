#pragma once

// GGML op validation and dispatch of the POTAL backend.
//
//   MUL_MAT I8 x I8 -> F32, M = N = 16, K = 16|32   validated single GEMM (matmul/pack.hpp)
//   MUL_MAT F32 x Q8_HP1 -> F32, any 2-D shape      tiled PoTal GEMM (matmul/dense.hpp)
//   NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE             metadata only
//
// Everything else is rejected by supports_op and runs on another backend.

#include "ggml.h"
#include "ggml-potal.h"
#include "matmul/weight_cache.hpp"

#include <cstdint>
#include <string>

namespace ggml::potal {

// Per backend instance.
struct backend_state {
    ggml_backend_potal_stats stats = {};
    std::string              model_arch; // device init params, used for layer names
    weight_cache             weights;    // prepared Q8_HP1 weights, freed with the backend
};

// supports_op contract for any op.
ggml_potal_reject check_op(const ggml_tensor * op);

// graph_compute contract: check_op plus allocated data.
ggml_potal_reject check_compute(const ggml_tensor * node);

// Compute one MUL_MAT node that passed check_compute. Never computes on the
// CPU: a request the hardware does not complete fails and its dst is NaN.
ggml_status compute_mul_mat(backend_state & state, ggml_tensor * node, int node_index);

// Called by supports_op for every MUL_MAT answer (device-wide CPU-fallback accounting).
void note_supports_mul_mat(const ggml_tensor * op, ggml_potal_reject reason);

// MUL_MAT supports_op rejections so far (the scheduler asks repeatedly per node).
uint64_t supports_mul_mat_rejected();

bool is_view_op(ggml_op op);

} // namespace ggml::potal
