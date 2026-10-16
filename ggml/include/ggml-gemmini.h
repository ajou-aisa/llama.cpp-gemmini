#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_gemmini_init(void);

// GGML_BACKEND_API bool ggml_backend_is_gemmini(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gemmini_reg(void);

struct ggml_gemmini_attention_stats {
    uint64_t qk_nodes, pv_nodes, main_dots, residual_dots;
    uint64_t main_fragments, residual_fragments, add_ops, raw_cycles;
};
GGML_BACKEND_API bool ggml_backend_gemmini_get_attention_stats(
    ggml_backend_t backend, struct ggml_gemmini_attention_stats * stats);

enum ggml_gemmini_attention_role {
    GGML_GEMMINI_ATTENTION_NONE = 0,
    GGML_GEMMINI_ATTENTION_QK = 0x5054514b,
    GGML_GEMMINI_ATTENTION_PV = 0x50545056,
};

// MUL_MAT precision owns op_params[0]. The role survives scheduler tensor copies.
static inline void ggml_gemmini_set_attention_role(struct ggml_tensor * tensor,
                                                  enum ggml_gemmini_attention_role role) {
    GGML_ASSERT(tensor->op == GGML_OP_MUL_MAT);
    tensor->op_params[1] = (int32_t) role;
}

static inline enum ggml_gemmini_attention_role ggml_gemmini_get_attention_role(const struct ggml_tensor * tensor) {
    if (tensor && tensor->op == GGML_OP_MUL_MAT) {
        if (tensor->op_params[1] == GGML_GEMMINI_ATTENTION_QK) return GGML_GEMMINI_ATTENTION_QK;
        if (tensor->op_params[1] == GGML_GEMMINI_ATTENTION_PV) return GGML_GEMMINI_ATTENTION_PV;
    }
    return GGML_GEMMINI_ATTENTION_NONE;
}

#ifdef __cplusplus
}
#endif
