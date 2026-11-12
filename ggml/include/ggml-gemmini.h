#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <string.h>

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

struct ggml_gemmini_kv_cache_info {
    uint64_t instance, epoch, revision;
    uint32_t layer, begin, end;
};

static inline void ggml_gemmini_set_attention_kv_cache(
        struct ggml_tensor * tensor, const struct ggml_gemmini_kv_cache_info * info) {
    GGML_ASSERT(tensor->op == GGML_OP_MUL_MAT);
    tensor->op_params[2] = 0x504b5631;
    memcpy(&tensor->op_params[3], &info->instance, sizeof(info->instance));
    memcpy(&tensor->op_params[5], &info->epoch, sizeof(info->epoch));
    memcpy(&tensor->op_params[7], &info->revision, sizeof(info->revision));
    memcpy(&tensor->op_params[9], &info->layer, sizeof(info->layer));
    memcpy(&tensor->op_params[10], &info->begin, sizeof(info->begin));
    memcpy(&tensor->op_params[11], &info->end, sizeof(info->end));
    memset(&tensor->op_params[12], 0, 4 * sizeof(tensor->op_params[0]));
}

static inline bool ggml_gemmini_get_attention_kv_cache(
        const struct ggml_tensor * tensor, struct ggml_gemmini_kv_cache_info * info) {
    if (!tensor || tensor->op != GGML_OP_MUL_MAT || tensor->op_params[2] != 0x504b5631) {
        return false;
    }
    for (int i = 12; i < 16; ++i) {
        if (tensor->op_params[i] != 0) return false;
    }
    struct ggml_gemmini_kv_cache_info value;
    memcpy(&value.instance, &tensor->op_params[3], sizeof(value.instance));
    memcpy(&value.epoch, &tensor->op_params[5], sizeof(value.epoch));
    memcpy(&value.revision, &tensor->op_params[7], sizeof(value.revision));
    memcpy(&value.layer, &tensor->op_params[9], sizeof(value.layer));
    memcpy(&value.begin, &tensor->op_params[10], sizeof(value.begin));
    memcpy(&value.end, &tensor->op_params[11], sizeof(value.end));
    if (!value.instance || !value.epoch || !value.revision || value.begin >= value.end) {
        return false;
    }
    *info = value;
    return true;
}

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
