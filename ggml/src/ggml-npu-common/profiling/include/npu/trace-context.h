#pragma once
#include <stdint.h>
#ifdef __cplusplus
#define GEMMINI_TRACE_NOEXCEPT noexcept
extern "C" {
#else
#define GEMMINI_TRACE_NOEXCEPT
#endif

enum gemmini_trace_flags {
    GEMMINI_TRACE_CAPTURED = 1u << 0,
    GEMMINI_TRACE_BOUND = 1u << 1,
    GEMMINI_TRACE_OPERATOR = 1u << 2,
    GEMMINI_TRACE_WORKER = 1u << 3,
    GEMMINI_TRACE_MULTITASK = 1u << 4,
};
enum gemmini_trace_role {
    GEMMINI_TRACE_ROLE_OPERATOR = 0,
    GEMMINI_TRACE_ROLE_DENSE = 1,
    GEMMINI_TRACE_ROLE_RESIDUAL = 2,
};
/* Value-owned metadata; no pointers into a request, tensor, queue slot or TLS.
 * Captured-empty is distinct from absent: a late sample must not acquire the
 * identity of whatever request happens to be active when it is serialized. */
typedef struct gemmini_trace_context {
    uint64_t request_id, inference_operation_id;
    uint64_t graph_id, operator_id, node_id;
    uint64_t task_id, parent_task_id, span_id, parent_span_id, worker_id;
    uint32_t flags;
    uint8_t phase, role;
    char operator_name[32];
} gemmini_trace_context;

gemmini_trace_context gemmini_trace_capture(void) GEMMINI_TRACE_NOEXCEPT;
gemmini_trace_context gemmini_trace_bind(gemmini_trace_context context) GEMMINI_TRACE_NOEXCEPT;
void gemmini_trace_restore(gemmini_trace_context previous) GEMMINI_TRACE_NOEXCEPT;
uint64_t gemmini_trace_reserve_ids(uint64_t count) GEMMINI_TRACE_NOEXCEPT;
gemmini_trace_context gemmini_trace_fork(gemmini_trace_context parent) GEMMINI_TRACE_NOEXCEPT;
gemmini_trace_context gemmini_trace_operator(gemmini_trace_context parent,
    uint64_t graph_id, uint64_t operator_id, uint64_t node_id,
    const char *operator_name, uint64_t worker_id, int worker_present,
    int multitask) GEMMINI_TRACE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef GEMMINI_TRACE_NOEXCEPT
