#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void *gemmini_cycle_sim_context_enter(const void *node_key);
void gemmini_cycle_sim_context_exit(void *scope);
#ifdef __cplusplus
}
#endif
