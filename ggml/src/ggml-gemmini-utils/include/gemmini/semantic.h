#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void *gemmini_semantic_enter(const void *node, uint64_t workers);
void gemmini_semantic_exit(void *scope);
#ifdef __cplusplus
}
#endif
