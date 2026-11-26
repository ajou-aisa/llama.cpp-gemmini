/*
 * Gemmini standalone NPU API (C11, usable from C++).
 *
 * Layers: gemmini_isa.h (RTL constants), gemmini_command.h (descriptors,
 * pure encoders, geometry validation), gemmini_transport.h (issue/drain:
 * RoCC and recording). No GGML, llama.cpp or external SDK dependency.
 */
#ifndef GEMMINI_API_H
#define GEMMINI_API_H

#include "gemmini_command.h"
#include "gemmini_isa.h"
#include "gemmini_transport.h"
#include "gemmini_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Counter helpers (counter.c); value is the full 64-bit response. */
gemmini_status_t gemmini_counter_read(const gemmini_transport_t *transport, uint32_t index, uint64_t *value);
gemmini_status_t gemmini_counter_configure(const gemmini_transport_t *transport, uint32_t index, uint32_t event);
gemmini_status_t gemmini_counter_snapshot_take(const gemmini_transport_t *transport);
gemmini_status_t gemmini_counter_snapshot_reset(const gemmini_transport_t *transport);
gemmini_status_t gemmini_counter_reset(const gemmini_transport_t *transport);

#ifdef __cplusplus
}
#endif

#endif /* GEMMINI_API_H */
