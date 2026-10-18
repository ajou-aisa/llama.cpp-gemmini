#pragma once

// PoTal backend diagnostics: one record per event in the existing Gemmini
// debug JSONL (ggml-gemmini-utils: $GEMMINI_LOG_DIR/debug-log.jsonl, else
// ./output/log/debug-log.jsonl), {"layer": <layer>, "msg": <text>}. Routine
// backend state never reaches stdout/stderr.
//
// Layers: potal.backend, potal.issuer, potal.tlb, potal.matmul,
// potal.supports_op, potal.summary; per-node records use the resolved matmul layer.

#include "ggml.h"

namespace ggml::potal::log {

// Default outputs of the shared logger (setup_default_outputs, once per process;
// an output a tool already set, e.g. --gemmini-debug-log, is kept). Called by
// every record below, so no PoTal record precedes the tool's own log setup
// unless the tool logs nothing before using the backend.
void ensure_outputs();

// One debug JSONL record (no-op when built with LOG_DEBUG=0).
void debug(const char * layer, const char * fmt, ...) GGML_ATTRIBUTE_FORMAT(2, 3);

// A failure record. With LOG_DEBUG=0 there is no debug sink: the same text goes
// to the GGML error log instead, so a failure is never silent. The caller still
// returns its failure status.
void failure(const char * layer, const char * fmt, ...) GGML_ATTRIBUTE_FORMAT(2, 3);

} // namespace ggml::potal::log
