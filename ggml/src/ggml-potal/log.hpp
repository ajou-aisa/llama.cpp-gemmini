#pragma once

// PoTal backend diagnostics: one record per event in the existing Gemmini
// debug JSONL (ggml-gemmini-utils: $GEMMINI_LOG_DIR/debug-log.jsonl, else
// ./output/log/debug-log.jsonl), {"layer": <layer>, "msg": <text>}. Routine
// backend state never reaches stdout/stderr.
//
// Layers: potal.backend, potal.issuer, potal.tlb, potal.matmul,
// potal.supports_op, potal.summary; per-node records use the resolved matmul layer.

#include "ggml.h"

#include <string>

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

// Inference identity of the running node, as the Gemmini backend's records
// carry it: "phase=... decode_ordinal=... request_id=... operation_id=...
// graph_id=... operator_id=... node_ordinal=...". Phase/decode come from the
// performance context, ids from the operator trace context bound on the
// calling thread (potal_graph_compute; the issuer binds a job's own context).
// phase=none outside an inference.
std::string trace_fields();

} // namespace ggml::potal::log
