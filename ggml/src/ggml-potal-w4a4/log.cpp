#include "log.hpp"

#include "ggml-impl.h"

#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#include <gemmini/trace-context.hpp>

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace ggml::potal::log {

void ensure_outputs() {
    static std::once_flag once;
    std::call_once(once, [] {
        const auto result = ggml::gemmini::log::setup_default_outputs();
        if (!result.cycle || !result.debug) {
            // The logger itself is unavailable: stderr is the only place left.
            GGML_LOG_WARN("[POTAL_BACKEND] default log outputs: cycle=%d debug=%d\n", result.cycle ? 1 : 0,
                          result.debug ? 1 : 0);
        }
    });
}

void debug(const char * layer, const char * fmt, ...) {
#if LOG_DEBUG
    ensure_outputs();
    va_list ap;
    va_start(ap, fmt);
    ggml::gemmini::log::debug.v_layer(layer, fmt, ap);
    va_end(ap);
#else
    (void) layer;
    (void) fmt;
#endif
}

void failure(const char * layer, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
#if LOG_DEBUG
    ensure_outputs();
    ggml::gemmini::log::debug.v_layer(layer, fmt, ap);
#else
    char text[1024];
    std::vsnprintf(text, sizeof(text), fmt, ap);
    GGML_LOG_ERROR("[%s] %s\n", layer, text);
#endif
    va_end(ap);
}

std::string trace_fields() {
    namespace perf                = ggml::gemmini::performance;
    const gemmini_trace_context t = gemmini_trace_capture();
    const perf::Context         p = ggml::gemmini::trace::inference_context(t);
    // The trace context has no decode ordinal; the published context has it
    // while the same operation is current.
    const perf::Context g      = perf::capture_context();
    const bool          decode = p.request_id && p.phase == perf::Phase::decode;
    const bool          known  = decode && g.request_id == p.request_id && g.operation_id == p.operation_id;
    const std::string   decode_ordinal = known ? std::to_string(g.decode_ordinal) : "null"; // prefill: null
    char                buf[256];
    std::snprintf(buf, sizeof(buf),
                  "phase=%s decode_ordinal=%s request_id=%" PRIu64 " operation_id=%" PRIu64 " graph_id=%" PRIu64
                  " operator_id=%" PRIu64 " node_ordinal=%" PRIu64,
                  p.request_id ? perf::phase_name(p.phase) : "none", decode_ordinal.c_str(), p.request_id,
                  p.operation_id, t.graph_id, t.operator_id, t.node_id);
    return buf;
}

} // namespace ggml::potal::log
