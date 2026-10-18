#include "log.hpp"

#include "ggml-impl.h"

#include <gemmini/log.hpp>

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

} // namespace ggml::potal::log
