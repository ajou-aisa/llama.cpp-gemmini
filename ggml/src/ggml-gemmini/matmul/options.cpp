#include "options.hpp"

#include <charconv>
#include <cstdlib>

namespace ggml::gemmini {

bool parse_positive_size(std::string_view text, size_t & value) {
    if (text.empty()) {
        return false;
    }
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
}

MatmulOptionsResolution resolve_matmul_options(const MatmulOptionOverrides & explicit_options) {
    MatmulOptionsResolution result;
    if (!config::ALLOW_RUNTIME_MATMUL_OVERRIDE && !explicit_options.rmd_backend &&
        std::getenv("GEMMINI_RMD_BACKEND") != nullptr) {
        result.error = MatmulOptionsError::runtime_override_disabled;
        return result;
    }
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        if (!explicit_options.mode)
            if (const char * value = std::getenv("GEMMINI_MATMUL_MODE")) {
                const std::string_view mode(value);
                if (mode == "FULL") {
                    result.options.mode = MatmulInvocationMode::full;
                } else if (mode == "STRIPE_PIPELINE") {
                    result.options.mode = MatmulInvocationMode::stripe_pipeline;
                } else {
                    result.error = MatmulOptionsError::invalid_mode;
                    return result;
                }
            }
        if (!explicit_options.job_capacity)
            if (const char * value = std::getenv("GEMMINI_STRIPE_JOB_CAPACITY")) {
                if (!parse_positive_size(value, result.options.job_capacity)) {
                    result.error = MatmulOptionsError::invalid_job_capacity;
                    return result;
                }
            }
        if (!explicit_options.rmd_backend)
            if (const char * value = std::getenv("GEMMINI_RMD_BACKEND")) {
                const std::string_view backend(value);
                if (backend == "CPU") {
                    result.options.rmd_backend = RmdBackend::cpu_direct;
                } else if (backend == "WS") {
                    result.options.rmd_backend = RmdBackend::gemmini_ws_compact;
                } else {
                    result.error = MatmulOptionsError::invalid_rmd_backend;
                    return result;
                }
                result.rmd_backend_source = MatmulOptionSource::environment;
            }
    }

    if (explicit_options.mode)
        result.options.mode = *explicit_options.mode;
    if (explicit_options.job_capacity)
        result.options.job_capacity = *explicit_options.job_capacity;
    if (explicit_options.rmd_backend) {
        result.options.rmd_backend = *explicit_options.rmd_backend;
        result.rmd_backend_source  = MatmulOptionSource::explicit_override;
    }
    result.options.dense_threads = explicit_options.dense_threads;
    result.options.validation    = explicit_options.validation;
    result.options.profiling     = explicit_options.profiling;

    if (explicit_options.job_capacity && *explicit_options.job_capacity == 0) {
        result.error = MatmulOptionsError::invalid_job_capacity;
        return result;
    }
    if (result.options.rmd_backend != RmdBackend::cpu_direct &&
        result.options.rmd_backend != RmdBackend::gemmini_ws_compact) {
        result.error = MatmulOptionsError::invalid_rmd_backend;
        return result;
    }
    if (result.options.mode != MatmulInvocationMode::full &&
        result.options.mode != MatmulInvocationMode::stripe_pipeline) {
        result.error = MatmulOptionsError::invalid_mode;
        return result;
    }
    if (result.options.mode == MatmulInvocationMode::stripe_pipeline &&
        (!config::ENABLE_STRIPE_MATMUL || !config::ENABLE_STRIPE_PIPELINE)) {
        result.error = MatmulOptionsError::disabled_mode;
    }
    return result;
}

} // namespace ggml::gemmini
