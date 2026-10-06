#pragma once

#include "ggml-gemmini-matmul-config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ggml::gemmini {

enum class MatmulInvocationMode {
    full,
    stripe_pipeline,
};

enum class RmdBackend : uint8_t {
    cpu_direct,
    gemmini_ws_compact,
};

enum class MatmulOptionSource : uint8_t {
    build_default,
    environment,
    explicit_override,
};

struct ResolvedMatmulOptions {
    MatmulInvocationMode mode = static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE);
    size_t               dense_threads = 0;
    bool                 validation    = false;
    bool                 profiling     = false;
    size_t               job_capacity  = config::DEFAULT_STRIPE_JOB_CAPACITY;
    RmdBackend           rmd_backend   = static_cast<RmdBackend>(config::DEFAULT_RMD_BACKEND);

    ResolvedMatmulOptions() {}
};

struct MatmulOptionOverrides {
    std::optional<MatmulInvocationMode> mode;
    size_t                              dense_threads = 0;
    bool                                validation    = false;
    bool                                profiling     = false;
    std::optional<size_t>               job_capacity;
    std::optional<RmdBackend>           rmd_backend;
};

using MatmulOptions = MatmulOptionOverrides;

enum class MatmulOptionsError : uint8_t {
    none,
    invalid_mode,
    invalid_job_capacity,
    invalid_rmd_backend,
    runtime_override_disabled,
    disabled_mode,
};

struct MatmulOptionsResolution {
    ResolvedMatmulOptions options;
    MatmulOptionsError    error              = MatmulOptionsError::none;
    MatmulOptionSource    rmd_backend_source = MatmulOptionSource::build_default;

    bool ok() const {
        return error == MatmulOptionsError::none;
    }
};

bool                    parse_positive_size(std::string_view text, size_t & value);
MatmulOptionsResolution resolve_matmul_options(const MatmulOptionOverrides & explicit_options = {});

} // namespace ggml::gemmini
