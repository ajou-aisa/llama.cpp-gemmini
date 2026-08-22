#include "ggml-gemmini-matmul.hpp"

#include <gemmini.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace ggml::gemmini;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

void clear_environment() {
    unsetenv("GEMMINI_MATMUL_MODE");
    unsetenv("GEMMINI_STRIPE_ROWS");
    unsetenv("GEMMINI_STRIPE_JOB_CAPACITY");
    unsetenv("GEMMINI_RMD_BACKEND");
}

ggml_gemmini_args_t make_args(std::vector<elem_t> & activation,
                              std::vector<elem_t> & weights,
                              std::vector<float> & output) {
    ggml_gemmini_args_t args{};
    args.I = 3;
    args.J = 2;
    args.K = 2;
    args.A.allocate(args.I, args.K, 8);
    for (size_t i = 0; i < args.I * args.K; ++i) {
        args.A.set(i / args.K, i % args.K, activation[i]);
    }
    args.B = weights.data();
    args.sA = args.K;
    args.sB = args.J;
    args.f_out = output.data();
    args.col_stride_f_out = 1;
    args.stride_f_out = args.J;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    return args;
}

bool test_generated_config_contract() {
    const bool valid_default_mode =
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::full) ||
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::stripe_pipeline);
    const bool default_rows_valid =
        !config::DEFAULT_STRIPE_ROWS.has_value() || *config::DEFAULT_STRIPE_ROWS > 0;
    const bool stripe_default_requires_support =
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::full) ||
        config::ENABLE_STRIPE_MATMUL;
    const bool pipeline_default_requires_support =
        config::DEFAULT_MATMUL_MODE != static_cast<int>(MatmulInvocationMode::stripe_pipeline) ||
        config::ENABLE_STRIPE_PIPELINE;
    const auto defaults = resolve_matmul_options();

    return check(valid_default_mode, "generated default matmul mode value") &&
        check(default_rows_valid, "generated stripe rows value") &&
        check(config::DEFAULT_STRIPE_JOB_CAPACITY > 0, "generated job capacity") &&
        check(stripe_default_requires_support, "stripe default requires stripe support") &&
        check(pipeline_default_requires_support, "pipeline default requires pipeline support") &&
        check(defaults.ok(), "generated defaults resolve") &&
        check(defaults.options.mode == static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE),
              "resolved default mode matches generated config") &&
        check(defaults.options.stripe_rows_auto == !config::DEFAULT_STRIPE_ROWS.has_value(),
              "resolved stripe-row auto flag matches generated config") &&
        check(defaults.options.stripe_rows == config::DEFAULT_STRIPE_ROWS.value_or(1),
              "resolved stripe rows match generated config") &&
        check(defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
              "resolved job capacity matches generated config") &&
        check(config::DEFAULT_RMD_BACKEND == static_cast<int>(RmdBackend::cpu_direct) ||
                  config::DEFAULT_RMD_BACKEND == static_cast<int>(RmdBackend::gemmini_ws_compact),
              "generated RMD backend default is valid") &&
        check(defaults.options.rmd_backend == static_cast<RmdBackend>(config::DEFAULT_RMD_BACKEND),
              "resolved RMD backend matches generated default") &&
        check(defaults.rmd_backend_source == MatmulOptionSource::build_default,
              "default RMD backend source is build default");
}

bool test_precedence() {
    clear_environment();
    const auto defaults = resolve_matmul_options();
    if (!check(defaults.ok() && defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
               "generated build defaults")) {
        return false;
    }

    setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
    setenv("GEMMINI_STRIPE_ROWS", "7", 1);
    setenv("GEMMINI_STRIPE_JOB_CAPACITY", "5", 1);
    setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    const auto environment = resolve_matmul_options();
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE &&
        !check(environment.ok() && environment.options.mode == MatmulInvocationMode::full &&
                   !environment.options.stripe_rows_auto && environment.options.stripe_rows == 7 &&
                   environment.options.job_capacity == 5 &&
                   environment.options.rmd_backend == RmdBackend::gemmini_ws_compact &&
                   environment.rmd_backend_source == MatmulOptionSource::environment,
               "environment overrides build defaults")) {
        return false;
    }

    MatmulOptionOverrides explicit_options{};
    explicit_options.mode = config::ENABLE_STRIPE_MATMUL && config::ENABLE_STRIPE_PIPELINE
        ? MatmulInvocationMode::stripe_pipeline : MatmulInvocationMode::full;
    explicit_options.stripe_rows = 11;
    explicit_options.job_capacity = 6;
    explicit_options.rmd_backend = RmdBackend::cpu_direct;
    setenv("GEMMINI_RMD_BACKEND", "DEC", 1);
    const auto explicit_result = resolve_matmul_options(explicit_options);
    clear_environment();
    return check(explicit_result.ok() && explicit_result.options.mode == *explicit_options.mode &&
                     explicit_result.options.stripe_rows == 11 &&
                     explicit_result.options.job_capacity == 6 &&
                     explicit_result.options.rmd_backend == RmdBackend::cpu_direct &&
                     explicit_result.rmd_backend_source == MatmulOptionSource::explicit_override,
                 "explicit options override environment");
}

bool test_invalid_environment() {
    struct InvalidEnvironment {
        const char * name;
        const char * value;
        MatmulOptionsError error;
    };
    const InvalidEnvironment cases[] = {
        { "GEMMINI_MATMUL_MODE", "AUTO", MatmulOptionsError::invalid_mode },
        { "GEMMINI_MATMUL_MODE", "full", MatmulOptionsError::invalid_mode },
        { "GEMMINI_MATMUL_MODE", "STRIPE_SEQUENTIAL", MatmulOptionsError::invalid_mode },
        { "GEMMINI_STRIPE_ROWS", "12x", MatmulOptionsError::invalid_stripe_rows },
        { "GEMMINI_STRIPE_ROWS", "-1", MatmulOptionsError::invalid_stripe_rows },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "+2", MatmulOptionsError::invalid_job_capacity },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "0", MatmulOptionsError::invalid_job_capacity },
        { "GEMMINI_RMD_BACKEND", "", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "cpu", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", " CPU", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "WS ", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "0", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "AUTO", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "INHERIT", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "OS", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "DEC", MatmulOptionsError::invalid_rmd_backend },
    };
    for (const auto & invalid : cases) {
        clear_environment();
        setenv(invalid.name, invalid.value, 1);
        const auto result = resolve_matmul_options();
        if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
            if (!check(!result.ok() && result.error == invalid.error, "invalid environment rejected")) {
                return false;
            }
        } else if (std::strcmp(invalid.name, "GEMMINI_RMD_BACKEND") == 0) {
            if (!check(!result.ok() && result.error == MatmulOptionsError::runtime_override_disabled,
                       "disabled RMD runtime override rejected")) {
                return false;
            }
        } else if (!check(result.ok(), "disabled runtime override ignores unrelated environment")) {
            return false;
        }
    }
    clear_environment();
    return true;
}

bool test_invalid_explicit_rmd_backend() {
    MatmulOptionOverrides options{};
    options.rmd_backend = static_cast<RmdBackend>(2);
    const auto result = resolve_matmul_options(options);
    return check(!result.ok() && result.error == MatmulOptionsError::invalid_rmd_backend,
                 "invalid explicit RMD backend rejected");
}

bool test_invalid_explicit_mode() {
    MatmulOptionOverrides options{};
    options.mode = static_cast<MatmulInvocationMode>(2);
    const auto result = resolve_matmul_options(options);
    return check(!result.ok() && result.error == MatmulOptionsError::invalid_mode,
                 "invalid explicit matmul mode rejected");
}

#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
bool test_execution_route_propagation() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    const tiled_matmul_type_t main_route = args.tiled_matmul_type;
    ResolvedMatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&args, options);
    const bool cpu = check(execution.status().ok() &&
                               args.residual_route == residual::ResidualRoute::cpu_direct,
                           "resolved CPU backend reaches execution residual sink") &&
        check(args.tiled_matmul_type == main_route,
              "CPU residual selection preserves main backend");

    options.rmd_backend = RmdBackend::gemmini_ws_compact;
    auto ws_execution = prepare_execution(&args, options);
#if defined(__riscv)
    const bool ws = check(ws_execution.status().ok(), "WS backend is available on target");
#else
    const bool ws = check(ws_execution.status().code == MatmulStatusCode::unsupported_backend,
                          "WS backend preflights as unavailable on host");
#endif
    return cpu && ws &&
        check(args.residual_route == residual::ResidualRoute::ws_packet,
              "resolved WS backend reaches execution residual sink") &&
        check(args.tiled_matmul_type == main_route,
              "WS residual selection preserves main backend");
}
#endif

bool test_disabled_runtime_rmd_environment() {
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        return true;
    }
    clear_environment();
    setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    const auto result = resolve_matmul_options();
    clear_environment();
    return check(!result.ok() && result.error == MatmulOptionsError::runtime_override_disabled &&
                     result.rmd_backend_source == MatmulOptionSource::build_default,
                 "disabled runtime override rejects valid RMD backend environment");
}

#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
bool test_disabled_mode_status_contract() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(12, 0.0f);
    bool passed = true;

    if (!config::ENABLE_STRIPE_MATMUL || !config::ENABLE_STRIPE_PIPELINE) {
        auto args = make_args(activation, weights, output);
        args.I = 6;
        args.K = 1;
        args.sA = 1;
        activation.resize(args.I * args.K, 1);
        args.A.allocate(args.I, args.K, 8);
        for (size_t i = 0; i < args.I * args.K; ++i) {
            args.A.set(i / args.K, i % args.K, activation[i]);
        }
        auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
        meta.theta.assign(args.I, 0);

        MatmulOptionOverrides options{};
        options.mode = MatmulInvocationMode::stripe_pipeline;
        const auto resolution = resolve_matmul_options(options);
        const auto execution = prepare_execution(args, options);
        passed = check(!resolution.ok() && resolution.error == MatmulOptionsError::disabled_mode,
                       "disabled stripe pipeline resolves as disabled mode") && passed;
        passed = check(execution.status().code == MatmulStatusCode::unsupported_invocation,
                       "disabled stripe pipeline maps to unsupported invocation") && passed;
    }

    return passed;
}
#endif

}

int main() {
    const bool ok = test_generated_config_contract() &&
        test_precedence() &&
        test_invalid_environment() &&
        test_invalid_explicit_rmd_backend() &&
        test_invalid_explicit_mode() &&
        test_disabled_runtime_rmd_environment()
#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
        && test_execution_route_propagation()
#endif
#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
        && test_disabled_mode_status_contract()
#endif
        ;
    return ok ? 0 : 1;
}
