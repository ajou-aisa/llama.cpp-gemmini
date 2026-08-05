#include "ggml-gemmini-matmul.hpp"

#include <cstdio>
#include <cstdlib>

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
    unsetenv("GEMMINI_RC_SHARDS");
    unsetenv("GEMMINI_STRIPE_JOB_CAPACITY");
}

bool test_precedence() {
    clear_environment();
    const auto defaults = resolve_matmul_options();
    if (!check(defaults.ok() && defaults.options.rc_shards == config::DEFAULT_RC_SHARDS &&
                   defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
               "generated build defaults")) {
        return false;
    }

    setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
    setenv("GEMMINI_STRIPE_ROWS", "7", 1);
    setenv("GEMMINI_RC_SHARDS", "3", 1);
    setenv("GEMMINI_STRIPE_JOB_CAPACITY", "5", 1);
    const auto environment = resolve_matmul_options();
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE &&
        !check(environment.ok() && environment.options.mode == MatmulInvocationMode::full &&
                   !environment.options.stripe_rows_auto && environment.options.stripe_rows == 7 &&
                   environment.options.rc_shards == 3 && environment.options.job_capacity == 5,
               "environment overrides build defaults")) {
        return false;
    }

    MatmulOptionOverrides explicit_options{};
    explicit_options.mode = config::ENABLE_STRIPE_MATMUL
        ? MatmulInvocationMode::stripe_sequential : MatmulInvocationMode::full;
    explicit_options.stripe_rows = 11;
    explicit_options.rc_shards = 9;
    explicit_options.job_capacity = 6;
    const auto explicit_result = resolve_matmul_options(explicit_options);
    clear_environment();
    return check(explicit_result.ok() && explicit_result.options.mode == *explicit_options.mode &&
                     explicit_result.options.stripe_rows == 11 && explicit_result.options.rc_shards == 9 &&
                     explicit_result.options.job_capacity == 6,
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
        { "GEMMINI_STRIPE_ROWS", "12x", MatmulOptionsError::invalid_stripe_rows },
        { "GEMMINI_STRIPE_ROWS", "-1", MatmulOptionsError::invalid_stripe_rows },
        { "GEMMINI_RC_SHARDS", "18446744073709551616", MatmulOptionsError::invalid_rc_shards },
        { "GEMMINI_RC_SHARDS", "", MatmulOptionsError::invalid_rc_shards },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "+2", MatmulOptionsError::invalid_job_capacity },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "0", MatmulOptionsError::invalid_job_capacity },
    };
    for (const auto & invalid : cases) {
        clear_environment();
        setenv(invalid.name, invalid.value, 1);
        const auto result = resolve_matmul_options();
        if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
            if (!check(!result.ok() && result.error == invalid.error, "invalid environment rejected")) {
                return false;
            }
        } else if (!check(result.ok(), "disabled runtime override ignores environment")) {
            return false;
        }
    }
    clear_environment();
    return true;
}

}

int main() {
    return test_precedence() && test_invalid_environment() ? 0 : 1;
}
