#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"

#include <cstdio>

#if defined(GGML_GEMMINI_TESTING)
#error "production host backend oracle must not enable GGML_GEMMINI_TESTING"
#endif

int main() {
    using ggml::gemmini::rmd::compact_rmd_backend_available;
    const bool production_host = compact_rmd_backend_available(false);
    const bool hardware_target = compact_rmd_backend_available(true);
    std::printf(
        "PRODUCTION_HOST_BACKEND hardware_host=%u hardware_target=%u\n",
        static_cast<unsigned>(production_host),
        static_cast<unsigned>(hardware_target));
    return !production_host && hardware_target ? 0 : 1;
}
