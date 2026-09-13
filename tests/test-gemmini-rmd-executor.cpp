#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp"

#include <array>
#include <cstdio>

using namespace ggml::gemmini::rmd;

namespace {
struct ExecutorProbe {
    size_t calls = 0;
    int fault = 0;
    static int execute(void * opaque, const im2p_matmul_desc_t * d,
                       im2p_work_stats_extended_t * stats) {
        auto & probe = *static_cast<ExecutorProbe *>(opaque);
        ++probe.calls;
        if (!d || d->vector_op != IM2P_VECTOR_BYPASS || d->output_domain != 0 ||
            d->m != 2 || d->n != 2 || d->k != 3 || !d->provider.write_output ||
            d->provider.read_scale || d->weights || d->output || probe.fault == 1)
            return IM2P_ERROR;
        // Test-only scalar executor. Production uses the injected transport.
        for (size_t i = 0; i < d->m; ++i) {
            std::array<int64_t, 2> sums{};
            for (size_t k = 0; k < d->k; ++k) {
                std::array<int8_t, 2> weights{};
                if (d->provider.read_weight_i8(d->provider.context, k, 0, 2,
                                               weights.data()) != 0) return IM2P_ERROR;
                const auto a = static_cast<const int8_t *>(d->activations)[
                    i * d->activation_row_stride_bytes + k];
                for (size_t j = 0; j < 2; ++j) sums[j] += int64_t(a) * weights[j];
            }
            if (probe.fault == 2 && i == 1) continue;
            if (d->provider.write_output(d->provider.context, 0, i, 0, 2,
                                         sums.data(), d->output_domain) != 0) return IM2P_ERROR;
            if (probe.fault == 3 &&
                d->provider.write_output(d->provider.context, 0, i, 0, 2,
                                         sums.data(), d->output_domain) != 0) return IM2P_ERROR;
        }
        stats->base.work_total_cycles = 7;
        stats->base.output_write_requests = 2;
        return IM2P_OK;
    }
};
}

int main() {
    const int32_t a[] = {1, -2, 3, 99, -3, 2, 1, 99};
    const int32_t w[] = {2, -1, 99, 3, 4, 99, -2, 5, 99};
    const detail::Im2pCompactDot dot{8, a, 2, 4, w, 2, 3, 3};
    for (int fault = 0; fault < 4; ++fault) {
        ExecutorProbe probe{0, fault};
        Im2pFullExecutor executor{&probe, ExecutorProbe::execute};
        std::array<int64_t, 6> output{73, 73, 73, 73, 73, 73};
        detail::Im2pProviderStatsAggregate stats{};
        const auto result = detail::execute_im2p_compact_dot(
            nullptr, dot, output.data(), 3, stats,
            Im2pProviderTestFault::none, &executor);
        const auto expected = fault == 0 ? RmdStatus::success :
            fault == 2 ? RmdStatus::invalid_packet : RmdStatus::execution_failed;
        if (result != expected || probe.calls != 1 ||
            (fault == 0 && (output != std::array<int64_t, 6>{-10, 6, 73, -2, 16, 73} ||
                            stats.stats.work_total_cycles() != 7)) ||
            (fault != 0 && stats.stats.work_total_cycles() != 0)) return 1;
    }
    detail::Im2pProviderStatsAggregate stats{};
    std::array<int64_t, 4> output{73, 73, 73, 73};
    if (detail::execute_im2p_compact_dot(nullptr, dot, output.data(), 2, stats) !=
        RmdStatus::invalid_arguments) return 1;
    std::puts("RMD_EXECUTOR_MOCK_PASS transport_calls=4 simulator_calls=0 duplicate_and_missing_rejected=1");
    return 0;
}
