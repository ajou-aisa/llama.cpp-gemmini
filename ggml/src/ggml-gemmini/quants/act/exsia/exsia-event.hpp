#pragma once

#include "types.hpp"
#include "../../../residual/direct/direct-types.hpp"
#include "../../../residual/rmd/rmd-types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#if CYCLE_SIM
#include <vector>
#endif

namespace ggml::gemmini::evaluation {
class Invocation;
}

namespace ggml::gemmini::quants::act::exsia {
struct StripeMetadataSnapshot {
    int16_t e_s   = std::numeric_limits<int16_t>::min();
    int16_t rho   = 6;
    int32_t sigma = GGML_GEMMINI_EXSIA_SIGMA;
    int16_t theta = std::numeric_limits<int16_t>::min();
};

struct StripeReadyEvent {
    uint64_t                              run_id    = 0;
    size_t                                stripe_id = 0;
    size_t                                slot      = 0;
    size_t                                row_begin = 0;
    size_t                                row_end   = 0;
    std::optional<StripeMetadataSnapshot> activation_metadata;
    uint64_t                              quantization_start    = 0;
    uint64_t                              quantization_end      = 0;
    uint64_t                              quantization_start_ns = 0;
    uint64_t                              quantization_end_ns   = 0;
    // Residual work for this stripe, or nullptr when the stripe has no residual.
    // The packet owns its buffers, so it stays valid after the ExSIA slot is released.
    ggml::gemmini::rmd::StripePacketHandle             rmd_packet;
    ggml::gemmini::residual::DirectStripePayloadHandle direct_residual;
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    std::shared_ptr<evaluation::Invocation> evaluation_context;
#endif
    uint64_t rmd_pack_ns                 = 0;
    uint64_t local_start_ns              = 0;
    uint64_t local_end_ns                = 0;
    uint64_t folding_start_ns            = 0;
    uint64_t folding_end_ns              = 0;
    uint64_t mask_assembly_start_ns      = 0;
    uint64_t mask_assembly_end_ns        = 0;
    uint64_t exponent_reduction_start_ns = 0;
    uint64_t exponent_reduction_end_ns   = 0;
    uint64_t folding_commit_ns           = 0;
#if CYCLE_SIM
    std::vector<uint64_t> cycle_sim_host_dependencies{};
#endif
    // The synchronous sink reports queue-capacity wait; null means it is not instrumented.
    bool                            collect_submission_timing = false;
    mutable std::optional<uint64_t> submission_wait_ns;
};

struct StripeReadySink {
    void * user_data                                   = nullptr;
    bool (*on_ready)(void *, const StripeReadyEvent &) = nullptr;
};

} // namespace ggml::gemmini::quants::act::exsia
