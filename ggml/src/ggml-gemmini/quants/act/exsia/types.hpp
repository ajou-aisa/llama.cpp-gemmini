#pragma once

#include "../types.hpp"
#include "../../../ggml-gemmini-config.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#ifndef GGML_GEMMINI_EXSIA_SIGMA
#define GGML_GEMMINI_EXSIA_SIGMA 2
#endif

#ifndef GGML_GEMMINI_EXSIA_LOCAL_WORKERS
#define GGML_GEMMINI_EXSIA_LOCAL_WORKERS 4
#endif

namespace ggml::gemmini::quants::act::exsia {
enum class FailureCode : uint8_t {
    None,
    InvalidInput,
    OpenMPUnavailable,
    WrongTeamSize,
    ExternalOpenMPRegionUnsupported,
    LocalBlockFailure,
    MaskAssemblyFailure,
    ExponentReductionFailure,
    FoldingFailure,
    ValidationSnapshotFailure,
    StripeReadySinkFailure,
    ProfileIntervalInvalid,
    ProfileFlushFailure,
    Exception,
};

static_assert(static_cast<uint8_t>(FailureCode::None) == 0);
static_assert(static_cast<uint8_t>(FailureCode::InvalidInput) == 1);
static_assert(static_cast<uint8_t>(FailureCode::OpenMPUnavailable) == 2);
static_assert(static_cast<uint8_t>(FailureCode::WrongTeamSize) == 3);
static_assert(static_cast<uint8_t>(FailureCode::ExternalOpenMPRegionUnsupported) == 4);
static_assert(static_cast<uint8_t>(FailureCode::LocalBlockFailure) == 5);
static_assert(static_cast<uint8_t>(FailureCode::MaskAssemblyFailure) == 6);
static_assert(static_cast<uint8_t>(FailureCode::ExponentReductionFailure) == 7);
static_assert(static_cast<uint8_t>(FailureCode::FoldingFailure) == 8);
static_assert(static_cast<uint8_t>(FailureCode::ValidationSnapshotFailure) == 9);
static_assert(static_cast<uint8_t>(FailureCode::StripeReadySinkFailure) == 10);
static_assert(static_cast<uint8_t>(FailureCode::ProfileIntervalInvalid) == 11);
static_assert(static_cast<uint8_t>(FailureCode::ProfileFlushFailure) == 12);
static_assert(static_cast<uint8_t>(FailureCode::Exception) == 13);

constexpr size_t EXSIA_PIPELINE_SLOT_COUNT = 2;
constexpr size_t EXSIA_LOCAL_WORKER_COUNT  = GGML_GEMMINI_EXSIA_LOCAL_WORKERS;
constexpr size_t EXSIA_OMP_THREAD_COUNT    = EXSIA_LOCAL_WORKER_COUNT + 1;
static_assert(EXSIA_LOCAL_WORKER_COUNT == 3 || EXSIA_LOCAL_WORKER_COUNT == 4,
              "ExSIA requires three or four Local workers");

struct Meta {
    std::optional<uint64_t> run_id; // originating quantization invocation, including run zero
    int16_t                 e_s   = std::numeric_limits<int16_t>::min();
    int16_t                 rho   = config::GGML_GEMMINI_ACTIVATION_RHO;
    int32_t                 sigma = GGML_GEMMINI_EXSIA_SIGMA;
    std::vector<int16_t>    theta;
    RmdPacketList           rmd_packets;
    DirectResidualList      direct_residuals;

    void reset() {
        run_id.reset();
        e_s   = std::numeric_limits<int16_t>::min();
        rho   = config::GGML_GEMMINI_ACTIVATION_RHO;
        sigma = GGML_GEMMINI_EXSIA_SIGMA;
        theta.clear();
        rmd_packets.clear();
        direct_residuals.clear();
    }

    int16_t resolve_stripe_theta(int stripe_idx) const {
        if (stripe_idx < 0 || static_cast<size_t>(stripe_idx) >= theta.size())
            return std::numeric_limits<int16_t>::min();

        return theta[static_cast<size_t>(stripe_idx)];
    }
};
} // namespace ggml::gemmini::quants::act::exsia
