#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>

#ifndef LOG_CYCLE
#define LOG_CYCLE 0
#endif

#ifndef CYCLE_SIM
#define CYCLE_SIM 0
#endif

namespace ggml::gemmini::semantic { struct Context; }

namespace ggml::gemmini::log {

struct CpuCorrelation {
    bool present = false;
    bool captured = false;
    uint64_t collection_run_id = 0;
    uint64_t phase_id = UINT64_MAX;
    uint64_t operation_id = UINT64_MAX;
    uint64_t target_node_id = UINT64_MAX;
    uint64_t parent_id = UINT64_MAX;
    uint64_t work_id = UINT64_MAX;
    uint64_t call_id = UINT64_MAX;
    uint64_t host_stage_id = UINT64_MAX;
    uint64_t worker_count = UINT64_MAX;
    std::shared_ptr<const semantic::Context> semantic_context{};
};

struct CpuExclusionSnapshot {
    uint64_t epoch = 0;
    bool active = false;
};

namespace detail {
#if CYCLE_SIM
inline thread_local uint64_t functional_emulation_epoch = 0;
inline thread_local unsigned functional_emulation_depth = 0;
#endif
}

CpuCorrelation current_cpu_correlation() noexcept;
CpuCorrelation exchange_cpu_correlation(CpuCorrelation value) noexcept;

class ScopedCpuCorrelation {
public:
    explicit ScopedCpuCorrelation(CpuCorrelation value) noexcept
        : previous_(exchange_cpu_correlation(value)) {}
    ~ScopedCpuCorrelation() { exchange_cpu_correlation(previous_); }
    ScopedCpuCorrelation(const ScopedCpuCorrelation &) = delete;
    ScopedCpuCorrelation &operator=(const ScopedCpuCorrelation &) = delete;
private:
    CpuCorrelation previous_;
};

inline CpuExclusionSnapshot capture_cpu_exclusion() noexcept {
#if CYCLE_SIM
    return {detail::functional_emulation_epoch, detail::functional_emulation_depth != 0};
#else
    return {};
#endif
}

inline const char *cpu_exclusion_since(CpuExclusionSnapshot start) noexcept {
    const auto end = capture_cpu_exclusion();
    return start.active || end.active || start.epoch != end.epoch ? "functional_emulation" : nullptr;
}

class ScopedFunctionalEmulationSuppression {
public:
    ScopedFunctionalEmulationSuppression() noexcept {
#if CYCLE_SIM
        ++detail::functional_emulation_epoch;
        ++detail::functional_emulation_depth;
#endif
    }
    ~ScopedFunctionalEmulationSuppression() {
#if CYCLE_SIM
        --detail::functional_emulation_depth;
#endif
    }
    ScopedFunctionalEmulationSuppression(const ScopedFunctionalEmulationSuppression &) = delete;
    ScopedFunctionalEmulationSuppression &operator=(const ScopedFunctionalEmulationSuppression &) = delete;
};

inline const char *cpu_service_exclusion(const char *operation) noexcept {
#if CYCLE_SIM
    if (capture_cpu_exclusion().active) return "functional_emulation";
    if (operation != nullptr) {
        // These envelopes may include another worker's emulation or a completion wait.
        for (const char *coarse : {"im2p.frontend_start_host_call", "im2p.fence_host_call",
                "im2p.stripe_submit_host_call", "im2p.residual_simulator_host_call",
                "dense_backend_host_call", "pipeline_drain_and_join", "exsia.stripe_ready_handoff",
                "exsia.stripe_total", "exsia.run_total"}) {
            if (std::strcmp(operation, coarse) == 0) return "npu_dependency_envelope";
        }
    }
#else
    (void) operation;
#endif
    return nullptr;
}

}
