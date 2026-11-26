#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>

#ifndef LOG_CYCLE
#define LOG_CYCLE 0
#endif


namespace ggml::npu::semantic { struct Context; }

namespace ggml::npu::log {

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
    return {};
}

inline const char *cpu_exclusion_since(CpuExclusionSnapshot start) noexcept {
    const auto end = capture_cpu_exclusion();
    return start.active || end.active || start.epoch != end.epoch ? "functional_emulation" : nullptr;
}

class ScopedFunctionalEmulationSuppression {
public:
    ScopedFunctionalEmulationSuppression() noexcept {
    }
    ~ScopedFunctionalEmulationSuppression() {
    }
    ScopedFunctionalEmulationSuppression(const ScopedFunctionalEmulationSuppression &) = delete;
    ScopedFunctionalEmulationSuppression &operator=(const ScopedFunctionalEmulationSuppression &) = delete;
};

inline const char *cpu_service_exclusion(const char *operation) noexcept {
    (void) operation;
    return nullptr;
}

}
