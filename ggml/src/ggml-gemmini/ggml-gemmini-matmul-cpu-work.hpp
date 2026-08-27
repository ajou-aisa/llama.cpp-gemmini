#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ggml::gemmini {

enum class CpuWorkCoverage : uint8_t {
    absent,
    fine_leaves,
    algorithm_cpu_leaves,
    coarse_same_thread_envelope,
    unavailable_route,
    invalid,
};

struct CpuWorkComponent {
    std::optional<uint64_t> cycles;
    std::string reason = "absent";
    std::string sample_reason;
    CpuWorkCoverage coverage = CpuWorkCoverage::absent;
    bool additive = false;
};

struct DenseCpuWorkInput {
    // The parent is scheduled calling-thread work only: userspace control,
    // caller-context syscalls/kernel work, active polling, and completion.
    // It excludes RTL, descheduled/blocking time, and every other CPU thread.
    CpuWorkComponent parent;
    std::vector<CpuWorkComponent> fine;
    bool fine_selected = false;
    bool external_marker = false;
};

CpuWorkComponent select_dense_cpu_work(const DenseCpuWorkInput & input);

enum class RmdCpuRoute : uint8_t {
    absent,
    cpu_direct,
    checked_software,
    native_accelerator,
};

struct RmdCpuWorkInput {
    RmdCpuRoute route = RmdCpuRoute::absent;
    CpuWorkComponent direct;
    CpuWorkComponent backend;
};

CpuWorkComponent select_rmd_cpu_work(const RmdCpuWorkInput & input);

struct RmdPostCpuWorkInput {
    CpuWorkComponent rmd;
    CpuWorkComponent compose;
    CpuWorkComponent finalize;
    CpuWorkComponent merge;
    bool packet = false;
    bool merge_succeeded = true;
};

struct RmdPostCpuWorkSelection {
    CpuWorkComponent rmd;
    CpuWorkComponent compose;
    CpuWorkComponent finalize;
    CpuWorkComponent merge;
    std::optional<uint64_t> finalize_canonical_cycles;
    std::optional<uint64_t> canonical_cycles;
    std::string reason;
};

RmdPostCpuWorkSelection select_rmd_post_cpu_work(
    const RmdPostCpuWorkInput & input);

struct MatmulCpuWorkMetrics {
    CpuWorkComponent dense;
    CpuWorkComponent rmd;
    CpuWorkComponent compose;
    CpuWorkComponent finalize;
    CpuWorkComponent merge;
};

} // namespace ggml::gemmini
