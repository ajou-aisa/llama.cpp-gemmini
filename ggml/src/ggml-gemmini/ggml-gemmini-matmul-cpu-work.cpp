#include "ggml-gemmini-matmul-cpu-work.hpp"

#include <limits>

namespace ggml::gemmini {
namespace {

CpuWorkComponent absent_component() {
    return {};
}

CpuWorkComponent invalid_component(std::string reason) {
    CpuWorkComponent result{};
    result.reason = std::move(reason);
    result.coverage = CpuWorkCoverage::invalid;
    return result;
}

bool checked_add(uint64_t value, uint64_t & total) {
    if (value > std::numeric_limits<uint64_t>::max() - total) return false;
    total += value;
    return true;
}

}

CpuWorkComponent select_dense_cpu_work(const DenseCpuWorkInput & input) {
    if (input.external_marker) return absent_component();
    if (!input.fine_selected) return input.parent;
    if (input.fine.empty()) return invalid_component("missing_fine_leaf");

    uint64_t total = 0;
    for (const CpuWorkComponent & leaf : input.fine) {
        if (leaf.coverage != CpuWorkCoverage::fine_leaves)
            return invalid_component("invalid_fine_coverage");
        if (!leaf.cycles.has_value()) {
            CpuWorkComponent invalid = leaf;
            invalid.coverage = CpuWorkCoverage::invalid;
            invalid.additive = false;
            return invalid;
        }
        if (!checked_add(*leaf.cycles, total)) return invalid_component("overflow");
    }
    CpuWorkComponent result{};
    result.cycles = total;
    result.reason = "complete";
    result.coverage = CpuWorkCoverage::fine_leaves;
    result.additive = true;
    return result;
}

CpuWorkComponent select_rmd_cpu_work(const RmdCpuWorkInput & input) {
    switch (input.route) {
        case RmdCpuRoute::absent:
            return absent_component();
        case RmdCpuRoute::cpu_direct:
            return input.direct;
        case RmdCpuRoute::checked_software:
            return input.backend;
        case RmdCpuRoute::native_accelerator: {
            CpuWorkComponent result{};
            result.reason = "unavailable_native_rmd_provider";
            result.coverage = CpuWorkCoverage::unavailable_route;
            return result;
        }
    }
    return invalid_component("invalid_route");
}

RmdPostCpuWorkSelection select_rmd_post_cpu_work(
        const RmdPostCpuWorkInput & input) {
    RmdPostCpuWorkSelection result{};
    result.rmd = input.rmd;
    result.compose = input.packet ? input.compose : absent_component();
    result.finalize = input.finalize;
    result.merge = input.merge;
    result.merge.additive = false;
    if (result.finalize.cycles.has_value()) {
        result.finalize_canonical_cycles = result.finalize.cycles;
    }
    if (!input.merge_succeeded) {
        result.finalize_canonical_cycles.reset();
        result.reason = "failed_operation";
        return result;
    }

    uint64_t total = 0;
    const CpuWorkComponent * required[] = {
        &result.rmd,
        input.packet ? &result.compose : nullptr,
        &result.finalize,
    };
    for (const CpuWorkComponent * component : required) {
        if (component == nullptr || component->coverage == CpuWorkCoverage::absent) continue;
        if (!component->cycles.has_value()) {
            result.reason = component->reason;
            return result;
        }
        if (!checked_add(*component->cycles, total)) {
            result.reason = "overflow";
            return result;
        }
    }
    result.canonical_cycles = total;
    result.reason = "complete";
    return result;
}

} // namespace ggml::gemmini
