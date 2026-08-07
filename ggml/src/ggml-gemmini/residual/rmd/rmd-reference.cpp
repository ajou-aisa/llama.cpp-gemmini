#include "rmd-reference.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_route.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <new>

namespace ggml::gemmini::rmd {

namespace {

namespace wroute = quants::wroute;

constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());

bool weight_code(const ggml_gemmini_args_t & args,
                 const wroute::WeightRoutePlan & plan,
                 size_t k, size_t j, int8_t & out) {
    if (plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks) {
        const block_q8_h1 * block = args.q8_h1_block(j, k / kBlockSize);
        if (block == nullptr) {
            return false;
        }
        out = static_cast<int8_t>(block->qs[k % kBlockSize]);
        return true;
    }
    const int8_t * dense = reinterpret_cast<const int8_t *>(args.B);
    if (dense == nullptr) {
        return false;
    }
    out = plan.layout == wroute::WeightLayout::JxK_ColMajor ?
        dense[j * plan.weight_stride + k] : dense[k * plan.weight_stride + j];
    return true;
}

RmdStatus accumulate(const ggml_gemmini_args_t & args,
                     size_t row_count,
                     const std::vector<ReferenceResidual> & residuals,
                     bool through_digits,
                     std::vector<OutputValue> & correction) {
    if (args.J == 0 || args.K == 0 || row_count == 0) {
        return RmdStatus::invalid_arguments;
    }
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }

    try {
        correction.assign(row_count * args.J, OutputValue{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    // Group by original weight block so the block scale is applied exactly once.
    std::map<uint32_t, std::vector<const ReferenceResidual *>> by_block;
    for (const ReferenceResidual & residual : residuals) {
        if (residual.local_row >= row_count || residual.k >= args.K) {
            return RmdStatus::invalid_arguments;
        }
        if (residual.residual == 0) {
            continue;
        }
        if (through_digits) {
            BalancedDigits digits{};
            if (!decompose_balanced_radix256(residual.residual, digits)) {
                return RmdStatus::residual_too_wide;
            }
            if (compose_balanced_radix256(digits) != residual.residual) {
                return RmdStatus::execution_failed;
            }
        }
        by_block[residual.k / static_cast<uint32_t>(kBlockSize)].push_back(&residual);
    }

    std::vector<int64_t> accumulator;
    try {
        accumulator.assign(row_count * args.J, int64_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (const auto & [block_id, entries] : by_block) {
        std::fill(accumulator.begin(), accumulator.end(), int64_t{0});
        for (const ReferenceResidual * residual : entries) {
            for (size_t j = 0; j < args.J; ++j) {
                int8_t code = 0;
                if (!weight_code(args, plan, residual->k, j, code)) {
                    return RmdStatus::execution_failed;
                }
                const __int128 product =
                    static_cast<__int128>(residual->residual) * static_cast<__int128>(code);
                const size_t index = residual->local_row * args.J + j;
                const __int128 sum = static_cast<__int128>(accumulator[index]) + product;
                if (sum > kInt64Max || sum < kInt64Min) {
                    return RmdStatus::overflow;
                }
                accumulator[index] = static_cast<int64_t>(sum);
            }
        }
        for (size_t j = 0; j < args.J; ++j) {
            const uint64_t scale = wroute::route_block_scale(plan, args, j, block_id);
            if (scale > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return RmdStatus::overflow;
            }
            for (size_t row = 0; row < row_count; ++row) {
                const size_t index = row * args.J + j;
                const __int128 scaled = static_cast<__int128>(accumulator[index]) *
                    static_cast<__int128>(static_cast<int64_t>(scale));
                const __int128 sum = static_cast<__int128>(correction[index]) + scaled;
                if (sum > kInt64Max || sum < kInt64Min) {
                    return RmdStatus::overflow;
                }
                correction[index] = static_cast<int64_t>(sum);
            }
        }
    }
    return RmdStatus::success;
}

}

RmdStatus reference_direct_correction(const ggml_gemmini_args_t & args,
                                      size_t row_count,
                                      const std::vector<ReferenceResidual> & residuals,
                                      std::vector<OutputValue> & correction) {
    return accumulate(args, row_count, residuals, false, correction);
}

RmdStatus reference_rmd_correction(const ggml_gemmini_args_t & args,
                                   size_t row_count,
                                   const std::vector<ReferenceResidual> & residuals,
                                   std::vector<OutputValue> & correction) {
    return accumulate(args, row_count, residuals, true, correction);
}

}
