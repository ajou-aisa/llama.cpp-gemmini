#pragma once

#include "quants/common/hp1_scu.hpp"
#include "quants/common/weight_reader.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ggml::gemmini::evaluation {

#if GGML_GEMMINI_SCALE_METRICS
inline void observe_scu_block(const ggml_gemmini_args_t &args,
                              const quants::wroute::WeightRoutePlan &plan,
                              size_t stripe, const char *work_type, size_t block,
                              size_t rows, size_t fragments) {
    if (!args.evaluation_context || !args.evaluation_context->scale_enabled()) return;
    if (!plan.valid || !plan.hp1_carriers)
        throw std::runtime_error("evaluation metrics: SCALE requires a validated HP1 weight route");
    if (!rows || !fragments || rows > UINT64_MAX / fragments)
        throw std::runtime_error("evaluation metrics: invalid SCU partial sum extent");
    const uint64_t partials = rows * fragments;
    for (size_t column = 0; column < args.J; ++column) {
        const auto scale = quants::wreader::read_hp1_carrier_validated(args, plan, column, block);
        if (!scale.ok()) throw std::runtime_error("evaluation metrics: SCALE requires HP1 carrier metadata");
        const bool zero = scale.carrier == quants::hp1::zero_carrier;
        const uint32_t shift = zero ? 0 : scale.carrier;
        const double block_scale = zero ? 0 : std::ldexp(static_cast<double>(scale.column_scale), shift);
        args.evaluation_context->scale_alignment(stripe, work_type, column, block,
            block_scale, scale.column_scale, shift, shift ? partials : 0, partials, zero);
    }
}

inline void observe_dense_scu(const ggml_gemmini_args_t &args, size_t stripe, size_t rows) {
    if (!args.evaluation_context || !args.evaluation_context->scale_enabled()) return;
    const auto plan = quants::wroute::resolve_weight_route_plan(
        args, quants::wroute::WeightScaleInfoMode::ResidualHp1Scu);
    for (size_t begin = 0; begin < args.K; begin += 32) {
        const size_t valid_k = std::min(size_t{32}, args.K - begin);
        observe_scu_block(args, plan, stripe, "DENSE", begin / 32, rows,
            (valid_k + DIM - 1) / DIM);
    }
}
#endif

}
