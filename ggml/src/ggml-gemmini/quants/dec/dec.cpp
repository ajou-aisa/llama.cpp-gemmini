#include "dec.hpp"
#include "dec_internal.hpp"
#include "dec_kernel.hpp"
#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"
#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#ifndef DEC_VALIDATION
#define DEC_VALIDATION 0
#endif

#if DEC_VALIDATION
#include <limits>
#endif

namespace ggml::gemmini::quants::dec { namespace
{
    struct RkTriplet
    {
        int k;
        int r;
        int32_t d;
    };

    struct ActivationDECScratch
    {
        std::vector<size_t> rk_counts;
        std::vector<size_t> rk_offs;
        std::vector<std::pair<int, int32_t>> rk_pairs;
        std::vector<int> unique_k;
        std::vector<RkTriplet> rk_stage;
        std::vector<int64_t> i1_delta_by_k;
        double i1_total_abs_residual = 0.0;
        std::vector<float> Wk_f;
        std::vector<float> Y_com;

        void resize_for_dims(size_t I, size_t K, size_t J)
        {
            rk_counts.assign(K + 1, 0);
            rk_offs.resize(K + 1);
            unique_k.clear();
            unique_k.reserve(K);
            rk_stage.clear();
            rk_stage.reserve(I);
            Wk_f.resize(J);
            Y_com.assign(I * J, 0.f);
        }

        void reset_i1_delta(size_t K)
        {
            if (i1_delta_by_k.size() != K)
                i1_delta_by_k.assign(K, int64_t {0});
            else
                std::fill(i1_delta_by_k.begin(), i1_delta_by_k.end(), int64_t {0});

            i1_total_abs_residual = 0.0;
        }
    };

    ActivationDECScratch &get_dec_scratch()
    {
        static thread_local ActivationDECScratch scratch;
        return scratch;
    }

    void log_dec_reject(const char *layer, const char *reason, const ggml_gemmini_args_t &args)
    {
#if LOG_DEBUG
        ggml::gemmini::log::debug(
            layer,
            "[dec.reject] reason=%s I=%zu K=%zu J=%zu format=%u",
            reason,
            args.I,
            args.K,
            args.J,
            static_cast<unsigned>(args.weight_format));
#else
        (void) layer;
        (void) reason;
        (void) args;
#endif
    }

#if DEC_VALIDATION
    bool checked_add_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
            return false;

        out = lhs + rhs;
        return true;
    }

    bool checked_mul_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if (lhs == 0 || rhs == 0)
        {
            out = 0;
            return true;
        }

        if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
            (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()))
            return false;

        if ((lhs > 0 && rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (lhs > 0 && rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs) ||
            (lhs < 0 && rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
            (lhs < 0 && rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
            return false;

        out = lhs * rhs;
        return true;
    }

    bool reference_weight_code(
        size_t k,
        size_t j,
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        int8_t &code)
    {
        if (plan.route == DecWeightRoute::Q8HP1)
        {
            const block_q8_hp1 *block = args.q8_hp1_block(j, k / QK8_HP);
            if (!block)
                return false;
            code = block->qs[k % QK8_HP];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8HP2)
        {
            const block_q8_hp2 *block = args.q8_hp2_block(j, k / QK8_HP);
            if (!block)
                return false;
            code = block->qs[k % QK8_HP];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8H2)
        {
            const block_q8_h2 *block = args.q8_h2_block(j, k / QK8_H2);
            if (!block)
                return false;
            code = block->qs[k % QK8_H2];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        {
            const block_q8_h1 *block = args.q8_h1_block(j, k / QK8_0);
            if (!block)
                return false;
            code = block->qs[k % QK8_0];
            return true;
        }

        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return false;

        code = plan.layout == WeightLayout::KxJ_RowMajor ?
            weights[k * plan.weight_stride + j] : weights[j * plan.weight_stride + k];
        return true;
    }

    bool accumulate_scalar_reference(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs,
        const std::pair<int, int32_t> *rk_pairs,
        const std::vector<int64_t> &delta_by_k,
        bool decode,
        float *reference_ycom)
    {
        const bool per_block_scale = !plan.scales.scalar_mode && !plan.scales.row_header_mode &&
            !plan.scales.channel_mode;
        const size_t route_count = per_block_scale ? plan.scales.cols : 1;
        std::vector<int64_t> accumulator(args.I * args.J, 0);

        for (size_t route = 0; route < route_count; ++route)
        {
            std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;

                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K ||
                    (per_block_scale && k_sz / plan.scales.block_size != route))
                    continue;

                for (size_t j = 0; j < args.J; ++j)
                {
                    int8_t weight_code = 0;
                    if (!reference_weight_code(k_sz, j, args, plan, weight_code))
                        return false;

                    if (decode)
                    {
                        if (k_sz >= delta_by_k.size())
                            return false;

                        int64_t product = 0;
                        int64_t updated = 0;
                        if (!checked_mul_i64(delta_by_k[k_sz], weight_code, product) ||
                            !checked_add_i64(accumulator[j], product, updated))
                            return false;
                        accumulator[j] = updated;
                        continue;
                    }

                    if (k_sz + 1 >= rk_offs.size())
                        return false;
                    for (size_t p = rk_offs[k_sz]; p < rk_offs[k_sz + 1]; ++p)
                    {
                        const int r = rk_pairs[p].first;
                        if (r < 0 || static_cast<size_t>(r) >= args.I)
                            return false;

                        int64_t product = 0;
                        int64_t updated = 0;
                        const size_t accumulator_index = static_cast<size_t>(r) * args.J + j;
                        // int32 residual * int8 code is <= 2^38; int64 permits at least 2^25 such terms per route.
                        if (!checked_mul_i64(rk_pairs[p].second, weight_code, product) ||
                            !checked_add_i64(accumulator[accumulator_index], product, updated))
                            return false;
                        accumulator[accumulator_index] = updated;
                    }
                }
            }

            for (size_t r = 0; r < args.I; ++r)
            {
                const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                for (size_t j = 0; j < args.J; ++j)
                {
                    const size_t index = r * args.J + j;
                    reference_ycom[index] += static_cast<float>(
                        static_cast<double>(accumulator[index]) * activation_scale *
                        dec_route_weight_scale(plan, args, j, route));
                }
            }
        }

        return true;
    }

    bool output_offset(const ggml_gemmini_args_t &args, size_t r, size_t j, size_t &offset)
    {
        const size_t row_stride = args.stride_f_out ? args.stride_f_out : args.J;
        const size_t col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
        if ((r != 0 && row_stride > std::numeric_limits<size_t>::max() / r) ||
            (j != 0 && col_stride > std::numeric_limits<size_t>::max() / j))
            return false;

        const size_t row_offset = r * row_stride;
        const size_t col_offset = j * col_stride;
        if (row_offset > std::numeric_limits<size_t>::max() - col_offset)
            return false;

        offset = row_offset + col_offset;
        return true;
    }

    bool capture_output(const ggml_gemmini_args_t &args, std::vector<float> &output)
    {
        output.resize(args.I * args.J);
        for (size_t r = 0; r < args.I; ++r)
        {
            for (size_t j = 0; j < args.J; ++j)
            {
                size_t offset = 0;
                if (!output_offset(args, r, j, offset))
                    return false;
                output[r * args.J + j] = args.f_out[offset];
            }
        }
        return true;
    }
#endif

    void load_weight_row_scaled(
        size_t k,
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        float *Wk_f)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        const size_t J = args.J;
        if ((!weights && !plan.native_weight_blocks) || !Wk_f || J == 0)
            return;

        if (plan.weight_stride == 0)
            return;

        if (plan.route == DecWeightRoute::Q8HP1)
        {
            const size_t block = k / QK8_HP;
            const size_t offset = k % QK8_HP;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_hp1_block(j, block)->qs[offset]);
        }
        else if (plan.route == DecWeightRoute::Q8HP2)
        {
            const size_t block = k / QK8_HP;
            const size_t offset = k % QK8_HP;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_hp2_block(j, block)->qs[offset]);
        }
        else if (plan.route == DecWeightRoute::Q8H2)
        {
            const size_t block = k / QK8_H2;
            const size_t offset = k % QK8_H2;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_h2_block(j, block)->qs[offset]);
        }
        else if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        {
            const size_t block = k / QK8_0;
            const size_t offset = k % QK8_0;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_h1_block(j, block)->qs[offset]);
        }
        else if (plan.layout == WeightLayout::KxJ_RowMajor)
        {
            const int8_t *row = weights + k * plan.weight_stride;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(row[j]);
        }
        else
        {
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(weights[j * plan.weight_stride + k]);
        }

        const size_t block_index = plan.scales.block_size ? k / plan.scales.block_size : 0;
        for (size_t j = 0; j < J; ++j)
        {
            Wk_f[j] *= dec_route_weight_scale(plan, args, j, block_index);
        }
    }

    size_t stage_from_outliers(
        const std::vector<QactOutlier> &outliers,
        size_t I,
        size_t K,
        ActivationDECScratch &scratch)
    {
        if (outliers.empty() || I == 0 || K == 0)
            return 0;

        size_t staged = 0;
        for (const auto &outlier : outliers)
        {
            const int k = outlier.col;
            const int r_idx = outlier.row;
            if (k < 0 || r_idx < 0)
                continue;

            const size_t r = static_cast<size_t>(r_idx);
            const size_t k_sz = static_cast<size_t>(k);
            if (r >= I || k_sz >= K)
                continue;

            scratch.rk_stage.push_back({k, static_cast<int>(r), outlier.residual});
            scratch.rk_counts[k_sz + 1]++;
            ++staged;
        }

#if LOG_DEBUG
        if (staged > 0)
        {
            double residual_sum = 0.0;
            double residual_sq_sum = 0.0;
            double residual_max = 0.0;
            const size_t start_idx = scratch.rk_stage.size() - staged;
            for (size_t i = start_idx; i < scratch.rk_stage.size(); ++i)
            {
                const double abs_res = std::fabs(static_cast<double>(scratch.rk_stage[i].d));
                residual_sum += abs_res;
                residual_sq_sum += abs_res * abs_res;
                residual_max = std::max(residual_max, abs_res);
            }

            const double residual_mean = residual_sum / staged;
            const double residual_std = std::sqrt((residual_sq_sum / staged) - (residual_mean * residual_mean));
            ggml::gemmini::log::debug(
                nullptr,
                "[dec.select.outlier] total_outliers=%zu staged=%zu mean_residual=%.6g std_residual=%.6g max_residual=%.6g",
                outliers.size(),
                staged,
                residual_mean,
                residual_std,
                residual_max);
        }
#endif

        return staged;
    }

    size_t stage_from_outliers_i1(
        const std::vector<QactOutlier> &outliers,
        size_t K,
        ActivationDECScratch &scratch)
    {
        if (outliers.empty() || K == 0)
            return 0;

        size_t staged = 0;
#if LOG_DEBUG
        double residual_sum = 0.0;
        double residual_sq_sum = 0.0;
        double residual_max = 0.0;
#endif

        for (const auto &outlier : outliers)
        {
            const int k = outlier.col;
            const int r_idx = outlier.row;
            if (k < 0 || r_idx != 0)
                continue;

            const size_t k_sz = static_cast<size_t>(k);
            if (k_sz >= K)
                continue;

            if (scratch.rk_counts[k_sz + 1] == 0)
                scratch.unique_k.push_back(k);

            scratch.i1_delta_by_k[k_sz] += static_cast<int64_t>(outlier.residual);
            const double abs_res = std::fabs(static_cast<double>(outlier.residual));
            scratch.i1_total_abs_residual += abs_res;
            scratch.rk_counts[k_sz + 1]++;
            ++staged;

#if LOG_DEBUG
            residual_sum += abs_res;
            residual_sq_sum += abs_res * abs_res;
            residual_max = std::max(residual_max, abs_res);
#endif
        }

#if LOG_DEBUG
        if (staged > 0)
        {
            const double residual_mean = residual_sum / staged;
            const double residual_std = std::sqrt((residual_sq_sum / staged) - (residual_mean * residual_mean));
            ggml::gemmini::log::debug(
                nullptr,
                "[dec.select.outlier] total_outliers=%zu staged=%zu mean_residual=%.6g std_residual=%.6g max_residual=%.6g",
                outliers.size(),
                staged,
                residual_mean,
                residual_std,
                residual_max);
        }
#endif

        return staged;
    }

    size_t build_rk_csc(size_t K, ActivationDECScratch &scratch)
    {
        if (K == 0)
            return 0;

        if (scratch.rk_offs.size() != K + 1)
            scratch.rk_offs.resize(K + 1);

        for (size_t k = 0; k <= K; ++k)
            scratch.rk_offs[k] = scratch.rk_counts[k];

        for (size_t k = 1; k <= K; ++k)
            scratch.rk_offs[k] += scratch.rk_offs[k - 1];

        const size_t nnz = scratch.rk_offs[K];
        scratch.rk_pairs.assign(nnz, {0, 0});

        if (nnz == 0)
        {
            scratch.unique_k.clear();
            scratch.rk_stage.clear();
            return nnz;
        }

        std::vector<size_t> pos(scratch.rk_offs.begin(), scratch.rk_offs.end() - 1);
        for (const auto &t : scratch.rk_stage)
        {
            const size_t k = static_cast<size_t>(t.k);
            if (k >= K)
                continue;

            const size_t dst = pos[k]++;
            if (dst < nnz)
                scratch.rk_pairs[dst] = {t.r, t.d};
        }

        scratch.unique_k.clear();
        scratch.unique_k.reserve(K);
        for (size_t k = 0; k < K; ++k)
        {
            if (scratch.rk_offs[k] != scratch.rk_offs[k + 1])
                scratch.unique_k.push_back(static_cast<int>(k));
        }

        scratch.rk_stage.clear();
        return nnz;
    }

}

ActivationDECResult compensate_activation_dec(
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer)
{
    uint64_t start = ggml::gemmini::cycle::read();

    ActivationDECResult result{};
    const DecRoutePlan plan = resolve_dec_route_plan(args, WeightScaleInfoMode::Dec);
    if (!plan.valid)
    {
        log_dec_reject(layer, plan.reject_reason, args);
        return result;
    }

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if ((!weights && !plan.native_weight_blocks) || !args.f_out)
    {
        log_dec_reject(layer, "missing weight or output buffer", args);
        return result;
    }

    const size_t I = args.I;
    const size_t K = args.K;
    const size_t J = args.J;

    if (I == 0 || K == 0 || J == 0)
    {
        log_dec_reject(layer, "zero dimension", args);
        return result;
    }

    if (plan.weight_stride == 0)
    {
        log_dec_reject(layer, "zero weight stride", args);
        return result;
    }

    const size_t minimum_weight_stride = plan.layout == WeightLayout::KxJ_RowMajor ? J : K;
    if (weights && !plan.native_weight_blocks && plan.weight_stride < minimum_weight_stride)
    {
        log_dec_reject(layer, "weight stride is shorter than logical row", args);
        return result;
    }

    auto activation_scales_vec = act::activation_scales(args, I);
    const float *activation_scales = activation_scales_vec.data();
    if (!dec_route_covers_k(plan, K))
    {
        log_dec_reject(layer, "unsupported or incomplete weight scale metadata", args);
        return result;
    }

    ActivationDECScratch &scr = get_dec_scratch();

    uint64_t end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Ready compensation", start, end);

    start = ggml::gemmini::cycle::read();

    const bool decode = I == 1;

    scr.resize_for_dims(I, K, J);
    if (decode)
        scr.reset_i1_delta(K);

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Initialize scratch", start, end);

    start = ggml::gemmini::cycle::read();

    size_t total_staged = 0;
    if (!outliers.empty())
    {
        if (decode)
            total_staged += stage_from_outliers_i1(outliers, K, scr);
        else
            total_staged += stage_from_outliers(outliers, I, K, scr);
    }

    result.total_selected = total_staged;
    if (total_staged == 0)
        return result;

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Stage outlier residuals", start, end);

    start = ggml::gemmini::cycle::read();

    if (decode)
    {
        result.nnz = result.total_selected;
        result.unique_k_count = scr.unique_k.size();
    }
    else
    {
        result.nnz = build_rk_csc(K, scr);
        result.unique_k_count = scr.unique_k.size();
    }

    if (result.unique_k_count == 0)
        return result;

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Build R_k CSC structure", start, end);

    start = ggml::gemmini::cycle::read();

    const bool use_jmajor_blocked =
        plan.route == DecWeightRoute::Dense &&
        !plan.scales.scalar_mode &&
        !plan.scales.row_header_mode &&
        plan.layout == WeightLayout::JxK_ColMajor &&
        plan.weight_stride >= K;
    const bool use_int64_scalar = plan.route == DecWeightRoute::Dense && plan.scales.scalar_mode;
    const bool use_int64_channel_direct = plan.route == DecWeightRoute::Q8ChannelDirect;
    const bool use_int64_channel_sidecar = plan.route == DecWeightRoute::Q8ChannelSidecar;
    const bool use_int64_h1 = plan.route == DecWeightRoute::Q8H1;
    const bool use_int64_block = !use_int64_h1 && !plan.scales.scalar_mode && !plan.scales.row_header_mode &&
        !plan.scales.channel_mode && (plan.route == DecWeightRoute::Dense ||
            plan.route == DecWeightRoute::Q8H1 || plan.route == DecWeightRoute::Q8H2 ||
            plan.route == DecWeightRoute::Q8HP1 || plan.route == DecWeightRoute::Q8HP2);
#if LOG_DEBUG
    const bool use_int64_kernel = use_int64_scalar || use_int64_channel_direct ||
        use_int64_channel_sidecar || use_int64_h1 || use_int64_block;
    const size_t j_tiles = dec_int64_j_tile_count(J);
    const int dec_threads = resolve_dec_threads(j_tiles);
    const char *route = decode ? (use_jmajor_blocked ? "decode-jmajor-blocked" : "decode-fallback") :
        (use_jmajor_blocked ? "prefill-jmajor-blocked" : "prefill-fallback");
    ggml::gemmini::log::debug(
        layer,
        "[dec.route] route=%s weight_route=%s kernel=%s I=%zu K=%zu J=%zu scale_mode=%s j_tiles=%zu threads=%d",
        route,
        dec_route_name(plan),
        use_int64_scalar ? "int64-scalar" :
            (use_int64_channel_direct ? "int64-channel-direct" :
                (use_int64_channel_sidecar ? "int64-channel-sidecar" :
                    (use_int64_h1 ? "int64-h1" : (use_int64_block ? "int64-block" :
                        (use_jmajor_blocked ? "fp-jmajor-blocked" : "fp-fallback"))))),
        I,
        K,
        J,
        dec_scale_mode_name(plan),
        j_tiles,
        use_int64_kernel ? dec_threads : 1);
    ggml::gemmini::log::debug(
        layer,
        "[dec.work] selected=%zu nnz=%zu unique_k=%zu j_tiles=%zu threads=%d output_stride_row=%zu output_stride_col=%zu",
        result.total_selected,
        result.nnz,
        result.unique_k_count,
        j_tiles,
        use_int64_kernel ? dec_threads : 1,
        args.stride_f_out ? args.stride_f_out : J,
        args.col_stride_f_out ? args.col_stride_f_out : 1);
#endif

    if (use_int64_scalar)
    {
        if (decode)
            accumulate_single_row_to_ycom_int64_scalar(
                args, plan, J, activation_scales, scr.unique_k, scr.i1_delta_by_k,
                scr.Y_com.data());
        else
            accumulate_to_ycom_int64_scalar(
                args, plan, I, J, activation_scales, scr.unique_k, scr.rk_offs, scr.rk_pairs.data(),
                scr.Y_com.data());
    }
    else if (use_int64_channel_direct)
    {
        if (decode)
            accumulate_single_row_to_ycom_int64_channel_direct(
                args, plan, J, activation_scales, scr.unique_k, scr.i1_delta_by_k,
                scr.Y_com.data());
        else
            accumulate_to_ycom_int64_channel_direct(
                args, plan, I, J, activation_scales, scr.unique_k, scr.rk_offs, scr.rk_pairs.data(),
                scr.Y_com.data());
    }
    else if (use_int64_channel_sidecar)
    {
        if (decode)
            accumulate_single_row_to_ycom_int64_channel_sidecar(
                args, plan, J, activation_scales, scr.unique_k, scr.i1_delta_by_k,
                scr.Y_com.data());
        else
            accumulate_to_ycom_int64_channel_sidecar(
                args, plan, I, J, activation_scales, scr.unique_k, scr.rk_offs, scr.rk_pairs.data(),
                scr.Y_com.data());
    }
    else if (use_int64_h1)
    {
        if (decode) accumulate_single_row_to_ycom_int64_h1(args, plan, J, activation_scales, scr.unique_k, scr.i1_delta_by_k, scr.Y_com.data());
        else accumulate_to_ycom_int64_h1(args, plan, I, J, activation_scales, scr.unique_k, scr.rk_offs, scr.rk_pairs.data(), scr.Y_com.data());
    }
    else if (use_int64_block)
    {
        if (decode)
            accumulate_single_row_to_ycom_int64_block(
                args, plan, J, activation_scales, scr.unique_k, scr.i1_delta_by_k,
                scr.Y_com.data());
        else
            accumulate_to_ycom_int64_block(
                args, plan, I, J, activation_scales, scr.unique_k, scr.rk_offs, scr.rk_pairs.data(),
                scr.Y_com.data());
    }
    else if (decode)
    {
        if (use_jmajor_blocked)
        {
            accumulate_single_row_to_ycom_jmajor_blocked(
                args,
                plan,
                J,
                activation_scales,
                scr.unique_k,
                scr.i1_delta_by_k,
                scr.Y_com.data());
        }
        else
        {
            for (int k : scr.unique_k)
            {
                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= scr.i1_delta_by_k.size())
                    continue;

                load_weight_row_scaled(
                    k_sz,
                    args,
                    plan,
                    scr.Wk_f.data());

                accumulate_single_row_delta_to_ycom(
                    scr.Wk_f.data(),
                    J,
                    scr.i1_delta_by_k[k_sz],
                    activation_scales,
                    scr.Y_com.data());
            }
        }
    }
    else if (use_jmajor_blocked)
    {
        accumulate_to_ycom_jmajor_blocked(
            args,
            plan,
            I,
            J,
            activation_scales,
            scr.unique_k,
            scr.rk_offs,
            scr.rk_pairs.data(),
            scr.Y_com.data());
    }
    else
    {
        for (int k : scr.unique_k)
        {
            const size_t k_sz = static_cast<size_t>(k);
            const size_t beg = scr.rk_offs[k_sz];
            const size_t rk_end = scr.rk_offs[k_sz + 1];

            load_weight_row_scaled(
                k_sz,
                args,
                plan,
                scr.Wk_f.data());

            accumulate_to_ycom(
                scr.Wk_f.data(),
                J,
                beg,
                rk_end,
                scr.rk_pairs.data(),
                activation_scales,
                scr.Y_com.data());
        }
    }

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Compute and accumulate compensation", start, end);

#if DEC_VALIDATION
    std::vector<float> validation_output_before;
    std::vector<float> reference_ycom(I * J, 0.0f);
    const bool validation_ready = capture_output(args, validation_output_before) &&
        accumulate_scalar_reference(
            args,
            plan,
            activation_scales,
            scr.unique_k,
            scr.rk_offs,
            scr.rk_pairs.data(),
            scr.i1_delta_by_k,
            decode,
            reference_ycom.data());
#endif

    start = ggml::gemmini::cycle::read();
    apply_ycom_to_output(scr.Y_com.data(), I, J, args);
    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Apply Y_com to output", start, end);

#if DEC_VALIDATION
    if (!validation_ready)
    {
        ggml::gemmini::log::debug(layer, "[dec.validation] skipped: reference setup failed");
    }
    else
    {
        float max_abs_error = 0.0f;
        float max_relative_error = 0.0f;
        bool finite = true;
        for (size_t r = 0; r < I; ++r)
        {
            for (size_t j = 0; j < J; ++j)
            {
                size_t offset = 0;
                if (!output_offset(args, r, j, offset))
                {
                    finite = false;
                    continue;
                }

                const size_t index = r * J + j;
                const float expected = validation_output_before[index] + reference_ycom[index];
                const float observed = args.f_out[offset];
                if (!std::isfinite(expected) || !std::isfinite(observed))
                {
                    finite = false;
                    continue;
                }

                const float absolute_error = std::fabs(observed - expected);
                const float relative_error = absolute_error /
                    std::max(std::fabs(expected), std::numeric_limits<float>::min());
                max_abs_error = std::max(max_abs_error, absolute_error);
                max_relative_error = std::max(max_relative_error, relative_error);
            }
        }
        ggml::gemmini::log::debug(
            layer,
            "[dec.validation] finite=%d max_abs_error=%.9g max_relative_error=%.9g",
            finite ? 1 : 0,
            max_abs_error,
            max_relative_error);
    }
#endif

#if LOG_DEBUG
    {
        ggml::gemmini::log::debug(
            layer,
            "[dec] I=%zu K=%zu J=%zu staged=%zu nnz=%zu unique_k=%zu",
            I,
            K,
            J,
            result.total_selected,
            result.nnz,
            result.unique_k_count);

        if (result.nnz > 0)
        {
            double total_compensation = 0.0;

            if (decode)
                total_compensation = scr.i1_total_abs_residual;
            else
            {
                for (int k : scr.unique_k)
                {
                    const size_t k_sz = static_cast<size_t>(k);
                    const size_t beg = scr.rk_offs[k_sz];
                    const size_t rk_end = scr.rk_offs[k_sz + 1];

                    for (size_t t = beg; t < rk_end; ++t)
                        total_compensation += std::fabs(static_cast<double>(scr.rk_pairs[t].second));
                }
            }

            const double avg_compensation_per_entry = total_compensation / result.nnz;
            const double sparsity = 100.0 * result.nnz / (I * K);
            const double comp_density = total_compensation / (I * J);

            ggml::gemmini::log::debug(
                layer,
                "[dec.summary] total_comp=%.6g avg_comp_per_entry=%.6g sparsity=%.6f%% comp_density=%.6g",
                total_compensation,
                avg_compensation_per_entry,
                sparsity,
                comp_density);
        }
    }
#endif
    return result;
}
} // namespace ggml::gemmini::quants::dec
