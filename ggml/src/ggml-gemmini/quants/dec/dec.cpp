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

    void load_weight_row_scaled(
        size_t k,
        const ggml_gemmini_args_t &args,
        const float *weight_scales,
        size_t scale_rows,
        size_t blocks_k,
        size_t block_size_k,
        bool scalar_mode,
        bool row_header_mode,
        float scalar_weight_scale,
        float *Wk_f)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        const size_t J = args.J;
        const bool native_h1 = is_q8_h1_args(args);
        const bool q8_hp1 = is_q8_hp1_args(args);
        const bool q8_hp2 = is_q8_hp2_args(args);
        const bool q8_h2 = is_q8_h2_args(args);
        if ((!weights && !native_h1 && !q8_hp1 && !q8_hp2 && !q8_h2) || !Wk_f || J == 0)
            return;

        const size_t weight_stride = resolve_weight_stride_elems(args);
        if (weight_stride == 0)
            return;

        if (q8_hp1)
        {
            const size_t block = k / QK8_HP;
            const size_t offset = k % QK8_HP;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_hp1_block(j, block)->qs[offset]);
        }
        else if (q8_hp2)
        {
            const size_t block = k / QK8_HP;
            const size_t offset = k % QK8_HP;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_hp2_block(j, block)->qs[offset]);
        }
        else if (q8_h2)
        {
            const size_t block = k / QK8_H2;
            const size_t offset = k % QK8_H2;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_h2_block(j, block)->qs[offset]);
        }
        else if (native_h1)
        {
            const size_t block = k / QK8_0;
            const size_t offset = k % QK8_0;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(args.q8_h1_block(j, block)->qs[offset]);
        }
        else if (resolve_weight_layout(args) == WeightLayout::KxJ_RowMajor)
        {
            const int8_t *row = weights + k * weight_stride;
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(row[j]);
        }
        else
        {
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] = static_cast<float>(weights[j * weight_stride + k]);
        }

        if (row_header_mode)
        {
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] *= args.q8_channel_scale(j);
        }
        else if (scalar_mode)
        {
            for (size_t j = 0; j < J; ++j)
                Wk_f[j] *= scalar_weight_scale;
        }
        else if (weight_scales && block_size_k > 0 && blocks_k > 0)
        {
            const size_t blk = k / block_size_k;

            for (size_t j = 0; j < J; ++j)
            {
                if (j < scale_rows && (is_q8_channel_dense_sidecar_args(args) || blk < blocks_k))
                    Wk_f[j] *= is_q8_channel_dense_sidecar_args(args) ? weight_scales[j] :
                        weight_scales[j * blocks_k + blk];
            }
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
    bool q8_hp1 = false;
    bool q8_hp2 = false;
    switch (args.weight_format)
    {
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
            q8_hp1 = has_q8_hp1_native_dec_contract(args);
            if (q8_hp1) {
                // Ponytail: gates against malformed padding/finiteness gracefully
                // (the generic ggml_validate_row_data(Q8_HP*) is now a no-op for
                // performance). Kept narrow: only padding + NaN/Inf, no per-qs loop.
                for (size_t i = 0; i < args.q8_hp1_block_count; ++i) {
                    const block_q8_hp1 & b = args.q8_hp1_blocks[i];
                    if (b.padding[0] != 0 || b.padding[1] != 0 ||
                        !std::isfinite(b.channel_scale)) {
                        return result;
                    }
                }
            } else {
                return result;
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
            q8_hp2 = has_q8_hp2_native_dec_contract(args);
            if (q8_hp2) {
                // Ponytail: same gating rationale as Q8_HP1 sibling above.
                for (size_t i = 0; i < args.q8_hp2_block_count; ++i) {
                    const block_q8_hp2 & b = args.q8_hp2_blocks[i];
                    if (b.padding[0] != 0 || b.padding[1] != 0 ||
                        !std::isfinite(b.channel_scale)) {
                        return result;
                    }
                }
            } else {
                return result;
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel:
            if (!args.has_q8_channel_direct_read_contract())
                return result;
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar:
            if (!args.has_q8_channel_dense_sidecar_contract())
                return result;
            break;
        default:
            break;
    }

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const bool native_h1 = args.has_q8_h1_im2p_contract();
    const bool q8_h2 = is_q8_h2_args(args);
    if ((!weights && !native_h1 && !q8_hp1 && !q8_hp2 && !q8_h2) || !args.f_out)
        return result;

    const size_t I = args.I;
    const size_t K = args.K;
    const size_t J = args.J;

    if (I == 0 || K == 0 || J == 0)
        return result;

    const size_t weight_stride = resolve_weight_stride_elems(args);
    if (weight_stride == 0)
        return result;

    const WeightLayout weight_layout = resolve_weight_layout(args);
    const WeightScaleInfo weight_scales = build_weight_scale_info(
        args,
        WeightScaleInfoMode::Dec);
    auto activation_scales_vec = act::activation_scales(args, I);
    const float *activation_scales = activation_scales_vec.data();
    if (!weight_scales.supported)
    {
        ggml::gemmini::log::debug(
            layer,
            "[dec] reject unsupported weight metadata path: stripe_J=%zu",
            args.stripe_J);
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
        !native_h1 &&
        !q8_hp1 &&
        !q8_hp2 &&
        !q8_h2 &&
        !is_q8_channel_dense_sidecar_args(args) &&
        !weight_scales.scalar_mode &&
        !weight_scales.row_header_mode &&
        weight_layout == WeightLayout::JxK_ColMajor &&
        weight_stride >= K;

    if (decode)
    {
        if (use_jmajor_blocked)
        {
            accumulate_single_row_to_ycom_jmajor_blocked(
                args,
                weight_scales.data,
                weight_scales.rows,
                weight_scales.cols,
                weight_scales.block_size,
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
                    weight_scales.data,
                    weight_scales.rows,
                    weight_scales.cols,
                    weight_scales.block_size,
                    weight_scales.scalar_mode,
                    weight_scales.row_header_mode,
                    weight_scales.scalar,
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
            weight_scales.data,
            weight_scales.rows,
            weight_scales.cols,
            weight_scales.block_size,
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
                weight_scales.data,
                weight_scales.rows,
                weight_scales.cols,
                weight_scales.block_size,
                weight_scales.scalar_mode,
                weight_scales.row_header_mode,
                weight_scales.scalar,
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

    start = ggml::gemmini::cycle::read();
    apply_ycom_to_output(scr.Y_com.data(), I, J, args);

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

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Apply Y_com to output", start, end);
    return result;
}
} // namespace ggml::gemmini::quants::dec
