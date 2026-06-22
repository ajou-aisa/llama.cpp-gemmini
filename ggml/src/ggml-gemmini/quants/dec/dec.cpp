#include "dec.hpp"
#include "dec_internal.hpp"
#include "dec_kernel.hpp"
#include "../act/dispatch.hpp"
#include "../../ggml-gemmini-args.h"
#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants
{
    std::vector<QactOutlier> activation_outliers(const ggml_gemmini_args_t &args);
}

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

        void reset_counts(size_t K)
        {
            if (rk_counts.size() != K + 1)
                rk_counts.assign(K + 1, 0);
            else
                std::fill(rk_counts.begin(), rk_counts.end(), size_t {0});
        }

        void reset_stage()
        {
            rk_stage.clear();
            unique_k.clear();
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

    struct Q80RDECScratch
    {
        std::vector<float> weight_scales;
    };

    Q80RDECScratch &get_q80_r_dec_scratch()
    {
        static thread_local Q80RDECScratch scratch;
        return scratch;
    }

    struct WeightScaleInfo
    {
        const float *data = nullptr;
        size_t rows = 0;
        size_t cols = 0;
        size_t block_size = 0;
        bool supported = true;
    };

    WeightScaleInfo build_weight_scale_info(const ggml_gemmini_args_t &args)
    {
        WeightScaleInfo result{};
        if (is_q80_r_weight_args(args))
        {
            auto &scratch = get_q80_r_dec_scratch();
            const size_t rows = args.blocks_J ? args.blocks_J : args.J;
            const size_t cols = args.blocks_per_row;
            const size_t block_size = 32;
            const bool stripe_mode = args.stripe_J > 1;

            if (rows == 0 || cols == 0 ||
                (args.block_size_k != 0 && args.block_size_k != block_size) ||
                rows > std::numeric_limits<size_t>::max() / cols)
            {
                return result;
            }

            if (stripe_mode && (!args.s_rf_stripe || !args.R_stripe))
            {
                result.supported = false;
                return result;
            }

            if (!stripe_mode && (!args.s_rf || !args.R))
                return result;

            scratch.weight_scales.resize(rows * cols);
            for (size_t j = 0; j < rows; ++j)
            {
                const size_t stripe_idx = stripe_mode ? (j / args.stripe_J) : 0;
                const float s_rf = stripe_mode ? args.s_rf_stripe[stripe_idx] : args.s_rf[j];
                const uint16_t R = stripe_mode ? args.R_stripe[stripe_idx] : args.R[j];

                for (size_t blk = 0; blk < cols; ++blk)
                {
                    const size_t idx = j * cols + blk;
                    const uint64_t c_eff =
                        static_cast<uint64_t>(static_cast<uint16_t>(args.c_b[idx])) +
                        static_cast<uint64_t>(R);
                    scratch.weight_scales[idx] = static_cast<float>(
                        static_cast<double>(s_rf) * static_cast<double>(c_eff));
                }
            }

            result.data = scratch.weight_scales.data();
            result.rows = rows;
            result.cols = cols;
            result.block_size = block_size;
            return result;
        }

        if (!args.B_scales)
            return result;

        result.data = args.B_scales;
        result.rows = args.blocks_J;
        result.cols = args.blocks_K;
        result.block_size = args.block_size_k ? args.block_size_k : QK8_0;
        return result;
    }

    void load_weight_row_scaled(
        size_t k,
        const ggml_gemmini_args_t &args,
        const float *weight_scales,
        size_t scale_rows,
        size_t blocks_k,
        size_t block_size_k,
        float *Wk_f)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        const size_t J = args.J;
        if (!weights || !Wk_f || J == 0)
            return;

        const size_t weight_stride = resolve_weight_stride_elems(args);
        if (weight_stride == 0)
            return;

        if (resolve_weight_layout(args) == WeightLayout::KxJ_RowMajor)
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

        if (weight_scales && block_size_k > 0 && blocks_k > 0)
        {
            const size_t blk = k / block_size_k;

            for (size_t j = 0; j < J; ++j)
            {
                if (j < scale_rows && blk < blocks_k)
                    Wk_f[j] *= weight_scales[j * blocks_k + blk];
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
    const ActivationDECConfig &cfg)
{
    uint64_t start = ggml::gemmini::cycle::read();
    const char *layer = cfg.layer;

    ActivationDECResult result{};
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights || !args.f_out)
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
    const WeightScaleInfo weight_scales = build_weight_scale_info(args);
    auto activation_scales_vec = act::activation_scales(args, I);
    const float *activation_scales = activation_scales_vec.data();
    if (!weight_scales.supported)
    {
        ggml::gemmini::log::debug(
            cfg.layer,
            "[dec] reject unsupported stripe metadata path: stripe_J=%zu missing shared stripe scales",
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
    {
        result.success = true;
        return result;
    }

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
    {
        result.success = true;
        return result;
    }

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Build R_k CSC structure", start, end);

    start = ggml::gemmini::cycle::read();

    const bool use_jmajor_blocked =
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

    result.success = true;

#if LOG_DEBUG
    if (cfg.record_stats)
    {
        ggml::gemmini::log::debug(
            cfg.layer,
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
                cfg.layer,
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

bool should_apply_dec(const ggml_gemmini_args_t &args) {
    if (!args.B || !args.f_out)
        return false;

    if (args.I == 0 || args.K == 0 || args.J == 0)
        return false;

    return true;
}

void append_activation_outliers(
    const ggml_gemmini_args_t &args,
    std::vector<QactOutlier> &outliers) {
    outliers = ggml::gemmini::quants::activation_outliers(args);
}
} // namespace ggml::gemmini::quants::dec
