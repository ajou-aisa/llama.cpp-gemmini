#include "dec.hpp"
#include "dec_kernel.hpp"
#include "dec_stage.hpp"
#include "dec_weight.hpp"
#include "../../ggml-gemmini-args.h"
#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants
{
    std::vector<QactOutlier> activation_outliers(const ggml_gemmini_args_t &args);
}

namespace ggml::gemmini::quants::dec { namespace
{
    struct Q80RDECScratch
    {
        std::vector<float> weight_scales;
    };

    Q80RDECScratch &get_q80_r_dec_scratch()
    {
        static thread_local Q80RDECScratch scratch;
        return scratch;
    }

    bool is_q80_r_weight_args(const ggml_gemmini_args_t &args)
    {
        return args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0;
    }

    WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args) || args.transpose_B)
            return WeightLayout::JxK_ColMajor;

        return WeightLayout::KxJ_RowMajor;
    }

    size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args))
            return args.K;

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
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
}

ActivationDECScratch &get_dec_scratch()
{
    static thread_local ActivationDECScratch scratch;
    return scratch;
}

ActivationDECResult compensate_activation_dec(
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const ActivationDECConfig &cfg,
    ActivationDECScratch *scratch)
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
    if (!weight_scales.supported)
    {
        ggml::gemmini::log::debug(
            cfg.layer,
            "[dec] reject unsupported stripe metadata path: stripe_J=%zu missing shared stripe scales",
            args.stripe_J);
        return result;
    }

    ActivationDECScratch &scr = scratch ? *scratch : get_dec_scratch();

    uint64_t end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Ready compensation", start, end);

    start = ggml::gemmini::cycle::read();

    const bool need_ycom = !cfg.fuse_apply;
    const bool decode = I == 1;

    scr.resize_for_dims(I, K, J, need_ycom);
    scr.reset_counts(K);
    scr.reset_stage();
    if (decode)
        scr.reset_i1_delta(K);

    if (need_ycom)
        scr.reset_ycom(I, J);

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
        build_rk_csc(K, scr);
        result.nnz = get_rk_nnz(K, scr);
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
        need_ycom &&
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

                const float delta = scr.i1_delta_by_k[k_sz];
                std::pair<int, float> single_pair[] = {{0, delta}};

                load_weight_row_scaled(
                    k_sz,
                    args,
                    weight_scales.data,
                    weight_scales.rows,
                    weight_scales.cols,
                    weight_scales.block_size,
                    scr.Wk_f.data());

                if (cfg.fuse_apply)
                {
                    accumulate_to_output(
                        scr.Wk_f.data(),
                        J,
                        0,
                        1,
                        single_pair,
                        args,
                        cfg.unroll8);
                }
                else
                {
                    accumulate_to_ycom(
                        scr.Wk_f.data(),
                        J,
                        0,
                        1,
                        single_pair,
                        scr.Y_com.data(),
                        cfg.unroll8);
                }
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

            if (cfg.fuse_apply)
            {
                accumulate_to_output(
                    scr.Wk_f.data(),
                    J,
                    beg,
                    rk_end,
                    scr.rk_pairs.data(),
                    args,
                    cfg.unroll8);
            }
            else
            {
                accumulate_to_ycom(
                    scr.Wk_f.data(),
                    J,
                    beg,
                    rk_end,
                    scr.rk_pairs.data(),
                    scr.Y_com.data(),
                    cfg.unroll8);
            }
        }
    }

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Compute and accumulate compensation", start, end);

    start = ggml::gemmini::cycle::read();
    if (!cfg.fuse_apply)
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
                        total_compensation += std::fabs(scr.rk_pairs[t].second);
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
