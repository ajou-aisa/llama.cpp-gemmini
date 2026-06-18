#include "dec_stage.hpp"
#include <gemmini/log.hpp>

#include <cmath>

namespace ggml::gemmini::quants::dec { size_t stage_from_outliers(
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

        const float residual = outlier.original - outlier.saturated;
        scratch.rk_stage.push_back({k, static_cast<int>(r), residual});
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
            const double abs_res = std::fabs(scratch.rk_stage[i].d);
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

        const float residual = outlier.original - outlier.saturated;
        if (scratch.rk_counts[k_sz + 1] == 0)
            scratch.unique_k.push_back(k);

        scratch.i1_delta_by_k[k_sz] += residual;
        scratch.i1_total_abs_residual += std::fabs(residual);
        scratch.rk_counts[k_sz + 1]++;
        ++staged;

#if LOG_DEBUG
        const double abs_res = std::fabs(residual);
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

void build_rk_csc(size_t K, ActivationDECScratch &scratch)
{
    if (K == 0)
        return;

    if (scratch.rk_offs.size() != K + 1)
        scratch.rk_offs.resize(K + 1);

    for (size_t k = 0; k <= K; ++k)
        scratch.rk_offs[k] = scratch.rk_counts[k];

    for (size_t k = 1; k <= K; ++k)
        scratch.rk_offs[k] += scratch.rk_offs[k - 1];

    const size_t nnz = scratch.rk_offs[K];
    scratch.rk_pairs.assign(nnz, {0, 0.f});

    if (nnz == 0)
    {
        scratch.unique_k.clear();
        return;
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
} }
