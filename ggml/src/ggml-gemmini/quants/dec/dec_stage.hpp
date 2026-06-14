#pragma once

#include "dec_types.hpp"

#include <cstddef>
#include <vector>

namespace ggml::gemmini::quants::dec
{
size_t stage_from_outliers(
    const std::vector<QactOutlier> &outliers,
    size_t I,
    size_t K,
    ActivationDECScratch &scratch);

size_t stage_from_outliers_i1(
    const std::vector<QactOutlier> &outliers,
    size_t K,
    ActivationDECScratch &scratch);

void build_rk_csc(size_t K, ActivationDECScratch &scratch);

inline size_t get_rk_nnz(size_t K, const ActivationDECScratch &scratch)
{
    if (scratch.rk_offs.size() <= K)
        return 0;

    return scratch.rk_offs[K];
}

} // namespace ggml::gemmini::quants::dec
