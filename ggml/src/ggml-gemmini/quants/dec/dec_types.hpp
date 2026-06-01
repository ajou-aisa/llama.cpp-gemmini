// DEC (Dynamic Error Compensation) common types - ggml-free
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants
{
    struct QactOutlier;
}

namespace ggml::gemmini::quants::dec
{
    enum class WeightLayout
    {
        KxJ_RowMajor,
        JxK_ColMajor,
    };

    struct RkTriplet
    {
        int k;
        int r;
        float d;
    };

    struct ActivationDECScratch
    {
        std::vector<size_t> rk_counts;
        std::vector<size_t> rk_offs;
        std::vector<std::pair<int, float>> rk_pairs;
        std::vector<int> unique_k;
        std::vector<RkTriplet> rk_stage;
        std::vector<float> i1_delta_by_k;
        double i1_total_abs_residual = 0.0;
        std::vector<float> Wk_f;
        std::vector<float> Y_com;

        void resize_for_dims(size_t I, size_t K, size_t J, bool need_ycom)
        {
            rk_counts.assign(K + 1, 0);
            rk_offs.resize(K + 1);
            unique_k.clear();
            unique_k.reserve(K);
            rk_stage.clear();
            rk_stage.reserve(I);
            Wk_f.resize(J);

            if (need_ycom)
            {
                Y_com.assign(I * J, 0.f);
            }
            else
            {
                Y_com.clear();
            }
        }

        void reset_counts(size_t K)
        {
            if (rk_counts.size() != K + 1)
            {
                rk_counts.assign(K + 1, 0);
            }
            else
            {
                std::fill(rk_counts.begin(), rk_counts.end(), size_t {0});
            }
        }

        void reset_stage()
        {
            rk_stage.clear();
            unique_k.clear();
        }

        void reset_i1_delta(size_t K)
        {
            if (i1_delta_by_k.size() != K)
            {
                i1_delta_by_k.assign(K, 0.f);
            }
            else
            {
                std::fill(i1_delta_by_k.begin(), i1_delta_by_k.end(), 0.f);
            }

            i1_total_abs_residual = 0.0;
        }

        void reset_ycom(size_t I, size_t J)
        {
            const size_t sz = I * J;
            if (Y_com.size() != sz)
            {
                Y_com.assign(sz, 0.f);
            }
            else
            {
                std::fill(Y_com.begin(), Y_com.end(), 0.f);
            }
        }
    };

    ActivationDECScratch &get_dec_scratch();

} // namespace ggml::gemmini::quants::dec
