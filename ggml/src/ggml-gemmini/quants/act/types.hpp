#pragma once

#include "../view/act_view.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ggml::gemmini::types
{
enum class ModelType : uint8_t;
enum class LayerType : uint8_t;
}

namespace ggml::gemmini::quants::act
{

struct L2Config
{
    int c = 1;
    int d = 1;
};

struct Result
{
    std::vector<::ggml::gemmini::quants::QactOutlier> outliers;
    int16_t e_s = std::numeric_limits<int16_t>::min();
    std::vector<int16_t> e_s_per_stripe;
    size_t num_padding_blocks = 0;
};

struct Config
{
    std::pair<ggml::gemmini::types::ModelType, ggml::gemmini::types::LayerType> preset = {
        static_cast<ggml::gemmini::types::ModelType>(0),
        static_cast<ggml::gemmini::types::LayerType>(0),
    };

    size_t block_size = 32;
    int16_t m = 6;
    int8_t qmax = 127;
    int delta = 0;
    bool l2_on = false;
    L2Config l2{};
    Result result{};
    size_t num_real_blocks = 0;
};

} // namespace ggml::gemmini::quants::act
