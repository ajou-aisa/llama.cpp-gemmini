#pragma once

#include "types.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::ethos
{
struct BlockState
{
    std::vector<int16_t> e; // block exponent list
    int16_t e_max = std::numeric_limits<int16_t>::min(); // block max exponent
    int16_t e_max2 = std::numeric_limits<int16_t>::min(); // block second max exponent;
    int16_t e_b = std::numeric_limits<int16_t>::min(); // block inliers's scale(max) exponent
    int16_t e_b2 = std::numeric_limits<int16_t>::min(); // block inlier's second max exponent
    int16_t s_e = std::numeric_limits<int16_t>::min();

    std::vector<float> x; // temporal
    std::vector<int> q;

    int64_t S = 0;
    int64_t S2 = 0;
    uint32_t outlier_mask = 0; // block outlier mask
};

struct StripeState
{
    size_t blk_num = 0; // total blocks

    int16_t e_min = std::numeric_limits<int16_t>::max();

    // stripe exponent
    int16_t e_s = std::numeric_limits<int16_t>::min();

    size_t num_real_blocks = 0;
};

struct Metadata
{
    std::vector<BlockState> block;
    StripeState stripe;
};

class Initializer
{
public:
    bool init(
        Config &cfg,
        Metadata &meta,
        const char *data_ptr,
        size_t stride_i_bytes,
        size_t stride_k_bytes,
        size_t rows,
        size_t cols);
};

class Quantizer
{
public:
    int quantize(float x, int16_t s_e);
};

class Clipper
{
public:
    std::pair<int8_t, bool> clip(int q, Config &cfg);
};

class L1Detector
{
public:
    void detect_l1(Config &cfg, Metadata &meta, int blk_idx, Quantizer &quantizer, Clipper &clipper);
};

class L2Detector
{
public:
    void detect_l2(Config &cfg, Metadata &meta, int blk_idx);
};

class Ethos
{
private:
    Metadata meta_;
    Initializer unit_i_;
    Quantizer unit_q_;
    Clipper unit_c_;
    L1Detector unit_l1_;
    L2Detector unit_l2_;

public:
    bool run(
        Config &cfg,
        const char *data_ptr,
        size_t stride_i_bytes,
        size_t stride_k_bytes,
        size_t rows,
        size_t cols,
        size_t row_offset,
        size_t col_offset,
        int8_t *dst);

    const Metadata &metadata() const
    {
        return meta_;
    }
};

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride);

} // namespace ggml::gemmini::quants::act::ethos
