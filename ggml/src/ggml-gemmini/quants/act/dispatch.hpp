#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act {

class ActivationMetadataView {
public:
    ActivationMetadataView(const ggml_gemmini_args_t &source,
                           size_t global_row_begin,
                           size_t global_row_end);

    bool valid() const;
    size_t row_count() const;
    size_t stripe_count() const;
    bool global_row(size_t local_row, size_t &global_row) const;
    bool global_stripe(size_t local_stripe, size_t &global_stripe) const;
    bool scale(size_t local_row, float &scale) const;
    bool theta(size_t local_stripe, int16_t &theta) const;

private:
    const ggml_gemmini_args_t *source_ = nullptr;
    size_t global_row_begin_ = 0;
    size_t global_row_end_ = 0;
    size_t global_stripe_begin_ = 0;
    size_t global_stripe_end_ = 0;
    size_t rows_per_stripe_ = 0;
    bool valid_ = false;
};

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args);

bool dequantize_activation(float *dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t &args);

const RmdPacketList &rmd_packets(const ggml_gemmini_args_t &args);
const DirectResidualList &direct_residuals(const ggml_gemmini_args_t &args);

std::vector<float> activation_scales(const ggml_gemmini_args_t &args, size_t row_count);

} // namespace ggml::gemmini::quants::act
