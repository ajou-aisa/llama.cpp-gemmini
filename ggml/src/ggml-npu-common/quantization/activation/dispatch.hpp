#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct npu_matmul_args;

namespace ggml::npu::quants::act {

class ActivationMetadataView {
  public:
    ActivationMetadataView(const npu_matmul_args & source,
                           size_t                      global_row_begin,
                           size_t                      global_row_end);

    bool   valid() const;
    size_t row_count() const;
    size_t stripe_count() const;
    bool   global_row(size_t local_row, size_t & global_row) const;
    bool   global_stripe(size_t local_stripe, size_t & global_stripe) const;
    bool   scale(size_t local_row, float & scale) const;
    bool   scale(size_t local_row, size_t k, float & scale) const;
    bool   theta(size_t local_stripe, int16_t & theta) const;

  private:
    const npu_matmul_args * source_              = nullptr;
    size_t                      global_row_begin_    = 0;
    size_t                      global_row_end_      = 0;
    size_t                      global_stripe_begin_ = 0;
    size_t                      global_stripe_end_   = 0;
    size_t                      rows_per_stripe_     = 0;
    bool                        valid_               = false;
};

bool quantize(const ggml_tensor * src, npu_matmul_args & args);

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const npu_matmul_args & args);

const RmdPacketList &      rmd_packets(const npu_matmul_args & args);
const DirectResidualList & direct_residuals(const npu_matmul_args & args);

std::vector<float> activation_scales(const npu_matmul_args & args, size_t row_count);

} // namespace ggml::npu::quants::act
