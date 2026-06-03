#include "tensor.hpp"

namespace ggml::gemmini::quants::act::tensor
{

void set_config(Config &cfg) {
    (void)cfg;
}

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    (void)args;
    (void)k_offset;
    (void)block_k;
    (void)acc32;
    (void)acc_stride;
}

}
