#include <cstddef>
#include <cstdint>

#include "../../ggml-gemmini-args.h"
#include "../../ggml-gemmini-config.hpp"
#include "../act/ethos/ethos.hpp"
#include "../act/tensor/tensor.hpp"

namespace ggml::gemmini {

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::ETHOS:
    default:
        quants::act::ethos::dequantize(args, k_offset, block_k, acc32, acc_stride);
        break;
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        quants::act::tensor::dequantize(args, k_offset, block_k, acc32, acc_stride);
        break;
    }
}

}
