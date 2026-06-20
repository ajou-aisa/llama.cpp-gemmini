#include "dequant.hpp"

#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"

namespace ggml::gemmini {

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    quants::act::dequantize(args, k_offset, block_k, acc32, acc_stride);
}

}
