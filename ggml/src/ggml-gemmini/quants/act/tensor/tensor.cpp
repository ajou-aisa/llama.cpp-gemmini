#include "tensor.hpp"

#include "../../../ggml-gemmini-args.h"

namespace ggml::gemmini::quants::act::tensor
{
    // per-tensor에서는 의미 없음. 
void set_config(Config &cfg)
{
    (void)cfg;
}

void set_scale(const ggml_gemmini_args_t &args, Config &cfg)
{
    // TODO: per-tensor scale 결정. outlier 리스트에 해당하는 값은 scale 계산에 포함하지 않도록 해야 함
    (void)args;
    (void)cfg;
}

bool quantize(
    Config &cfg,
    const ggml_tensor *src,
    const ggml_gemmini_args_t &args,
    int8_t *dst)
{
    // TODO: per-tensor quantization. 
    // Outlier는 옵션에 따라 처리. 
    (void)cfg;
    (void)src;
    (void)args;
    (void)dst;

    #if ERROR_COMPENSATION
        // TODO 3sigma rule (mean, sigma) 적용하여 outlier selection
        // inlier만 quantization.
    #endif
    // TODO: outlier selection하지 않고 모든 값을 inlier로 간주.

    // TODO: quantization 완료 시 true 반환, 실패 시 false 반환
    return false;
}

void dequantize(
    const ggml_gemmini_args_t &args,
    const int32_t *acc32,
    size_t acc_stride)
{
    // TODO: per-tensor dequantization.
    (void)args;
    (void)acc32;
    (void)acc_stride;
}

}
