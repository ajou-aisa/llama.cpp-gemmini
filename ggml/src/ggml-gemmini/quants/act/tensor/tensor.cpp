#include "tensor.hpp"

#include "../../../ggml-gemmini-args.h"

#include <algorithm>
#include <variant>

namespace ggml::gemmini::quants::act::tensor
{
    // per-tensor에서는 의미 없음. 
void set_config(Meta &meta)
{
    (void)meta;
}

void set_scale(const ggml_gemmini_args_t &args, Meta &meta)
{
    // TODO: per-tensor scale 결정. outlier 리스트에 해당하는 값은 scale 계산에 포함하지 않도록 해야 함
    (void)args;
    (void)meta;
}

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    // TODO: per-tensor quantization. 
    // Outlier는 옵션에 따라 처리. 
    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0) {
        return false;
    }

    auto *meta = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta) {
        return false;
    }

    #if ERROR_COMPENSATION
        // TODO 3sigma rule (mean, sigma) 적용하여 outlier selection
        // inlier만 quantization.
    #endif
    // TODO: outlier selection하지 않고 모든 값을 inlier로 간주.

    // TODO: int32_t 타입으로 양자화 후 [-128, 127]을 초과하는 값의 clipping된 residual을 index와 함께 outlier로 저장.
    // TODO: quantization 완료 시 true 반환, 실패 시 false 반환
    return false;
}

bool dequantize_activation(
    float *dst,
    size_t dst_row_stride,
    size_t dst_col_stride,
    size_t rows,
    size_t cols,
    const ggml_gemmini_args_t &args)
{
    const int8_t *src = reinterpret_cast<const int8_t *>(args.A);
    if (!src || !dst || args.I == 0 || args.K == 0 ||
        dst_row_stride == 0 || dst_col_stride == 0 ||
        rows == 0 || cols == 0) {
        return false;
    }

    const auto *meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta_ptr) {
        return false;
    }
    const Meta &meta = *meta_ptr;

    const size_t src_row_stride = args.sA != 0 ? args.sA : args.K;
    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            const int8_t q = src[row * src_row_stride + col];
            dst[row * dst_row_stride + col * dst_col_stride] =
                static_cast<float>(q) * meta.scale;
        }
    }

    return true;
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
