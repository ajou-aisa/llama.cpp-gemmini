// Copied from ggml-gemmini/quants/act/quantize.cpp. The Gemmini geometry
// pre-check and the dispatch over EXSIA/TOKEN/BLOCK/STRIPE are replaced by the
// single TENSOR algorithm the build selects (matmul/options.hpp).

#include "quantize.hpp"
#include "../../ggml-potal-args.h"
#include "../common/tensor_util.hpp"
#include "tensor/tensor.hpp"

#include <variant>

namespace ggml::potal::quants {

namespace {

void reset_activation_output(ggml_potal_args_t & args) {
    if (!args.A.valid() || args.I == 0 || args.K == 0) {
        return;
    }
    args.A.zero_fill();
}

} // namespace

bool quantize_activation(const ggml_tensor * src, ggml_potal_args_t & args) {
    reset_activation_quant_state(args);

    if (!src || src->type != GGML_TYPE_F32 || !args.A.valid() || args.I == 0 || args.K == 0) {
        reset_activation_output(args);
        return false;
    }

    if (!ggml::potal::activation_data(src)) {
        reset_activation_output(args);
        return false;
    }

    // GGML_GEMMINI_ACTIVATION_QUANT=TENSOR (the only value ggml-potal builds with).
    auto & meta = args.act_quant.storage().emplace<act::tensor::Meta>();
    act::tensor::set_config(meta);
    if (!act::tensor::quantize(src, args)) {
        reset_activation_quant_state(args);
        reset_activation_output(args);
        return false;
    }
    return true;
}

bool dequantize_activation(float *                   dst,
                           size_t                    dst_row_stride,
                           size_t                    dst_col_stride,
                           size_t                    rows,
                           size_t                    cols,
                           const ggml_potal_args_t & args) {
    if (!dst || dst_row_stride == 0 || dst_col_stride == 0 || rows == 0 || cols == 0) {
        return false;
    }

    return act::tensor::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
}

void reset_activation_quant_state(ggml_potal_args_t & args) {
    args.act_quant.reset();
}

} // namespace ggml::potal::quants
