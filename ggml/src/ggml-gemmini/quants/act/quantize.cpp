#include "dispatch.hpp"
#include "quantize.hpp"
#include "../../ggml-gemmini-args.h"
#include "../common/tensor_util.hpp"

#include <vector>
#include <variant>

namespace ggml::gemmini::quants {

void quantize_activation(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    reset_activation_quant_state(args);

    args.gemmini_call_k_logical = args.K;
    args.gemmini_call_k_aligned = args.K;
    args.gemmini_call_tile_k_elems = args.tile_K > 0 ? args.tile_K * DIM : args.K;

    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0) {
        return;
    }

    if (!ggml::gemmini::activation_data(src)) {
        return;
    }

    act::quantize(src, args);
}

bool dequantize_activation(float *dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t &args)
{
    if (!dst || dst_row_stride == 0 || dst_col_stride == 0 || rows == 0 || cols == 0) {
        return false;
    }

    return act::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
}

void reset_activation_quant_state(ggml_gemmini_args_t &args) {
    args.act_quant.reset();
    args.gemmini_call_k_logical = 0;
    args.gemmini_call_k_aligned = 0;
    args.gemmini_call_tile_k_elems = 0;
}

void init_exsia_meta(ggml_gemmini_args_t &args)
{
    args.act_quant.storage().emplace<act::exsia::Meta>();
}

const act::exsia::Meta &get_exsia_meta(const ggml_gemmini_args_t &args)
{
    return std::get<act::exsia::Meta>(args.act_quant.storage());
}

act::exsia::Meta &get_exsia_meta_mut(ggml_gemmini_args_t &args)
{
    return std::get<act::exsia::Meta>(args.act_quant.storage());
}

std::vector<QactOutlier> activation_outliers(const ggml_gemmini_args_t &args)
{
    return act::outliers(args);
}
} // namespace ggml::gemmini::quants
