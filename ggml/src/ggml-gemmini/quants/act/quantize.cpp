#include "dispatch.hpp"
#include "quantize.hpp"
#include "../../ggml-gemmini-args.h"
#include "../common/tensor_util.hpp"

#include <vector>

namespace ggml::gemmini::quants {

namespace {

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }

    out = lhs * rhs;
    return true;
}

void reset_activation_output(ggml_gemmini_args_t &args)
{
    if (!args.A.valid() || args.I == 0 || args.K == 0) {
        return;
    }
    args.A.zero_fill();
}

}

bool quantize_activation(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    reset_activation_quant_state(args);

    if (!src || src->type != GGML_TYPE_F32 || !args.A.valid() || args.I == 0 || args.K == 0) {
        reset_activation_output(args);
        return false;
    }

    if (!ggml::gemmini::activation_data(src)) {
        reset_activation_output(args);
        return false;
    }

    if (!act::quantize(src, args)) {
        reset_activation_quant_state(args);
        reset_activation_output(args);
        return false;
    }

    args.gemmini_call_k_logical = args.K;
    args.gemmini_call_k_aligned = args.K;
    args.gemmini_call_tile_k_elems = args.tile_K > 0 ? args.tile_K * DIM : args.K;
    return true;
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

const act::RmdPacketList &activation_rmd_packets(const ggml_gemmini_args_t &args)
{
    return act::rmd_packets(args);
}

const act::DirectResidualList &activation_direct_residuals(const ggml_gemmini_args_t &args)
{
    return act::direct_residuals(args);
}
} // namespace ggml::gemmini::quants
