#include "dispatch.hpp"
#include "quantize.hpp"
#include "../../types/matmul.hpp"
#include "../common/tensor_util.hpp"

#include <vector>

namespace ggml::npu::quants {

namespace {

void reset_activation_output(npu_matmul_args & args) {
    if (!args.A.valid() || args.I == 0 || args.K == 0) {
        return;
    }
    args.A.zero_fill();
}

} // namespace

bool quantize_activation(const ggml_tensor * src, npu_matmul_args & args) {
    ggml::npu::NpuGeometry geometry;
    if (!args.activation_quant_geometry_matches(geometry)) {
        reset_activation_quant_state(args);
        return false;
    }

    reset_activation_quant_state(args);

    if (!src || src->type != GGML_TYPE_F32 || !args.A.valid() || args.I == 0 || args.K == 0) {
        reset_activation_output(args);
        return false;
    }

    if (!ggml::npu::activation_data(src)) {
        reset_activation_output(args);
        return false;
    }

    if (!act::quantize(src, args)) {
        reset_activation_quant_state(args);
        reset_activation_output(args);
        return false;
    }

    args.gemmini_call_k_logical    = args.K;
    args.gemmini_call_k_aligned    = args.K;
    args.gemmini_call_tile_k_elems = args.tile_K > 0 ? args.tile_K * NPU_D : args.K;
    return true;
}

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const npu_matmul_args & args) {
    if (!dst || dst_row_stride == 0 || dst_col_stride == 0 || rows == 0 || cols == 0) {
        return false;
    }

    return act::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
}

void reset_activation_quant_state(npu_matmul_args & args) {
    args.act_quant.reset();
    args.gemmini_call_k_logical    = 0;
    args.gemmini_call_k_aligned    = 0;
    args.gemmini_call_tile_k_elems = 0;
}

const act::RmdPacketList & activation_rmd_packets(const npu_matmul_args & args) {
    return act::rmd_packets(args);
}

const act::DirectResidualList & activation_direct_residuals(const npu_matmul_args & args) {
    return act::direct_residuals(args);
}
} // namespace ggml::npu::quants
