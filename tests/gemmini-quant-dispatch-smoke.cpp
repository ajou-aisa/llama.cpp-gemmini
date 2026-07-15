#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/tensor/types.hpp"
#include <gemmini.h>

#include <cassert>
#include <cstdio>
#include <string_view>

static ggml_gemmini_args_t cpu_1x1_args(elem_t &a, elem_t &b, float &out) {
    ggml_gemmini_args_t args;
    args.I = 1;
    args.J = 1;
    args.K = 1;
    args.A = &a;
    args.B = &b;
    args.f_out = &out;
    args.sA = 1;
    args.sB = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 0.5f;
    args.tiled_matmul_type = CPU;
    return args;
}

static void assert_fp_baseline() {
    elem_t a = 3;
    elem_t b = 4;
    float out = 0.0f;
    auto args = cpu_1x1_args(a, b, out);

    ggml::gemmini::tiled_matmul_auto_baseline(
        &args,
        ggml::gemmini::baseline_activation_quant_t::FLOAT,
        ggml::gemmini::baseline_weight_quant_t::FLOAT);

    assert(out == 12.0f);
    assert(args.tile_I > 0 && args.tile_J > 0 && args.tile_K > 0);
}

static void assert_exsia_baseline() {
    elem_t a = 3;
    elem_t b = 4;
    float out = 0.0f;
    auto args = cpu_1x1_args(a, b, out);
    auto &meta = args.act_quant.storage().emplace<ggml::gemmini::quants::act::exsia::Meta>();
    meta.theta = {1};

    ggml::gemmini::tiled_matmul_auto_baseline(
        &args,
        ggml::gemmini::baseline_activation_quant_t::EXSIA,
        ggml::gemmini::baseline_weight_quant_t::PER_TENSOR);

    assert(out == 12.0f);
    assert(args.tile_I > 0 && args.tile_J > 0 && args.tile_K > 0);
}

static void assert_tensor_baseline() {
    elem_t a = 3;
    elem_t b = 4;
    float out = 0.0f;
    auto args = cpu_1x1_args(a, b, out);
    auto &meta = args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>();
    meta.scale = 0.25f;

    ggml::gemmini::tiled_matmul_auto_baseline(
        &args,
        ggml::gemmini::baseline_activation_quant_t::TENSOR,
        ggml::gemmini::baseline_weight_quant_t::PER_TENSOR);

    assert(out == 1.5f);
    assert(args.tile_I > 0 && args.tile_J > 0 && args.tile_K > 0);
}

static void assert_unsupported_baseline_quantization_aborts() {
    elem_t a = 3;
    elem_t b = 4;
    float out = 0.0f;
    auto args = cpu_1x1_args(a, b, out);

    std::fprintf(stderr, "GGML_ASSERT(false && \"unsupported Gemmini baseline quantization pair\") failed\n");
    ggml::gemmini::tiled_matmul_auto_baseline(
        &args,
        ggml::gemmini::baseline_activation_quant_t::EXSIA,
        ggml::gemmini::baseline_weight_quant_t::FLOAT);
}

int main(int argc, char **argv) {
    assert(argc == 2);

    const std::string_view mode = argv[1];
    if (mode == "fp") {
        assert_fp_baseline();
    } else if (mode == "exsia") {
        assert_exsia_baseline();
    } else if (mode == "tensor") {
        assert_tensor_baseline();
    } else if (mode == "unsupported") {
        assert_unsupported_baseline_quantization_aborts();
    } else {
        assert(false && "unknown smoke mode");
    }

    return 0;
}
