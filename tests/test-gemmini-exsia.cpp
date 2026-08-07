#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"

#include <gemmini.h>

#include <cstdio>

namespace {

using namespace ggml::gemmini;

bool check(bool value, const char * message) {
    if (!value)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

bool test_exsia_baseline() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = 1;
    args.A = &activation;
    args.B = &weight;
    args.f_out = &output;
    args.sA = 1;
    args.sB = 1;
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    tiled_matmul_auto_baseline(&args, baseline_activation_quant_t::EXSIA,
                               baseline_weight_quant_t::TENSOR);
    return check(output == 12.0f, "ExSIA baseline output") &&
        check(meta.rmd_packets.empty(), "empty residual has no RMD packet");
}

bool test_dispatch_modes() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = args.J = args.K = 1;
    args.A = &activation;
    args.B = &weight;
    args.f_out = &output;
    args.sA = args.sB = args.stride_f_out = args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    const auto & packets = quants::act::rmd_packets(args);
    return check(packets.empty(), "dispatch exposes empty RMD packet list");
}

}

int main() {
    const bool ok = test_exsia_baseline() && test_dispatch_modes();
    if (ok)
        std::puts("PASS: ExSIA baseline, RMD packet empty-residual handoff, dispatch capability");
    return ok ? 0 : 1;
}
