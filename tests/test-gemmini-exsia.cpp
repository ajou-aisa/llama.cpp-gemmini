#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"

#include <ggml.h>
#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
#include <gemmini.h>
#endif

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace ggml::gemmini;

bool check(bool value, const char * message) {
    if (!value)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
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
#endif

bool profile_output_routing(const std::filesystem::path & expected, bool invalid_parent) {
    if (invalid_parent) {
        std::ofstream("blocked") << "not a directory";
    }

    std::vector<float> source(64);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = index % 4 == 0 ? -0.5f : 0.5f;
    }
    std::vector<elem_t> quantized(64);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = source.size();
    args.A = quantized.data();
    args.sA = source.size();

    const bool wrote = quants::quantize_activation(&tensor, args);
    if (invalid_parent) {
        return check(!wrote, "invalid ExSIA profile parent reports existing writer failure") &&
            check(!std::filesystem::exists(expected), "invalid ExSIA profile path is absent") &&
            check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "invalid ExSIA profile does not fall back to legacy log");
    }

    std::ifstream input(expected);
    std::string line;
    const bool parsed = std::getline(input, line) && !line.empty() && line.front() == '{' &&
        line.back() == '}' && line.find("\"record_type\":\"TIMELINE\"") != std::string::npos;
    return check(wrote, "ExSIA profile writer succeeds") &&
        check(parsed, "ExSIA profile writer emits non-empty JSONL") &&
        check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "ExSIA profile writer does not create legacy log");
}

}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--profile-output") {
        const bool invalid_parent = argc == 4 && std::string(argv[3]) == "--invalid-parent";
        return profile_output_routing(argv[2], invalid_parent) ? 0 : 1;
    }
#ifdef GEMMINI_EXSIA_WRITER_TEST_ONLY
    return 1;
#else
    const bool ok = test_exsia_baseline() && test_dispatch_modes();
    if (ok)
        std::puts("PASS: ExSIA baseline, RMD packet empty-residual handoff, dispatch capability");
    return ok ? 0 : 1;
#endif
}
