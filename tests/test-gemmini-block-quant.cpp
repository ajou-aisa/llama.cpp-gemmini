#include "../ggml/include/ggml.h"

#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/block/block.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace act = ggml::gemmini::quants::act;
namespace block = ggml::gemmini::quants::act::block;

namespace {

bool metadata_tests() {
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.K = 64;
    args.activation_row_offset = 1;
    auto & meta = args.act_quant.storage().emplace<block::Meta>();
    meta.rows = 2;
    meta.cols = 64;
    meta.scales = {1.0f, 2.0f, 3.0f, 4.0f};
    meta.run_id = 0;

    act::ActivationMetadataView view(args, 1, 2);
    float scale = 0.0f;
    if (!view.valid() ||
        !view.scale(0, 0, scale) || scale != 3.0f ||
        !view.scale(0, 63, scale) || scale != 4.0f ||
        view.scale(0, 64, scale) || view.scale(0, scale)) {
        std::fputs("FAIL metadata: K-aware/offset lookup contract\n", stderr);
        return false;
    }

    const auto rejects = [&](std::vector<float> scales, size_t rows, size_t cols) {
        meta.scales = std::move(scales);
        meta.rows = rows;
        meta.cols = cols;
        return !act::ActivationMetadataView(args, 1, 2).valid();
    };
    if (!rejects({1.0f, 2.0f, 3.0f}, 2, 64) ||
        !rejects({1.0f, 2.0f, 3.0f, 4.0f}, 1, 64) ||
        !rejects({1.0f, 2.0f, 3.0f, 4.0f}, 2, 63)) {
        std::fputs("FAIL metadata: shape/cardinality accepted\n", stderr);
        return false;
    }
    meta.rows = 2;
    meta.cols = 64;
    meta.scales = {1.0f, 2.0f, 0.0f, 4.0f};
    if (act::ActivationMetadataView(args, 1, 2).valid()) {
        std::fputs("FAIL metadata: zero scale accepted\n", stderr);
        return false;
    }
    meta.scales[2] = std::numeric_limits<float>::quiet_NaN();
    if (act::ActivationMetadataView(args, 1, 2).valid()) {
        std::fputs("FAIL metadata: NaN scale accepted\n", stderr);
        return false;
    }

    meta.scales = {1.0f};
    meta.rows = meta.cols = 7;
    meta.rmd_packets.emplace_back();
    meta.direct_residuals.emplace_back();
    meta.reset();
    if (!meta.scales.empty() || !meta.rmd_packets.empty() ||
        !meta.direct_residuals.empty() || meta.rows != 0 || meta.cols != 0 ||
        meta.run_id.has_value()) {
        std::fputs("FAIL metadata: reset left stale state\n", stderr);
        return false;
    }
    ggml_gemmini_args_t token_args{};
    token_args.I = 2;
    token_args.K = 64;
    auto & token = token_args.act_quant.storage().emplace<act::token::Meta>();
    token.scales = {5.0f, 6.0f};
    act::ActivationMetadataView token_view(token_args, 0, 2);
    if (!token_view.scale(1, scale) || scale != 6.0f ||
        !token_view.scale(1, 63, scale) || scale != 6.0f) {
        std::fputs("FAIL metadata: non-BLOCK lookup changed\n", stderr);
        return false;
    }
    std::puts("metadata: shape/scale/offset/reset checks passed");
    return true;
}

bool quantize_snapshot(const std::vector<float> & source,
                       size_t rows,
                       size_t cols,
                       std::vector<float> & scales,
                       std::vector<int32_t> & codes) {
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = const_cast<float *>(source.data());
    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = 1;
    args.K = cols;
    args.sA = cols;
    args.A.allocate(rows, cols, GGML_GEMMINI_ACTIVATION_BITS);
    args.act_quant.storage().emplace<block::Meta>();
    if (!block::quantize(&tensor, args)) return false;
    scales = std::get<block::Meta>(args.act_quant.storage()).scales;
    codes.resize(rows * cols);
    for (size_t row = 0; row < rows; ++row)
        for (size_t k = 0; k < cols; ++k)
            codes[row * cols + k] = args.A.get(row, k);
    return true;
}

bool isolation_test() {
    constexpr size_t rows = 2;
    constexpr size_t cols = 64;
    std::vector<float> base(rows * cols, 1.0f);
    std::fill(base.begin() + 32, base.begin() + 64, 10.0f);
    std::fill(base.begin() + 64, base.end(), 100.0f);
    std::vector<float> row_changed = base;
    std::fill(row_changed.begin() + 64, row_changed.end(), 10000.0f);
    std::vector<float> block_changed = base;
    std::fill(block_changed.begin() + 32, block_changed.begin() + 64, -1000.0f);
    std::vector<float> base_scales, row_scales, block_scales;
    std::vector<int32_t> base_codes, row_codes, block_codes;
    if (!quantize_snapshot(base, rows, cols, base_scales, base_codes) ||
        !quantize_snapshot(row_changed, rows, cols, row_scales, row_codes) ||
        !quantize_snapshot(block_changed, rows, cols, block_scales, block_codes))
        return false;
    return base_scales[0] == row_scales[0] && base_scales[1] == row_scales[1] &&
        std::equal(base_codes.begin(), base_codes.begin() + cols, row_codes.begin()) &&
        base_scales[0] == block_scales[0] &&
        std::equal(base_codes.begin(), base_codes.begin() + 32, block_codes.begin());
}

bool quant_case(size_t cols, act::ResidualRoute route) {
    const int32_t qmax = ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX;
    std::vector<float> source(cols, 1.0f);
    for (size_t k = 32; k < cols; ++k) source[k] = -10.0f;
    if (cols >= 17) source[16] = 100.0f;
    if (cols >= 3) {
        source[0] = std::numeric_limits<float>::quiet_NaN();
        source[1] = std::numeric_limits<float>::infinity();
        source[2] = -std::numeric_limits<float>::infinity();
    }

    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = cols;
    args.sA = cols;
    args.residual_route = route;
    args.A.allocate(1, cols, GGML_GEMMINI_ACTIVATION_BITS);
    args.act_quant.storage().emplace<block::Meta>();
    if (!block::quantize(&tensor, args)) return false;

    auto & meta = std::get<block::Meta>(args.act_quant.storage());
    if (!meta.run_id.has_value())
        return false;
    const auto first_run_id = meta.run_id;
    if (!block::quantize(&tensor, args) || !meta.run_id.has_value() ||
        meta.run_id == first_run_id)
        return false;
    const size_t block_count = cols / block::kGroupSize + (cols % block::kGroupSize != 0);
    if (meta.rows != 1 || meta.cols != cols || meta.scales.size() != block_count)
        return false;
    const float first_scale =
#if GGML_GEMMINI_ENABLE_RMD
        1.0f / static_cast<float>(qmax);
#else
        100.0f / static_cast<float>(qmax);
#endif
    if (std::fabs(meta.scales[0] - first_scale) > first_scale * 1e-5f)
        return false;
    if (block_count > 1) {
        const float second_scale = 10.0f / static_cast<float>(qmax);
        if (std::fabs(meta.scales[1] - second_scale) > second_scale * 1e-5f)
            return false;
    }
    if (args.A.get(0, 0) != 0 || args.A.get(0, 1) != 0 || args.A.get(0, 2) != 0)
        return false;

#if GGML_GEMMINI_ENABLE_RMD
    const bool packet = route == act::ResidualRoute::ws_packet;
    if (cols >= 17 && (packet ? meta.rmd_packets.empty() || !meta.direct_residuals.empty()
                              : meta.direct_residuals.empty() || !meta.rmd_packets.empty()))
        return false;
#else
    if (!meta.rmd_packets.empty() || !meta.direct_residuals.empty())
        return false;
#endif

    std::vector<float> restored(cols, -1.0f);
    if (!block::dequantize_activation(restored.data(), cols, 1, 1, cols, args))
        return false;
    for (size_t k = 0; k < cols; ++k) {
        const float scale = meta.scales[k / block::kGroupSize];
        const double scaled = std::isfinite(source[k]) ?
            static_cast<double>(source[k]) / scale : 0.0;
        const int32_t q32 = scaled <= std::numeric_limits<int32_t>::min() ?
            std::numeric_limits<int32_t>::min() :
            scaled >= std::numeric_limits<int32_t>::max() ?
                std::numeric_limits<int32_t>::max() :
                static_cast<int32_t>(std::nearbyint(scaled));
        const float expected = static_cast<float>(q32) * scale;
        if (std::fabs(restored[k] - expected) > std::max(1e-5f, std::fabs(expected) * 1e-5f))
            return false;
    }
#if GGML_GEMMINI_ENABLE_RMD
    if (packet && !meta.rmd_packets.empty()) {
        auto malformed = std::make_shared<ggml::gemmini::rmd::StripePacket>(
            *meta.rmd_packets.front());
        malformed->version = 0;
        meta.rmd_packets = {malformed};
        if (block::dequantize_activation(restored.data(), cols, 1, 1, cols, args))
            return false;
    }
#endif
    return true;
}

bool quant_tests() {
    size_t cases = 0;
    for (size_t cols : {size_t{31}, size_t{32}, size_t{33}, size_t{64}, size_t{65}}) {
        for (act::ResidualRoute route : {act::ResidualRoute::ws_packet,
                                        act::ResidualRoute::cpu_direct}) {
            if (!quant_case(cols, route)) {
                std::fprintf(stderr, "FAIL quant: K=%zu route=%s\n", cols,
                    route == act::ResidualRoute::ws_packet ? "packet" : "direct");
                return false;
            }
            ++cases;
        }
    }
    if (!isolation_test()) {
        std::fputs("FAIL quant: adjacent row/K block changed target block\n", stderr);
        return false;
    }

    std::vector<float> source(33, 1.0f);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = 33;
    args.sA = 33;
    args.A.allocate(1, 33, GGML_GEMMINI_ACTIVATION_BITS);
    if (!ggml::gemmini::quants::quantize_activation(&tensor, args)) return false;
    if (args.act_quant.kind() != act::MetaKind::block) {
        std::fputs("FAIL quant: public dispatch did not select BLOCK\n", stderr);
        return false;
    }
    tensor.type = GGML_TYPE_I32;
    if (ggml::gemmini::quants::quantize_activation(&tensor, args) ||
        args.act_quant.kind() != act::MetaKind::none) {
        std::fputs("FAIL quant: public failure did not reset metadata\n", stderr);
        return false;
    }
    for (size_t k = 0; k < args.K; ++k) {
        if (args.A.get(0, k) != 0) {
            std::fputs("FAIL quant: public failure did not zero payload\n", stderr);
            return false;
        }
    }
    const int32_t qmax = ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX;
    const float first_scale =
#if GGML_GEMMINI_ENABLE_RMD
        1.0f / static_cast<float>(qmax);
#else
        100.0f / static_cast<float>(qmax);
#endif
    const float second_scale = 10.0f / static_cast<float>(qmax);
    std::printf(
        "quant: %zu K/route cases; G=%zu qmax=%d first_scale=%.9g second_scale=%.9g; "
        "isolation/nonfinite/malformed/reset passed\n",
        cases, block::kGroupSize, qmax, first_scale, second_scale);
    return true;
}

}

int main(int argc, char ** argv) {
    bool metadata = argc == 1;
    bool quant = argc == 1;
    for (int i = 1; i < argc; ++i) {
        metadata |= std::strcmp(argv[i], "--metadata") == 0;
        quant |= std::strcmp(argv[i], "--quant") == 0;
    }
    if ((!metadata && !quant) || (metadata && !metadata_tests()) ||
        (quant && !quant_tests())) return 1;
    return 0;
}
