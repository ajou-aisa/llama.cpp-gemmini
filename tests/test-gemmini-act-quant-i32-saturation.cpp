#include "../ggml/include/ggml.h"

#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/stripe/stripe.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/tensor/tensor.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/token/token.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"

#include <cfloat>
#include <cfenv>
#include <cstdio>
#include <limits>
#include <vector>

namespace act = ggml::gemmini::quants::act;

template<typename Meta, typename Quantize>
bool check_quantizer(const char *name, Quantize quantize)
{
    auto run = [&](float outlier) {
        std::vector<float> source(17, 1.0f);
        source.back() = outlier;
        std::vector<elem_t> quantized(source.size(), 0);

        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();

        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = source.size();
        args.A = quantized.data();
        args.sA = source.size();
        args.act_quant.storage().template emplace<Meta>();
        return quantize(&tensor, args);
    };

    if (!run(1000000.0f)) {
        std::fprintf(stderr, "FAIL: %s rejected a representable finite outlier\n", name);
        return false;
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    const bool accepted_overflow = run(FLT_MAX);
    if (std::fetestexcept(FE_INVALID | FE_OVERFLOW) != 0) {
        std::fprintf(stderr, "FAIL: %s raised a floating exception before clamping to INT32\n", name);
        return false;
    }
    if (accepted_overflow) {
        std::fprintf(stderr, "FAIL: %s accepted a finite outlier beyond the INT32/RMD domain\n", name);
        return false;
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    if (!run(-FLT_MAX)) {
        std::fprintf(stderr, "FAIL: %s rejected a negative finite outlier representable after INT32 clamp\n", name);
        return false;
    }
    if (std::fetestexcept(FE_INVALID | FE_OVERFLOW) != 0) {
        std::fprintf(stderr, "FAIL: %s raised a floating exception while clamping to INT32_MIN\n", name);
        return false;
    }
    return true;
}

template<typename Meta, typename Quantize>
bool check_quantizer_zeroes_nonfinite(const char *name, Quantize quantize)
{
    const float values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : values) {
        std::vector<float> source(17, 1.0f);
        source[8] = value;
        std::vector<elem_t> quantized(source.size(), 42);
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = source.size();
        args.A = quantized.data();
        args.sA = source.size();
        args.act_quant.storage().template emplace<Meta>();
        if (!quantize(&tensor, args) || quantized[8] != 0) {
            std::fprintf(stderr, "FAIL: %s did not convert NaN/Inf to zero\n", name);
            return false;
        }
    }
    return true;
}

static bool check_public_quantizer_zeroes_nonfinite()
{
    const float values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : values) {
        std::vector<float> source(17, 1.0f);
        source[8] = value;
        std::vector<elem_t> quantized(source.size(), 42);
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.data = source.data();
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = source.size();
        args.A = quantized.data();
        args.sA = source.size();

        if (!ggml::gemmini::quants::quantize_activation(&tensor, args)) {
            std::fputs("FAIL: public activation quantizer rejected NaN/Inf instead of zeroing it\n", stderr);
            return false;
        }
        if (args.act_quant.kind() == act::MetaKind::none) {
            std::fputs("FAIL: accepted NaN/Inf did not commit activation metadata\n", stderr);
            return false;
        }
        if (quantized[8] != 0) {
            std::fputs("FAIL: NaN/Inf was not converted to zero\n", stderr);
            return false;
        }
    }
    return true;
}

static bool check_public_quantizer_rejects_too_wide_without_commit()
{
    std::vector<float> source(17, 1.0f);
    source.back() = FLT_MAX;
    std::vector<elem_t> quantized(source.size(), 42);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = source.size();
    args.A = quantized.data();
    args.sA = source.size();

    if (ggml::gemmini::quants::quantize_activation(&tensor, args)) {
        std::fputs("FAIL: public activation quantizer accepted a residual beyond four lanes\n", stderr);
        return false;
    }
    if (args.act_quant.kind() != act::MetaKind::none) {
        std::fputs("FAIL: failed quantization committed activation metadata\n", stderr);
        return false;
    }
    for (elem_t value : quantized) {
        if (value != 0) {
            std::fputs("FAIL: failed quantization left partial activation output\n", stderr);
            return false;
        }
    }
    return true;
}

static bool check_exact_packet_slice_reuses_handle()
{
    ggml::gemmini::rmd::RmdStripeBuilder builder;
    builder.reset(7, 16, 2, 32, 3);
    if (!builder.add_residual(0, 5, 257)) {
        std::fputs("FAIL: exact-slice fixture build failed\n", stderr);
        return false;
    }
    const auto packet = builder.finish();
    if (!packet) {
        std::fputs("FAIL: exact-slice fixture packet missing\n", stderr);
        return false;
    }
    ggml::gemmini::rmd::RmdStatus status{};
    const auto sliced = ggml::gemmini::rmd::slice_packets({packet}, 16, 18, 7, status);
    if (status != ggml::gemmini::rmd::RmdStatus::success || sliced != packet) {
        std::fputs("FAIL: exact packet slice rebuilt instead of reusing its handle\n", stderr);
        return false;
    }
    return true;
}

static bool check_radix_compose_allows_final_int64_cancellation()
{
    ggml::gemmini::rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 32, 1);
    if (!builder.add_residual(0, 0, 65793)) {
        return false;
    }
    const auto packet = builder.finish();
    if (!packet || packet->blocks.size() != 1) {
        return false;
    }
    const auto &block = packet->blocks.front();
    ggml::gemmini::rmd::CompressedOutput output;
    output.domain = ggml::gemmini::rmd::CompressedOutput::Domain::block_scaled_int64;
    output.j_padded = packet->j_padded;
    output.values.assign(packet->total_output_values, 0);
    for (uint8_t position = 0; position < block.active_lane_count; ++position) {
        const size_t offset = block.output_value_offset +
            static_cast<size_t>(position) * block.lane_stride_values;
        switch (block.lane_ids[position]) {
        case 0: output.values[offset] = std::numeric_limits<int64_t>::max(); break;
        case 1: output.values[offset] = 1; break;
        case 2: output.values[offset] = -1; break;
        default: break;
        }
    }
    std::vector<ggml::gemmini::rmd::OutputValue> correction;
    const auto status = ggml::gemmini::rmd::compose_rmd_output(*packet, output, correction);
    const int64_t expected = std::numeric_limits<int64_t>::max() - 65280;
    if (status != ggml::gemmini::rmd::RmdStatus::success || correction != std::vector<int64_t>{expected}) {
        std::fputs("FAIL: radix compose rejected an in-range final INT64 after lane cancellation\n", stderr);
        return false;
    }
    for (uint8_t position = 0; position < block.active_lane_count; ++position) {
        if (block.lane_ids[position] == 2) {
            const size_t offset = block.output_value_offset +
                static_cast<size_t>(position) * block.lane_stride_values;
            output.values[offset] = 0;
        }
    }
    if (ggml::gemmini::rmd::compose_rmd_output(*packet, output, correction) !=
        ggml::gemmini::rmd::RmdStatus::overflow) {
        std::fputs("FAIL: radix compose accepted a final value beyond INT64\n", stderr);
        return false;
    }
    return true;
}

int main()
{
    const bool ok =
        check_quantizer<act::tensor::Meta>("TENSOR", act::tensor::quantize) &&
        check_quantizer<act::token::Meta>("TOKEN", act::token::quantize) &&
        check_quantizer<act::stripe::Meta>("STRIPE", act::stripe::quantize) &&
        check_quantizer_zeroes_nonfinite<act::tensor::Meta>("TENSOR", act::tensor::quantize) &&
        check_quantizer_zeroes_nonfinite<act::token::Meta>("TOKEN", act::token::quantize) &&
        check_quantizer_zeroes_nonfinite<act::stripe::Meta>("STRIPE", act::stripe::quantize) &&
        check_public_quantizer_zeroes_nonfinite() &&
        check_public_quantizer_rejects_too_wide_without_commit() &&
        check_exact_packet_slice_reuses_handle() &&
        check_radix_compose_allows_final_int64_cancellation();
    if (ok)
        std::puts("PASS: activation bounds/non-finite, exact packet reuse, and widened radix compose");
    return ok ? 0 : 1;
}
