#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "metal-quantized-fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace metal_quantized_fixtures;
constexpr float kUnpublished = -12345.5f;
constexpr int32_t kUnpublishedInteger = 123456789;

bool check(bool condition, const std::string & message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    return condition;
}

uint32_t float_bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool compare(const char * field, const std::vector<int32_t> & actual,
             const std::vector<int32_t> & expected, const std::string & label) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: %s %s[%zu] actual=%d expected=%d\n",
                         label.c_str(), field, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

bool run_fixture(Fixture & fixture, const std::string & label, bool expect_failure = false) {
    fixture.bind();
    const auto expected = reference(fixture.view, fixture.request_ptrs);
    if (expected.valid && fixture.view.request_count != 0) {
        const auto parallel = reference(fixture.view, fixture.request_ptrs, true, 4);
        bool same = parallel.valid && parallel.raw == expected.raw && parallel.scaled == expected.scaled &&
                    parallel.dense == expected.dense && parallel.lanes == expected.lanes &&
                    parallel.correction == expected.correction && parallel.output.size() == expected.output.size();
        for (size_t i = 0; same && i < expected.output.size(); ++i)
            same &= float_bits(parallel.output[i]) == float_bits(expected.output[i]);
        if (!check(same, label + " parallel CPU oracle preserves every ordered lane and output bit")) return false;
    }
    std::vector<float> output(expected.output.size(), kUnpublished);
    std::vector<int32_t> raw(expected.raw.size(), kUnpublishedInteger);
    std::vector<int32_t> scaled(expected.scaled.size(), kUnpublishedInteger);
    std::vector<int32_t> dense(expected.dense.size(), kUnpublishedInteger);
    std::vector<int32_t> lanes(expected.lanes.size(), kUnpublishedInteger);
    std::vector<int64_t> correction(expected.correction.size(), kUnpublishedInteger);
    ggml_metal_quantized_trace trace{raw.data(), scaled.data(), raw.size(), dense.data(), correction.data(),
                                    dense.size(), lanes.data(), lanes.size()};
    const auto before = ggml_metal_quantized_get_stats();
    const bool executed = ggml_metal_quantized_execute(&fixture.view, fixture.request_ptrs.data(),
                                                      output.data(), &trace);
    if (expect_failure) {
        bool ok = check(!executed, label + " rejected");
        for (float value : output) ok &= check(float_bits(value) == float_bits(kUnpublished), label + " output unpublished");
        for (int32_t value : raw) ok &= check(value == kUnpublishedInteger, label + " raw trace unpublished");
        for (int32_t value : scaled) ok &= check(value == kUnpublishedInteger, label + " SCU trace unpublished");
        for (int32_t value : dense) ok &= check(value == kUnpublishedInteger, label + " dense trace unpublished");
        for (int32_t value : lanes) ok &= check(value == kUnpublishedInteger, label + " lane trace unpublished");
        for (int64_t value : correction) ok &= check(value == kUnpublishedInteger, label + " correction unpublished");
        std::printf("METAL_CASE %s expected_rejection=%s diagnostic=%s\n", label.c_str(), ok ? "PASS" : "FAIL",
                    ggml_metal_quantized_last_error());
        return ok;
    }
    if (!check(expected.valid && executed, label + " executes: " + ggml_metal_quantized_last_error())) return false;
    const auto after = ggml_metal_quantized_get_stats();
    bool ok = check(after.dense_launches > before.dense_launches, label + " launched a real Metal dense kernel");
    if (fixture.view.request_count)
        ok &= check(after.residual_launches > before.residual_launches, label + " launched a real Metal residual kernel");
    ok &= compare("raw", raw, expected.raw, label);
    ok &= compare("SCU", scaled, expected.scaled, label);
    ok &= compare("dense", dense, expected.dense, label);
    ok &= compare("lane", lanes, expected.lanes, label);
    for (size_t i = 0; i < output.size(); ++i) {
        if (correction[i] != expected.correction[i]) {
            std::fprintf(stderr, "FAIL: %s correction[%zu] actual=%lld expected=%lld\n", label.c_str(), i,
                         static_cast<long long>(correction[i]), static_cast<long long>(expected.correction[i]));
            ok = false;
        }
        if (float_bits(output[i]) != float_bits(expected.output[i])) {
            std::fprintf(stderr, "FAIL: %s F32[%zu] actual=%a (0x%08x) expected=%a (0x%08x)\n", label.c_str(), i,
                         output[i], float_bits(output[i]), expected.output[i], float_bits(expected.output[i]));
            ok = false;
        }
    }
    std::printf("METAL_CASE %s m=%zu n=%zu k=%zu bits=%u dim=%u integer=exact f32=bitwise %s\n",
                label.c_str(), fixture.view.m, fixture.view.n, fixture.view.k,
                fixture.view.profile.bits, fixture.view.profile.dim, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_without_trace(Fixture & fixture, const std::string & label) {
    fixture.bind();
    const auto expected = reference(fixture.view, fixture.request_ptrs, false);
    std::vector<float> actual(expected.output.size(), kUnpublished);
    const auto before = ggml_metal_quantized_get_stats();
    bool ok = check(ggml_metal_quantized_execute(&fixture.view, fixture.request_ptrs.data(), actual.data(), nullptr),
                    label + " executes with trace disabled");
    ok &= check(ggml_metal_quantized_get_stats().dense_launches > before.dense_launches,
                label + " launches a real Metal kernel");
    for (size_t i = 0; i < actual.size(); ++i)
        ok &= check(float_bits(actual[i]) == float_bits(expected.output[i]), label + " F32 bits match without tracing");
    std::printf("METAL_NO_TRACE %s bits=%u dim=%u %s\n", label.c_str(), fixture.view.profile.bits,
                fixture.view.profile.dim, ok ? "PASS" : "FAIL");
    return ok;
}

bool low_level_cases() {
    bool ok = true;
    for (uint32_t bits : {4u, 8u}) {
        for (uint32_t dim : {16u, 32u, 64u}) {
            for (size_t k : {31u, 32u, 33u, 63u, 64u, 65u, 4096u}) {
                auto block = make_block(bits, dim, k == 32 ? 1 : 3, dim + 3, k, k % 2);
                ok &= run_fixture(block, k % 32 ? "block-kernel-only-tail" : "block-aligned");
                if (k == 32 || k == 33) ok &= run_without_trace(block, "block");
            }
            auto hp1 = make_hp1(bits, dim);
            ok &= run_fixture(hp1, "hp1-ordered-fragments");
            ok &= run_without_trace(hp1, "hp1-dense");
            hp1.view.profile.residual_enabled = true;
            hp1.requests.push_back(make_request(bits, hp1.view.n, false));
            match_request_weights(hp1);
            ok &= run_fixture(hp1, "hp1-sparse-original-lanes-nonconsecutive-runs");
            ok &= run_without_trace(hp1, "hp1-residual");
            auto next_stripe = make_request(bits, hp1.view.n, false);
            next_stripe.request.source_row_begin = 2;
            next_stripe.request.source_row_count = 1;
            next_stripe.request.m = 1;
            next_stripe.rows = {{2, 0}};
            next_stripe.activation.resize(next_stripe.request.k);
            hp1.requests.push_back(std::move(next_stripe));
            ok &= run_fixture(hp1, "hp1-multiple-stripes");

            hp1.requests[0].runs[0].union_k_mask ^= UINT32_C(0x80000000);
            ok &= run_fixture(hp1, "hp1-invalid-original-K-map", true);

            auto overflow = make_hp1(bits, dim, 1, 1);
            overflow.view.profile.residual_enabled = true;
            overflow.requests.push_back(make_request(bits, 1, true));
            match_request_weights(overflow);
            overflow.bind();
            ok &= check(!reference(overflow.view, overflow.request_ptrs).valid, "independent radix oracle detects INT64 overflow");
            ok &= run_fixture(overflow, "hp1-radix-overflow", true);

            auto invalid = make_hp1(bits, dim);
            invalid.carriers[0] = 32768;
            ok &= run_fixture(invalid, "hp1-invalid-carrier", true);
        }
    }
    auto invalid_scale = make_block(4, 16, 1, 1, 32, false);
    invalid_scale.weight_scale[0] = std::numeric_limits<float>::infinity();
    ok &= run_fixture(invalid_scale, "block-invalid-scale", true);
    auto invalid_code = make_block(4, 16, 1, 1, 32, false);
    invalid_code.activation[0] = 8;
    ok &= run_fixture(invalid_code, "a4-invalid-code", true);
    auto unsupported = make_block(4, 16, 1, 1, 32, false);
    unsupported.view.profile.bits = 16;
    ok &= run_fixture(unsupported, "A16-admission-disabled", true);
    auto subnormal = make_block(4, 16, 1, 5, 32, false);
    std::fill(subnormal.activation.begin(), subnormal.activation.end(), 0);
    std::fill(subnormal.weight.begin(), subnormal.weight.end(), 0);
    subnormal.activation[0] = 1;
    subnormal.activation_scale[0] = std::ldexp(1.0f, -125);
    for (size_t column = 0; column < 5; ++column) {
        subnormal.weight[column * 32] = column % 2 ? -1 : 1;
        subnormal.weight_scale[column] = std::ldexp(float(column + 1), -24);
    }
    ok &= run_fixture(subnormal, "block-F32-subnormal-no-FTZ");
    subnormal.activation_scale[0] = std::ldexp(1.0f, -126);
    subnormal.weight[0] = -1;
    ok &= run_fixture(subnormal, "block-F32-underflow-negative-zero");
    auto all_residual = make_block(8, 64, 3, 7, 64, true);
    std::fill(all_residual.residual.begin(), all_residual.residual.end(), 129);
    ok &= run_fixture(all_residual, "block-residual-at-every-position");
    return ok;
}

int stored_code(uint32_t bits, size_t column, size_t k, bool minimum_codes) {
    const int half_range = 1 << (bits - 1);
    return minimum_codes ? -half_range : int((column * 3 + k * 11) % (2 * half_range)) - half_range;
}

float stored_scale(size_t column, size_t block) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16((block % 2 ? 1.0f : -1.0f) *
        (0.25f + float(column) / 32.0f) * (1.0f + float(block) / 8.0f)));
}

std::vector<uint8_t> encode_original(ggml_type type, uint32_t bits, size_t k, size_t n, bool minimum_codes) {
    const bool hp1 = type == GGML_TYPE_Q4_HP1 || type == GGML_TYPE_Q8_HP1;
    const size_t blocks = k / 32;
    const size_t block_bytes = ggml_type_size(type);
    std::vector<uint8_t> data(n * blocks * block_bytes, 0);
    for (size_t column = 0; column < n; ++column) {
        for (size_t block = 0; block < blocks; ++block) {
            uint8_t * dst = data.data() + (column * blocks + block) * block_bytes;
            uint8_t * codes = dst + (hp1 ? 0 : sizeof(ggml_half));
            for (size_t local = 0; local < 32; ++local) {
                const int code = stored_code(bits, column, block * 32 + local, minimum_codes);
                if (bits == 4) {
                    const unsigned shift = local < 16 ? 0 : 4;
                    codes[local % 16] |= static_cast<uint8_t>((code + 8) << shift);
                } else codes[local] = static_cast<uint8_t>(static_cast<int8_t>(code));
            }
            if (hp1) {
                const int16_t exponent = block == 0 ? std::numeric_limits<int16_t>::min() : int16_t(block % 2);
                const size_t exponent_offset = bits == 4 ? offsetof(block_q4_hp1, m) : offsetof(block_q8_hp1, m);
                const size_t scale_offset = bits == 4 ? offsetof(block_q4_hp1, channel_scale) : offsetof(block_q8_hp1, channel_scale);
                const float scale = 0.010000001f * float(column + 1);
                std::memcpy(dst + exponent_offset, &exponent, sizeof(exponent));
                std::memcpy(dst + scale_offset, &scale, sizeof(scale));
            } else {
                const ggml_half scale = ggml_fp32_to_fp16(stored_scale(column, block));
                std::memcpy(dst, &scale, sizeof(scale));
            }
        }
    }
    return data;
}

bool verify_original(const ggml_metal_quantized_view & v, const ggml_tensor * weights, bool minimum_codes) {
    bool ok = check(v.original_weights == weights->data &&
                    v.original_weight_bytes == ggml_nbytes(weights) &&
                    v.original_weight_type == weights->type, "producer preserves original GGUF storage identity");
    const bool hp1 = v.profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    for (size_t column = 0; column < v.n; ++column) {
        for (size_t k = 0; k < v.k; ++k)
            ok &= check(v.weights[column * v.k + k] == stored_code(v.profile.bits, column, k, minimum_codes),
                        "producer decodes original signed code including Q4 nibble offset");
        for (size_t block = 0; block < v.k / 32; ++block) {
            if (hp1) {
                ok &= check(v.carriers[block * v.n + column] == (block == 0 ? kZeroCarrier : uint32_t(block % 2)),
                            "producer reads original exponent and zero sentinel");
                ok &= check(float_bits(v.column_scales[column]) == float_bits(0.010000001f * float(column + 1)),
                            "producer preserves FP32 column scale bits");
            } else {
                ok &= check(float_bits(v.weight_scales[column * (v.k / 32) + block]) ==
                            float_bits(stored_scale(column, block)), "producer preserves negative FP16 block scale");
            }
        }
    }
    return ok;
}

bool public_graph_case(ggml_backend_t backend, size_t rows, bool minimum_codes) {
    const auto profile = ggml_metal_quantized_get_profile();
    const bool hp1 = profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    const ggml_type type = hp1 ? (profile.bits == 4 ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1)
                               : (profile.bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
    const size_t k = minimum_codes ? 32 : 128;
    const size_t n = profile.dim + 3;
    ggml_context * context = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true});
    if (!check(context != nullptr, "public graph context")) return false;
    const size_t row_spacing = minimum_codes ? 1 : 2;
    ggml_tensor * weight_storage = ggml_new_tensor_2d(context, type, k, n * row_spacing + 1);
    ggml_tensor * activation_storage = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, rows * row_spacing + 1);
    ggml_tensor * weights = ggml_view_2d(context, weight_storage, k, n, weight_storage->nb[1] * row_spacing, weight_storage->nb[1]);
    ggml_tensor * activation = ggml_view_2d(context, activation_storage, k, rows, activation_storage->nb[1] * row_spacing, activation_storage->nb[1]);
    ggml_tensor * output = ggml_mul_mat(context, weights, activation);
    output->nb[1] = (n + 3) * sizeof(float);
    output->nb[2] = output->nb[1] * rows;
    output->nb[3] = output->nb[2];
    ggml_set_name(weights, "blk.0.attn_q.weight");
    ggml_set_name(activation, "blk.0.attn_input");
    ggml_set_name(output, "blk.0.attn_q.result");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!check(buffer != nullptr, "Metal public graph buffer")) {
        ggml_free(context);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    const auto encoded = encode_original(type, profile.bits, k, n, minimum_codes);
    std::vector<uint8_t> encoded_rows(ggml_nbytes(weights), UINT8_C(0xa5));
    for (size_t column = 0; column < n; ++column)
        std::copy_n(encoded.data() + column * ggml_row_size(type, k), ggml_row_size(type, k),
                    encoded_rows.data() + column * weights->nb[1]);
    ggml_backend_tensor_set(weights, encoded_rows.data(), 0, encoded_rows.size());
    bool ok = check(ggml_backend_is_metal(backend), "public backend is Metal") &&
              check(ggml_backend_supports_op(backend, output), "Metal admits the quantized graph") &&
              check(ggml_metal_quantized_supports_op(output), "custom Metal arithmetic admits graph");
    ggml_metal_quantized_payload * retained = nullptr;
    std::vector<int8_t> retained_codes;
    std::vector<float> retained_scales;
    std::vector<float> previous_output;

    for (size_t pass = 0; pass < 3 && ok; ++pass) {
        std::vector<float> source(rows * k, pass == 1 ? -2.0f : 1.0f);
        if (pass == 2 && !minimum_codes) {
            for (size_t row = 0; row < rows; ++row) {
                source[row * k + 3] = 64.0f * float(row + 1);
                source[row * k + 3 * 32 + 21] = -257.0f;
            }
        }
        std::vector<float> source_rows(ggml_nbytes(activation) / sizeof(float), 1.0e20f);
        for (size_t row = 0; row < rows; ++row)
            std::copy_n(source.data() + row * k, k, source_rows.data() + row * k * row_spacing);
        ggml_backend_tensor_set(activation, source_rows.data(), 0, source_rows.size() * sizeof(float));
        ggml_metal_quantized_payload * payload = nullptr;
        char error[256]{};
        if (!check(ggml_metal_quantized_prepare(weights, activation, &payload, error, sizeof(error)),
                   std::string("public producer prepares: ") + error)) {
            ok = false;
            break;
        }
        const auto & view = *ggml_metal_quantized_get_view(payload);
        ok &= verify_original(view, weights, minimum_codes);
        std::vector<const ggml_metal_quantized_request *> requests;
        for (size_t i = 0; i < view.request_count; ++i) requests.push_back(ggml_metal_quantized_get_request(payload, i));
        const auto expected = reference(view, requests);
        ok &= check(expected.valid, "independent ordered oracle is finite");
        if (!hp1 && pass < 2) {
            const int qmax = (1 << (profile.bits - 1)) - 1;
            const int expected_code = pass == 0 ? qmax : -qmax;
            const float expected_scale = (pass == 0 ? 1.0f : 2.0f) / float(qmax);
            for (size_t i = 0; i < rows * k; ++i)
                ok &= check(view.activations[i] == expected_code, "independent constant BLOCK code oracle");
            for (size_t i = 0; i < rows * (k / 32); ++i)
                ok &= check(float_bits(view.activation_scales[i]) == float_bits(expected_scale),
                            "independent constant BLOCK scale oracle");
        }
        const auto before = ggml_metal_quantized_get_stats();
        std::vector<float> output_storage(ggml_nbytes(output) / sizeof(float), kUnpublished);
        ggml_backend_tensor_set(output, output_storage.data(), 0, ggml_nbytes(output));
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ok &= check(status == GGML_STATUS_SUCCESS, std::string("Metal public graph executes: ") + ggml_metal_quantized_last_error());
        const auto after = ggml_metal_quantized_get_stats();
        ok &= check(after.dense_launches > before.dense_launches &&
                    (hp1 ? after.hp1_calls > before.hp1_calls : after.block_calls > before.block_calls),
                    "public graph executed the custom Metal kernel without CPU fallback");
        std::vector<float> actual(rows * n, kUnpublished);
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(output, output_storage.data(), 0, ggml_nbytes(output));
            for (size_t row = 0; row < rows; ++row) {
                std::copy_n(output_storage.data() + row * (n + 3), n, actual.data() + row * n);
                if (row + 1 < rows)
                    for (size_t column = n; column < n + 3; ++column)
                        ok &= check(output_storage[row * (n + 3) + column] == kUnpublished,
                                    "public graph preserves output row padding");
            }
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (float_bits(actual[i]) != float_bits(expected.output[i])) {
                std::fprintf(stderr, "FAIL: public graph pass=%zu output=%zu actual=%a expected=%a\n",
                             pass, i, actual[i], expected.output[i]);
                ok = false;
            }
        }
        if (pass == 0 && minimum_codes && !hp1 && profile.bits == 4)
            ok &= check(actual[0] == 64.0f, "Q4_0 negative d and code -8 preserve exact output 64");
        if (pass == 1 && (!hp1 || !minimum_codes))
            ok &= check(actual != previous_output, "changed input changes public graph result");
        previous_output = actual;
        if (retained) {
            const auto & previous = *ggml_metal_quantized_get_view(retained);
            ok &= check(std::equal(retained_codes.begin(), retained_codes.end(), previous.activations),
                        "next producer call leaves retained activation payload intact");
            ok &= check(std::equal(retained_scales.begin(), retained_scales.end(), previous.activation_scales),
                        "next producer call leaves retained scales intact");
        }
        if (pass == 0) {
            retained = payload;
            retained_codes.assign(view.activations, view.activations + rows * k);
            retained_scales.assign(view.activation_scales, view.activation_scales + rows * (hp1 ? 1 : k / 32));
        } else ggml_metal_quantized_free(payload);
        std::printf("METAL_PUBLIC_GRAPH rows=%zu k=%zu bits=%u mode=%u offset_view=1 input_row_spacing=%zu output_stride=%zu pass=%zu f32=bitwise %s\n",
                    rows, k, profile.bits, static_cast<unsigned>(profile.mode), row_spacing, n + 3, pass, ok ? "PASS" : "FAIL");
    }
    if (retained && hp1 && !minimum_codes) {
        auto invalid_bytes = encoded_rows;
        const size_t scale_offset = profile.bits == 4 ? offsetof(block_q4_hp1, channel_scale)
                                                     : offsetof(block_q8_hp1, channel_scale);
        const float changed_scale = 1.0f;
        std::memcpy(invalid_bytes.data() + ggml_type_size(type) + scale_offset, &changed_scale, sizeof(changed_scale));
        ggml_backend_tensor_set(weights, invalid_bytes.data(), 0, invalid_bytes.size());
        ggml_metal_quantized_payload * unchanged = retained;
        char error[256]{};
        ok &= check(!ggml_metal_quantized_prepare(weights, activation, &unchanged, error, sizeof(error)) && unchanged == retained,
                    "producer rejects inconsistent HP1 column scales without publishing a payload");
        invalid_bytes = encoded_rows;
        const size_t exponent_offset = profile.bits == 4 ? offsetof(block_q4_hp1, m) : offsetof(block_q8_hp1, m);
        const int16_t invalid_exponent = -1;
        std::memcpy(invalid_bytes.data() + exponent_offset, &invalid_exponent, sizeof(invalid_exponent));
        ggml_backend_tensor_set(weights, invalid_bytes.data(), 0, invalid_bytes.size());
        ok &= check(!ggml_metal_quantized_prepare(weights, activation, &unchanged, error, sizeof(error)) && unchanged == retained,
                    "producer rejects a negative nonsentinel HP1 exponent without publishing a payload");
    }
    if (retained) ggml_metal_quantized_free(retained);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool public_graph_cases() {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!check(backend != nullptr, "actual Metal device available; CPU substitution is forbidden")) return false;
    bool ok = public_graph_case(backend, 1, true);
    ok &= public_graph_case(backend, 17, false);
    ggml_backend_free(backend);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const bool kernels_only = argc == 2 && std::strcmp(argv[1], "--kernels-only") == 0;
    if (argc != 1 && !kernels_only) {
        std::fprintf(stderr, "Usage: %s [--kernels-only]\n", argv[0]);
        return 2;
    }
    const auto profile = ggml_metal_quantized_get_profile();
    std::printf("METAL_QUANTIZED_PROFILE mode=%u bits=%u dim=%u residual=%u\n",
                static_cast<unsigned>(profile.mode), profile.bits, profile.dim, profile.residual_enabled);
    if (!check(ggml_metal_quantized_enabled(), "custom Metal quantized path is enabled")) return 1;
    bool ok = low_level_cases();
    if (!kernels_only) ok &= public_graph_cases();
    const auto stats = ggml_metal_quantized_get_stats();
    std::printf("METAL_QUANTIZED_RESULT scope=%s dense_launches=%llu residual_launches=%llu merge_launches=%llu %s\n",
                kernels_only ? "kernels" : "kernels-and-public-graphs",
                static_cast<unsigned long long>(stats.dense_launches),
                static_cast<unsigned long long>(stats.residual_launches),
                static_cast<unsigned long long>(stats.merge_launches), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
