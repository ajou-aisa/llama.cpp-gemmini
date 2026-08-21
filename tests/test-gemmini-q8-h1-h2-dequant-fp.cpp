#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <gemmini.h>
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

constexpr int64_t K = 64;
constexpr int64_t J = 2;
constexpr int64_t I = 1;
constexpr float TOLERANCE = 1e-5f;
static_assert(K % QK8_0 == 0 && K % QK8_H2 == 0);

enum class malformed_case {
    shape,
    stride,
    view_bounds,
    alignment,
};

const char * type_name(ggml_type type) {
    return ggml_type_name(type);
}

const char * malformed_name(malformed_case kind) {
    switch (kind) {
        case malformed_case::shape:        return "shape";
        case malformed_case::stride:       return "stride";
        case malformed_case::view_bounds:  return "view-bounds";
        case malformed_case::alignment:     return "alignment";
    }

    return "unknown";
}

std::vector<float> make_weights() {
    std::vector<float> values(J * K);
    for (int64_t j = 0; j < J; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            const float sign = ((j + k) % 3 == 0) ? -1.0f : 1.0f;
            values[j * K + k] = sign * (0.25f + 0.0078125f * static_cast<float>((3 * j + k) % 19));
        }
    }
    return values;
}

std::vector<float> make_activations() {
    std::vector<float> values(I * K);
    for (int64_t k = 0; k < K; ++k) {
        values[k] = (k % 4 == 0) ? -0.5f : 0.5f;
    }
    return values;
}

std::vector<uint8_t> quantize_weights(ggml_type type, const std::vector<float> & values) {
    std::vector<uint8_t> encoded(J * ggml_row_size(type, K));
    size_t written = 0;
    if (type == GGML_TYPE_Q8_H1) {
        written = quantize_q8_h1(values.data(), encoded.data(), J, K, nullptr);
    } else {
        written = quantize_q8_h2(values.data(), encoded.data(), J, K, nullptr);
    }

    if (written != encoded.size()) {
        std::fprintf(stderr, "quantization size mismatch for %s\n", type_name(type));
        encoded.clear();
    }
    return encoded;
}

std::vector<float> scalar_dequantize_weights(ggml_type type, const std::vector<uint8_t> & encoded) {
    std::vector<float> decoded(J * K);
    const size_t row_size = ggml_row_size(type, K);
    for (int64_t row = 0; row < J; ++row) {
        const uint8_t * row_data = encoded.data() + row * row_size;
        if (type == GGML_TYPE_Q8_H1) {
            dequantize_row_q8_h1(
                reinterpret_cast<const block_q8_h1 *>(row_data),
                decoded.data() + row * K,
                K);
        } else {
            dequantize_row_q8_h2(
                reinterpret_cast<const block_q8_h2 *>(row_data),
                decoded.data() + row * K,
                K);
        }
    }
    return decoded;
}

bool dequantize_activations(const ggml_tensor * activation, std::vector<float> & decoded) {
    std::vector<int8_t> quantized(I * K);
    ggml_gemmini_args_t args;
    args.I = I;
    args.J = J;
    args.K = K;
    args.A.allocate(I, K, 8);
    args.sA = K;
    if (!ggml::gemmini::quants::quantize_activation(activation, args)) {
        return false;
    }

    decoded.resize(I * K);
    return ggml::gemmini::quants::dequantize_activation(
        decoded.data(), K, 1, I, K, args);
}

void free_case(ggml_backend_buffer_t buffer, ggml_context * ctx) {
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

bool run_valid_case(ggml_backend_t backend, ggml_type type) {
    ggml_init_params params = {
        ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s: ggml_init failed\n", type_name(type));
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, K, J);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, I);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, activation);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "%s: backend tensor allocation failed\n", type_name(type));
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> weight_source = make_weights();
    const std::vector<float> activation_source = make_activations();
    const std::vector<uint8_t> encoded = quantize_weights(type, weight_source);
    if (encoded.empty()) {
        free_case(buffer, ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, encoded.data(), 0, encoded.size());
    ggml_backend_tensor_set(activation, activation_source.data(), 0, activation_source.size() * sizeof(float));

    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr,
                     "RED: GEMMINI DEQUANT_FP_TEST rejects valid %s at supports_op\n",
                     type_name(type));
        free_case(buffer, ctx);
        return false;
    }

    const std::vector<float> weight_dequantized = scalar_dequantize_weights(type, encoded);
    std::vector<float> activation_dequantized;
    if (!dequantize_activations(activation, activation_dequantized)) {
        std::fprintf(stderr, "%s: activation dequantization failed\n", type_name(type));
        free_case(buffer, ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: backend graph compute failed: %s\n", type_name(type), ggml_status_to_string(status));
        free_case(buffer, ctx);
        return false;
    }

    std::vector<float> actual(I * J);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int64_t i = 0; i < I; ++i) {
        for (int64_t j = 0; j < J; ++j) {
            float expected = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                expected += activation_dequantized[i * K + k] * weight_dequantized[j * K + k];
            }

            const float observed = actual[i * J + j];
            const float abs_error = std::fabs(observed - expected);
            const float rel_error = abs_error / std::max(std::fabs(expected), 1e-12f);
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, rel_error);
            if (!std::isfinite(observed) || !std::isfinite(expected)) {
                std::fprintf(stderr, "%s: non-finite result at [%lld,%lld]\n", type_name(type), (long long) i, (long long) j);
                free_case(buffer, ctx);
                return false;
            }
        }
    }

    const bool ok = max_abs <= TOLERANCE && max_rel <= TOLERANCE;
    std::printf("%s valid: max_abs=%g max_rel=%g %s\n", type_name(type), max_abs, max_rel, ok ? "PASS" : "FAIL");
    free_case(buffer, ctx);
    return ok;
}

bool run_malformed_case(ggml_backend_t backend, ggml_type type, malformed_case kind) {
    ggml_init_params params = {
        ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s/%s: ggml_init failed\n", type_name(type), malformed_name(kind));
        return false;
    }

    ggml_tensor * base = ggml_new_tensor_2d(ctx, type, K, J);
    ggml_tensor * weights = base;
    if (kind == malformed_case::view_bounds || kind == malformed_case::alignment) {
        weights = ggml_view_2d(ctx, base, K, 1, base->nb[1], 0);
    }
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, I);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, activation);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "%s/%s: backend tensor allocation failed\n", type_name(type), malformed_name(kind));
        ggml_free(ctx);
        return false;
    }

    switch (kind) {
        case malformed_case::shape:
            weights->ne[0] = K - 1;
            activation->ne[0] = K - 1;
            activation->nb[1] = sizeof(float) * activation->ne[0];
            activation->nb[2] = activation->nb[1] * activation->ne[1];
            activation->nb[3] = activation->nb[2];
            break;
        case malformed_case::stride:
            weights->nb[0] += 1;
            break;
        case malformed_case::view_bounds:
            weights->view_offs = ggml_nbytes(base) + 1;
            break;
        case malformed_case::alignment:
            weights->view_offs = 1;
            weights->data = static_cast<char *>(base->data) + 1;
            break;
    }

    const bool rejected = !ggml_backend_supports_op(backend, output);
    std::printf("%s %s: %s\n", type_name(type), malformed_name(kind), rejected ? "REJECTED (PASS)" : "ACCEPTED (FAIL)");
    free_case(buffer, ctx);
    return rejected;
}

bool run_malformed_cases(ggml_backend_t backend) {
    bool ok = true;
    for (ggml_type type : { GGML_TYPE_Q8_H1, GGML_TYPE_Q8_H2 }) {
        for (malformed_case kind : {
                 malformed_case::shape,
                 malformed_case::stride,
                 malformed_case::view_bounds,
                 malformed_case::alignment,
             }) {
            ok = run_malformed_case(backend, type, kind) && ok;
        }
    }
    return ok;
}

}

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--malformed")) {
        std::fprintf(stderr, "usage: %s [--malformed]\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("GEMMINI", nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "BLOCKED: GEMMINI backend is not registered\n");
        return 2;
    }

    const bool ok = argc == 2
        ? run_malformed_cases(backend)
        : [&]() {
            const bool h1_ok = run_valid_case(backend, GGML_TYPE_Q8_H1);
            const bool h2_ok = run_valid_case(backend, GGML_TYPE_Q8_H2);
            return h1_ok && h2_ok;
        }();

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
