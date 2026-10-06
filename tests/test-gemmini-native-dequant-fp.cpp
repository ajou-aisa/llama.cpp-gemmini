#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-quants.h"
#include <gemmini.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

constexpr int64_t K         = 64;
constexpr int64_t J         = 2;
constexpr int64_t I         = 2;
constexpr float   TOLERANCE = 1e-5f;

#if GGML_GEMMINI_WEIGHT_BITS == 4
constexpr ggml_type H0_TYPE           = GGML_TYPE_Q4_0;
constexpr ggml_type PRIMARY_TYPE      = GGML_TYPE_Q4_HP1;
constexpr int64_t   NATIVE_BLOCK_SIZE = QK4_0;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
constexpr ggml_type H0_TYPE           = GGML_TYPE_Q8_0;
constexpr ggml_type PRIMARY_TYPE      = GGML_TYPE_Q8_HP1;
constexpr ggml_type H2_TYPE           = GGML_TYPE_Q8_H2;
constexpr int64_t   NATIVE_BLOCK_SIZE = QK8_0;
#elif GGML_GEMMINI_WEIGHT_BITS == 16
constexpr ggml_type H0_TYPE           = GGML_TYPE_Q16_0;
constexpr ggml_type PRIMARY_TYPE      = GGML_TYPE_Q16_H1;
constexpr int64_t   NATIVE_BLOCK_SIZE = QK16_0;
#else
#error "Unsupported Gemmini weight width"
#endif

static_assert(K % NATIVE_BLOCK_SIZE == 0);
#if GGML_GEMMINI_WEIGHT_BITS == 8
static_assert(K % QK8_H2 == 0);
#endif

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
    case malformed_case::shape:
        return "shape";
    case malformed_case::stride:
        return "stride";
    case malformed_case::view_bounds:
        return "view-bounds";
    case malformed_case::alignment:
        return "alignment";
    }

    return "unknown";
}

std::vector<float> make_weights() {
    std::vector<float> values(J * K);
    for (int64_t j = 0; j < J; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            const float sign  = ((j + k) % 3 == 0) ? -1.0f : 1.0f;
            values[j * K + k] = sign * (0.25f + 0.0078125f * static_cast<float>((3 * j + k) % 19));
        }
    }
    return values;
}

std::vector<float> make_activations() {
    std::vector<float> values(I * K);
    for (int64_t i = 0; i < I; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            values[i * K + k] = ((i + k) % 4 == 0) ? -0.5f : 0.5f;
        }
    }
    return values;
}

std::vector<uint8_t> quantize_weights(ggml_type type, const std::vector<float> & values) {
    std::vector<uint8_t> encoded(J * ggml_row_size(type, K));
    const size_t         written =
        ggml_quantize_chunk(type, values.data(), encoded.data(), 0, J, K, nullptr);

    if (written != encoded.size()) {
        std::fprintf(stderr, "quantization size mismatch for %s\n", type_name(type));
        encoded.clear();
    }
    return encoded;
}

std::vector<float> scalar_dequantize_weights(ggml_type type, const std::vector<uint8_t> & encoded) {
    std::vector<float> decoded(J * K);
    const size_t       row_size = ggml_row_size(type, K);
    for (int64_t row = 0; row < J; ++row) {
        const uint8_t * row_data = encoded.data() + row * row_size;
        ggml_get_type_traits(type)->to_float(row_data, decoded.data() + row * K, K);
    }
    return decoded;
}

bool dequantize_activations(const ggml_tensor * activation, std::vector<float> & decoded) {
    std::vector<int8_t> quantized(I * K);
    ggml_gemmini_args_t args;
    args.I = I;
    args.J = J;
    args.K = K;
    args.A.allocate(I, K, GGML_GEMMINI_ACTIVATION_BITS);
    args.sA = K;
    if (!ggml::gemmini::quants::quantize_activation(activation, args)) {
        return false;
    }

    decoded.resize(I * K);
    return ggml::gemmini::quants::dequantize_activation(decoded.data(), K, 1, I, K, args);
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

    ggml_tensor *         weights    = ggml_new_tensor_2d(ctx, type, K, J);
    ggml_tensor *         activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, I);
    ggml_tensor *         output     = ggml_mul_mat(ctx, weights, activation);
    ggml_backend_buffer_t buffer     = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "%s: backend tensor allocation failed\n", type_name(type));
        ggml_free(ctx);
        return false;
    }

    const std::vector<float>   weight_source     = make_weights();
    const std::vector<float>   activation_source = make_activations();
    const std::vector<uint8_t> encoded           = quantize_weights(type, weight_source);
    if (encoded.empty()) {
        free_case(buffer, ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, encoded.data(), 0, encoded.size());
    ggml_backend_tensor_set(
        activation, activation_source.data(), 0, activation_source.size() * sizeof(float));

#if defined(IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1) && !CYCLE_SIM && GGML_GEMMINI_WEIGHT_BITS == 16
    if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                      ggml::gemmini::config::ComputeType::INT &&
                  !ggml::gemmini::config::DEQUANT_FP_TEST && OPTION != CPU) {
        if (type == PRIMARY_TYPE) {
            using namespace ggml::gemmini::im2p_adapter;
            const bool         unsupported = !ggml_backend_supports_op(backend, output);
            std::vector<float> actual(I * J, 12345.0f);
            ggml_backend_tensor_set(output, actual.data(), 0, actual.size() * sizeof(float));
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            test_reset();
            const auto status   = ggml_backend_graph_compute(backend, graph);
            const auto counters = test_counters();
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            const bool rejected =
                unsupported && status != GGML_STATUS_SUCCESS &&
                counters.production_error == Error::unsupported_route && counters.full == 0 &&
                counters.pipeline == 0 && counters.provider_dot_attempts == 0 &&
                counters.commit == 0 &&
                std::all_of(actual.begin(), actual.end(), [](float v) { return v == 12345.0f; });
            std::printf("H1 GEMMINI_HP1 supports_op=false typed=unsupported_route dispatch=0 "
                        "sentinel=unchanged: %s\n",
                        rejected ? "PASS" : "FAIL");
            test_reset();
            free_case(buffer, ctx);
            return rejected;
        }
    }
#endif

    if constexpr (ggml::gemmini::config::CURRENT_COMPUTE_TYPE ==
                      ggml::gemmini::config::ComputeType::INT &&
                  !ggml::gemmini::config::DEQUANT_FP_TEST && OPTION != CPU) {
        if (type == GGML_TYPE_Q8_H2) {
            const bool rejected = !ggml_backend_supports_op(backend, output);
            std::printf("q8_h2 INT/WS: %s\n", rejected ? "REJECTED (PASS)" : "ACCEPTED (FAIL)");
            free_case(buffer, ctx);
            return rejected;
        }
    }
    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr,
                     "RED: GEMMINI DEQUANT_FP_TEST rejects valid %s at supports_op\n",
                     type_name(type));
        free_case(buffer, ctx);
        return false;
    }

    const std::vector<float> weight_dequantized = scalar_dequantize_weights(type, encoded);
    std::vector<float>       activation_dequantized;
    if (!dequantize_activations(activation, activation_dequantized)) {
        std::fprintf(stderr, "%s: activation dequantization failed\n", type_name(type));
        free_case(buffer, ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "%s: backend graph compute failed: %s\n",
                     type_name(type),
                     ggml_status_to_string(status));
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

            const float observed  = actual[i * J + j];
            const float abs_error = std::fabs(observed - expected);
            const float rel_error = abs_error / std::max(std::fabs(expected), 1e-12f);
            max_abs               = std::max(max_abs, abs_error);
            max_rel               = std::max(max_rel, rel_error);
            if (!std::isfinite(observed) || !std::isfinite(expected)) {
                std::fprintf(stderr,
                             "%s: non-finite result at [%lld,%lld]\n",
                             type_name(type),
                             (long long)i,
                             (long long)j);
                free_case(buffer, ctx);
                return false;
            }
        }
    }

    const bool ok = max_abs <= TOLERANCE && max_rel <= TOLERANCE;
    std::printf("%s valid: max_abs=%g max_rel=%g %s\n",
                type_name(type),
                max_abs,
                max_rel,
                ok ? "PASS" : "FAIL");
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

    ggml_tensor * base    = ggml_new_tensor_2d(ctx, type, K, J);
    ggml_tensor * weights = base;
    if (kind == malformed_case::view_bounds || kind == malformed_case::alignment) {
        weights = ggml_view_2d(ctx, base, K, 1, base->nb[1], 0);
    }
    ggml_tensor *         activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, I);
    ggml_tensor *         output     = ggml_mul_mat(ctx, weights, activation);
    ggml_backend_buffer_t buffer     = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr,
                     "%s/%s: backend tensor allocation failed\n",
                     type_name(type),
                     malformed_name(kind));
        ggml_free(ctx);
        return false;
    }

    switch (kind) {
    case malformed_case::shape:
        weights->ne[0]    = K - 1;
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
        weights->data      = static_cast<char *>(base->data) + 1;
        break;
    }

    const bool rejected = !ggml_backend_supports_op(backend, output);
    std::printf("%s %s: %s\n",
                type_name(type),
                malformed_name(kind),
                rejected ? "REJECTED (PASS)" : "ACCEPTED (FAIL)");
    free_case(buffer, ctx);
    return rejected;
}

bool run_malformed_cases(ggml_backend_t backend) {
    bool ok = true;
#if GGML_GEMMINI_WEIGHT_BITS == 8
    for (ggml_type type : {H0_TYPE, PRIMARY_TYPE, H2_TYPE}) {
#else
    for (ggml_type type : {H0_TYPE, PRIMARY_TYPE}) {
#endif
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

} // namespace

int main(int argc, char ** argv) {
    const std::string_view selection = argc == 2 ? argv[1] : "";
    if (argc > 2 || (argc == 2 && selection != "--malformed" && selection != "--h0" &&
                     selection != "--native" && selection != "--native-full" &&
                     selection != "--native-pipeline")) {
        std::fprintf(stderr,
                     "usage: %s [--malformed|--h0|--native|--native-full|--native-pipeline]\n",
                     argv[0]);
        return 2;
    }

    if (selection == "--native-full") {
        const auto resolution = ggml::gemmini::resolve_matmul_options();
        if (!resolution.ok() ||
            resolution.options.mode != ggml::gemmini::MatmulInvocationMode::full) {
            std::fputs("dequant test did not resolve FULL\n", stderr);
            return 1;
        }
    }

    if (selection == "--native-pipeline") {
        const auto resolution = ggml::gemmini::resolve_matmul_options();
        if (!resolution.ok() ||
            resolution.options.mode != ggml::gemmini::MatmulInvocationMode::stripe_pipeline) {
            std::fputs("deferred dequant test did not resolve STRIPE_PIPELINE\n", stderr);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("GEMMINI", nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "BLOCKED: GEMMINI backend is not registered\n");
        return 2;
    }

    const bool ok = selection == "--malformed" ? run_malformed_cases(backend)
                    : selection == "--h0"      ? run_valid_case(backend, H0_TYPE)
                    : selection == "--native" || selection == "--native-full" ||
                            selection == "--native-pipeline"
                        ? run_valid_case(backend, PRIMARY_TYPE)
                        : [&]() {
                              const bool native_ok = run_valid_case(backend, PRIMARY_TYPE);
#if GGML_GEMMINI_WEIGHT_BITS == 4
                              const bool h0_ok = run_valid_case(backend, H0_TYPE);
                              return h0_ok && native_ok;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
                              const bool h2_ok = run_valid_case(backend, GGML_TYPE_Q8_H2);
                              const bool h0_ok = run_valid_case(backend, H0_TYPE);
                              return native_ok && h2_ok && h0_ok;
#else
                              return native_ok && run_valid_case(backend, H0_TYPE) &&
                                     run_valid_case(backend, GGML_TYPE_Q16_HP1);
#endif
                          }();

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
