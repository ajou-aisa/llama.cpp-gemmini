// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_mxint_round_nearest_even() {
    const int saved_mode = fegetround();
    int failures = 0;
    for (const auto type : {GGML_TYPE_MXINT4, GGML_TYPE_MXINT8}) {
        const bool four = type == GGML_TYPE_MXINT4;
        const int limit = four ? 7 : 127;
        const float boundary = four ? 8.0f : 128.0f;
        const float ties[] = {0.5f, 1.5f, 2.5f, 3.5f, limit - 0.5f};
        const int expected[] = {0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, limit-1, limit-1, limit};
        float input[6*32] = {};
        // Three different block scales share a minimum channel anchor of 0.5.
        for (int b = 0; b < 3; ++b) {
            const float scale = ldexpf(1.0f, b-1);
            for (int j = 0; j < 15; ++j) {
                const float tie = ties[j/3];
                const float v = j%3 == 0 ? nextafterf(tie, 0.0f) :
                                j%3 == 1 ? tie : nextafterf(tie, INFINITY);
                input[b*32+j] = v*scale;
                input[b*32+16+j] = -v*scale;
            }
            input[b*32+15] = (limit + 0.75f)*scale;
            input[b*32+31] = -input[b*32+15];
        }
        // The actual amax crosses a power of two; the last block stays empty.
        for (int b = 3; b < 5; ++b) {
            input[b*32] = 1.5f;
            input[b*32+16] = -1.5f;
            input[b*32+15] = b == 3 ? nextafterf(boundary, 0.0f) : boundary;
            input[b*32+31] = -input[b*32+15];
        }
        for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            assert(fesetround(mode) == 0);
            for (int reference = 0; reference < 2; ++reference) {
                block_q4_hp1 q4[6] = {};
                block_q8_hp1 q8[6] = {};
                void * output = four ? static_cast<void *>(q4) : static_cast<void *>(q8);
                if (reference) {
                    ggml_get_type_traits(type)->from_float_ref(input, output, 6*32);
                } else {
                    assert(ggml_quantize_chunk(type, input, output, 0, 1, 6*32, nullptr) == ggml_row_size(type, 6*32));
                }
                for (int b = 0; b < 6; ++b) {
                    assert((four ? q4[b].channel_scale : q8[b].channel_scale) == 0.5f);
                    assert((four ? q4[b].m : q8[b].m) == (b < 3 ? b : b == 3 ? 1 : b == 4 ? 2 : INT16_MIN));
                    assert((four ? q4[b].padding[0] : q8[b].padding[0]) == 0);
                    assert((four ? q4[b].padding[1] : q8[b].padding[1]) == 0);
                    for (int j = 0; j < 16; ++j) {
                        const int code = b < 3 ? (j < 15 ? expected[j] : limit) :
                                         b < 5 ? (j == 0 ? (b == 3 ? 2 : 1) :
                                                  j == 15 ? (b == 3 ? limit : (four ? 4 : 64)) : 0) : 0;
                        const bool ok = four ? q4[b].qs[j] == (b == 5 ? 0 : ((code+8) | ((8-code)<<4))) :
                                              q8[b].qs[j] == code && q8[b].qs[j+16] == -code;
                        if (!ok) {
                            fprintf(stderr, "%s RN mode=%d reference=%d block=%d pair=%d expected=%d\n",
                                    ggml_type_name(type), mode, reference, b, j, code);
                            ++failures;
                        }
                    }
                }
                assert(fegetround() == mode);
            }
        }
    }
    assert(fesetround(saved_mode) == 0);
    printf("MXINT4/8 RN ties, adjacent floats, scales, packing and fenv: %d failures\n", failures);
    return failures != 0;
}

int main(int argc, char * argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--mxint-rn") {
        return test_mxint_round_nearest_even();
    }
    const int rn_failed = test_mxint_round_nearest_even();
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();

    int num_failed = rn_failed;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS : MAX_QUANTIZATION_TOTAL_ERROR;
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
