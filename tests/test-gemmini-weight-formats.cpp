#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

float max_abs_error(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float error = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        error = std::max(error, std::fabs(lhs[i] - rhs[i]));
    }
    return error;
}

bool round_trip(enum ggml_type type, float error_limit) {
    constexpr int64_t rows = 2;
    constexpr int64_t columns = 64;
    std::vector<float> source(rows * columns);
    for (int64_t i = 0; i < rows * columns; ++i) {
        source[i] = std::sin(static_cast<float>(i) * 0.17f) * 9.0f +
                    std::cos(static_cast<float>(i) * 0.031f);
    }
    source[0] = 0.0f;
    source[17] = -12.0f;
    source[63] = 11.5f;

    const size_t row_size = ggml_row_size(type, columns);
    std::vector<uint8_t> quantized(rows * row_size);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), quantized.data(), 0, rows, columns, nullptr);
    if (!check(written == quantized.size(), "quantized byte count mismatch")) {
        return false;
    }
    if (!check(
            ggml_validate_row_data(type, quantized.data(), quantized.size()),
            "quantized row validation failed")) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!check(traits != nullptr && traits->to_float != nullptr, "missing to_float trait")) {
        return false;
    }

    std::vector<float> decoded(source.size());
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(
            quantized.data() + row * row_size,
            decoded.data() + row * columns,
            columns);
    }

    return check(
        max_abs_error(source, decoded) <= error_limit,
        "round-trip error exceeded format limit");
}

} // namespace

int main() {
    bool ok = true;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_H1) == 32, "Q4_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q4_HP1) == 32, "Q4_HP1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_0) == 32, "Q16_0 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_H1) == 32, "Q16_H1 block size") && ok;
    ok = check(ggml_blck_size(GGML_TYPE_Q16_HP1) == 32, "Q16_HP1 block size") && ok;

    ok = round_trip(GGML_TYPE_Q4_H1, 2.5f) && ok;
    ok = round_trip(GGML_TYPE_Q4_HP1, 2.5f) && ok;
    ok = round_trip(GGML_TYPE_Q16_0, 0.01f) && ok;
    ok = round_trip(GGML_TYPE_Q16_H1, 0.02f) && ok;
    ok = round_trip(GGML_TYPE_Q16_HP1, 0.02f) && ok;
    return ok ? 0 : 1;
}
