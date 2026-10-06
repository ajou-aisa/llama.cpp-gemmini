#pragma once

#include "../ggml/src/ggml-metal/ggml-metal-quantized.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <vector>

namespace metal_quantized_fixtures {

inline constexpr uint32_t kZeroCarrier = UINT32_C(0x80000000);

inline int32_t bounded(__int128 value) {
    return static_cast<int32_t>(std::max<__int128>(INT32_MIN, std::min<__int128>(INT32_MAX, value)));
}

// Deliberately independent of hp1_scu.hpp and the Metal implementation.
inline int32_t scu(int64_t raw, uint32_t exponent) {
    if (raw == 0 || exponent == kZeroCarrier) return 0;
    if (exponent >= 32) return raw < 0 ? INT32_MIN : INT32_MAX;
    return bounded(static_cast<__int128>(raw) * (static_cast<__int128>(1) << exponent));
}

struct OwnedRequest {
    ggml_metal_quantized_request request{};
    std::vector<ggml_metal_quantized_run> runs;
    std::vector<std::vector<uint16_t>> original_k;
    std::vector<ggml_metal_quantized_row> rows;
    std::vector<int8_t> activation;
    std::vector<int32_t> weight;
    std::vector<uint32_t> carriers;

    const ggml_metal_quantized_request * bind() {
        for (size_t i = 0; i < runs.size(); ++i) runs[i].original_local_k = original_k[i].data();
        request.run_count = runs.size();
        request.runs = runs.data();
        request.rows = rows.data();
        request.activations = activation.data();
        request.weights = weight.data();
        request.carriers = carriers.data();
        return &request;
    }
};

struct Fixture {
    ggml_metal_quantized_view view{};
    std::vector<int8_t> activation;
    std::vector<int8_t> weight;
    std::vector<float> activation_scale;
    std::vector<float> weight_scale;
    std::vector<int32_t> residual;
    std::vector<uint32_t> carriers;
    std::vector<float> column_scale;
    std::vector<int16_t> theta;
    std::vector<OwnedRequest> requests;
    std::vector<const ggml_metal_quantized_request *> request_ptrs;

    void bind() {
        view.activations = activation.data();
        view.weights = weight.data();
        view.activation_scales = activation_scale.data();
        view.weight_scales = weight_scale.empty() ? nullptr : weight_scale.data();
        view.block_residual = residual.empty() ? nullptr : residual.data();
        view.carriers = carriers.empty() ? nullptr : carriers.data();
        view.column_scales = column_scale.empty() ? nullptr : column_scale.data();
        view.stripe_theta = theta.empty() ? nullptr : theta.data();
        view.stripe_count = theta.size();
        view.request_count = requests.size();
        request_ptrs.clear();
        for (auto & request : requests) request_ptrs.push_back(request.bind());
    }
};

struct Expected {
    std::vector<int32_t> raw;
    std::vector<int32_t> scaled;
    std::vector<int32_t> dense;
    std::vector<int32_t> lanes;
    std::vector<int64_t> correction;
    std::vector<float> output;
    bool valid = true;
};

inline Expected reference(const ggml_metal_quantized_view & v,
                   const std::vector<const ggml_metal_quantized_request *> & requests,
                   bool traces = true, unsigned workers = 1) {
    Expected result;
    const bool hp1 = v.profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    const size_t fragment = hp1 ? std::min<size_t>(32, v.profile.dim) : 32;
    const size_t fragments = (v.k + fragment - 1) / fragment;
    const size_t blocks = (v.k + 31) / 32;
    result.raw.resize(traces ? v.m * v.n * fragments : 0);
    result.scaled.resize(result.raw.size());
    result.dense.assign(traces ? v.m * v.n : 0, 0);
    result.correction.assign(v.m * v.n, 0);
    result.output.resize(v.m * v.n);

    std::vector<__int128> correction(v.m * v.n, 0);
    for (const auto * request : requests) {
        const auto lane_value = [&](size_t row, size_t column) {
            int32_t acc = 0;
            for (size_t run_index = 0; run_index < request->run_count; ++run_index) {
                const auto & run = request->runs[run_index];
                for (size_t begin = 0; begin < run.compact_k_count; begin += fragment) {
                    int64_t raw = 0;
                    for (size_t k = begin; k < std::min(run.compact_k_count, begin + fragment); ++k) {
                        const size_t compact = run.compact_k_begin + k;
                        raw += int64_t(request->activations[row * request->k + compact]) *
                               request->weights[compact * request->n + column];
                    }
                    acc = bounded(static_cast<__int128>(acc) +
                                  scu(raw, request->carriers[run_index * request->n + column]));
                }
            }
            return acc;
        };
        const size_t lane_count = request->m * request->n;
        std::vector<int32_t> parallel_lanes;
        if (workers > 1) {
            parallel_lanes.resize(lane_count);
            const auto evaluate_lanes = [&](size_t first, size_t last) {
                for (size_t index = first; index < last; ++index)
                    parallel_lanes[index] = lane_value(index / request->n, index % request->n);
            };
            const size_t worker_count = std::min<size_t>(workers, lane_count);
            std::vector<std::future<void>> tasks;
            for (size_t worker = 1; worker < worker_count; ++worker)
                tasks.push_back(std::async(std::launch::async, evaluate_lanes,
                    lane_count * worker / worker_count, lane_count * (worker + 1) / worker_count));
            evaluate_lanes(0, lane_count / worker_count);
            for (auto & task : tasks) task.get();
        }
        for (size_t row = 0; row < request->m; ++row) {
            for (size_t column = 0; column < request->n; ++column) {
                const int32_t acc = workers > 1 ? parallel_lanes[row * request->n + column] : lane_value(row, column);
                if (traces) result.lanes.push_back(acc);
                const auto & map = request->rows[row];
                const size_t output = (request->source_row_begin + map.source_row) * v.n + column;
                correction[output] += static_cast<__int128>(acc) *
                    (static_cast<__int128>(1) << (v.profile.bits * map.original_lane_id));
            }
        }
    }
    for (size_t i = 0; i < correction.size(); ++i) {
        if (correction[i] < INT64_MIN || correction[i] > INT64_MAX) result.valid = false;
        else result.correction[i] = static_cast<int64_t>(correction[i]);
    }

    const auto evaluate = [&](size_t first, size_t last) {
        bool valid = true;
        for (size_t output = first; output < last; ++output) {
            const size_t row = output / v.n;
            const size_t column = output % v.n;
            int32_t acc = 0;
            double dense = 0;
            double block_correction = 0;
            for (size_t f = 0; f < fragments; ++f) {
                const size_t begin = f * fragment;
                int64_t raw = 0;
                int64_t residual_raw = 0;
                for (size_t k = begin; k < std::min(v.k, begin + fragment); ++k) {
                    raw += int64_t(v.activations[row * v.k + k]) * v.weights[column * v.k + k];
                    if (!hp1 && v.block_residual)
                        residual_raw += int64_t(v.block_residual[row * v.k + k]) * v.weights[column * v.k + k];
                }
                const size_t index = output * fragments + f;
                const int32_t scaled = hp1 ? scu(raw, v.carriers[(begin / 32) * v.n + column])
                                          : static_cast<int32_t>(raw);
                if (traces) {
                    result.raw[index] = static_cast<int32_t>(raw);
                    result.scaled[index] = scaled;
                }
                if (hp1) acc = bounded(static_cast<__int128>(acc) + scaled);
                else {
                    const double weight_scale = v.weight_scales[column * blocks + begin / 32];
                    const double activation_scale = v.activation_scales[row * blocks + begin / 32];
                    dense += static_cast<double>(raw) * weight_scale * activation_scale;
                    block_correction += static_cast<double>(residual_raw) * weight_scale * activation_scale;
                    result.correction[output] += residual_raw;
                }
            }
            if (hp1) {
                if (traces) result.dense[output] = acc;
                dense = static_cast<double>(acc) * static_cast<double>(v.column_scales[column]) *
                        static_cast<double>(v.activation_scales[row]);
                block_correction = static_cast<double>(result.correction[output]) *
                    static_cast<double>(v.column_scales[column]) * static_cast<double>(v.activation_scales[row]);
            }
            const float rounded_dense = static_cast<float>(dense);
            const float rounded_correction = static_cast<float>(block_correction);
            bool merge = !hp1 && v.block_residual;
            for (const auto * request : requests)
                merge |= row >= request->source_row_begin &&
                         row < request->source_row_begin + request->source_row_count;
            result.output[output] = merge ? rounded_dense + rounded_correction : rounded_dense;
            valid &= std::isfinite(result.output[output]);
        }
        return valid;
    };
    const size_t count = v.m * v.n;
    workers = static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(workers, count)));
    std::vector<std::future<bool>> tasks;
    for (unsigned worker = 1; worker < workers; ++worker) {
        tasks.push_back(std::async(std::launch::async, evaluate,
            count * worker / workers, count * (worker + 1) / workers));
    }
    result.valid &= evaluate(0, count / workers);
    for (auto & task : tasks) result.valid &= task.get();
    return result;
}

inline Fixture make_block(uint32_t bits, uint32_t dim, size_t m, size_t n, size_t k, bool residual) {
    Fixture f;
    f.view.profile = {GGML_METAL_QUANTIZED_BLOCK, bits, dim, residual};
    f.view.m = m; f.view.n = n; f.view.k = k;
    f.view.activation_rows_per_stripe = m;
    const size_t blocks = (k + 31) / 32;
    f.activation.resize(m * k);
    f.weight.resize(n * k);
    f.activation_scale.resize(m * blocks);
    f.weight_scale.resize(n * blocks);
    if (residual) f.residual.assign(m * k, 0);
    const int range = 1 << bits;
    for (size_t row = 0; row < m; ++row) {
        for (size_t i = 0; i < k; ++i) {
            f.activation[row * k + i] = static_cast<int8_t>((i * 7 + row * 13) % range - range / 2);
            if (residual && (i + row) % 19 == 0)
                f.residual[row * k + i] = ((i + row) % 2 ? -1 : 1) * static_cast<int32_t>(257 + row);
        }
        for (size_t b = 0; b < blocks; ++b)
            f.activation_scale[row * blocks + b] = std::ldexp(0.10000001f * float(row + 1), int(b % 11) - 5);
    }
    for (size_t column = 0; column < n; ++column) {
        for (size_t i = 0; i < k; ++i)
            f.weight[column * k + i] = static_cast<int8_t>((i * 11 + column * 3) % range - range / 2);
        for (size_t b = 0; b < blocks; ++b) {
            const float sign = (column + b) % 2 ? -1.0f : 1.0f;
            f.weight_scale[column * blocks + b] = ggml_fp16_to_fp32(
                ggml_fp32_to_fp16(sign * (0.103f + float(column % 5) * 0.037f) * std::ldexp(1.0f, int(b % 7) - 3)));
        }
    }
    return f;
}

inline Fixture make_hp1(uint32_t bits, uint32_t dim, size_t m = 3, size_t n = 7) {
    Fixture f;
    f.view.profile = {GGML_METAL_QUANTIZED_HP1_EXSIA, bits, dim, false};
    f.view.m = m; f.view.n = n; f.view.k = 192;
    f.view.activation_rows_per_stripe = 2;
    f.activation.resize(m * f.view.k);
    f.weight.resize(n * f.view.k);
    f.activation_scale.resize(m);
    f.column_scale.resize(n);
    f.carriers.resize(6 * n);
    f.theta.resize((m + 1) / 2);
    const uint32_t exponent[] = {31, 31, 32, 32767, kZeroCarrier, 0};
    const int qmax = (1 << (bits - 1)) - 1;
    for (size_t row = 0; row < m; ++row) {
        f.activation_scale[row] = std::ldexp(1.0f, int(row / 2) - 9);
        f.theta[row / 2] = static_cast<int16_t>(int(row / 2) - 9);
        for (size_t k = 0; k < f.view.k; ++k)
            f.activation[row * f.view.k + k] = static_cast<int8_t>((k % 5 == 0) ? 0 : (row + 1));
    }
    for (size_t column = 0; column < n; ++column) {
        f.column_scale[column] = 0.00300000003f * float(column + 1);
        for (size_t b = 0; b < 6; ++b) f.carriers[b * n + column] = exponent[(b + column) % 6];
        for (size_t k = 0; k < f.view.k; ++k) {
            const int sign = ((k / 16 + column) % 2) ? -1 : 1;
            f.weight[column * f.view.k + k] = static_cast<int8_t>(sign * std::min<int>(qmax, int(column + 1)));
        }
    }
    return f;
}

inline OwnedRequest make_request(uint32_t bits, size_t n, bool overflow) {
    OwnedRequest owned;
    auto & r = owned.request;
    r.m = overflow ? 2 : 3; r.n = n; r.k = overflow ? 2 : 39;
    r.source_row_begin = 0; r.source_row_count = overflow ? 1 : 2;
    r.tile_i = r.tile_j = r.tile_k = 1;
    if (overflow) {
        owned.rows = {{0, 0}, {32 / bits, 0}};
        owned.runs = {{0, 0, 1, 0, 1, nullptr}, {1, 32, 1, 1, 1, nullptr}};
        owned.original_k = {{0}, {0}};
        owned.activation = {0, 1, 1, 0};
        owned.weight.assign(2 * n, -1);
        owned.carriers.resize(2 * n);
        for (size_t column = 0; column < n; ++column) {
            owned.carriers[column] = 32;
            owned.carriers[n + column] = 0;
        }
    } else {
        owned.rows = {{0, 0}, {2, 0}, {32 / bits, 1}};
        owned.runs = {{0, 0, UINT32_C(0x1ffff), 0, 17, nullptr},
                      {3, 96, UINT32_C(0x1ffff), 17, 17, nullptr},
                      {5, 160, UINT32_C(0x1f), 34, 5, nullptr}};
        owned.original_k.resize(3);
        for (size_t run = 0; run < 3; ++run)
            for (size_t k = 0; k < owned.runs[run].compact_k_count; ++k)
                owned.original_k[run].push_back(static_cast<uint16_t>(k));
        owned.activation.assign(r.m * r.k, 1);
        owned.weight.resize(r.k * n);
        owned.carriers.resize(3 * n);
        for (size_t k = 0; k < r.k; ++k)
            for (size_t column = 0; column < n; ++column)
                owned.weight[k * n + column] = k < 17 ? 1 : (k < 34 ? -1 : 1);
        for (size_t run = 0; run < 3; ++run)
            for (size_t column = 0; column < n; ++column)
                owned.carriers[run * n + column] = run < 2 ? 31 : 0;
        // The highest carry lane is present without overflowing its radix place.
        for (size_t k = 0; k < 34; ++k) owned.activation[2 * r.k + k] = 0;
    }
    return owned;
}

inline void match_request_weights(Fixture & fixture) {
    for (const auto & owned : fixture.requests) {
        for (size_t run_index = 0; run_index < owned.runs.size(); ++run_index) {
            const auto & run = owned.runs[run_index];
            for (size_t column = 0; column < fixture.view.n; ++column) {
                fixture.carriers[run.original_block_id * fixture.view.n + column] =
                    owned.carriers[run_index * fixture.view.n + column];
                for (size_t k = 0; k < run.compact_k_count; ++k)
                    fixture.weight[column * fixture.view.k + run.original_global_k_begin + owned.original_k[run_index][k]] =
                        static_cast<int8_t>(owned.weight[(run.compact_k_begin + k) * fixture.view.n + column]);
            }
        }
    }
}

} // namespace metal_quantized_fixtures
