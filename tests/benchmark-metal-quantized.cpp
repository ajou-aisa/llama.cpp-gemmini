#include "metal-quantized-fixtures.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace metal_quantized_fixtures;
using timer = std::chrono::steady_clock;

struct shape { const char * name; size_t m, n, k; };
struct samples {
    std::vector<double> cpu, wall, dense, residual, merge, staging;
};

double milliseconds(timer::time_point start) {
    return std::chrono::duration<double, std::milli>(timer::now() - start).count();
}

Fixture fixture_for(const shape & s, uint32_t bits, uint32_t dim, bool hp1, unsigned density) {
    auto f = make_block(bits, dim, s.m, s.n, s.k, false);
    f.view.profile.residual_enabled = density != 0;
    if (!hp1) {
        if (density) {
            f.residual.assign(s.m * s.k, 0);
            for (size_t m = 0; m < s.m; ++m) {
                for (size_t k = 0; k < s.k; ++k) {
                    if (density == 100 || k % 16 == 0) {
                        const int sign = (m + k / 16) % 2 ? -1 : 1;
                        f.residual[m * s.k + k] = sign * ((1 << bits) + 1);
                    }
                }
            }
        }
        f.bind();
        return f;
    }
    f.view.profile.mode = GGML_METAL_QUANTIZED_HP1_EXSIA;
    f.view.activation_rows_per_stripe = std::min(s.m, size_t(64));
    const size_t stripes = (s.m + 63) / 64;
    f.weight_scale.clear();
    f.activation_scale.resize(s.m);
    f.theta.resize(stripes);
    for (size_t m = 0; m < s.m; ++m) {
        const int exponent = -8 + int((m / 64) % 3);
        f.activation_scale[m] = std::ldexp(1.0f, exponent);
        f.theta[m / 64] = static_cast<int16_t>(exponent);
    }
    f.column_scale.resize(s.n);
    f.carriers.resize((s.k / 32) * s.n);
    for (size_t n = 0; n < s.n; ++n) {
        f.column_scale[n] = 0.00300000003f * float(1 + n % 7);
        for (size_t b = 0; b < s.k / 32; ++b) { f.carriers[b * s.n + n] = uint32_t((b + n) % 5); }
    }
    if (density) {
        const size_t step = density == 100 ? 1 : 16;
        const size_t per_block = 32 / step;
        f.requests.reserve(stripes);
        for (size_t stripe = 0; stripe < stripes; ++stripe) {
            OwnedRequest owned;
            auto & q = owned.request;
            const size_t source_begin = stripe * 64;
            const size_t source_count = std::min(size_t(64), s.m - source_begin);
            q.m = source_count * 2; q.n = s.n; q.k = s.k / step;
            q.source_row_begin = source_begin; q.source_row_count = source_count;
            q.tile_i = q.tile_j = q.tile_k = 1;
            owned.runs.resize(s.k / 32);
            owned.original_k.resize(s.k / 32);
            for (size_t b = 0; b < s.k / 32; ++b) {
                uint32_t mask = 0;
                for (size_t k = 0; k < 32; k += step) {
                    mask |= uint32_t(1) << k;
                    owned.original_k[b].push_back(static_cast<uint16_t>(k));
                }
                owned.runs[b] = {uint32_t(b), uint32_t(b * 32), mask, b * per_block, per_block, nullptr};
            }
            owned.rows.reserve(q.m);
            owned.activation.resize(q.m * q.k);
            for (uint32_t lane = 0; lane < 2; ++lane) {
                for (size_t m = 0; m < source_count; ++m) {
                    owned.rows.push_back({lane, uint32_t(m)});
                    for (size_t k = 0; k < q.k; ++k) {
                        owned.activation[(size_t(lane) * source_count + m) * q.k + k] =
                            (source_begin + m + (k * step) / 16) % 2 ? -1 : 1;
                    }
                }
            }
            owned.weight.resize(q.k * s.n);
            for (size_t k = 0; k < q.k; ++k) {
                for (size_t n = 0; n < s.n; ++n) { owned.weight[k * s.n + n] = f.weight[n * s.k + k * step]; }
            }
            owned.carriers = f.carriers;
            f.requests.push_back(std::move(owned));
        }
    }
    f.bind();
    return f;
}

void exact(const std::vector<float> & expected, const std::vector<float> & actual) {
    if (expected.size() != actual.size() || std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float))) {
        throw std::runtime_error("CPU oracle / Metal F32 bitwise mismatch");
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

void sample_json(const char * name, const std::vector<double> & values) {
    std::cout << '"' << name << "\":{\"min\":" << *std::min_element(values.begin(), values.end())
              << ",\"median\":" << median(values) << ",\"samples\":[";
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << "]}";
}

void measure(const shape & s, uint32_t bits, uint32_t dim, bool hp1, unsigned density,
             unsigned warmup, unsigned repeats, unsigned cpu_workers) {
    auto f = fixture_for(s, bits, dim, hp1, density);
    // A move may change owning vector addresses. Bind after final placement.
    f.bind();
    auto expected = reference(f.view, f.request_ptrs, false, cpu_workers);
    if (!expected.valid) { throw std::runtime_error("Invalid independent oracle fixture"); }
    std::vector<float> output(s.m * s.n);
    for (unsigned i = 0; i < warmup; ++i) {
        if (!ggml_metal_quantized_execute(&f.view, f.request_ptrs.data(), output.data(), nullptr)) {
            throw std::runtime_error(ggml_metal_quantized_last_error());
        }
        exact(expected.output, output);
        const auto cpu = reference(f.view, f.request_ptrs, false, cpu_workers);
        if (!cpu.valid) { throw std::runtime_error("CPU oracle warmup failed"); }
        exact(expected.output, cpu.output);
    }
    samples times;
    for (unsigned i = 0; i < repeats; ++i) {
        auto cpu_run = [&] {
            const auto start = timer::now();
            const auto result = reference(f.view, f.request_ptrs, false, cpu_workers);
            times.cpu.push_back(milliseconds(start));
            if (!result.valid) { throw std::runtime_error("CPU oracle timed execution failed"); }
            exact(expected.output, result.output);
        };
        auto gpu_run = [&] {
            const auto before = ggml_metal_quantized_get_stats();
            const auto start = timer::now();
            const bool ok = ggml_metal_quantized_execute(&f.view, f.request_ptrs.data(), output.data(), nullptr);
            times.wall.push_back(milliseconds(start));
            const auto after = ggml_metal_quantized_get_stats();
            if (!ok) { throw std::runtime_error(ggml_metal_quantized_last_error()); }
            exact(expected.output, output);
            if (after.dense_launches != before.dense_launches + 1 || after.fallback_calls != before.fallback_calls) {
                throw std::runtime_error("Missing dedicated Metal dense dispatch or unexpected fallback");
            }
            times.dense.push_back((after.dense_gpu_seconds - before.dense_gpu_seconds) * 1e3);
            times.residual.push_back((after.residual_gpu_seconds - before.residual_gpu_seconds) * 1e3);
            times.merge.push_back((after.merge_gpu_seconds - before.merge_gpu_seconds) * 1e3);
            times.staging.push_back((after.transfer_seconds - before.transfer_seconds) * 1e3);
        };
        if (i % 2) { gpu_run(); cpu_run(); } else { cpu_run(); gpu_run(); }
    }
    std::vector<double> kernel, host_rest;
    for (size_t i = 0; i < times.wall.size(); ++i) {
        kernel.push_back(times.dense[i] + times.residual[i] + times.merge[i]);
        host_rest.push_back(times.wall[i] - kernel[i] - times.staging[i]);
    }
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"kind\":\"measurement\",\"shape\":\"" << s.name
              << "\",\"mode\":\"" << (hp1 ? "HP1_EXSIA" : "BLOCK")
              << "\",\"bits\":" << bits << ",\"dim\":" << dim
              << ",\"m\":" << s.m << ",\"n\":" << s.n << ",\"k\":" << s.k
              << ",\"residual_density\":" << (density == 100 ? 1.0 : density ? 0.0625 : 0.0)
              << ",\"residual_lanes\":" << (hp1 && density ? 2 : 0)
              << ",\"residual_requests\":" << f.view.request_count
              << ",\"producer_included\":false,\"fixture\":\"synthetic-signed-codes-v1\""
              << ",\"cpu_workers\":" << cpu_workers << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
              << ",\"f32_bitwise\":true,\"milliseconds\":{";
    sample_json("cpu_oracle", times.cpu); std::cout << ',';
    sample_json("metal_host_total", times.wall); std::cout << ',';
    sample_json("metal_gpu_total", kernel); std::cout << ',';
    sample_json("dense_gpu", times.dense); std::cout << ',';
    sample_json("residual_gpu", times.residual); std::cout << ',';
    sample_json("merge_gpu", times.merge); std::cout << ',';
    sample_json("shared_buffer_staging", times.staging); std::cout << ',';
    sample_json("host_validation_submission_wait_and_other", host_rest);
    std::cout << "},\"cpu_oracle_over_metal_host\":" << median(times.cpu) / median(times.wall)
              << ",\"cpu_oracle_over_metal_gpu\":" << median(times.cpu) / median(kernel) << "}\n" << std::flush;
    std::fprintf(stderr, "%s %s A%uW%u DIM%u residual=%g cpu=%.3f ms metal_host=%.3f ms metal_gpu=%.3f ms exact\n",
                 s.name, hp1 ? "HP1" : "BLOCK", bits, bits, dim, density == 100 ? 1.0 : density ? 0.0625 : 0.0,
                 median(times.cpu), median(times.wall), median(kernel));
}
} // namespace

int main(int argc, char ** argv) {
    unsigned repeats = 5, warmup = 1, cpu_workers = 1, case_limit = 0, case_start = 1;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) { throw std::runtime_error("Expected integer argument after benchmark option"); }
            const int value = std::stoi(argv[++i]);
            if (value <= 0) { throw std::runtime_error("Benchmark options must be positive integers"); }
            if (option == "--repeats") repeats = unsigned(value);
            else if (option == "--warmup") warmup = unsigned(value);
            else if (option == "--cpu-workers") cpu_workers = unsigned(value);
            else if (option == "--case-limit") case_limit = unsigned(value);
            else if (option == "--case-start") case_start = unsigned(value);
            else throw std::runtime_error("Unknown benchmark option");
        }
        const shape shapes[] = {{"gpt2-decode-attn", 1, 768, 768},
                                {"gpt2-decode-ffn", 1, 3072, 768},
                                {"gpt2-prefill-attn", 256, 768, 768}};
        std::cout << "{\"schema_version\":1,\"kind\":\"method\",\"cpu_baseline\":\"independent ordered arithmetic oracle, not production CPU backend\","
                     "\"gpu_timing\":\"MTLCommandBuffer GPU timestamps\",\"host_timing\":\"validation, shared-buffer allocation/copy, GPU submission/wait, output publish\","
                     "\"staging\":\"CPU copies into unified shared buffers and output copy; not measured PCIe H2D/D2H\","
                     "\"excluded\":\"CPU block/ExSIA producer and initial shader compilation\",\"compiler\":\"" << __VERSION__ << "\"}\n";
        unsigned cases = 0, position = 0;
        for (const auto & s : shapes) {
            for (uint32_t bits : {4u, 8u}) {
                for (unsigned mode : {0u, 1u, 2u}) {
                    for (unsigned density : {0u, 6u, 100u}) {
                        if (++position < case_start) continue;
                        measure(s, bits, mode == 2 ? 64 : 16, mode != 0, density, warmup, repeats, cpu_workers);
                        if (case_limit && ++cases >= case_limit) return 0;
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "METAL_BENCHMARK_FAILED: %s\n", error.what());
        return 1;
    }
}
