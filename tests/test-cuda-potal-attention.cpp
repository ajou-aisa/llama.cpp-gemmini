#include "../ggml/src/ggml-cuda/cpu-exact/ggml-cuda-potal-attention.h"
#include "../ggml/src/ggml-gemmini/quants/common/hp1_scu.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace hp1 = ggml::gemmini::quants::hp1;
using fragment = ggml_cuda_potal_attention_fragment;
using handle = std::unique_ptr<ggml_cuda_potal_attention_weights, decltype(&ggml_cuda_potal_attention_destroy)>;

static void require(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(std::string(message)+": "+ggml_cuda_potal_attention_error());
}

static size_t check_case(unsigned bits, size_t rows, size_t columns, size_t k, bool compact) {
    std::mt19937 random(unsigned(rows*3+columns*13+k*17+bits));
    const int bound = 1 << (bits-1);
    std::vector<int32_t> low(k*columns), high(k*columns);
    for (size_t i = 0; i < low.size(); ++i) {
        low[i] = int(random()%(2*bound))-bound;
        high[i] = int(random()%(2*bound))-bound;
    }
    constexpr std::array<uint32_t, 8> shifts{0, 1, 7, 15, 31, 32, 32767, hp1::zero_carrier};
    std::vector<uint32_t> carriers(k/32*columns);
    for (auto & carrier : carriers) carrier = shifts[random()%shifts.size()];
    handle w(ggml_cuda_potal_attention_create(low.data(), high.data(), carriers.data(), columns, k, bits),
             ggml_cuda_potal_attention_destroy);
    require(bool(w), "prepare weights");
    std::vector<fragment> maps;
    size_t stride = k;
    if (compact) {
        stride = 0;
        for (unsigned f = 0; f < k/32; ++f) {
            fragment map{};
            map.compact_begin = uint32_t(stride);
            map.count = f%3 == 0 ? 1 : f%3 == 1 ? 17 : 31;
            map.original_block = uint32_t((k/32-1)-f);
            for (unsigned lane = 0; lane < map.count; ++lane)
                map.original_k[lane] = map.original_block*32+(lane*7+3)%32;
            stride += map.count+3;
            maps.push_back(map);
        }
    }
    std::vector<int8_t> activations(rows*stride);
    for (auto & a : activations) a = int8_t(int(random()%(2*bound))-bound);
    const size_t first = compact && columns > 4 ? 3 : 0, width = columns-first;
    for (unsigned pass = 0; pass < 2; ++pass) {
        std::vector<int32_t> actual(rows*width), expected(rows*width);
        require(ggml_cuda_potal_attention_dot(w.get(), pass, activations.data(), rows, stride, first, width,
                compact ? maps.data() : nullptr, maps.size(), compact, actual.data()), "CUDA integer fragment GEMM");
        const auto & weights = pass ? high : low;
        for (size_t row = 0; row < rows; ++row) for (size_t col = 0; col < width; ++col) {
            int32_t acc = 0;
            for (size_t f = 0; f < k/32; ++f) {
                const size_t count = compact ? maps[f].count : 32;
                const size_t begin = compact ? maps[f].compact_begin : f*32;
                const size_t original_block = compact ? maps[f].original_block : f;
                int32_t raw = 0;
                for (size_t lane = 0; lane < count; ++lane) {
                    const size_t original = compact ? maps[f].original_k[lane] : begin+lane;
                    raw += int32_t(activations[row*stride+begin+lane])*weights[original*columns+first+col];
                }
                acc = hp1::accumulate(acc, hp1::apply_validated(raw, carriers[original_block*columns+first+col]));
            }
            expected[row*width+col] = acc;
        }
        require(actual == expected, "CPU/CUDA bitwise integer mismatch");
    }
    if (compact) {
        auto invalid = maps;
        invalid[0].original_k[0] = uint32_t(k);
        std::vector<int32_t> ignored(rows*width);
        require(!ggml_cuda_potal_attention_dot(w.get(), 0, activations.data(), rows, stride, first, width,
                invalid.data(), invalid.size(), true, ignored.data()), "accepted invalid K mapping");
    }
    return rows*width*2;
}

int main() {
    try {
        const auto start = std::chrono::steady_clock::now();
        size_t values = 0, cases = 0;
        for (unsigned bits : {4u, 8u}) for (size_t rows : {1u, 5u, 32u, 79u})
            for (size_t columns : {1u, 7u, 33u, 65u}) for (size_t k : {32u, 64u, 160u})
                for (bool compact : {false, true}) {
                    values += check_case(bits, rows, columns, k, compact); ++cases;
                }
        std::vector<int32_t> invalid(32, 8);
        uint32_t carrier = 0;
        handle bad(ggml_cuda_potal_attention_create(invalid.data(), nullptr, &carrier, 1, 32, 4),
                   ggml_cuda_potal_attention_destroy);
        require(!bad, "accepted out-of-range INT4 weight");
        std::fill(invalid.begin(), invalid.end(), 1);
        carrier = 0x80000001u;
        bad.reset(ggml_cuda_potal_attention_create(invalid.data(), nullptr, &carrier, 1, 32, 8));
        require(!bad, "accepted invalid SCU carrier");
        const auto stats = ggml_cuda_potal_attention_get_stats();
        require(stats.main_dot_calls && stats.residual_dot_calls && stats.gemm_batches, "missing CUDA coverage");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::printf("PASS PoTal attention: %zu cases, %zu bitwise INT32 values, %.3f seconds\n", cases, values, seconds);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL PoTal CUDA attention: %s\n", error.what()); return 1;
    }
}
