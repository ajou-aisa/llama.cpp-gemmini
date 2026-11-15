#include "../exec.hpp"
#include "golden.hpp"
#include <array>
#include <fstream>
#include <string>

namespace {
namespace fp32 = ggml::gemmini::fp32;
float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
std::vector<float> load(const std::string & path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> result(count);
    file.read(reinterpret_cast<char *>(result.data()), std::streamsize(count * sizeof(float)));
    golden_fp32::check(bool(file) && file.peek() == std::char_traits<char>::eof(), path.c_str());
    return result;
}
uint64_t model_mismatches = 0, golden_mismatches = 0, vector_words = 0, probes = 0;
void compare(const char * name, const std::vector<float> & a, const std::vector<float> & b,
             const std::vector<float> & expected, size_t m, size_t n, size_t k, bool vector) {
    std::vector<float> actual(m * n);
    // Poison the reused accumulator row: the next absent-D GEMM must overwrite it.
    float seed_a = 3, seed_b = 7, seed_c = 0;
    fp32::matmul(&seed_a, &seed_b, &seed_c, 1, 1, 1);
    golden_fp32::check(seed_c == 21, "accumulator poison");
    fp32::matmul(a.data(), b.data(), actual.data(), m, n, k);
    uint64_t wrong = 0, subnormal = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const uint32_t bits = golden_fp32::bits(expected[i]);
        subnormal += (bits & 0x7f800000) == 0 && (bits & 0x007fffff) != 0;
        if (golden_fp32::bits(actual[i]) != bits) {
            if (wrong < 3)
                std::printf("RTL_MODEL_MISMATCH case=%s word=%zu got=%08x expected=%08x\n",
                            name, i, golden_fp32::bits(actual[i]), bits);
            ++wrong;
        }
    }
    const auto oracle = golden_fp32::compare(a.data(), b.data(), expected.data(), m, n, k, 0);
    model_mismatches += wrong;
    golden_mismatches += oracle.mismatches;
    if (vector) vector_words += actual.size();
    else ++probes;
    std::printf("RTL_VECTOR case=%s words=%zu model_mismatches=%llu golden_mismatches=%llu subnormal=%llu\n",
                name, actual.size(), (unsigned long long)wrong,
                (unsigned long long)oracle.mismatches, (unsigned long long)subnormal);
}
}
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    golden_fp32::check(std::fesetround(FE_TONEAREST) == 0, "RNE");
    fp32::set_order(fp32::Order::chunk_fma);
    const std::array<size_t, 3> shapes[] = {
        {1,8,9}, {1,8,896}, {1,8,4864}, {32,8,19}, {512,8,17}, {32,16,896},
    };
    for (unsigned corpus = 0; corpus < 2; ++corpus) {
        for (size_t shape = 0; shape < std::size(shapes); ++shape) {
            const auto [m,n,k] = shapes[shape];
            // v2 contains all four modes; v1 contributes only its original subnormal mode.
            for (unsigned mode = corpus == 0 ? 0 : 3; mode < 4; ++mode) {
                const std::string base = std::string(argv[corpus + 1]) + "/case-" +
                    std::to_string(shape) + "-mode-" + std::to_string(mode);
                compare(base.c_str(), load(base + ".a.bin", m*k), load(base + ".b.bin", k*n),
                        load(base + ".expected.bin", m*n), m,n,k,true);
            }
        }
    }
    // Analytic IEEE/RNE probes complement the exact externally RTL-checked corpus.
    for (uint32_t bits : {1u, 0x80000001u, 0x007fffffu, 0x807fffffu}) {
        std::vector<float> a(8, 0.0f), b(8, 0.0f);
        a[7] = from_bits(bits); b[7] = 1.0f;
        compare(("unity-subnormal-" + std::to_string(bits)).c_str(),
                a,b,{from_bits(bits)},1,1,8,false);
    }
    std::vector<float> a(8, 0.0f), b(8, 0.0f);
    a[7] = from_bits(0x80000001); b[7] = 0.5f;
    compare("mvout-negative-underflow-zero", a,b,{0.0f},1,1,8,false);
    compare("tail-underflow-then-seven-zero-fmas",
            {from_bits(0x80000001)},{0.5f},{0.0f},1,1,1,false);
    compare("negative-zero-transfers", std::vector<float>(9, -0.0f),
            std::vector<float>(9, 1.0f), {0.0f},1,1,9,false);
    std::printf("FP32_RTL_VECTORS %s cases=30 words_per_implementation=%llu comparisons=%llu "
                "analytic_probes=%llu model_mismatches=%llu golden_mismatches=%llu\n",
                model_mismatches || golden_mismatches ? "FAIL" : "PASS",
                (unsigned long long)vector_words, (unsigned long long)(2 * vector_words),
                (unsigned long long)probes, (unsigned long long)model_mismatches,
                (unsigned long long)golden_mismatches);
    return model_mismatches || golden_mismatches ? 1 : 0;
}
