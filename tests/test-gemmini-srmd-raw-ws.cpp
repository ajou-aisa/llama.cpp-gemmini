#include "../../RISC-V-DynDNN-gemmini-include/gemmini_params.h"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using namespace ggml::gemmini::rmd;

struct GemmCase {
    size_t m;
    size_t j;
    size_t k;
};

bool fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return false;
}

bool checked_accumulator(__int128 value, acc_t & output) {
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        return fail("oracle accumulator does not fit int64_t");
    }
    if (value < static_cast<__int128>(std::numeric_limits<acc_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<acc_t>::max())) {
        return false;
    }
    output = static_cast<acc_t>(value);
    return true;
}

bool checked_int64(__int128 value, int64_t & output) {
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        return fail("radix reconstruction does not fit int64_t");
    }
    output = static_cast<int64_t>(value);
    return true;
}

bool cpu_gemm(const std::vector<elem_t> & a, const std::vector<elem_t> & b,
              const GemmCase & shape, std::vector<acc_t> & output) {
    output.assign(shape.m * shape.j, 0);
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t column = 0; column < shape.j; ++column) {
            __int128 sum = 0;
            for (size_t packed_k = 0; packed_k < shape.k; ++packed_k) {
                sum += static_cast<__int128>(a[row * shape.k + packed_k]) *
                    b[column * shape.k + packed_k];
            }
            if (!checked_accumulator(sum, output[row * shape.j + column])) {
                return fail("CPU accumulator does not fit acc_t");
            }
        }
    }
    return true;
}

bool exact_oracle(const std::vector<elem_t> & a, const std::vector<elem_t> & b,
                  const GemmCase & shape, std::vector<acc_t> & output) {
    output.assign(shape.m * shape.j, 0);
    for (size_t cell = 0; cell < output.size(); ++cell) {
        const size_t row = cell / shape.j;
        const size_t column = cell % shape.j;
        const elem_t * const a_row = a.data() + row * shape.k;
        const elem_t * const b_row = b.data() + column * shape.k;
        __int128 sum = 0;
        for (size_t packed_k = 0; packed_k < shape.k; ++packed_k) {
            sum += static_cast<__int128>(a_row[packed_k]) * b_row[packed_k];
        }
        if (!checked_accumulator(sum, output[cell])) {
            return fail("oracle accumulator does not fit acc_t");
        }
    }
    return true;
}

bool run_cpu_case(const GemmCase & shape) {
    static constexpr elem_t values[] = {-128, -127, -65, -1, 0, 1, 63, 127};
    std::vector<elem_t> a(shape.m * shape.k);
    std::vector<elem_t> b(shape.j * shape.k);
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t packed_k = 0; packed_k < shape.k; ++packed_k) {
            a[row * shape.k + packed_k] = values[(row * 3 + packed_k * 5) % 8];
        }
    }
    for (size_t column = 0; column < shape.j; ++column) {
        for (size_t packed_k = 0; packed_k < shape.k; ++packed_k) {
            b[column * shape.k + packed_k] = values[(column * 5 + packed_k * 3 + 1) % 8];
        }
    }

    std::vector<acc_t> actual;
    std::vector<acc_t> expected;
    if (!cpu_gemm(a, b, shape, actual) || !exact_oracle(a, b, shape, expected)) {
        return false;
    }
    for (size_t cell = 0; cell < actual.size(); ++cell) {
        if (actual[cell] != expected[cell]) {
            std::fprintf(stderr, "FAIL: cell=%zu actual=%d expected=%d\n", cell,
                         static_cast<int>(actual[cell]), static_cast<int>(expected[cell]));
            return false;
        }
    }
    std::printf("CPU GEMM exact: M=%zu J=%zu K=%zu cells=%zu\n", shape.m, shape.j,
                shape.k, actual.size());
    return true;
}

bool direct_residual_dot(const std::vector<int32_t> & residuals, const std::vector<elem_t> & b,
                         const GemmCase & shape, size_t row, size_t column, int64_t & output) {
    __int128 sum = 0;
    for (size_t packed_k = 0; packed_k < shape.k; ++packed_k) {
        sum += static_cast<__int128>(residuals[row * shape.k + packed_k]) *
            b[column * shape.k + packed_k];
    }
    return checked_int64(sum, output);
}

void report_radix_mismatch(size_t row, size_t column, const std::array<size_t, 5> & original_k,
                           const std::vector<BalancedDigits> & digits, const std::vector<acc_t> & lane_raw,
                           size_t columns, int64_t reconstructed, int64_t direct) {
    std::fprintf(stderr, "FAIL: row=%zu J=%zu", row, column);
    for (size_t packed_k = 0; packed_k < original_k.size(); ++packed_k) {
        const BalancedDigits & cell = digits[row * original_k.size() + packed_k];
        std::fprintf(stderr, " original_K=%zu packed_K=%zu digits=[%d,%d,%d,%d]", original_k[packed_k],
                     packed_k, static_cast<int>(cell.digits[0]), static_cast<int>(cell.digits[1]),
                     static_cast<int>(cell.digits[2]), static_cast<int>(cell.digits[3]));
    }
    std::fprintf(stderr, " lane_raw=[%d,%d,%d,%d] reconstructed=%lld direct=%lld\n",
                 static_cast<int>(lane_raw[(row * 4 + 0) * columns + column]),
                 static_cast<int>(lane_raw[(row * 4 + 1) * columns + column]),
                 static_cast<int>(lane_raw[(row * 4 + 2) * columns + column]),
                 static_cast<int>(lane_raw[(row * 4 + 3) * columns + column]),
                 static_cast<long long>(reconstructed), static_cast<long long>(direct));
}

bool run_cpu_radix_case() {
    constexpr std::array<int32_t, 17> values = {
        std::numeric_limits<int32_t>::min(), -16777217, -129, -128, -1, 0, 1, 127, 128,
        129, 255, 256, 65535, 65536, 16777215, 16777216, 2139062143,
    };
    constexpr std::array<size_t, 5> original_k = {1, 4, 9, 17, 31};
    constexpr size_t scale_group_begin = 0;
    constexpr size_t scale_group_width = 32;
    constexpr size_t columns = 3;
    const GemmCase shape = {values.size(), columns, original_k.size()};
    const std::array<elem_t, columns * original_k.size()> physical_b = {
        -128, 127, -3, 1, 64,
        127, -128, 5, -1, -64,
        -7, 11, 127, -128, 1,
    };
    const std::vector<elem_t> b(physical_b.begin(), physical_b.end());
    std::vector<int32_t> residuals(shape.m * shape.k, 0);
    std::vector<BalancedDigits> digits(shape.m * shape.k);
    std::vector<elem_t> a_stacked(shape.m * 4 * shape.k, 0);

    for (size_t packed_k = 0; packed_k < original_k.size(); ++packed_k) {
        if (original_k[packed_k] < scale_group_begin ||
            original_k[packed_k] >= scale_group_begin + scale_group_width ||
            (packed_k != 0 && original_k[packed_k - 1] >= original_k[packed_k])) {
            return fail("compact K is not sorted, unique, and in one 32-K scale group");
        }
    }
    for (size_t row = 0; row < values.size(); ++row) {
        const size_t packed_k = (row * 3 + 1) % shape.k;
        residuals[row * shape.k + packed_k] = values[row];
        for (size_t kp = 0; kp < shape.k; ++kp) {
            BalancedDigits & cell = digits[row * shape.k + kp];
            if (!decompose_balanced_radix256(residuals[row * shape.k + kp], cell)) {
                return fail("accepted radix fixture residual rejected");
            }
            for (size_t lane = 0; lane < 4; ++lane) {
                a_stacked[(row * 4 + lane) * shape.k + kp] = cell.digits[lane];
            }
        }
    }

    BalancedDigits rejected{};
    rejected.digits.fill(1);
    rejected.lane_mask = 0x0f;
    if (decompose_balanced_radix256(std::numeric_limits<int32_t>::max(), rejected) ||
        rejected.lane_mask != 0 || rejected.digits != std::array<int8_t, 4>{}) {
        return fail("INT32_MAX decomposition must reject and clear digits");
    }

    std::vector<acc_t> lane_raw;
    std::vector<acc_t> lane_oracle;
    if (!cpu_gemm(a_stacked, b, {shape.m * 4, shape.j, shape.k}, lane_raw) ||
        !exact_oracle(a_stacked, b, {shape.m * 4, shape.j, shape.k}, lane_oracle)) {
        return false;
    }
    for (size_t cell = 0; cell < lane_raw.size(); ++cell) {
        if (lane_raw[cell] != lane_oracle[cell]) {
            return fail("checked CPU lane GEMM differs from independent oracle");
        }
    }
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t column = 0; column < shape.j; ++column) {
            __int128 reconstructed_sum = 0;
            __int128 place = 1;
            for (size_t lane = 0; lane < 4; ++lane) {
                reconstructed_sum += static_cast<__int128>(lane_raw[(row * 4 + lane) * shape.j + column]) * place;
                place *= 256;
            }
            int64_t reconstructed = 0;
            int64_t direct = 0;
            if (!checked_int64(reconstructed_sum, reconstructed) ||
                !direct_residual_dot(residuals, b, shape, row, column, direct)) {
                return false;
            }
            if (reconstructed != direct) {
                report_radix_mismatch(row, column, original_k, digits, lane_raw, shape.j, reconstructed, direct);
                return false;
            }
        }
    }
    std::printf("CPU radix exact: rows=%zu J=%zu compact_K=%zu scale_group=[%zu,%zu)\n", shape.m,
                shape.j, shape.k, scale_group_begin, scale_group_begin + scale_group_width);
    return true;
}

bool run_cpu() {
    return run_cpu_case({1, 1, 1}) && run_cpu_case({DIM + 3, DIM + 5, 2 * DIM + 3}) &&
        run_cpu_radix_case();
}

void pack_kj(const std::vector<elem_t> & source_jk, const GemmCase & shape,
             std::vector<elem_t> & packed_kj) {
    for (size_t k = 0; k < shape.k; ++k) {
        for (size_t column = 0; column < shape.j; ++column) {
            packed_kj[k * shape.j + column] = source_jk[column * shape.k + k];
        }
    }
}

void pack_jk(const std::vector<elem_t> & source_jk, const GemmCase & shape,
             std::vector<elem_t> & packed_jk) {
    for (size_t k = 0; k < shape.k; ++k) {
        for (size_t column = 0; column < shape.j; ++column) {
            packed_jk[column * shape.k + k] = source_jk[column * shape.k + k];
        }
    }
}

void gemm_kj(const std::vector<elem_t> & a, const std::vector<elem_t> & b_kj,
             const GemmCase & shape, std::vector<int64_t> & output) {
    std::fill(output.begin(), output.end(), int64_t{0});
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t k = 0; k < shape.k; ++k) {
            const int64_t activation = a[row * shape.k + k];
            if (activation == 0) {
                continue;
            }
            for (size_t column = 0; column < shape.j; ++column) {
                output[row * shape.j + column] += activation * b_kj[k * shape.j + column];
            }
        }
    }
}

void gemm_jk(const std::vector<elem_t> & a, const std::vector<elem_t> & b_jk,
             const GemmCase & shape, std::vector<int64_t> & output) {
    std::fill(output.begin(), output.end(), int64_t{0});
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t column = 0; column < shape.j; ++column) {
            int64_t sum = 0;
            for (size_t k = 0; k < shape.k; ++k) {
                const int64_t activation = a[row * shape.k + k];
                if (activation != 0) {
                    sum += activation * b_jk[column * shape.k + k];
                }
            }
            output[row * shape.j + column] = sum;
        }
    }
}

template<typename Operation>
double benchmark_ns(size_t iterations, Operation operation) {
    volatile int64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        checksum += operation(iteration);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) / iterations;
}

bool benchmark_layout_case(const char * name, std::vector<elem_t> a,
                           const std::vector<elem_t> & source_jk, const GemmCase & shape) {
    std::vector<elem_t> b_kj(shape.k * shape.j);
    std::vector<elem_t> b_jk(shape.j * shape.k);
    std::vector<int64_t> output_a(shape.m * shape.j);
    std::vector<int64_t> output_b(shape.m * shape.j);
    std::vector<acc_t> expected;
    pack_kj(source_jk, shape, b_kj);
    pack_jk(source_jk, shape, b_jk);
    gemm_kj(a, b_kj, shape, output_a);
    gemm_jk(a, b_jk, shape, output_b);
    if (!exact_oracle(a, source_jk, shape, expected)) {
        return false;
    }
    for (size_t cell = 0; cell < expected.size(); ++cell) {
        if (output_a[cell] != expected[cell] || output_b[cell] != expected[cell]) {
            return fail("layout A or B GEMM differs from exact oracle");
        }
    }

    constexpr size_t pack_iterations = 200000;
    constexpr size_t gemm_iterations = 20000;
    const double pack_a = benchmark_ns(pack_iterations, [&](size_t iteration) {
        pack_kj(source_jk, shape, b_kj);
        return static_cast<int64_t>(b_kj[iteration % b_kj.size()]);
    });
    const double pack_b = benchmark_ns(pack_iterations, [&](size_t iteration) {
        pack_jk(source_jk, shape, b_jk);
        return static_cast<int64_t>(b_jk[iteration % b_jk.size()]);
    });
    const double gemm_a = benchmark_ns(gemm_iterations, [&](size_t iteration) {
        gemm_kj(a, b_kj, shape, output_a);
        return output_a[iteration % output_a.size()];
    });
    const double gemm_b = benchmark_ns(gemm_iterations, [&](size_t iteration) {
        gemm_jk(a, b_jk, shape, output_b);
        return output_b[iteration % output_b.size()];
    });
    const double total_a = pack_a + gemm_a;
    const double total_b = pack_b + gemm_b;
    std::printf(
        "%s exact: yes\n"
        "  A [K][J], transpose_B=false: pack=%.1f ns gemm=%.1f ns total=%.1f ns\n"
        "  B [J][K], transpose_B=true:  pack=%.1f ns gemm=%.1f ns total=%.1f ns\n"
        "  B/A total ratio: %.3fx (%s faster on CPU)\n",
        name, pack_a, gemm_a, total_a, pack_b, gemm_b, total_b, total_b / total_a,
        total_a <= total_b ? "A" : "B");
    return true;
}

bool run_layout_benchmark() {
    const GemmCase shape = {DIM, DIM, DIM};
    std::vector<elem_t> source_jk(shape.j * shape.k);
    std::vector<elem_t> dense_a(shape.m * shape.k);
    std::vector<elem_t> sparse_a(shape.m * shape.k, 0);
    for (size_t column = 0; column < shape.j; ++column) {
        for (size_t k = 0; k < shape.k; ++k) {
            source_jk[column * shape.k + k] = static_cast<elem_t>((column * 11 + k * 7) % 255 - 127);
        }
    }
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t k = 0; k < shape.k; ++k) {
            const elem_t value = static_cast<elem_t>((row * 5 + k * 3) % 127 + 1);
            dense_a[row * shape.k + k] = value;
            if ((row * shape.k + k) % 5 == 0) {
                sparse_a[row * shape.k + k] = value;
            }
        }
    }
    std::printf("CPU layout benchmark: M=%zu J=%zu K=%zu (best-effort host proxy, not FPGA)\n",
                shape.m, shape.j, shape.k);
    return benchmark_layout_case("dense", dense_a, source_jk, shape) &&
        benchmark_layout_case("RMD-like sparse (20% nonzero)", sparse_a, source_jk, shape);
}

int usage(const char * message) {
    std::fprintf(stderr, "%s\nusage: %s --cpu|--bench-layouts|--ws\n", message,
                 "test-gemmini-srmd-raw-ws");
    return 2;
}

}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return usage("expected exactly one mode");
    }
    if (std::strcmp(argv[1], "--cpu") == 0) {
        return run_cpu() ? 0 : 1;
    }
    if (std::strcmp(argv[1], "--bench-layouts") == 0) {
        return run_layout_benchmark() ? 0 : 1;
    }
    if (std::strcmp(argv[1], "--ws") == 0) {
        std::fputs("WS mode is not implemented; real RISC-V Gemmini hardware is required.\n", stderr);
        return 2;
    }
    return usage("unknown or malformed mode");
}
