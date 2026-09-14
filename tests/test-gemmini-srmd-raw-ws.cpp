#include <gemmini_params.h>
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-types.hpp"

#include <array>
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
                           const std::vector<NativeBalancedDigits> & digits, const std::vector<acc_t> & lane_raw,
                           size_t columns, int64_t reconstructed, int64_t direct) {
    const size_t lane_count = balanced_radix_contract(8).lane_capacity;
    std::fprintf(stderr, "FAIL: row=%zu J=%zu", row, column);
    for (size_t packed_k = 0; packed_k < original_k.size(); ++packed_k) {
        const NativeBalancedDigits & cell = digits[row * original_k.size() + packed_k];
        std::fprintf(stderr, " original_K=%zu packed_K=%zu digits=[", original_k[packed_k], packed_k);
        for (size_t lane = 0; lane < lane_count; ++lane) {
            std::fprintf(stderr, "%s%d", lane == 0 ? "" : ",", static_cast<int>(cell.digits[lane]));
        }
        std::fputs("]", stderr);
    }
    std::fputs(" lane_raw=[", stderr);
    for (size_t lane = 0; lane < lane_count; ++lane) {
        std::fprintf(stderr, "%s%d", lane == 0 ? "" : ",",
                     static_cast<int>(lane_raw[(row * lane_count + lane) * columns + column]));
    }
    std::fprintf(stderr, "] reconstructed=%lld direct=%lld\n",
                 static_cast<long long>(reconstructed), static_cast<long long>(direct));
}

bool run_cpu_radix_case() {
    constexpr std::array<int32_t, 20> values = {
        std::numeric_limits<int32_t>::min(), -16777217, -129, -128, -1, 0, 1, 127, 128,
        129, 255, 256, 65535, 65536, 16777215, 16777216, 2139062143,
        2139062144, std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::max(),
    };
    constexpr std::array<size_t, 5> original_k = {1, 4, 9, 17, 31};
    constexpr size_t scale_group_begin = 0;
    constexpr size_t scale_group_width = 32;
    constexpr size_t columns = 3;
    const size_t lane_count = balanced_radix_contract(8).lane_capacity;
    const GemmCase shape = {values.size(), columns, original_k.size()};
    const std::array<elem_t, columns * original_k.size()> physical_b = {
        -128, 127, -3, 1, 64,
        127, -128, 5, -1, -64,
        -7, 11, 127, -128, 1,
    };
    const std::vector<elem_t> b(physical_b.begin(), physical_b.end());
    std::vector<int32_t> residuals(shape.m * shape.k, 0);
    std::vector<NativeBalancedDigits> digits(shape.m * shape.k);
    std::vector<elem_t> a_stacked(shape.m * lane_count * shape.k, 0);

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
            NativeBalancedDigits & cell = digits[row * shape.k + kp];
            if (decompose_balanced_radix(residuals[row * shape.k + kp], 8, cell) != RmdStatus::success) {
                return fail("accepted radix fixture residual rejected");
            }
            for (size_t lane = 0; lane < lane_count; ++lane) {
                a_stacked[(row * lane_count + lane) * shape.k + kp] = static_cast<elem_t>(cell.digits[lane]);
            }
        }
    }

    NativeBalancedDigits zero{};
    zero.digits.fill(1);
    zero.active_lane_count = static_cast<uint8_t>(lane_count);
    if (decompose_balanced_radix(0, 8, zero) != RmdStatus::success ||
        zero.active_lane_count != 0 || zero.digits != std::array<int32_t, kMaxNativeRadixLanes>{}) {
        return fail("zero decomposition must clear all reused digits and the active lane count");
    }

    std::vector<acc_t> lane_raw;
    if (!cpu_gemm(a_stacked, b, {shape.m * lane_count, shape.j, shape.k}, lane_raw)) {
        return false;
    }
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t column = 0; column < shape.j; ++column) {
            __int128 reconstructed_sum = 0;
            __int128 place = 1;
            for (size_t lane = 0; lane < lane_count; ++lane) {
                reconstructed_sum += static_cast<__int128>(
                    lane_raw[(row * lane_count + lane) * shape.j + column]) * place;
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

int usage(const char * message) {
    std::fprintf(stderr, "%s\nusage: %s --cpu|--ws\n", message,
                 "test-gemmini-srmd-raw-ws");
    return 2;
}

}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return usage("expected exactly one mode");
    }
    if (std::strcmp(argv[1], "--help") == 0) {
        std::fputs("usage: test-gemmini-srmd-raw-ws --cpu|--ws\n", stdout);
        return 0;
    }
    if (std::strcmp(argv[1], "--cpu") == 0) {
        return run_cpu_radix_case() ? 0 : 1;
    }
    if (std::strcmp(argv[1], "--ws") == 0) {
        std::fputs("WS mode is not implemented; real RISC-V Gemmini hardware is required.\n", stderr);
        return 2;
    }
    return usage("unknown or malformed mode");
}
