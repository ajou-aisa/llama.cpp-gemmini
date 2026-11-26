// Golden-reference fixture: runs the verbatim legacy SDK tiler (extracted by
// extract-legacy-sdk.py) with its RoCC issue bridge replaced by a recorder,
// and prints the normalized command stream, the selected tiles and the CPU
// reference results. Build: see README.md in this directory.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "gemmini_params.h"
#include "golden-format.hpp"

namespace capture {
struct Entry {
    bool     fence;
    uint32_t funct;
    uint64_t rs1, rs2;
};
static std::vector<Entry> log;
static bool full_stream = false;
static void issue(uint64_t rs1, uint64_t rs2, uint32_t funct) { log.push_back({ false, funct, rs1, rs2 }); }
static void fence() { log.push_back({ true, 0, 0, 0 }); }
} // namespace capture

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) capture::issue((uint64_t) (rs1), (uint64_t) (rs2), funct);
#define gemmini_fence() capture::fence()

struct ggml_gemmini_args_t {
    size_t      I = 0, J = 0, K = 0;
    int         act = 0;
    size_t      tile_I = 0, tile_J = 0, tile_K = 0;
    std::string matmul_layer = "golden";
};
namespace ggml::gemmini::cycle {
static uint64_t read() { return 0; }
} // namespace ggml::gemmini::cycle
#define gemmini_log_debug_layer(...) ((void) 0)

#include "legacy-extract.inc"

static void emit_case(const golden::Case &c) {
    const size_t rows_a = c.a_transpose ? c.K : c.I, cols_a = c.a_transpose ? c.I : c.K;
    const size_t rows_b = c.b_transpose ? c.J : c.K, cols_b = c.b_transpose ? c.K : c.J;
    std::vector<int8_t>  A(rows_a * cols_a), B(rows_b * cols_b);
    std::vector<int32_t> D32(c.repeating_bias ? c.J : c.I * c.J);
    std::vector<int8_t>  D8(D32.size());
    std::vector<int32_t> C32(c.I * c.J);
    std::vector<int8_t>  C8(c.I * c.J);
    golden::fill_i8(A.data(), A.size(), 1);
    golden::fill_i8(B.data(), B.size(), 2);
    golden::fill_i32(D32.data(), D32.size(), 3);
    golden::fill_i8(D8.data(), D8.size(), 4);
    const void *D = !c.bias ? nullptr : (c.low_d ? (const void *) D8.data() : (const void *) D32.data());
    void       *C = c.full_c ? (void *) C32.data() : (void *) C8.data();
    const std::vector<golden::Buffer> buffers = {
        { "A", (uint64_t) A.data(), A.size() },
        { "B", (uint64_t) B.data(), B.size() },
        { "D", (uint64_t) D, D ? (c.low_d ? D8.size() : D32.size() * 4) : 0 },
        { "C", (uint64_t) C, c.full_c ? C32.size() * 4 : C8.size() },
    };

    ggml_gemmini_args_t args;
    args.I   = c.I;
    args.J   = c.J;
    args.K   = c.K;
    args.act = c.act;
    gemmini_set_tile_ws(&args);
    std::printf("CASE %s DIM=%d I=%zu J=%zu K=%zu\n", c.name, DIM, c.I, c.J, c.K);
    std::printf("SET_TILE_WS %zu %zu %zu\n", args.tile_I, args.tile_J, args.tile_K);

    capture::log.clear();
    tiled_matmul_auto(c.I, c.J, c.K, A.data(), B.data(), D, C, cols_a, cols_b, c.J, c.J, MVIN_SCALE_IDENTITY,
                      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, c.act, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                      c.repeating_bias, c.a_transpose, c.b_transpose, c.full_c, c.low_d, 0, WS);
    uint64_t     hash = golden::fnv1a(nullptr, 0);
    const size_t head = capture::full_stream ? capture::log.size() : 24;
    std::vector<std::string> lines;
    for (const auto &e : capture::log) {
        lines.push_back(e.fence ? std::string("FENCE") : golden::command_line(buffers, e.funct, e.rs1, e.rs2));
        hash = golden::fnv1a(lines.back().data(), lines.back().size(), hash);
        hash = golden::fnv1a("\n", 1, hash);
    }
    std::printf("WS_COMMANDS %zu HASH %016llx\n", lines.size(), (unsigned long long) hash);
    for (size_t i = 0; i < lines.size() && i < head; ++i) {
        std::printf("  %s\n", lines[i].c_str());
    }

    if (c.I * c.J * c.K <= (size_t) 1 << 22 && !c.low_d) {
        std::fill(C32.begin(), C32.end(), 0);
        std::fill(C8.begin(), C8.end(), 0);
        tiled_matmul_auto(c.I, c.J, c.K, A.data(), B.data(), D, C8.data(), cols_a, cols_b, c.J, c.J,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, c.act, ACC_SCALE_IDENTITY,
                          ACC_SCALE_IDENTITY, c.repeating_bias, c.a_transpose, c.b_transpose, false, c.low_d, 0, CPU);
        std::printf("CPU_REFERENCE HASH %016llx\n", (unsigned long long) golden::fnv1a(C8.data(), C8.size()));
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--full-stream") == 0) {
        capture::full_stream = true;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--full-stream]\n", argv[0]);
        return 2;
    }
    std::printf("GEOMETRY DIM=%d BANK_NUM=%d BANK_ROWS=%d ACC_ROWS=%d MAX_BYTES=%d\n", DIM, BANK_NUM, BANK_ROWS,
                ACC_ROWS, MAX_BYTES);
    for (const auto &c : golden::cases()) {
        emit_case(c);
    }
    return 0;
}
