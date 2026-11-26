// Functional checks of the internal Gemmini path without hardware:
//  1. the WS tiler's command stream, executed by the LoopMatmul host model
//     (pb/pb-rocc-model.cpp, an independent decoder of CONFIG/LOOP_WS), must
//     produce the integer GEMM of a naive CPU loop;
//  2. the CPU reference matmul (matmul/reference.hpp) must reproduce the
//     CPU_REFERENCE hashes captured from the legacy SDK for this geometry.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gemmini_params.h"
#include "golden-format.hpp"
#include "matmul/reference.hpp"
#include "tiling.hpp"

#include "pb/pb-rocc-model.hpp"

namespace {

int failures = 0;

void fail(const std::string &what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
}



void fill_boundary(int8_t *dst, size_t n, int pattern) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = pattern == 0 ? int8_t(-128) : pattern == 1 ? int8_t(127) : int8_t(i % 2 ? 127 : -128);
    }
}

void run_ws_case(size_t I, size_t J, size_t K, int fill) {
    ggml::gemmini::pb::FunctionalTransport model;
    std::vector<int8_t>  A(I * K), B(K * J);
    std::vector<int32_t> C(I * J, 0x5a5a5a5a), ref(I * J, 0);
    if (fill < 0) {
        golden::fill_i8(A.data(), A.size(), 11);
        golden::fill_i8(B.data(), B.size(), 12);
    } else {
        fill_boundary(A.data(), A.size(), fill);
        fill_boundary(B.data(), B.size(), fill == 2 ? 2 : 1 - fill);
    }
    for (size_t i = 0; i < I; ++i)
        for (size_t j = 0; j < J; ++j) {
            int64_t acc = 0;
            for (size_t k = 0; k < K; ++k) acc += int32_t(A[i * K + k]) * int32_t(B[k * J + j]);
            ref[i * J + j] = int32_t(acc);
        }
    const gemmini_hw_config_t &cfg = *gemmini_hw_config_build();
    ggml::gemmini::tiling::Matmul mm;
    mm.dim_i = I;
    mm.dim_j = J;
    mm.dim_k = K;
    mm.a = (uint64_t) A.data();
    mm.b = (uint64_t) B.data();
    mm.c = (uint64_t) C.data();
    mm.stride_a = K;
    mm.stride_b = J;
    mm.stride_d = 0; // no bias: the model accepts only an absent D
    mm.stride_c = J;
    mm.a_scale_bits = mm.b_scale_bits = mm.d_scale_bits = 0x3f800000u;
    mm.acc_scale_bits = mm.config_ex_acc_scale_bits = 0x3f800000u;
    mm.full_c = true;
    const auto tiles = ggml::gemmini::tiling::select_ws_tiles(cfg, I, J, K, GEMMINI_ACT_NONE);
    const gemmini_status_t s = ggml::gemmini::tiling::run_ws(&model.transport(), cfg, mm, tiles);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "ws %zux%zux%zu fill=%d", I, J, K, fill);
    if (s != GEMMINI_OK) {
        fail(std::string(tag) + ": " + gemmini_status_name(s));
    } else if (C != ref) {
        fail(std::string(tag) + ": LoopMatmul model result differs from the integer GEMM");
    }
}

void check_cpu_reference(const char *golden_dir) {
    char name[128];
    std::snprintf(name, sizeof(name), "%s/gemmini-d%d-%dx%d-acc%d.golden", golden_dir, DIM, BANK_NUM, BANK_ROWS, ACC_ROWS);
    std::ifstream in(name);
    if (!in) {
        const std::string d16 = std::string(golden_dir) + "/gemmini-d16-4x4096-acc1024.golden";
        if (!std::ifstream(d16)) {
            fail(std::string("golden directory not found: ") + golden_dir);
            return;
        }
        std::printf("CPU reference: NOT APPLICABLE (no golden capture for this geometry: %s)\n", name);
        return;
    }
    std::string line, current;
    size_t      checked = 0;
    while (std::getline(in, line)) {
        char case_name[64], hash[32];
        if (std::sscanf(line.c_str(), "CASE %63s", case_name) == 1) {
            current = case_name;
        } else if (std::sscanf(line.c_str(), "CPU_REFERENCE HASH %31s", hash) == 1) {
            for (const auto &c : golden::cases()) {
                if (current != c.name) continue;
                const size_t rows_a = c.a_transpose ? c.K : c.I, cols_a = c.a_transpose ? c.I : c.K;
                const size_t rows_b = c.b_transpose ? c.J : c.K, cols_b = c.b_transpose ? c.K : c.J;
                std::vector<int8_t>  A(rows_a * cols_a), B(rows_b * cols_b), C8(c.I * c.J, 0);
                std::vector<int32_t> D32(c.repeating_bias ? c.J : c.I * c.J);
                golden::fill_i8(A.data(), A.size(), 1);
                golden::fill_i8(B.data(), B.size(), 2);
                golden::fill_i32(D32.data(), D32.size(), 3);
                matmul_cpu(c.a_transpose, c.b_transpose, c.I, c.J, c.K, A.data(), B.data(),
                           c.bias ? D32.data() : nullptr, C8.data(), cols_a, cols_b, c.J, c.J, MVIN_SCALE_IDENTITY,
                           MVIN_SCALE_IDENTITY, (scale_acc_t) MVIN_SCALE_IDENTITY, c.act, ACC_SCALE_IDENTITY,
                           ACC_SCALE_IDENTITY, c.repeating_bias);
                char got[32];
                std::snprintf(got, sizeof(got), "%016llx", (unsigned long long) golden::fnv1a(C8.data(), C8.size()));
                if (current.empty() || std::strcmp(got, hash) != 0) {
                    fail("CPU reference " + current + ": " + got + " != " + hash);
                }
                ++checked;
            }
        }
    }
    std::printf("CPU reference: %zu cases match the legacy SDK capture\n", checked);
    if (checked == 0) {
        fail("no CPU reference case checked");
    }
}

void check_transport_isolation() {
    // Interleave complete command programs with different operands. A shared
    // decoder/accumulator would mix the addresses even on a single thread.
    ggml::gemmini::pb::FunctionalTransport models[2];
    const int8_t a[2] = {2, -7}, b[2] = {3, 5};
    int32_t output[2] = {99, 99};
    gemmini_record_t records[2][12]{};
    gemmini_recorder_t recorders[2];
    const auto &cfg = *gemmini_hw_config_build();
    for (size_t n = 0; n < 2; ++n) {
        gemmini_recorder_init(&recorders[n], records[n], 12);
        ggml::gemmini::tiling::Matmul mm;
        mm.dim_i = mm.dim_j = mm.dim_k = 1;
        mm.a = reinterpret_cast<uintptr_t>(&a[n]);
        mm.b = reinterpret_cast<uintptr_t>(&b[n]);
        mm.c = reinterpret_cast<uintptr_t>(&output[n]);
        mm.stride_a = mm.stride_b = mm.stride_c = 1;
        mm.a_scale_bits = mm.b_scale_bits = mm.d_scale_bits = 0x3f800000;
        mm.acc_scale_bits = mm.config_ex_acc_scale_bits = 0x3f800000;
        mm.full_c = true;
        if (ggml::gemmini::tiling::run_ws(&recorders[n].transport, cfg, mm, {1, 1, 1}) != GEMMINI_OK ||
            recorders[n].count != 12) {
            fail("isolation fixture must contain 11 commands and one drain");
            return;
        }
    }
    for (size_t i = 0; i < 12; ++i) {
        for (size_t n = 0; n < 2; ++n) {
            const auto &record = records[n][i];
            const auto status = record.kind == GEMMINI_RECORD_DRAIN
                ? gemmini_drain(&models[n].transport())
                : gemmini_issue(&models[n].transport(), &record.command, nullptr);
            if (status != GEMMINI_OK) fail("interleaved independent model stream");
        }
    }
    if (output[0] != 6 || output[1] != -35) fail("model contexts must not share operand or accumulator state");
    for (const auto &model : models) {
        if (model.command_count(GEMMINI_FUNCT_CONFIG) != 5 ||
            model.command_count(GEMMINI_FUNCT_LOOP_WS) != 1 || model.drain_count() != 1)
            fail("actual decoded command counts");
    }
    ggml::gemmini::pb::FunctionalTransport broken;
    if (gemmini_issue(&broken.transport(), &records[0][10].command, nullptr) != GEMMINI_ERR_TRANSPORT ||
        gemmini_drain(&broken.transport()) != GEMMINI_ERR_TRANSPORT ||
        gemmini_drain(&models[0].transport()) != GEMMINI_OK)
        fail("unfinished command sequence faults only its owning model");
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <golden dir>\n", argv[0]);
        return 2;
    }
    const size_t shapes[][3] = { { 16, 16, 16 }, { 1, 16, 16 }, { 3, 5, 7 },    { 37, 45, 53 },
                                 { 1, 300, 520 }, { 64, 96, 1000 }, { 130, 70, 33 } };
    for (const auto &s : shapes) {
        run_ws_case(s[0], s[1], s[2], -1);
    }
    for (int fill = 0; fill < 3; ++fill) {
        run_ws_case(33, 17, 4096, fill);
    }
    check_cpu_reference(argv[1]);
    check_transport_isolation();
    if (failures) {
        std::fprintf(stderr, "gemmini-functional-test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("GEMMINI_FUNCTIONAL_TEST PASS\n");
    return 0;
}
