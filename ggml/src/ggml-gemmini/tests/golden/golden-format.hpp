// Shared by the legacy-SDK capture fixture (legacy-capture.cpp) and the
// in-tree tiling test, so both print command streams the same way.
#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace golden {

// Matmul cases replayed through tiled_matmul_auto(WS) and matmul_cpu. Shapes
// cover single/multiple tiles, sub-DIM, non-multiple-of-DIM, M=1 decode and
// SPAD/ACC boundaries; flags cover the backend's transpose_B / full_C / low_D /
// bias / repeating-bias / RELU variants.
struct Case {
    const char *name;
    size_t      I, J, K;
    bool        a_transpose, b_transpose, full_c, low_d, bias, repeating_bias;
    int         act;
};

inline const std::vector<Case> &cases() {
    static const std::vector<Case> list = {
        { "single_tile", 16, 16, 16, false, false, true, false, false, false, 0 },
        { "sub_dim", 3, 5, 7, false, false, true, false, false, false, 0 },
        { "non_multiple", 37, 45, 53, false, true, true, false, false, false, 0 },
        { "decode_m1", 1, 4096, 4096, false, true, true, false, false, false, 0 },
        { "decode_m1_wide", 1, 11008, 4096, false, true, true, false, false, false, 0 },
        { "prefill", 128, 4096, 4096, false, true, true, false, false, false, 0 },
        { "multi_tile", 300, 260, 520, false, true, true, false, false, false, 0 },
        { "residual_like", 64, 32, 96, false, false, true, false, false, false, 0 },
        { "bias_low_d", 48, 80, 64, false, false, false, true, true, false, 0 },
        { "bias_acc_repeat", 33, 70, 40, false, false, true, false, true, true, 0 },
        { "a_transpose", 40, 24, 72, true, false, true, false, false, false, 0 },
        { "relu_elem_out", 20, 36, 44, false, false, false, false, false, false, 1 },
        { "large_k", 16, 64, 16384, false, true, true, false, false, false, 0 },
    };
    return list;
}

// DRAM operands inside a registered matrix print as "<buffer>+<byte offset>"
// (allocation addresses differ run to run); no other bit-field is masked.
struct Buffer {
    const char *name;
    uint64_t    base;
    uint64_t    size;
};

inline std::string addr(const std::vector<Buffer> &buffers, uint64_t value) {
    char text[64];
    for (const Buffer &b : buffers) {
        if (b.size != 0 && value >= b.base && value < b.base + b.size) {
            std::snprintf(text, sizeof(text), "%s+%" PRIu64, b.name, value - b.base);
            return text;
        }
    }
    std::snprintf(text, sizeof(text), "0x%" PRIx64, value);
    return text;
}

inline bool is_dram_operand(uint32_t funct, int which) {
    switch (funct) {
        case 1: case 2: case 3: case 14: return which == 1; // MVIN2/MVIN/MVOUT/MVIN3 rs1
        case 10: case 11: return true;                     // LOOP_WS_CONFIG_ADDRS_AB/DC
        default: return false;
    }
}

inline std::string command_line(const std::vector<Buffer> &buffers, uint32_t funct, uint64_t rs1, uint64_t rs2) {
    char        text[160];
    std::string a = is_dram_operand(funct, 1) ? addr(buffers, rs1) : std::string();
    std::string b = is_dram_operand(funct, 2) ? addr(buffers, rs2) : std::string();
    if (a.empty()) {
        std::snprintf(text, sizeof(text), "0x%016" PRIx64, rs1);
        a = text;
    }
    if (b.empty()) {
        std::snprintf(text, sizeof(text), "0x%016" PRIx64, rs2);
        b = text;
    }
    std::snprintf(text, sizeof(text), "CMD f=%u rs1=%s rs2=%s", funct, a.c_str(), b.c_str());
    return text;
}

inline uint64_t fnv1a(const void *data, size_t bytes, uint64_t h = 1469598103934665603ull) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

// Deterministic signed int8 fill in [-128, 127].
inline void fill_i8(int8_t *dst, size_t n, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        dst[i] = (int8_t) (x & 0xff);
    }
}

inline void fill_i32(int32_t *dst, size_t n, uint32_t seed) {
    uint32_t x = seed * 2246822519u + 7u;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        dst[i] = (int32_t) (x % 2000001u) - 1000000;
    }
}

} // namespace golden
