#include "exec.hpp"
#include <gemmini_params.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
using ggml::gemmini::fp32::Order;
static_assert(DIM == 8 && sizeof(elem_t) == 4 && sizeof(acc_t) == 4);
struct Model {
    std::array<float, BANK_NUM * BANK_ROWS * DIM> sp{};
    std::array<float, ACC_ROWS * DIM> acc{};
    std::array<float, DIM * DIM> weight{};
    size_t ld[3]{}, st = 0;
    uint32_t output = 0;
    size_t output_rows = 0, output_cols = 0;
    unsigned config = 0;
    bool preload = false, have_weights = false;
    uint64_t commands = 0;
} model;
void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FP32 RoCC model: %s command=%llu\n", message,
                     static_cast<unsigned long long>(model.commands));
        std::abort();
    }
}
size_t rows(uint64_t x) { return x >> 48; }
size_t cols(uint64_t x) { return (x >> 32) & 65535; }
size_t acc_row(uint32_t x) { return x & ((uint32_t(1) << 29) - 1); }

void load(uint64_t address, uint64_t shape, unsigned id) {
    const size_t r = rows(shape), c = cols(shape), base = uint32_t(shape);
    require(address && r && r <= DIM && c && !(base >> 31) && model.ld[id] % 4 == 0,
            "unsupported mvin");
    const auto * src = reinterpret_cast<const float *>(address);
    const size_t stride = model.ld[id] / 4;
    require(stride >= c && base + ((c + DIM - 1) / DIM) * DIM <= BANK_NUM * BANK_ROWS,
            "mvin out of scratchpad/DRAM row");
    for (size_t block = 0; block < (c + DIM - 1) / DIM; ++block) {
        float * dest = model.sp.data() + (base + block * DIM) * DIM;
        std::fill_n(dest, DIM * DIM, 0.0f);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < DIM && block * DIM + j < c; ++j)
                // Unity MVIN is still a hardware FP multiply-add with positive zero.
                dest[i * DIM + j] = std::fma(src[i * stride + block * DIM + j], 1.0f, 0.0f);
    }
}
void preload(uint64_t b, uint64_t c) {
    require(!model.preload && rows(c) && rows(c) <= DIM && cols(c) && cols(c) <= DIM &&
            (uint32_t(c) >> 31), "invalid preload/output ownership");
    model.output = uint32_t(c);
    model.output_rows = rows(c); model.output_cols = cols(c);
    if (uint32_t(b) != UINT32_MAX) {
        const size_t base = uint32_t(b), r = rows(b), n = cols(b);
        require(base + DIM <= BANK_NUM * BANK_ROWS && r <= DIM && n <= DIM, "invalid weight tile");
        model.weight.fill(0.0f);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < n; ++j)
                model.weight[i * DIM + j] = model.sp[(base + i) * DIM + j];
        model.have_weights = true;
    }
    require(model.have_weights, "weight reuse before preload");
    model.preload = true;
}
void compute(uint64_t a, uint64_t d) {
    require(model.config == 31 && model.preload && uint32_t(d) == UINT32_MAX,
            "incomplete configuration or unsupported D operand");
    const size_t r = rows(a), k = cols(a), base = uint32_t(a), out = acc_row(model.output);
    require(r == model.output_rows && r <= DIM && k && k <= DIM &&
            base + DIM <= BANK_NUM * BANK_ROWS && out + DIM <= ACC_ROWS, "invalid compute bounds");
    const bool add = model.output & (uint32_t(1) << 30);
    const Order order = ggml::gemmini::fp32::get_order();
    for (size_t i = 0; i < r; ++i) {
        std::array<float, DIM> partial{};
        if (order == Order::serial_fma && add)
            std::copy_n(model.acc.data() + (out + i) * DIM, DIM, partial.data());
        for (size_t lane = 0; lane < DIM; ++lane) {
            const size_t z = order == Order::chunk_reverse_fma ? DIM - 1 - lane : lane;
            const float av = z < k ? model.sp[(base + i) * DIM + z] : 0.0f;
            for (size_t j = 0; j < DIM; ++j)
                partial[j] = std::fma(av, model.weight[z * DIM + j], partial[j]);
        }
        for (size_t j = 0; j < model.output_cols; ++j) {
            float & dest = model.acc[(out + i) * DIM + j];
            dest = !add || order == Order::serial_fma ? partial[j] : dest + partial[j];
        }
    }
    model.preload = false;
}
void store(uint64_t address, uint64_t shape) {
    const size_t r = rows(shape), c = cols(shape), base = acc_row(uint32_t(shape));
    require(address && (uint32_t(shape) >> 29) == 7 && r && r <= DIM && c && c <= DIM &&
            base + r <= ACC_ROWS && model.st >= c * 4 && model.st % 4 == 0 && !model.preload,
            "unsupported mvout");
    auto * dest = reinterpret_cast<float *>(address);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            dest[i * model.st / 4 + j] = std::fma(model.acc[(base + i) * DIM + j], 1.0f, 0.0f);
}
}

void gemmini_fp32_model_issue(uint64_t a, uint64_t b, uint32_t f) {
    ++model.commands;
    constexpr uint64_t one = uint64_t(0x3f800000) << 32;
    switch (f) {
    case 0:
        switch (a & 3) {
        case 0:
            require(std::fegetround() == FE_TONEAREST, "FP32 requires RNE");
            require(a == (one | (uint64_t(1) << 16) | 4) && b == (uint64_t(1) << 48) &&
                    !model.preload, "unsupported EX configuration");
            model.config = 1; break;
        case 2:
            require(a == 2 && (b >> 32) == 0x3f800000, "unsupported ST configuration");
            model.st = uint32_t(b); model.config |= 2; break;
        case 1: {
            const unsigned id = (a >> 3) & 3;
            require(id < 3 && a == (one | (uint64_t(DIM) << 16) | (uint64_t(1) << 8) |
                    (uint64_t(id) << 3) | 1), "unsupported LD configuration");
            model.ld[id] = b; model.config |= 4u << id; break;
        }
        default: require(false, "unsupported CONFIG");
        }
        break;
    case 7: require(!model.preload && a == 0 && b == 0, "invalid flush"); break;
    case 2: load(a, b, 0); break;
    case 1: load(a, b, 1); break;
    case 6: preload(a, b); break;
    case 4: case 5: compute(a, b); break;
    case 3: store(a, b); break;
    default: require(false, "unrecognized RoCC command");
    }
}
void gemmini_fp32_model_fence() {
    require(!model.preload, "fence before completion");
}
