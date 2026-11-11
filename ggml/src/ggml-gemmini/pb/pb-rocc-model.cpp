#if defined(GGML_GEMMINI_PB) && !defined(__riscv)
#include <gemmini.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
[[noreturn]] void reject(const char * reason) {
    std::fprintf(stderr, "gemmini-pb RoCC model: %s\n", reason);
    std::abort();
}
void require(bool condition, const char * reason) {
    if (!condition) reject(reason);
}

struct Operand {
    std::vector<int8_t> codes;
    size_t rows = 0, cols = 0;
};

struct Model {
    size_t i = 0, j = 0, k = 0, pad_i = 0, pad_j = 0, pad_k = 0;
    uint64_t a = 0, b = 0, d = 0, c = 0;
    size_t sa = 0, sb = 0, sd = 0, sc = 0;
    size_t load_stride[3]{}, store_stride = 0;
    unsigned config_mask = 0, loop_mask = 0;
    std::array<Operand, 3> cached_a, cached_b;
    std::vector<int32_t> accumulator;
    size_t acc_rows = 0, acc_cols = 0;
    bool pending = false;
} state;

const Operand & load(std::array<Operand, 3> & cache, unsigned id, uint64_t address,
                     size_t rows, size_t cols, size_t stride) {
    require(id < cache.size(), "invalid operand reuse ID");
    Operand & operand = cache[id];
    if (address) {
        require(stride >= cols, "invalid operand DRAM stride");
        const auto * source = reinterpret_cast<const int8_t *>(address);
        operand.codes.resize(rows * cols);
        operand.rows = rows;
        operand.cols = cols;
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                operand.codes[row * cols + col] = source[row * stride + col];
    } else {
        require(id != 0 && operand.rows == rows && operand.cols == cols,
                "operand reuse without a matching DRAM load");
    }
    return operand;
}

void run(uint64_t rs1, uint64_t rs2) {
    require(state.config_mask == 31 && state.loop_mask == 31, "incomplete WS configuration");
    require(rs2 == 0 && (rs1 & ~uint64_t{0xf0003}) == 0 && (rs1 & 2),
            "only full-width, non-transposed, no-activation WS matmul is modeled");
    require(state.d == 0 && state.sd == 0, "bias is not modeled");
    require(state.i && state.j && state.k &&
            state.pad_i < DIM && state.pad_j < DIM && state.pad_k < DIM,
            "invalid loop bounds");
    const size_t rows = state.i * DIM - state.pad_i;
    const size_t cols = state.j * DIM - state.pad_j;
    const size_t depth = state.k * DIM - state.pad_k;
    require(state.sa == state.load_stride[0] && state.sb == state.load_stride[1] &&
            state.sc * sizeof(int32_t) == state.store_stride && state.sc >= cols,
            "loop/config stride mismatch");
    const auto & a = load(state.cached_a, (rs1 >> 18) & 3, state.a, rows, depth, state.sa);
    const auto & b = load(state.cached_b, (rs1 >> 16) & 3, state.b, depth, cols, state.sb);
    const bool accumulate = rs1 & 1;
    if (accumulate) {
        require(state.pending && state.acc_rows == rows && state.acc_cols == cols,
                "accumulate without a preceding K tile");
    } else {
        require(!state.pending, "overwriting an unfinished output tile");
        state.accumulator.assign(rows * cols, 0);
        state.acc_rows = rows;
        state.acc_cols = cols;
    }
    // LoopMatmul.scala:427: overwrite at k=0 unless ex_accumulate, then accumulate.
    for (size_t ti = 0; ti < state.i; ++ti) {
        for (size_t tj = 0; tj < state.j; ++tj) {
            for (size_t tk = 0; tk < state.k; ++tk) {
                for (size_t i = ti * DIM; i < std::min(rows, (ti + 1) * DIM); ++i) {
                    for (size_t j = tj * DIM; j < std::min(cols, (tj + 1) * DIM); ++j) {
                        int64_t value = (tk == 0 && !accumulate) ? 0 : state.accumulator[i * cols + j];
                        for (size_t k = tk * DIM; k < std::min(depth, (tk + 1) * DIM); ++k)
                            value += int32_t(a.codes[i * depth + k]) * int32_t(b.codes[k * cols + j]);
                        require(value >= INT32_MIN && value <= INT32_MAX, "INT32 overflow");
                        state.accumulator[i * cols + j] = static_cast<int32_t>(value);
                    }
                }
            }
        }
    }
    state.pending = true;
    if (state.c) {
        auto * output = reinterpret_cast<int32_t *>(state.c);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                output[row * state.sc + col] = state.accumulator[row * cols + col];
        state.pending = false;
    }
    state.loop_mask = 0;
}
} // namespace

void gemmini_pb_rocc_model_issue(uint64_t rs1, uint64_t rs2, uint32_t funct) {
    switch (funct) {
    case k_CONFIG:
        switch (rs1 & 3) {
        case CONFIG_EX:
            require(rs1 == ((uint64_t{0x3f800000} << 32) | (uint64_t{1} << 16) |
                            (uint64_t{WEIGHT_STATIONARY} << 2) | CONFIG_EX) &&
                    rs2 == (uint64_t{1} << 48), "unsupported execute configuration");
            require(!state.pending && state.loop_mask == 0, "execute config during unfinished loop");
            state.config_mask = 1;
            break;
        case CONFIG_ST:
            require(rs1 == CONFIG_ST && (rs2 >> 32) == 0x3f800000, "unsupported store configuration");
            state.store_stride = static_cast<uint32_t>(rs2);
            state.config_mask |= 2;
            break;
        case CONFIG_LD: {
            const unsigned id = (rs1 >> 3) & 3;
            require(id < 3 &&
                    rs1 == ((uint64_t{0x3f800000} << 32) | (uint64_t{DIM} << 16) |
                            (uint64_t{1} << 8) | (uint64_t{id} << 3) | CONFIG_LD),
                    "unsupported load configuration");
            state.load_stride[id] = rs2;
            state.config_mask |= 4U << id;
            break;
        }
        default: reject("unsupported CONFIG command");
        }
        return;
    case k_LOOP_WS_CONFIG_BOUNDS:
        require((rs1 >> 48) == 0 && (rs2 >> 48) == 0, "invalid bounds encoding");
        state.pad_i = rs1 & 0xffff; state.pad_j = (rs1 >> 16) & 0xffff; state.pad_k = (rs1 >> 32) & 0xffff;
        state.i = rs2 & 0xffff; state.j = (rs2 >> 16) & 0xffff; state.k = (rs2 >> 32) & 0xffff;
        state.loop_mask |= 1;
        return;
    case k_LOOP_WS_CONFIG_ADDRS_AB:
        state.a = rs1; state.b = rs2; state.loop_mask |= 2;
        return;
    case k_LOOP_WS_CONFIG_ADDRS_DC:
        state.d = rs1; state.c = rs2; state.loop_mask |= 4;
        return;
    case k_LOOP_WS_CONFIG_STRIDES_AB:
        state.sa = rs1; state.sb = rs2; state.loop_mask |= 8;
        return;
    case k_LOOP_WS_CONFIG_STRIDES_DC:
        state.sd = rs1; state.sc = rs2; state.loop_mask |= 16;
        return;
    case k_LOOP_WS:
        run(rs1, rs2);
        return;
    case k_FLUSH:
        return;
    default:
        reject("unrecognized command");
    }
}

void gemmini_pb_rocc_model_fence(void) {
    require(state.loop_mask == 0 && !state.pending, "fence with unfinished output");
}
#endif
