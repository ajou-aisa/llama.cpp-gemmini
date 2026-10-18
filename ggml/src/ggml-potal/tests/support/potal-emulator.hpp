#pragma once

// Test double of the PoTal accelerator behind potal_ws_ops: the RTL semantics
// of LOOP_WS, the SCU and the Sat32 accumulator (LoopMatmul, PoTalSCU,
// PoTalScaleMemory, AccPipe), shared by the host tests. Passing on it proves
// the software contract, not the hardware (FireSim smoke tests do that).
// Depends on potal.h and tiling.hpp only, so API-only builds can use it too.

#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace potal_test {

using ggml::potal::potal_ws_ops;

constexpr uint32_t EMU_DIM = POTAL_HW_DIM;

struct emulator {
    struct entry {
        uint32_t generation;
        uint32_t carrier;
    };
    std::map<std::pair<uint32_t, uint32_t>, entry> scales; // (slot, column) -> valid entry
    std::vector<int32_t> acc[2];
    int                  half          = 0;
    bool                 in_flight     = false;
    bool                 scale_miss    = false; // hardware would hang
    int                  fail_write_at = -1;    // inject a SCALE_WRITE failure on this write
    bool                 fail_fence    = false;
    uint64_t             writes = 0, releases = 0, loops = 0, gemms = 0, fences = 0;
    std::vector<uint32_t> loop_generations; // scale_generation of every LOOP_WS, in order

    emulator() {
        acc[0].assign((size_t) POTAL_HW_ACC_ROWS / 2 * EMU_DIM, 0);
        acc[1].assign((size_t) POTAL_HW_ACC_ROWS / 2 * EMU_DIM, 0);
    }

    // PoTalSCU.scala: left shift with Sat32, zero carrier -> 0.
    static int32_t scu(int32_t partial, uint32_t carrier) {
        if (carrier == POTAL_CARRIER_ZERO || partial == 0) {
            return 0;
        }
        if (carrier >= 32) {
            return partial < 0 ? INT32_MIN : INT32_MAX;
        }
        const int64_t pos = (int64_t) INT32_MAX >> carrier;
        const int64_t neg = (int64_t) INT32_MIN >> carrier;
        if (partial > pos) {
            return INT32_MAX;
        }
        if (partial < neg) {
            return INT32_MIN;
        }
        return (int32_t) ((int64_t) partial * ((int64_t) 1 << carrier)); // no shift of a negative value (UB)
    }

    // AccumulatorMem saturateInt32.
    static int32_t sat_add(int32_t a, int32_t b) {
        const int64_t s = (int64_t) a + b;
        return s > INT32_MAX ? INT32_MAX : s < INT32_MIN ? INT32_MIN : (int32_t) s;
    }

    potal_status write(uint32_t slot, uint32_t gen, uint32_t col, uint32_t carrier) {
        if ((int64_t) writes == fail_write_at) {
            return POTAL_HW_INVALID_SCALE;
        }
        if (slot >= POTAL_SCALE_ENTRIES || col >= EMU_DIM || gen > POTAL_GENERATION_MAX ||
            !potal_carrier_is_valid(carrier)) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
        if (scales.count({ slot, col })) {
            return POTAL_HW_SCALE_CONFLICT;
        }
        scales[{ slot, col }] = { gen, carrier };
        writes++;
        return POTAL_OK;
    }

    potal_status release(uint32_t slot, uint32_t gen, uint32_t col) {
        auto it = scales.find({ slot, col });
        if (it == scales.end() || it->second.generation != gen) {
            return POTAL_HW_RELEASE_MISS;
        }
        scales.erase(it);
        releases++;
        return POTAL_OK;
    }

    potal_status run(const potal_loop_ws & l) {
        if (in_flight) {
            return POTAL_ERR_BUSY;
        }
        const potal_status st = potal_loop_ws_check(&l);
        if (st != POTAL_OK) {
            return st;
        }
        const bool accumulate = (l.flags & POTAL_GEMM_ACCUMULATE) != 0;
        auto &     a_cc       = acc[half];
        // LoopMatmul EX order: k outer, then j, then i.
        for (uint32_t k = 0; k < l.k_tiles; ++k) {
            const uint32_t k_valid = EMU_DIM - (k == l.k_tiles - 1 ? l.pad_k : 0);
            for (uint32_t j = 0; j < l.j_tiles; ++j) {
                const uint32_t cols = EMU_DIM - (j == l.j_tiles - 1 ? l.pad_j : 0);
                for (uint32_t i = 0; i < l.i_tiles; ++i) {
                    const uint32_t rows = EMU_DIM - (i == l.i_tiles - 1 ? l.pad_i : 0);
                    for (uint32_t c = 0; c < cols; ++c) {
                        const uint32_t slot = l.scale_base + (k / l.passes_per_block) * l.j_tiles + j;
                        auto           e    = scales.find({ slot, c });
                        if (e == scales.end() || e->second.generation != l.scale_generation) {
                            scale_miss = true;
                            continue;
                        }
                        for (uint32_t r = 0; r < rows; ++r) {
                            int32_t partial = 0;
                            for (uint32_t kk = 0; kk < k_valid; ++kk) {
                                partial += (int32_t) l.a[(size_t) (i * EMU_DIM + r) * l.lda + k * EMU_DIM + kk] *
                                           l.b[(size_t) (k * EMU_DIM + kk) * l.ldb + j * EMU_DIM + c];
                            }
                            const int32_t v   = scu(partial, e->second.carrier);
                            int32_t &     out = a_cc[((size_t) (i * l.j_tiles + j) * EMU_DIM + r) * EMU_DIM + c];
                            out               = (!accumulate && k == 0) ? v : sat_add(out, v);
                        }
                    }
                }
            }
        }
        if (l.c != nullptr) {
            for (uint32_t i = 0; i < l.i_tiles; ++i) {
                const uint32_t rows = EMU_DIM - (i == l.i_tiles - 1 ? l.pad_i : 0);
                for (uint32_t j = 0; j < l.j_tiles; ++j) {
                    const uint32_t cols = EMU_DIM - (j == l.j_tiles - 1 ? l.pad_j : 0);
                    for (uint32_t r = 0; r < rows; ++r) {
                        for (uint32_t c = 0; c < cols; ++c) {
                            l.c[(size_t) (i * EMU_DIM + r) * l.ldc + j * EMU_DIM + c] =
                                a_cc[((size_t) (i * l.j_tiles + j) * EMU_DIM + r) * EMU_DIM + c];
                        }
                    }
                }
            }
            half ^= 1; // the ACC half advances only when C is stored
        }
        in_flight = true;
        loops++;
        loop_generations.push_back(l.scale_generation);
        return POTAL_OK;
    }

    static potal_status op_write(void * ctx, uint32_t s, uint32_t g, uint32_t c, uint32_t v) {
        return static_cast<emulator *>(ctx)->write(s, g, c, v);
    }
    static potal_status op_release(void * ctx, uint32_t s, uint32_t g, uint32_t c) {
        return static_cast<emulator *>(ctx)->release(s, g, c);
    }
    static potal_status op_gemm(void * ctx, const potal_gemm_ws * g) {
        auto *             e  = static_cast<emulator *>(ctx);
        const potal_status st = potal_gemm_ws_check(g);
        if (st != POTAL_OK) {
            return st;
        }
        const potal_loop_ws l = { g->a, g->b, g->c, g->lda, g->ldb, g->ldc, g->m / (uint32_t) EMU_DIM,
                                  g->n / (uint32_t) EMU_DIM, g->k / (uint32_t) EMU_DIM, 0, 0, 0, g->scale_base,
                                  g->scale_generation, g->passes_per_block, 0 };
        e->gemms++;
        return e->run(l);
    }
    static potal_status op_loop(void * ctx, const potal_loop_ws * l) {
        return static_cast<emulator *>(ctx)->run(*l);
    }
    static potal_status op_fence(void * ctx) {
        auto * e = static_cast<emulator *>(ctx);
        if (e->fail_fence) {
            return POTAL_HW_BAD_COMMAND;
        }
        e->in_flight = false;
        e->fences++;
        return POTAL_OK;
    }

    potal_ws_ops ops() { return { op_write, op_release, op_gemm, op_loop, op_fence, this }; }
};

} // namespace potal_test
