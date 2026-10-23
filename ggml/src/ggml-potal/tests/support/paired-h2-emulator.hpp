#pragma once

// Command-level test double of the H2 PoTal accelerator behind a
// potal_paired_sink: it interprets the raw command stream of the paired session
// (CONFIG, LOOP_WS_CONFIG_*, paired LOOP_WS with LdA / LdB / indexed PRELOAD,
// INDEX_CONFIG / INDEX_WRITE, SCALE_WRITE / RELEASE, MVOUT, FENCE, FLUSH) on an
// explicit scratchpad, two ACC banks, the scale cache and two Index Tables, with
// the SCU / Sat32 arithmetic of support/potal-emulator.hpp. Passing on it shows
// that the emitted commands compute the contract on the documented RTL
// semantics; it models no RTL timing or concurrency.

#include "potal.h"
#include "potal_hw_params.h"
#include "support/potal-emulator.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace potal_test {

struct h2_emulator {
    static constexpr uint32_t D         = POTAL_HW_DIM;
    static constexpr size_t   sp_rows   = (size_t) POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS;
    static constexpr size_t   half_rows = sp_rows / 2;
    static constexpr size_t   acc_rows  = POTAL_HW_ACC_ROWS;

    struct index_table {
        bool     configured = false;
        uint32_t b_base = 0, j_tiles = 0, source_k_rows = 0, active_rows = 0;
        std::vector<uint32_t> entry;
        std::vector<uint8_t>  valid;
    };
    struct scale_entry {
        bool     valid = false;
        uint32_t generation = 0, carrier = 0;
    };

    std::vector<int8_t>   sp  = std::vector<int8_t>(sp_rows * D, 0);
    std::vector<int32_t>  acc = std::vector<int32_t>(acc_rows * D, 0);
    scale_entry           scales[POTAL_SCALE_ENTRIES][D];
    index_table           tables[2];
    uint32_t              selected_table = 0;
    uint64_t              ld_stride[2] = { 0, 0 }, st_stride = 0;
    uint64_t              bounds_rs1 = 0, bounds_rs2 = 0, a_dram = 0, b_dram = 0, d_dram = 0, c_dram = 0,
                          a_stride = 0, b_stride = 0, c_stride_elems = 0;
    // Counters and records.
    uint64_t    main_loops = 0, residual_loops = 0, index_configs = 0, index_writes = 0, scale_writes = 0,
                scale_releases = 0, mvouts = 0, fences = 0, flushes = 0, b_loads = 0;
    std::vector<potal_paired_cmd_record> records;
    std::string                          errors;      // contract violations seen (empty: none)
    bool                                 scale_miss = false;
    uint32_t                             fail_funct = 0xFFFFFFFFu; // answer this xd funct with fail_status, once
    potal_status                         fail_status = POTAL_OK;

    void err(const std::string & e) {
        errors += e;
        errors += "; ";
    }

    potal_paired_sink sink() { return { &h2_emulator::on_cmd, this }; }

    static potal_status on_cmd(void * ctx, const potal_paired_cmd_record * rec) {
        return static_cast<h2_emulator *>(ctx)->command(*rec);
    }

    potal_status command(const potal_paired_cmd_record & r) {
        records.push_back(r);
        if (r.funct == fail_funct) {
            fail_funct = 0xFFFFFFFFu;
            return fail_status;
        }
        switch (r.funct) {
            case POTAL_FUNCT_CONFIG: return config(r.rs1, r.rs2);
            case POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS: bounds_rs1 = r.rs1; bounds_rs2 = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB: a_dram = r.rs1; b_dram = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC: d_dram = r.rs1; c_dram = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB: a_stride = r.rs1; b_stride = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC: c_stride_elems = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS: return loop_ws(r.rs1, r.rs2);
            case POTAL_FUNCT_MVOUT: return mvout(r.rs1, r.rs2);
            case POTAL_FUNCT_SCALE_WRITE: return scale_write(r.rs1, r.rs2);
            case POTAL_FUNCT_SCALE_RELEASE: return scale_release(r.rs1);
            case POTAL_FUNCT_INDEX_CONFIG: return index_config(r.rs1, r.rs2);
            case POTAL_FUNCT_INDEX_WRITE: return index_write(r.rs1, r.rs2);
            case POTAL_FUNCT_FENCE: fences++; return POTAL_OK;
            case POTAL_FUNCT_FLUSH: flushes++; return POTAL_OK;
            default: err("unknown funct " + std::to_string(r.funct)); return POTAL_HW_BAD_COMMAND;
        }
    }

    potal_status config(uint64_t rs1, uint64_t rs2) {
        switch (rs1 & 3) {
            case 1: ld_stride[(rs1 >> 3) & 3] = rs2; break; // CONFIG_LD: state id at [4:3], row bytes in rs2
            case 2: st_stride = rs2; break;                 // CONFIG_ST: row bytes
            default: break;                                 // CONFIG_EX
        }
        return POTAL_OK;
    }

    potal_status scale_write(uint64_t rs1, uint64_t rs2) {
        const uint32_t slot = rs1 & 0xFF, gen = (rs1 >> 8) & 0xFF, col = (rs1 >> 16) & 0x3F;
        if (col >= D) {
            return POTAL_HW_INVALID_COLUMN;
        }
        if (!potal_carrier_is_valid((uint32_t) rs2)) {
            return POTAL_HW_INVALID_SCALE;
        }
        if (scales[slot][col].valid) {
            return POTAL_HW_SCALE_CONFLICT;
        }
        scales[slot][col] = { true, gen, (uint32_t) rs2 };
        scale_writes++;
        return POTAL_OK;
    }

    potal_status scale_release(uint64_t rs1) {
        const uint32_t slot = rs1 & 0xFF, gen = (rs1 >> 8) & 0xFF, col = (rs1 >> 16) & 0x3F;
        if (col >= D) {
            return POTAL_HW_INVALID_COLUMN;
        }
        if (!scales[slot][col].valid || scales[slot][col].generation != gen) {
            return POTAL_HW_RELEASE_MISS;
        }
        scales[slot][col].valid = false;
        scale_releases++;
        return POTAL_OK;
    }

    // PoTalScaleCommandIngress: the buffer from bBase; the region inside its weight bank.
    potal_status index_config(uint64_t rs1, uint64_t rs2) {
        index_table t;
        t.b_base        = (uint32_t) (rs1 & 0xFFFFFFFFull);
        t.j_tiles       = (uint32_t) (rs1 >> 32);
        t.source_k_rows = (uint32_t) (rs2 & 0xFFFFFFFFull);
        t.active_rows   = (uint32_t) (rs2 >> 32);
        const uint32_t buffer = t.b_base >= half_rows ? 1 : 0;
        const uint64_t begin  = buffer == 0 ? sp_rows / 4 : 3 * sp_rows / 4;
        const uint64_t end    = buffer == 0 ? sp_rows / 2 : sp_rows;
        const uint64_t region = (uint64_t) ((t.source_k_rows + D - 1) / D) * t.j_tiles * D;
        if (t.j_tiles == 0 || t.source_k_rows == 0 || t.active_rows == 0 || t.active_rows > POTAL_HW_INDEX_ENTRIES ||
            t.b_base < begin || t.b_base >= end || t.b_base + region > end) {
            return POTAL_HW_INVALID_ADDRESS;
        }
        t.configured = true;
        t.entry.assign(POTAL_HW_INDEX_ENTRIES, 0);
        t.valid.assign(POTAL_HW_INDEX_ENTRIES, 0);
        tables[buffer]  = t;
        selected_table = buffer;
        index_configs++;
        return POTAL_OK;
    }

    potal_status index_write(uint64_t rs1, uint64_t rs2) {
        if ((rs1 >> 32) != 0 || (rs2 >> 32) != 0) {
            return POTAL_HW_INVALID_ENCODING;
        }
        index_table & t = tables[selected_table];
        const uint32_t slot = (uint32_t) rs1, k = (uint32_t) rs2;
        if (!t.configured || slot >= t.active_rows || slot >= POTAL_HW_INDEX_ENTRIES || k >= t.source_k_rows) {
            return POTAL_HW_INVALID_ADDRESS;
        }
        t.entry[slot] = k;
        t.valid[slot] = 1;
        index_writes++;
        return POTAL_OK;
    }

    // LoopMatmul paired LOOP_WS: LdA (and LdB unless resident), then EX in k, j, i order with the
    // SCU per pass and Sat32 into the paired ACC bank; no C store.
    potal_status loop_ws(uint64_t rs1, uint64_t rs2) {
        const bool     accumulate = (rs1 & 1) != 0;
        const bool     enable     = ((rs1 >> 20) & 1) != 0;
        const bool     paired     = ((rs1 >> 61) & 1) != 0;
        const bool     residual   = ((rs1 >> 62) & 1) != 0;
        const bool     indexed    = ((rs1 >> 63) & 1) != 0;
        const uint32_t b_spad     = (rs1 >> 16) & 3, a_spad = (rs1 >> 18) & 3;
        const uint32_t base       = (rs1 >> 21) & 0xFFFF, gen = (rs1 >> 37) & 0xFF, ppb = (rs1 >> 45) & 0xFFFF;
        const uint32_t offset_tiles = (uint32_t) (rs2 >> 3);
        const uint32_t I = bounds_rs2 & 0xFFFF, J = (bounds_rs2 >> 16) & 0xFFFF, K = (bounds_rs2 >> 32) & 0xFFFF;
        const uint32_t pad_i = bounds_rs1 & 0xFFFF, pad_j = (bounds_rs1 >> 16) & 0xFFFF, pad_k = (bounds_rs1 >> 32) & 0xFFFF;
        if (!enable || !paired || a_spad == 0 || b_spad == 0 || a_spad != b_spad || I == 0 || J == 0 || K == 0 ||
            ppb == 0 || K % ppb != 0 || pad_i >= D || pad_j >= D || pad_k >= D) {
            err("LOOP_WS fields");
            return POTAL_HW_BAD_COMMAND;
        }
        if (indexed && (!residual || b_dram != 0)) {
            err("indexed LOOP_WS needs residual mode and B DRAM 0");
            return POTAL_HW_BAD_COMMAND;
        }
        if (!indexed && b_dram == 0) {
            err("Main LOOP_WS without B");
            return POTAL_HW_BAD_COMMAND;
        }
        const uint32_t pair      = a_spad - 1;
        const size_t   a_base    = (size_t) pair * half_rows + (size_t) offset_tiles * D;
        const size_t   bank_rows = POTAL_HW_BANK_ROWS;
        if ((size_t) offset_tiles * D + (size_t) I * K * D > bank_rows) {
            err("A beyond the activation bank");
            return POTAL_HW_BAD_COMMAND;
        }
        // LdA: tile (i, k) at sp row a_base + (i * K + k) * D, padded rows / columns zero.
        const int8_t * a = reinterpret_cast<const int8_t *>((uintptr_t) a_dram);
        for (uint32_t i = 0; i < I; ++i) {
            const uint32_t rows = D - (i + 1 == I ? pad_i : 0);
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t cols = D - (k + 1 == K ? pad_k : 0);
                for (uint32_t r = 0; r < D; ++r) {
                    int8_t * dst = &sp[(a_base + ((size_t) i * K + k) * D + r) * D];
                    for (uint32_t c = 0; c < D; ++c) {
                        dst[c] = (r < rows && c < cols) ? a[((size_t) i * D + r) * a_stride + (size_t) k * D + c] : 0;
                    }
                }
            }
        }
        // LdB (Main): rear-placed in the pair's weight bank; tile (k, j) at b_start + (k * J + j) * D.
        const size_t b_start = (size_t) (pair + 1) * half_rows - (size_t) K * J * D;
        if (!indexed) {
            if ((size_t) K * J * D > bank_rows) {
                err("B beyond the weight bank");
                return POTAL_HW_BAD_COMMAND;
            }
            const int8_t * b = reinterpret_cast<const int8_t *>((uintptr_t) b_dram);
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t rows = D - (k + 1 == K ? pad_k : 0);
                for (uint32_t j = 0; j < J; ++j) {
                    const uint32_t cols = D - (j + 1 == J ? pad_j : 0);
                    for (uint32_t kk = 0; kk < D; ++kk) {
                        int8_t * dst = &sp[(b_start + ((size_t) k * J + j) * D + kk) * D];
                        for (uint32_t c = 0; c < D; ++c) {
                            dst[c] = (kk < rows && c < cols) ? b[((size_t) k * D + kk) * b_stride + (size_t) j * D + c] : 0;
                        }
                    }
                }
            }
            b_loads++;
        }
        const index_table & t = tables[pair];
        if (indexed && (!t.configured || t.active_rows != (uint32_t) (K * D - pad_k))) {
            err("indexed LOOP_WS without a matching Index window");
            return POTAL_HW_BAD_COMMAND;
        }
        const size_t acc_base = residual ? acc_rows / 2 : 0;
        if ((size_t) I * J * D > acc_rows / 2) {
            err("paired ACC exceeds one bank");
            return POTAL_HW_BAD_COMMAND;
        }
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t k_valid = D - (k + 1 == K ? pad_k : 0);
            for (uint32_t j = 0; j < J; ++j) {
                if (indexed && j >= t.j_tiles) {
                    err("indexed J tile out of range");
                    return POTAL_HW_BAD_COMMAND;
                }
                const uint32_t cols = D - (j + 1 == J ? pad_j : 0);
                // Stationary weight tile: resident rows, through the Index Table when indexed.
                int8_t w[64][64];
                for (uint32_t kk = 0; kk < D; ++kk) {
                    size_t row = 0;
                    bool   zero = kk >= k_valid;
                    if (!zero && indexed) {
                        const uint32_t slot = k * D + kk;
                        if (!t.valid[slot]) {
                            err("indexed preload reads an unwritten slot");
                            return POTAL_HW_BAD_COMMAND;
                        }
                        const uint32_t src = t.entry[slot];
                        row = (size_t) t.b_base + ((size_t) (src / D) * t.j_tiles + j) * D + src % D;
                    } else if (!zero) {
                        row = b_start + ((size_t) k * J + j) * D + kk;
                    }
                    for (uint32_t c = 0; c < D; ++c) {
                        w[kk][c] = zero ? 0 : sp[row * D + c];
                    }
                }
                for (uint32_t i = 0; i < I; ++i) {
                    const uint32_t rows = D - (i + 1 == I ? pad_i : 0);
                    const uint32_t slot = base + (k / ppb) * J + j;
                    for (uint32_t c = 0; c < cols; ++c) {
                        const scale_entry & e = scales[slot][c];
                        if (!e.valid || e.generation != gen) {
                            scale_miss = true;
                            continue;
                        }
                        for (uint32_t r = 0; r < rows; ++r) {
                            const int8_t * ar = &sp[(a_base + ((size_t) i * K + k) * D + r) * D];
                            int32_t partial = 0;
                            for (uint32_t kk = 0; kk < k_valid; ++kk) {
                                partial += (int32_t) ar[kk] * (int32_t) w[kk][c];
                            }
                            const int32_t v   = emulator::scu(partial, e.carrier);
                            int32_t &     out = acc[(acc_base + ((size_t) i * J + j) * D + r) * D + c];
                            out               = (!accumulate && k == 0) ? v : emulator::sat_add(out, v);
                        }
                    }
                }
            }
        }
        if (residual) {
            residual_loops++;
        } else {
            main_loops++;
        }
        return POTAL_OK;
    }

    potal_status mvout(uint64_t rs1, uint64_t rs2) {
        const uint32_t rows = (rs2 >> 48) & 0xFFFF, cols = (rs2 >> 32) & 0xFFFF;
        const uint32_t addr = (uint32_t) rs2;
        if ((addr >> 31) == 0 || ((addr >> 29) & 1) == 0 || rows == 0 || rows > D || cols == 0 || cols > D) {
            err("MVOUT address / shape");
            return POTAL_HW_BAD_COMMAND;
        }
        const size_t row0 = addr & ((1u << 29) - 1);
        if (row0 + rows > acc_rows) {
            err("MVOUT beyond the ACC");
            return POTAL_HW_BAD_COMMAND;
        }
        int32_t * c = reinterpret_cast<int32_t *>((uintptr_t) rs1);
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(reinterpret_cast<uint8_t *>(c) + (size_t) r * st_stride, &acc[(row0 + r) * D], cols * sizeof(int32_t));
        }
        mvouts++;
        return POTAL_OK;
    }
};

} // namespace potal_test
