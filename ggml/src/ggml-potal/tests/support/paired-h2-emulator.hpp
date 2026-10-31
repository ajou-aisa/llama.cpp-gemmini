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
    std::vector<uint64_t> sp_written = std::vector<uint64_t>(sp_rows, 0);
    std::vector<int32_t>  acc = std::vector<int32_t>(acc_rows * D, 0);
    scale_entry           scales[POTAL_SCALE_ENTRIES][D];
    index_table           tables[2];
    uint32_t              selected_table = 0;
    uint64_t              ld_stride[3] = { 0, 0, 0 }, st_stride = 0;
    uint64_t              ld_config[3] = { 0, 0, 0 };
    uint64_t              bounds_rs1 = 0, bounds_rs2 = 0, a_dram = 0, b_dram = 0, d_dram = 0, c_dram = 0,
                          a_stride = 0, b_stride = 0, d_stride = 0, c_stride_elems = 0;
    // Counters and records.
    uint64_t    main_loops = 0, residual_loops = 0, index_configs = 0, index_writes = 0, scale_writes = 0,
                scale_releases = 0, mvouts = 0, fences = 0, flushes = 0, b_loads = 0;
    uint64_t    raw_b_mvins = 0, raw_b_bytes = 0, resident_main_loops = 0;
    uint64_t    ldb_only_loads = 0, ldb_only_bytes = 0;
    uint64_t    ldb_only_capability = D == 32 ? POTAL_LDB_ONLY_CAPABILITY : 0;
    uint64_t    stream_capability = D == 32 ? POTAL_STREAM_CAPABILITY : 0;
    uint64_t    stream_early_capability = D == 32 ? POTAL_STREAM_EARLY_CAPABILITY : 0;
    bool        stream_active = false, stream_continuous = false, stream_early = false;
    uint64_t    stream_waits = 0, stream_releases = 0, load3_loops = 0;
    struct borrowed_matrix {
        const int8_t *source;
        size_t rows, cols, stride;
        std::vector<int8_t> bytes;
    };
    struct stream_pair {
        uint64_t incarnation = 0, submitted = 0, completed = 0;
        uint64_t b_source = 0;
        bool scheduled = false, dma_done = false;
        bool producer = false;
        uint32_t b_k = 0, b_j = 0, b_pad_j = 0;
        uint64_t b_pitch = 0;
        std::vector<borrowed_matrix> borrowed;
    } stream_pairs[2];
    std::vector<borrowed_matrix> completed_sources;
    uint8_t scale_pairs[POTAL_SCALE_ENTRIES] = {};
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

    static potal_status on_counter(void * ctx, const potal_paired_cmd_record * rec, uint64_t * value) {
        auto & e = *static_cast<h2_emulator *>(ctx);
        e.records.push_back(*rec);
        if (rec->funct != POTAL_FUNCT_COUNTER || rec->xd != 1 || rec->rs2 ||
            (rec->rs1 != UINT64_C(0xA) && rec->rs1 != POTAL_LDB_ONLY_TAG &&
             rec->rs1 != POTAL_STREAM_TAG && rec->rs1 != POTAL_STREAM_EARLY_TAG)) {
            e.err("capability query");
            return POTAL_HW_BAD_COMMAND;
        }
        *value = rec->rs1 == POTAL_STREAM_EARLY_TAG ? e.stream_early_capability :
                 rec->rs1 == POTAL_STREAM_TAG ? e.stream_capability :
                 rec->rs1 == POTAL_LDB_ONLY_TAG ? e.ldb_only_capability : 0;
        return POTAL_OK;
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
            case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC: d_stride = r.rs1; c_stride_elems = r.rs2; return POTAL_OK;
            case POTAL_FUNCT_LOOP_WS:
                if (stream_active && r.xd != 1) {
                    err("stream LOOP_WS requires checked admission response");
                    return POTAL_HW_INVALID_ENCODING;
                }
                return loop_ws(r.rs1, r.rs2);
            case POTAL_FUNCT_STREAM: return stream_control(r);
            case POTAL_FUNCT_LDB_ONLY: return ldb_only(r);
            case POTAL_FUNCT_MVIN: return mvin_b(r.rs1, r.rs2);
            case POTAL_FUNCT_MVOUT: return mvout(r.rs1, r.rs2);
            case POTAL_FUNCT_SCALE_WRITE: return scale_write(r.rs1, r.rs2);
            case POTAL_FUNCT_SCALE_RELEASE: {
                if (stream_active) { complete_pair(0); complete_pair(1); }
                const auto status = scale_release(r.rs1);
                if (status == POTAL_OK && stream_active) {
                    const size_t slot = r.rs1 & 255;
                    bool live = false;
                    for (const auto & entry : scales[slot]) live = live || entry.valid;
                    if (!live) scale_pairs[slot] = 0;
                }
                return status;
            }
            case POTAL_FUNCT_INDEX_CONFIG: return index_config(r.rs1, r.rs2);
            case POTAL_FUNCT_INDEX_WRITE: return index_write(r.rs1, r.rs2);
            case POTAL_FUNCT_FENCE:
                if (stream_active) { complete_pair(0); complete_pair(1); }
                fences++; return POTAL_OK;
            case POTAL_FUNCT_FLUSH: flushes++; return POTAL_OK;
            default: err("unknown funct " + std::to_string(r.funct)); return POTAL_HW_BAD_COMMAND;
        }
    }

    // Arithmetic is synchronous; completion observations are deliberately not.
    // These leases prove command/source lifecycle only, never physical overlap.
    void borrow(uint32_t pair, uint64_t address, size_t rows, size_t cols, size_t stride) {
        borrowed_matrix m{reinterpret_cast<const int8_t *>(uintptr_t(address)), rows, cols, stride, {}};
        m.bytes.reserve(rows * cols);
        for (size_t r = 0; r < rows; ++r)
            m.bytes.insert(m.bytes.end(), m.source + r * stride, m.source + r * stride + cols);
        stream_pairs[pair].borrowed.push_back(std::move(m));
    }

    void complete_pair(uint32_t pair) {
        auto & p = stream_pairs[pair];
        for (const auto & m : p.borrowed)
            for (size_t r = 0; r < m.rows; ++r)
                if (std::memcmp(m.source + r * m.stride, m.bytes.data() + r * m.cols, m.cols))
                    err("immutable stream source changed before completion");
        for (auto & m : p.borrowed) completed_sources.push_back(std::move(m));
        p.borrowed.clear();
        p.completed = p.submitted;
        p.dma_done = p.scheduled || p.producer;
    }

    potal_status stream_control(const potal_paired_cmd_record & r) {
        const uint32_t op = r.rs1 & 15, pair = (r.rs1 >> 4) & 1;
        if (D != 32 || r.xd != 1 || op < 1 || op > 6 ||
            (r.rs1 >> (op == 4 ? 5 : 4)) || (op == 5 ? r.rs2 >> 21 : r.rs2))
            return POTAL_HW_INVALID_ENCODING;
        if (op <= 2 || op == 6) {
            if (stream_active) return POTAL_HW_BAD_COMMAND;
            for (auto & p : stream_pairs) p = stream_pair{};
            stream_active = true;
            stream_continuous = op != 1;
            stream_early = op == 6;
            return POTAL_OK;
        }
        if (!stream_active) return POTAL_HW_BAD_COMMAND;
        if (op == 4) {
            complete_pair(pair);
            ++stream_waits;
            return POTAL_OK;
        }
        if (op == 5) {
            const uint32_t slot = r.rs2 & 255;
            const uint8_t owners = scale_pairs[slot];
            for (uint32_t p = 0; p < 2; ++p)
                if ((owners & (1u << p)) && stream_pairs[p].submitted != stream_pairs[p].completed) {
                    err("local release before observed pair completion");
                    return POTAL_HW_BAD_COMMAND;
                }
            const auto status = scale_release(r.rs2);
            if (status == POTAL_OK) {
                --scale_releases; // local, not the legacy globally draining opcode
                ++stream_releases;
                bool live = false;
                for (const auto & entry : scales[slot]) live = live || entry.valid;
                if (!live) scale_pairs[slot] = 0;
            }
            return status;
        }
        for (const auto & p : stream_pairs)
            if (p.submitted != p.completed || (p.producer && !p.dma_done)) return POTAL_HW_BAD_COMMAND;
        for (const auto & slot : scales)
            for (const auto & entry : slot)
                if (entry.valid) return POTAL_HW_BAD_COMMAND;
        for (const auto & m : completed_sources)
            for (size_t row = 0; row < m.rows; ++row)
                if (std::memcmp(m.source + row * m.stride, m.bytes.data() + row * m.cols, m.cols))
                    err("immutable stream source changed before stream end");
        completed_sources.clear();
        stream_active = false;
        stream_continuous = stream_early = false;
        return POTAL_OK;
    }

    potal_status config(uint64_t rs1, uint64_t rs2) {
        switch (rs1 & 3) {
            case 1: { // CONFIG_LD: state id at [4:3], row bytes in rs2
                const size_t state = (rs1 >> 3) & 3;
                if (state >= 3) return POTAL_HW_BAD_COMMAND;
                if (stream_active && (stream_pairs[0].submitted || stream_pairs[1].submitted)) {
                    err("stream changes static load configuration");
                    return POTAL_HW_BAD_COMMAND;
                }
                ld_stride[state] = rs2;
                ld_config[state] = rs1;
                break;
            }
            case 2: st_stride = rs2; break;                 // CONFIG_ST: row bytes
            default: break;                                 // CONFIG_EX
        }
        return POTAL_OK;
    }

    // Independent LOAD2-only mapping. Admission is synchronous in this software
    // double; neither copied bytes nor its response are a completion certificate.
    // No A, ACC, Execute, Scale or Index operation participates.
    potal_status ldb_only(const potal_paired_cmd_record & command) {
        const uint32_t pair = command.rs1 & 1;
        const uint32_t I = bounds_rs2 & 0xFFFF, J = (bounds_rs2 >> 16) & 0xFFFF;
        const uint32_t K = (bounds_rs2 >> 32) & 0xFFFF;
        const uint32_t pad_j = (bounds_rs1 >> 16) & 0xFFFF;
        const uint64_t rows = uint64_t(K) * D, columns = uint64_t(J) * D;
        const uint64_t footprint = rows * J;
        const uint64_t canonical_ld = (uint64_t(D) << 16) | UINT64_C(0x109);
        if (D != 32 || command.xd != 1 || command.rs1 != (UINT64_C(0x10) | pair) || command.rs2 ||
            I != 1 || !J || !K || bounds_rs2 >> 48 || bounds_rs1 != uint64_t(pad_j) << 16 ||
            pad_j >= D || a_dram || d_dram || c_dram || a_stride || d_stride || c_stride_elems ||
            !b_dram || b_dram % D || b_stride < columns - pad_j ||
            footprint > POTAL_HW_BANK_ROWS || ld_config[1] != canonical_ld || ld_stride[1] != b_stride) {
            err("LdB-only fields, LOAD2 configuration or bank extent");
            return POTAL_HW_INVALID_ENCODING;
        }
        if (stream_active) {
            auto & p = stream_pairs[pair];
            const auto & current = stream_pairs[1 - pair];
            if (!stream_early || p.producer || p.submitted != p.completed ||
                !current.scheduled || current.submitted != 1 || current.completed != 0) {
                err("stream B-only requires free target and opposite pending Main");
                return POTAL_HW_BAD_COMMAND;
            }
            for (const auto owners : scale_pairs)
                if (owners & (1u << pair)) return POTAL_HW_BAD_COMMAND;
            p.submitted = p.completed = 0;
            p.scheduled = p.dma_done = false;
            p.producer = true;
            p.b_source = b_dram;
            p.b_k = K;
            p.b_j = J;
            p.b_pad_j = pad_j;
            p.b_pitch = b_stride;
            borrow(pair, b_dram, rows, columns - pad_j, b_stride);
        }
        const size_t bank_end = (2 * size_t(pair) + 2) * POTAL_HW_BANK_ROWS;
        const size_t start = bank_end - size_t(footprint);
        const auto * source = reinterpret_cast<const int8_t *>(uintptr_t(b_dram));
        for (size_t k = 0; k < rows; ++k) {
            for (size_t j = 0; j < columns; ++j) {
                const size_t local_row = start + (k / D * J + j / D) * D + k % D;
                sp[local_row * D + j % D] = j < columns - pad_j ? source[k * b_stride + j] : 0;
                sp_written[local_row] |= uint64_t(1) << (j % D);
            }
        }
        ++ldb_only_loads;
        ldb_only_bytes += rows * (columns - pad_j); // logical bytes, not measured DMA traffic
        return POTAL_OK;
    }

    // Ordinary state-0 MVIN of one B tile. This copies real bytes into the
    // explicit SP model; it models no DMA timing or completion certificate.
    potal_status mvin_b(uint64_t source, uint64_t shape) {
        const uint32_t row = uint32_t(shape);
        const uint32_t cols = (shape >> 32) & 0xFFFF;
        const uint32_t rows = shape >> 48;
        const uint64_t expected = (uint64_t(D) << 16) | (uint64_t(1) << 8) | 1;
        if (!source || source % D || rows == 0 || rows > D || cols == 0 || cols > D ||
            row >= sp_rows || rows > sp_rows - row || row % D ||
            row / POTAL_HW_BANK_ROWS % 2 != 1 ||
            (row + rows - 1) / POTAL_HW_BANK_ROWS != row / POTAL_HW_BANK_ROWS ||
            ld_config[0] != expected || ld_stride[0] < cols) {
            err("raw B MVIN fields or layout");
            return POTAL_HW_INVALID_ADDRESS;
        }
        const auto * src = reinterpret_cast<const int8_t *>(uintptr_t(source));
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                sp[(size_t(row) + r) * D + c] = src[r * ld_stride[0] + c];
                sp_written[row + r] |= uint64_t(1) << c;
            }
        }
        ++raw_b_mvins;
        raw_b_bytes += uint64_t(rows) * cols;
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
        if (stream_active) {
            const auto & p = stream_pairs[buffer];
            if (!p.scheduled || (p.submitted > 1 && p.completed != p.submitted)) {
                err("Index rewrite before previous Residual completed");
                return POTAL_HW_BAD_COMMAND;
            }
        }
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
        const uint32_t pair      = a_spad - 1;
        if (pair >= 2) return POTAL_HW_BAD_COMMAND;
        if (stream_active) {
            const size_t a_state = residual ? 2 : 0;
            const uint64_t canonical = (uint64_t(D) << 16) | UINT64_C(0x101);
            auto & p = stream_pairs[pair];
            if (ld_config[a_state] != (canonical | (a_state << 3)) ||
                ld_stride[a_state] != a_stride || !a_dram ||
                (!residual && (ld_config[1] != (canonical | 8) || ld_stride[1] != b_stride ||
                               (!b_dram && !p.producer)))) {
                err("stream static LOAD/LOAD2/LOAD3 state differs from descriptor");
                return POTAL_HW_INVALID_ENCODING;
            }
            if (!residual) {
                if (p.producer && (!stream_early || !p.dma_done || b_dram ||
                    p.b_k != K || p.b_j != J || p.b_pad_j != pad_j || p.b_pitch != b_stride)) {
                    err("prefetched Main requires completed equivalent B without reload");
                    return POTAL_HW_BAD_COMMAND;
                }
                if (p.completed != p.submitted) {
                    err("pair reused before observed completion");
                    return POTAL_HW_BAD_COMMAND;
                }
                for (const auto owners : scale_pairs) if (owners & (1u << pair)) {
                    err("pair reused with live generation lease");
                    return POTAL_HW_BAD_COMMAND;
                }
                ++p.incarnation;
                p.submitted = p.completed = 0;
                if (!p.producer) p.b_source = b_dram;
                p.scheduled = true;
                p.dma_done = false;
                if (!p.producer) borrow(pair, b_dram, K * D - pad_k, J * D - pad_j, ld_stride[1]);
                p.producer = false;
            } else if (!p.scheduled || (p.submitted > 1 && p.completed != p.submitted)) {
                err("Residual missing producer or reusing pending A/Index window");
                return POTAL_HW_BAD_COMMAND;
            }
            const uint32_t slots = K / ppb * J;
            if (base + slots > POTAL_SCALE_ENTRIES) return POTAL_HW_INVALID_ADDRESS;
            for (uint32_t slot = base; slot < base + slots; ++slot) {
                if (scale_pairs[slot] & ~(1u << pair)) {
                    err("Scale allocation leased by another pair");
                    return POTAL_HW_BAD_COMMAND;
                }
                scale_pairs[slot] |= 1u << pair;
            }
            borrow(pair, a_dram, I * D - pad_i, K * D - pad_k, ld_stride[a_state]);
            ++p.submitted;
            if (residual) ++load3_loops;
        }
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
        }
        if (!indexed && b_dram != 0) {
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
                        sp_written[b_start + ((size_t) k * J + j) * D + kk] = ~uint64_t(0) >> (64 - D);
                    }
                }
            }
            b_loads++;
        } else if (!indexed) {
            // Null-B Main is legal only when every used physical B byte exists.
            // The C11 state-machine tests separately prove generation ownership.
            for (uint32_t k = 0; k < K; ++k) {
                for (uint32_t j = 0; j < J; ++j) {
                    const uint32_t cols = D - (j + 1 == J ? pad_j : 0);
                    const uint64_t mask = ~uint64_t(0) >> (64 - cols);
                    for (uint32_t r = 0; r < D - (k + 1 == K ? pad_k : 0); ++r) {
                        if ((sp_written[b_start + ((size_t) k * J + j) * D + r] & mask) != mask) {
                            err("resident Main reads unwritten B");
                            return POTAL_HW_INVALID_ADDRESS;
                        }
                    }
                }
            }
            ++resident_main_loops;
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
