#ifndef POTAL_E4B2_EMULATOR_HPP
#define POTAL_E4B2_EMULATOR_HPP

#include "potal.h"
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace e4b2 {

/* Software-only command interpreter. DMA/compute effects retire at a modeled
 * global drain, not at command acknowledgement. It never issues an instruction.
 */
struct machine {
    struct index {
        uint32_t base = 0, j = 0, k = 0, active = 0, valid = 0;
        std::array<uint32_t, 32> source{};
    };
    struct scale {
        uint32_t carrier = 0, generation = 0;
        bool live = false;
    };
    struct loop {
        uint64_t a = 0, b = 0, c = 0, d = 0, lda = 0, ldb = 0, ldc = 0;
        uint32_t i = 0, j = 0, k = 0, pi = 0, pj = 0, pk = 0, mask = 0;
    } staged;
    struct resident_view {
        uint32_t base = 0, pitch = 0, origin = 0, acc = 0, pair = 0, bank = 0;
        bool indexed = false, valid = false;
    } view;
    std::vector<int8_t> sp = std::vector<int8_t>(16384 * 16);
    std::vector<int32_t> acc = std::vector<int32_t>(1024 * 16, INT32_C(0x12345678));
    std::array<index, 2> indices{};
    std::array<std::array<scale, 16>, 256> scales{};
    std::array<uint64_t, 3> load_stride{};
    uint64_t store_stride = 0;
    uint32_t selected_index = 0;
    std::array<uint32_t, 2> readers{};
    std::vector<std::function<void()>> work;
    std::vector<potal_paired_cmd_record> records;
    std::function<void()> on_command;
    bool supported = true;
    uint32_t fault_funct = UINT32_MAX, fault_occurrence = 0;
    potal_status fault_status = POTAL_ERR_BAD_RESPONSE;
    bool fault_after_effect = true;
    uint64_t drains = 0, issued_loops = 0, b_loads = 0;
    std::vector<uint32_t> preload_rows;

    static int32_t limit(int64_t value) {
        if (value > INT32_MAX) return INT32_MAX;
        if (value < INT32_MIN) return INT32_MIN;
        return static_cast<int32_t>(value);
    }
    static int32_t shifted(int64_t dot, uint32_t carrier) {
        if (carrier == UINT32_C(0x80000000) || dot == 0) return 0;
        for (uint32_t n = 0; n < carrier; ++n) {
            dot *= 2;
            if (dot > INT32_MAX) return INT32_MAX;
            if (dot < INT32_MIN) return INT32_MIN;
        }
        return static_cast<int32_t>(dot);
    }
    void drain() {
        for (auto &effect : work) effect();
        work.clear();
        readers = {};
        ++drains;
    }
    potal_status counter(uint64_t rs1, uint64_t rs2, uint64_t *response) {
        records.push_back({126, 1, rs1, rs2});
        *response = supported && rs1 == UINT64_C(0x504f544100000000) && rs2 == 0 ?
            UINT64_C(0x504f544102100007) : 0;
        return POTAL_OK;
    }
    potal_status command(const potal_paired_cmd_record &record) {
        records.push_back(record);
        if (on_command) {
            auto inspect = std::move(on_command);
            on_command = {};
            inspect();
        }
        const bool fault = record.funct == fault_funct &&
            fault_occurrence && --fault_occurrence == 0;
        if (fault && !fault_after_effect) return fault_status;
        const auto status = apply(record);
        return fault ? fault_status : status;
    }
    potal_status apply(const potal_paired_cmd_record &r) {
        const uint64_t a = r.rs1, b = r.rs2;
        switch (r.funct) {
            case 0:
                if ((a & 3) == 1) load_stride[(a >> 3) & 3] = b;
                if ((a & 3) == 2) store_stride = uint32_t(b);
                return POTAL_OK;
            case 7:
                view = {};
                staged.mask = 0;
                return POTAL_OK;
            case 9:
                staged.i = uint16_t(b); staged.j = uint16_t(b >> 16); staged.k = uint16_t(b >> 32);
                staged.pi = uint16_t(a); staged.pj = uint16_t(a >> 16); staged.pk = uint16_t(a >> 32);
                staged.mask |= 1;
                return POTAL_OK;
            case 10:
                staged.a = a; staged.b = b; staged.mask |= 2;
                return POTAL_OK;
            case 11:
                staged.d = a; staged.c = b; staged.mask |= 4;
                return POTAL_OK;
            case 12:
                staged.lda = a; staged.ldb = b; staged.mask |= 8;
                return POTAL_OK;
            case 13:
                staged.ldc = b; staged.mask |= 16;
                return POTAL_OK;
            case 28: {
                view = {};
                if (!r.xd || (b >> 8)) return POTAL_HW_INVALID_ENCODING;
                if (a == 0 && b == 0) return POTAL_OK;
                if (!(b & 1)) return POTAL_HW_INVALID_ENCODING;
                if (((b >> 4) & 15) != 1) return POTAL_HW_BAD_COMMAND;
                resident_view v;
                v.base = uint16_t(a); v.pitch = uint16_t(a >> 16);
                v.origin = uint16_t(a >> 32); v.acc = uint16_t(a >> 48);
                v.pair = (b >> 1) & 1; v.bank = (b >> 3) & 1; v.indexed = (b >> 2) & 1;
                if (!v.indexed && v.origin) return POTAL_HW_INVALID_ENCODING;
                const uint32_t begin = v.pair * 8192 + 4096;
                if (v.base < begin || v.base >= begin + 4096 || v.base % 16 ||
                    !v.pitch || v.pitch % 16 || v.pitch > 144 ||
                    v.origin >= 9 || v.acc % 16 || v.acc >= 512)
                    return POTAL_HW_INVALID_ADDRESS;
                v.valid = true;
                view = v;
                return POTAL_OK;
            }
            case 23: {
                const uint32_t slot = a & 255, generation = (a >> 8) & 255, column = (a >> 16) & 63;
                if ((a >> 22) || (b >> 32) || column >= 16) return POTAL_HW_INVALID_ENCODING;
                if (!(b <= 32767 || b == UINT64_C(0x80000000))) return POTAL_HW_INVALID_SCALE;
                auto &s = scales[slot][column];
                if (s.live) return POTAL_HW_SCALE_CONFLICT;
                s = {uint32_t(b), generation, true};
                return POTAL_OK;
            }
            case 24: {
                const uint32_t slot = a & 255, generation = (a >> 8) & 255, column = (a >> 16) & 63;
                if ((a >> 22) || b || column >= 16) return POTAL_HW_INVALID_ENCODING;
                auto &s = scales[slot][column];
                if (!s.live || s.generation != generation) return POTAL_HW_RELEASE_MISS;
                drain();
                s.live = false;
                return POTAL_OK;
            }
            case 25:
                if (a || b || !r.xd) return POTAL_HW_INVALID_ENCODING;
                drain();
                return POTAL_OK;
            case 27: {
                const uint32_t base = uint32_t(a), pair = base >= 8192;
                if (readers[pair]) return POTAL_ERR_BUSY;
                index next;
                next.base = base; next.j = a >> 32; next.k = uint32_t(b); next.active = b >> 32;
                if (!next.j || !next.k || !next.active || next.active > 32 ||
                    base < pair * 8192 + 4096 ||
                    uint64_t(base) + ((next.k + 15) / 16) * uint64_t(next.j) * 16 > (pair + 1) * 8192)
                    return POTAL_HW_INVALID_ADDRESS;
                selected_index = pair;
                indices[pair] = next;
                return POTAL_OK;
            }
            case 26: {
                auto &ix = indices[selected_index];
                if (readers[selected_index]) return POTAL_ERR_BUSY;
                if (a >= ix.active || a >= 32 || b >= ix.k) return POTAL_HW_INVALID_ADDRESS;
                ix.source[a] = uint32_t(b);
                ix.valid |= uint32_t(1) << a;
                return POTAL_OK;
            }
            case 2: {
                const uint32_t rows = b >> 48, cols = uint16_t(b >> 32), address = uint32_t(b);
                const uint64_t stride = load_stride[0];
                if (address & UINT32_C(0x80000000)) return POTAL_ERR_UNSUPPORTED;
                if (!a || !rows || rows > 16 || !cols || cols > 16 ||
                    uint64_t(address) + rows > 16384) return POTAL_HW_INVALID_ADDRESS;
                if ((address / 4096) & 1) ++b_loads;
                work.push_back([this, a, rows, cols, address, stride] {
                    const auto *source = reinterpret_cast<const int8_t *>(uintptr_t(a));
                    for (uint32_t i = 0; i < rows; ++i)
                        for (uint32_t j = 0; j < 16; ++j)
                            sp[(address + i) * 16 + j] = j < cols ? source[i * stride + j] : 0;
                });
                return POTAL_OK;
            }
            case 3: {
                const uint32_t rows = b >> 48, cols = uint16_t(b >> 32), row = uint32_t(b) & 1023;
                const uint64_t stride = store_stride / sizeof(int32_t);
                if (!a || !rows || rows > 16 || !cols || cols > 16 || row + rows > 1024 ||
                    !(b & UINT64_C(0x80000000))) return POTAL_HW_INVALID_ADDRESS;
                work.push_back([this, a, rows, cols, row, stride] {
                    auto *destination = reinterpret_cast<int32_t *>(uintptr_t(a));
                    for (uint32_t i = 0; i < rows; ++i)
                        for (uint32_t j = 0; j < cols; ++j)
                            destination[i * stride + j] = acc[(row + i) * 16 + j];
                });
                return POTAL_OK;
            }
            case 8:
                return launch(r);
            default:
                return POTAL_HW_BAD_COMMAND;
        }
    }
    potal_status launch(const potal_paired_cmd_record &r) {
        const auto l = staged;
        const auto v = view;
        staged.mask = 0;
        view.valid = false;
        if (!(r.rs1 & 8) || !r.xd) return POTAL_ERR_UNSUPPORTED;
        const uint32_t pair_a = (r.rs1 >> 18) & 3, pair_b = (r.rs1 >> 16) & 3;
        const uint32_t scale_base = (r.rs1 >> 21) & 65535, generation = (r.rs1 >> 37) & 255;
        const bool indexed = r.rs1 >> 63, accumulate = r.rs1 & 1;
        const uint32_t path = (r.rs1 >> 62) & 1, a_offset = (r.rs2 >> 3) & UINT32_C(0x1fffffff);
        if (!v.valid || l.mask != 31 || pair_a != v.pair + 1 || pair_b != v.pair + 1 ||
            (r.rs1 & UINT64_C(0xfff4)) || (r.rs2 >> 32) ||
            !(r.rs1 & (UINT64_C(1) << 20)) || !(r.rs1 & (UINT64_C(1) << 61)) ||
            ((r.rs1 >> 45) & 65535) != 1 || (r.rs2 & 7) ||
            v.indexed != indexed || v.bank != path || (path && !indexed) ||
            l.i != 1 || !l.k || l.k > 28 || !l.j || l.j > 9 ||
            l.pi >= 16 || l.pj >= 16 || l.pk >= 16 || !l.a || l.b || l.c || l.d)
            return POTAL_HW_INVALID_ENCODING;
        if (scale_base + l.k * l.j > 256 || v.acc + l.j * 16 > 512 ||
            a_offset + l.k > 256 || v.pitch < l.j * 16)
            return POTAL_HW_INVALID_ADDRESS;
        const auto ix = indices[v.pair];
        if (indexed) {
            const uint32_t active = l.k * 16 - l.pk;
            const uint32_t mask = ix.active == 32 ? UINT32_MAX :
                ix.active ? (uint32_t(1) << ix.active) - 1 : 0;
            if (ix.active != active || !mask || ix.valid != mask ||
                ix.base != v.base || ix.j * 16 != v.pitch || v.origin + l.j > ix.j)
                return POTAL_HW_INVALID_ADDRESS;
        } else if (uint64_t(v.base) + (l.k - 1) * v.pitch + l.j * 16 > (v.pair + 1) * 8192)
            return POTAL_HW_INVALID_ADDRESS;
        for (uint32_t p = 0; p < l.k; ++p)
            for (uint32_t j = 0; j < l.j; ++j)
                for (uint32_t c = 0; c < 16 - (j + 1 == l.j ? l.pj : 0); ++c) {
                    const auto &s = scales[scale_base + p * l.j + j][c];
                    if (!s.live || s.generation != generation) return POTAL_HW_INVALID_SCALE;
                }
        if (indexed) ++readers[v.pair];
        ++issued_loops;
        work.push_back([this, l, v, ix, a_offset, scale_base, accumulate, indexed] {
            const auto *source_a = reinterpret_cast<const int8_t *>(uintptr_t(l.a));
            for (uint32_t p = 0; p < l.k; ++p)
                for (uint32_t i = 0; i < 16 - l.pi; ++i)
                    for (uint32_t x = 0; x < 16; ++x)
                        sp[(v.pair * 8192 + (a_offset + p) * 16 + i) * 16 + x] =
                            x < 16 - (p + 1 == l.k ? l.pk : 0) ?
                            source_a[i * l.lda + p * 16 + x] : 0;
            for (uint32_t p = 0; p < l.k; ++p)
                for (uint32_t j = 0; j < l.j; ++j) {
                    preload_rows.push_back(v.base + p * v.pitch + j * 16);
                    for (uint32_t i = 0; i < 16 - l.pi; ++i)
                        for (uint32_t c = 0; c < 16 - (j + 1 == l.j ? l.pj : 0); ++c) {
                            int64_t dot = 0;
                            for (uint32_t x = 0; x < 16 - (p + 1 == l.k ? l.pk : 0); ++x) {
                                const uint32_t k = indexed ? ix.source[p * 16 + x] : p * 16 + x;
                                const uint32_t b_row = indexed ?
                                    ix.base + (k / 16 * ix.j + v.origin + j) * 16 + k % 16 :
                                    v.base + p * v.pitch + j * 16 + x;
                                dot += int64_t(sp[(v.pair * 8192 + (a_offset + p) * 16 + i) * 16 + x]) *
                                    sp[b_row * 16 + c];
                            }
                            const auto value = shifted(dot, scales[scale_base + p * l.j + j][c].carrier);
                            auto &target = acc[(v.bank * 512 + v.acc + j * 16 + i) * 16 + c];
                            target = !accumulate && p == 0 ? value : limit(int64_t(target) + value);
                        }
                }
        });
        return POTAL_OK;
    }
};

} // namespace e4b2
#endif
