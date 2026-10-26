// Test-only ACC value contexts. No LOOP arithmetic, SCU replay or inference.
// Default (also on RISC-V): software-only command/data emulator.
// --hardware: RISC-V Linux issuer_owner, no sink and no software fallback.
//
// Link potal-e3-context.c INSTEAD OF potal.c / potal-api. See the handoff:
// issuer_owner also needs the backend-source dependencies, with their API
// dependency replaced, not a second copy of the production API.
// Unknown DMA buffers below have static process lifetime. A failed fence is
// never retried. Host ownership poison runs LAST; no injection on hardware.

#include "potal-e3-context.h"
#include "potal_hw_params.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#if defined(__riscv) && defined(__linux__)
#include "matmul/execution.hpp"
#include <sys/mman.h>
#endif

namespace {
constexpr size_t D = POTAL_HW_DIM;
constexpr size_t stride = 3 * D; // larger than valid columns, tile aligned
constexpr size_t elements = 2 * D * stride;
constexpr uint32_t acc_bit = UINT32_C(1) << 31;
constexpr uint32_t add_bit = UINT32_C(1) << 30;
constexpr uint32_t full_bit = UINT32_C(1) << 29;
constexpr int32_t guard = 0x13579bdf;
static_assert(D == 16, "This bounded E3 prototype targets audited DIM16 only");
static_assert(4 * D <= POTAL_HW_ACC_ROWS / 2, "Two by two tiles must fit");

struct dma_arena {
    alignas(64) std::array<int32_t, elements> source[2];
    alignas(64) std::array<int32_t, elements> saved[2];
    alignas(64) std::array<int32_t, elements> output[2];
    alignas(64) std::array<int32_t, elements> archive[2];
    alignas(64) std::array<int32_t, elements> archive_expected[2];
    alignas(64) std::array<int32_t, elements> clobber;
    alignas(64) std::array<int32_t, 2 * D> bias;
};
// Never free/resize this storage, including after an unknown fence.
static dma_arena buffers;

struct failure : std::exception {
    const char *message;
    explicit failure(const char *m) : message(m) {}
    const char *what() const noexcept override { return message; }
};
void check(bool ok, const char *message) {
    if (!ok) throw failure(message); // explicit, independent of NDEBUG
}
void status(potal_status actual, potal_status wanted, const char *message) {
    if (actual != wanted) {
        std::fprintf(stderr, "%s: got %s, expected %s\n", message,
                     potal_status_name(actual), potal_status_name(wanted));
        throw failure(message);
    }
}
int32_t sat_add(int32_t a, int32_t b) {
    const int64_t sum = int64_t(a) + int64_t(b);
    return sum > INT32_MAX ? INT32_MAX : sum < INT32_MIN ? INT32_MIN : int32_t(sum);
}

// A deliberately small command model: full signed32 MVIN/MVOUT plus fence.
// It does NOT claim RTL concurrency, DMA bus traffic or hardware timing.
struct emulator {
    std::array<int32_t, POTAL_HW_ACC_ROWS * D> acc{};
    std::vector<potal_paired_cmd_record> expected;
    size_t cursor = 0;
    uint64_t load_stride = 0, store_stride = 0;
    uint64_t calls = 0, mvins = 0, mvouts = 0, fences = 0;
    uint64_t restore_bytes = 0, bias_bytes = 0, readback_bytes = 0;
    uint64_t fail_mvin = 0;
    bool fail_fence = false, bad = false;

    void push(uint32_t funct, uint32_t xd, uint64_t a, uint64_t b) {
        expected.push_back({funct, xd, a, b});
    }
    // Derive tile origins from logical row/column coordinates; do not call
    // production encoders or arithmetic for either command or data goldens.
    void matrix_commands(uint32_t bank, uint32_t pi, uint32_t pj,
                         const int32_t *p, bool output, bool bias) {
        push(POTAL_FUNCT_CONFIG, 0, output ? 2 : (uint64_t(D) << 16) | (1u << 8) | 1u,
             output ? stride * 4 : bias ? 0 : stride * 4);
        for (size_t y = 0; y < 2 * D - pi; y += D) {
            for (size_t x = 0; x < 2 * D - pj; x += D) {
                const uint64_t rows = std::min(D, 2 * D - pi - y);
                const uint64_t cols = std::min(D, 2 * D - pj - x);
                const uint32_t row = bank * (POTAL_HW_ACC_ROWS / 2) + (y / D * 2 + x / D) * D;
                const uint32_t local = acc_bit | (output ? full_bit : bias ? add_bit : 0) | row;
                const auto *origin = p + (bias ? x : y * stride + x);
                push(output ? POTAL_FUNCT_MVOUT : POTAL_FUNCT_MVIN, 0,
                     uint64_t(reinterpret_cast<uintptr_t>(origin)),
                     (rows << 48) | (cols << 32) | local);
            }
        }
    }
    void fence_command() { push(POTAL_FUNCT_FENCE, 1, 0, 0); }
    bool complete() const { return !bad && cursor == expected.size(); }

    static potal_status sink(void *ctx, const potal_paired_cmd_record *r) {
        return static_cast<emulator *>(ctx)->command(*r);
    }
    potal_status command(const potal_paired_cmd_record &r) {
        ++calls;
        if (cursor == expected.size()) { bad = true; return POTAL_ERR_BAD_RESPONSE; }
        const auto &e = expected[cursor++];
        if (e.funct != r.funct || e.xd != r.xd || e.rs1 != r.rs1 || e.rs2 != r.rs2) {
            bad = true;
            return POTAL_ERR_BAD_RESPONSE;
        }
        if (r.funct == POTAL_FUNCT_CONFIG) {
            if ((r.rs1 & 3) == 1) load_stride = r.rs2;
            else if (r.rs1 == 2) store_stride = r.rs2;
            else { bad = true; return POTAL_ERR_BAD_RESPONSE; }
            return POTAL_OK;
        }
        if (r.funct == POTAL_FUNCT_FENCE) {
            ++fences;
            return fail_fence ? POTAL_HW_BAD_COMMAND : POTAL_OK;
        }
        const bool load = r.funct == POTAL_FUNCT_MVIN;
        if (!load && r.funct != POTAL_FUNCT_MVOUT) {
            bad = true;
            return POTAL_ERR_BAD_RESPONSE;
        }
        const uint32_t local = uint32_t(r.rs2);
        const size_t row = local & ((UINT32_C(1) << 29) - 1);
        const size_t nr = r.rs2 >> 48, nc = (r.rs2 >> 32) & 0xffff;
        if (!(local & acc_bit) || nr == 0 || nr > D || nc == 0 || nc > D ||
            row + nr > POTAL_HW_ACC_ROWS ||
            (load ? bool(local & full_bit) : !(local & full_bit) || bool(local & add_bit))) {
            bad = true;
            return POTAL_ERR_BAD_RESPONSE;
        }
        if (load) ++mvins; else ++mvouts;
        const bool partial = load && fail_mvin == mvins;
        const uint64_t pitch = load ? load_stride : store_stride;
        for (size_t y = 0; y < (partial ? 1 : nr); ++y) {
            auto *dram = reinterpret_cast<int32_t *>(uintptr_t(r.rs1 + y * pitch));
            for (size_t x = 0; x < nc; ++x) {
                auto &v = acc[(row + y) * D + x];
                if (load) v = local & add_bit ? sat_add(v, dram[x]) : dram[x];
                else dram[x] = v;
            }
        }
        if (load && (local & add_bit)) bias_bytes += nr * nc * 4;
        else if (load) restore_bytes += nr * nc * 4;
        else readback_bytes += nr * nc * 4;
        return partial ? POTAL_ERR_BAD_RESPONSE : POTAL_OK;
    }
};

struct harness {
    bool hardware;
    emulator model;
    potal_paired_session *session = nullptr;
    uint64_t commands = 0, mvins = 0, mvouts = 0, fences = 0;
    uint64_t restore_bytes = 0, bias_bytes = 0, readback_bytes = 0;

    explicit harness(bool hw) : hardware(hw) {}
    void open() {
        potal_paired_sink sink{emulator::sink, &model};
        status(potal_paired_session_open(hardware ? nullptr : &sink, &session), POTAL_OK, "open");
    }
    void fence() {
        if (!hardware) model.fence_command();
        ++commands; ++fences;
        status(potal_paired_fence(session), POTAL_OK, "checked fence");
        check(potal_paired_session_query(session).outstanding_acc_updates == 0, "fence drains ACC updates");
    }
    potal_e3_acc_restore descriptor(uint32_t bank, uint32_t pi, uint32_t pj, const int32_t *p) {
        return {bank, 2, 2, pi, pj, p, stride};
    }
    void restore(uint32_t bank, uint32_t pi, uint32_t pj, const int32_t *p) {
        if (!hardware) model.matrix_commands(bank, pi, pj, p, false, false);
        const auto r = descriptor(bank, pi, pj, p);
        commands += 5; mvins += 4;
        restore_bytes += (2 * D - pi) * (2 * D - pj) * 4;
        status(potal_e3_acc_restore_issue(session, &r), POTAL_OK, "restore issue");
        check(potal_paired_session_query(session).outstanding_acc_updates == 4, "restore pending tiles");
    }
    void read(uint32_t bank, uint32_t pi, uint32_t pj, int32_t *p) {
        if (!hardware) model.matrix_commands(bank, pi, pj, p, true, false);
        potal_acc_readback r{bank, 2, 2, pi, pj, p, stride};
        commands += 5; mvouts += 4;
        readback_bytes += (2 * D - pi) * (2 * D - pj) * 4;
        status(potal_paired_acc_readback_issue(session, &r), POTAL_OK, "existing full readback");
        const uint64_t before = potal_paired_session_query(session).commands;
        const auto overwrite = descriptor(bank, pi, pj, buffers.source[bank].data());
        status(potal_e3_acc_restore_issue(session, &overwrite), POTAL_ERR_BUSY, "readback reader blocks overwrite");
        status(potal_paired_session_close(session), POTAL_ERR_BUSY, "pending readback blocks close");
        check(potal_paired_session_query(session).commands == before, "reader exclusion issues nothing");
        fence(); // CPU comparison only after successful completion
    }
    void bias(uint32_t bank, uint32_t pi, uint32_t pj) {
        if (!hardware) model.matrix_commands(bank, pi, pj, buffers.bias.data(), false, true);
        potal_acc_bias b{bank, 2, 2, pi, pj, buffers.bias.data()};
        commands += 5; mvins += 4;
        bias_bytes += (2 * D - pi) * (2 * D - pj) * 4;
        status(potal_paired_acc_bias_issue(session, &b), POTAL_OK, "checked Sat32 bias resume");
        fence();
    }
    void blocked(uint32_t bank, uint32_t pi, uint32_t pj) {
        const uint64_t before = potal_paired_session_query(session).commands;
        const auto r = descriptor(bank, pi, pj, buffers.source[bank].data());
        potal_acc_readback out{bank, 2, 2, pi, pj, buffers.output[bank].data(), stride};
        potal_paired_loop_ws loop{};
        potal_acc_bias b{bank, 2, 2, pi, pj, buffers.bias.data()};
        status(potal_e3_acc_restore_issue(session, &r), POTAL_ERR_BUSY, "pending restore refused");
        status(potal_paired_acc_readback_issue(session, &out), POTAL_ERR_BUSY, "pending read refused");
        status(potal_paired_loop_ws_issue(session, &loop), POTAL_ERR_BUSY, "pending LOOP refused");
        status(potal_paired_acc_bias_issue(session, &b), POTAL_ERR_BUSY, "pending bias refused");
        status(potal_paired_tlb_flush(session), POTAL_ERR_BUSY, "pending flush refused");
        status(potal_paired_session_close(session), POTAL_ERR_BUSY, "pending healthy close refused");
        check(potal_paired_session_query(session).commands == before, "busy checks issue nothing");
    }
    void refused() {
        const auto good = descriptor(0, 0, 0, buffers.source[0].data());
        const uint64_t before = potal_paired_session_query(session).commands;
        status(potal_e3_acc_restore_issue(nullptr, &good), POTAL_ERR_INVALID_ARGUMENT, "null session");
        status(potal_e3_acc_restore_issue(session, nullptr), POTAL_ERR_INVALID_ARGUMENT, "null descriptor");
        auto reject = [&](potal_e3_acc_restore r, potal_status wanted) {
            status(potal_e3_acc_restore_issue(session, &r), wanted, "descriptor preflight");
        };
        auto r = good; r.acc_bank = 2; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.i_tiles = 0; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.j_tiles = 65536; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.pad_i = D; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.pad_j = D; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.stride = 2 * D - 1; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.stride = UINT32_MAX / 4u + size_t(1); reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.matrix = nullptr; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.matrix += 1; reject(r, POTAL_ERR_INVALID_ARGUMENT);
        r = good; r.matrix = reinterpret_cast<const int32_t *>(UINTPTR_MAX & ~uintptr_t(63));
        reject(r, POTAL_ERR_INVALID_ARGUMENT); // overflow, never dereferenced
        r = good; r.i_tiles = 65535; r.j_tiles = 65535; r.stride = size_t(65535) * D;
        reject(r, POTAL_ERR_INVALID_ARGUMENT); // implied extent exceeds 32-bit DMA window
        r = good; r.i_tiles = POTAL_HW_ACC_ROWS / (2 * D) + 1; r.j_tiles = 1;
        reject(r, POTAL_ERR_CAPACITY);
        check(potal_paired_session_query(session).commands == before, "preflight issues zero commands");
        potal_paired_session *other = nullptr;
        status(potal_paired_session_open(nullptr, &other), POTAL_ERR_BUSY, "single session owner");
        check(other == nullptr, "busy open returns no session");
    }
    // Return fatal to issuer_owner when completion cannot be established.
    // All DMA storage, including the sink object, outlives this settlement.
    bool settle() {
        if (!session) return false;
        const auto q = potal_paired_session_query(session);
        potal_status drained = POTAL_OK;
        if (q.outstanding_loops || q.outstanding_readbacks || q.outstanding_acc_updates || q.live_scale_slots) {
            drained = potal_paired_session_cleanup(session); // never retries failed fence
        }
        const bool fatal = drained != POTAL_OK;
        if (fatal) return true; // no close while pending/unknown; retain session too
        const auto closed = potal_paired_session_close(session);
        if (closed == POTAL_ERR_BUSY) return true;
        session = nullptr;
        return closed != (q.failed ? q.error : POTAL_OK);
    }
};

void exact(const std::array<int32_t, elements> &a, const std::array<int32_t, elements> &b) {
    check(std::memcmp(a.data(), b.data(), sizeof(a)) == 0, "exact signed32 matrix and padding memcmp");
}
void prepare(uint32_t epoch, uint32_t pi, uint32_t pj) {
    const int32_t patterns[] = {INT32_MIN, INT32_MAX, -1, 0, 1, -65537, 0x12345678, -123456789};
    for (uint32_t bank = 0; bank < 2; ++bank) {
        buffers.source[bank].fill(guard);
        buffers.saved[bank].fill(guard);
        buffers.output[bank].fill(guard);
        for (size_t y = 0; y < 2 * D - pi; ++y) {
            for (size_t x = 0; x < 2 * D - pj; ++x) {
                // Each logical group/epoch and bank has a distinct context.
                const size_t tag = (y * 3 + x + bank * 5 + epoch * 7) % 8;
                buffers.source[bank][y * stride + x] = patterns[tag];
            }
        }
    }
    buffers.clobber.fill(-0x2468ace);
    for (size_t x = 0; x < 2 * D; ++x)
        buffers.bias[x] = x % 2 ? -1234567 : 7654321;
}

void positive(harness &h) {
    h.open();
    h.refused();
    // Two disjoint software J-group/epoch identities, full and tailed shapes.
    // No assertion that these software identities resurrect sealed work IDs.
    for (uint32_t epoch = 0; epoch < 2; ++epoch) {
        const uint32_t pi = epoch ? 3 : 0, pj = epoch ? 5 : 0;
        prepare(epoch, pi, pj);
        for (uint32_t bank = 0; bank < 2; ++bank) {
            h.restore(bank, pi, pj, buffers.source[bank].data());
            h.blocked(bank, pi, pj);
            h.fence();
            h.read(bank, pi, pj, buffers.saved[bank].data());
            exact(buffers.saved[bank], buffers.source[bank]);
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            // Save -> clobber -> verify clobber -> restore saved values.
            h.restore(bank, pi, pj, buffers.clobber.data()); h.fence();
            h.read(bank, pi, pj, buffers.output[bank].data());
            auto clobbered = buffers.source[bank];
            for (size_t y = 0; y < 2 * D - pi; ++y)
                for (size_t x = 0; x < 2 * D - pj; ++x) clobbered[y * stride + x] = -0x2468ace;
            exact(buffers.output[bank], clobbered);
            h.restore(bank, pi, pj, buffers.saved[bank].data()); h.fence();
            h.read(bank, pi, pj, buffers.output[bank].data());
            exact(buffers.output[bank], buffers.saved[bank]);
            // Other bank must survive the complete save/clobber/restore cycle.
            h.read(1 - bank, pi, pj, buffers.output[1 - bank].data());
            exact(buffers.output[1 - bank], buffers.source[1 - bank]);
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            h.bias(bank, pi, pj);
            h.read(bank, pi, pj, buffers.output[bank].data());
            auto resumed = buffers.saved[bank];
            for (size_t y = 0; y < 2 * D - pi; ++y)
                for (size_t x = 0; x < 2 * D - pj; ++x) {
                    const int64_t wide = int64_t(resumed[y * stride + x]) + buffers.bias[x];
                    resumed[y * stride + x] = int32_t(std::clamp<int64_t>(wide, INT32_MIN, INT32_MAX));
                }
            exact(buffers.output[bank], resumed);
            if (epoch == 0) {
                buffers.archive[bank] = buffers.saved[bank];
                buffers.archive_expected[bank] = buffers.source[bank];
            }
        }
    }
    // The first group's saved DMA matrices survive reuse by the second group.
    // Switch value epochs back, with no Scale/SCU work or arithmetic replay.
    for (uint32_t bank = 0; bank < 2; ++bank) {
        buffers.output[bank].fill(guard);
        h.restore(bank, 0, 0, buffers.archive[bank].data()); h.fence();
        h.read(bank, 0, 0, buffers.output[bank].data());
        exact(buffers.output[bank], buffers.archive_expected[bank]);
    }
    check(potal_paired_session_query(h.session).commands == h.commands, "exact session command count");
    if (!h.hardware) {
        check(h.model.complete(), "all encoded config flags and addresses");
        check(h.model.mvins == h.mvins && h.model.mvouts == h.mvouts && h.model.fences == h.fences,
              "exact command-class counts");
        check(h.model.restore_bytes == h.restore_bytes && h.model.bias_bytes == h.bias_bytes &&
              h.model.readback_bytes == h.readback_bytes, "logical byte counts");
    }
    status(potal_paired_session_close(h.session), POTAL_OK, "clean close");
    h.session = nullptr;
}

void partial_failure(harness &h) {
    h.open();
    h.model.fail_mvin = 2;
    // Only the config and two potentially partial commands reach the sink.
    h.model.matrix_commands(0, 0, 0, buffers.source[0].data(), false, false);
    h.model.expected.resize(3);
    const auto r = h.descriptor(0, 0, 0, buffers.source[0].data());
    status(potal_e3_acc_restore_issue(h.session, &r), POTAL_ERR_BAD_RESPONSE, "partial MVIN failure");
    const auto q = potal_paired_session_query(h.session);
    check(q.failed && q.outstanding_acc_updates == 2 && q.commands == 3 &&
          std::strcmp(q.error_stage, "e3_acc_restore") == 0, "partial issue retains pending and original error");
    check(h.model.acc[D * D] == buffers.source[0][D] && h.model.acc[(D + 1) * D] == 0,
          "partial MVIN actually changes only its first modeled row");
    status(potal_e3_acc_restore_issue(h.session, nullptr), q.error, "sticky error precedes validation");
    potal_acc_readback out{0, 2, 2, 0, 0, buffers.output[0].data(), stride};
    status(potal_paired_acc_readback_issue(h.session, &out), q.error, "failed readback refuses");
    check(potal_paired_session_query(h.session).commands == q.commands, "failure issues nothing further");
    h.model.fence_command();
    status(potal_paired_session_cleanup(h.session), POTAL_OK, "one cleanup drain");
    const auto cleaned = potal_paired_session_query(h.session);
    check(cleaned.failed && cleaned.error == q.error && cleaned.outstanding_acc_updates == 0 &&
          std::strcmp(cleaned.error_stage, "e3_acc_restore") == 0, "cleanup preserves sticky failure");
    status(potal_paired_session_cleanup(h.session), POTAL_OK, "cleanup is idempotent after drain");
    check(potal_paired_session_query(h.session).commands == cleaned.commands, "no redundant cleanup fence");
    status(potal_paired_session_close(h.session), q.error, "failed but drained close");
    h.session = nullptr;
    check(h.model.complete() && h.model.fences == 1, "partial failure exact cleanup stream");
    // Reopening in the next test proves successful cleanup released ownership.
}

void poison_last(harness &h) {
    h.open();
    h.restore(1, 3, 5, buffers.source[1].data());
    h.model.fail_fence = true;
    h.model.fence_command();
    status(potal_paired_fence(h.session), POTAL_HW_BAD_COMMAND, "injected fence failure");
    const auto q = potal_paired_session_query(h.session);
    check(q.failed && q.outstanding_acc_updates == 4 && std::strcmp(q.error_stage, "fence") == 0,
          "failed fence keeps ownership and buffers live");
    status(potal_paired_session_cleanup(h.session), q.error, "failed fence is not retried");
    status(potal_e3_acc_restore_issue(h.session, nullptr), q.error, "poisoned restore refuses");
    check(potal_paired_session_query(h.session).commands == q.commands, "no failed fence retry command");
    // Do not close a session with unknown pending DMA. Retain it, the sink and
    // every buffer through process exit; this case has no recovery API.
    potal_paired_session *other = nullptr;
    status(potal_paired_session_open(nullptr, &other), POTAL_ERR_BUSY, "unknown owner remains poisoned");
    check(other == nullptr && h.model.complete() && h.model.fences == 1, "poison last stream");
}

void report(const harness &h, const char *mode) {
    std::printf("POTAL_E3_CONTEXT %s PASS commands=%llu MVIN=%llu MVOUT=%llu fence=%llu "
                "logical_restore_bytes=%llu logical_bias_lane_bytes=%llu logical_readback_bytes=%llu\n",
                mode, (unsigned long long)h.commands, (unsigned long long)h.mvins,
                (unsigned long long)h.mvouts, (unsigned long long)h.fences,
                (unsigned long long)h.restore_bytes, (unsigned long long)h.bias_bytes,
                (unsigned long long)h.readback_bytes);
    std::puts("Bytes are logical/command_declared, not measured DMA bus traffic; no overlap or timing claim.");
}
} // namespace

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--hardware") != 0)) {
        std::fprintf(stderr, "Usage: %s [--hardware]\n", argv[0]);
        return 2;
    }
    const bool hardware = argc == 2;
    // Static sink storage as well as DMA storage survives unknown completion.
    static harness positive_case(hardware), partial_case(false), poison_case(false);
    if (hardware) {
#if defined(__riscv) && defined(__linux__)
        if (mlock(&buffers, sizeof(buffers)) != 0) {
            std::perror("mlock E3 DMA arena");
            return 1; // no commands issued, no pageable hardware fallback
        }
        ggml::potal::issuer_owner owner;
        std::string why;
        potal_status started_status = POTAL_OK;
        bool ok = false, fatal = false;
        double checked_context_ms = 0;
        const bool started = owner.start([&](const ggml::potal::owner_runners &) {
            try {
                const auto start = std::chrono::steady_clock::now();
                positive(positive_case);
                checked_context_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                ok = true;
            }
            catch (const std::exception &e) { std::fprintf(stderr, "E3 hardware FAIL: %s\n", e.what()); }
            fatal = positive_case.settle();
            return ggml::potal::consume_outcome{fatal, ok ? POTAL_OK : POTAL_ERR_BAD_RESPONSE, "e3_context"};
        }, why, started_status);
        owner.join();
        if (!fatal) (void)munlock(&buffers, sizeof(buffers));
        if (!started || !ok) {
            std::fprintf(stderr, "POTAL_E3_CONTEXT actual RoCC FAIL status=%s detail=%s\n",
                         potal_status_name(started_status), why.c_str());
            return 1;
        }
        report(positive_case, "actual RoCC");
        std::printf("POTAL_E3_CONTEXT_COST cpu_observed_checked_test_ms=%.6f "
                    "includes_validation_and_host_comparisons=1 DMA_time=Not_measured "
                    "pipeline_latency=Not_measured\n", checked_context_ms);
        std::puts("Hardware failure injection not run.");
        return 0;
#else
        std::fputs("--hardware requires RISC-V Linux issuer_owner; no software fallback.\n", stderr);
        return 2;
#endif
    }
    try {
        positive(positive_case);
        partial_failure(partial_case);
        poison_last(poison_case); // irreversible software ownership poison: MUST remain last
        report(positive_case, "software-only");
        std::printf("software-only failure cases PASS partial_commands=%llu partial_fences=%llu "
                    "poison_commands=%llu poison_fences=%llu; unknown session and static buffers retained\n",
                    (unsigned long long)partial_case.model.calls, (unsigned long long)partial_case.model.fences,
                    (unsigned long long)poison_case.model.calls, (unsigned long long)poison_case.model.fences);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "POTAL_E3_CONTEXT software-only FAIL: %s\n", e.what());
        // Unexpected sink mismatch: retain all storage, do not guess completion.
        return 1;
    }
}
