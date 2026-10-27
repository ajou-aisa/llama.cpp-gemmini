#include "potal.h"
#include "matmul/execution.hpp"
#include "support/e4b2-fixtures.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char *stage) {
    if (!condition) throw std::runtime_error(stage);
}
void ok(potal_status status, const char *stage) {
    if (status != POTAL_OK)
        throw std::runtime_error(std::string(stage) + ": " + potal_status_name(status));
}

/* No software sink or command model exists in this executable. Unexpected
 * issue/completion failure leaves this raw session pointer and its owned pools
 * retained; the issuer receives a fatal outcome instead of guessing recovery.
 */
struct hardware {
    enum stage {
        prepare, b_load_issue, b_completion, resident_handoff, index_setup, epoch_setup,
        scale_setup, binding_check, physical_encode, loop_issue, acc_completion,
        readback_issue, readback_completion, result_receipt, scale_release, retirement,
        close, stage_count
    };
    struct interval { uint64_t ns = 0, calls = 0; };
    bool measuring = false;
    std::array<interval, stage_count> intervals{};
    potal_paired_session *session = nullptr;
    std::array<potal_resident_ws_load, 2> loads{};
    std::array<potal_resident_ws_ref, 2> residents{};
    std::array<std::array<potal_resident_ws_ref, 32>, 2> epochs{};
    std::array<std::array<uint64_t, 32>, 2> progress{};
    std::array<std::array<std::array<int32_t, 256>, 32>, 2> golden{};
    uint32_t generation = 1;
    uint64_t cells = 0, runs = 0;

    template<typename Call> potal_status timed(stage which, Call call) {
        if (!measuring) return call();
        const auto begin = std::chrono::steady_clock::now();
        const auto status = call();
        auto &value = intervals[which];
        value.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
        ++value.calls;
        return status;
    }
    void print_intervals(unsigned sample) const {
        static const char *names[stage_count] = {
            "prepare", "b_load_issue", "b_completion", "resident_handoff", "index_setup",
            "epoch_setup", "scale_setup", "binding_check", "physical_encode",
            "generated_loop_issue", "acc_completion", "readback_issue",
            "readback_completion", "result_receipt", "scale_release", "retirement", "close"
        };
        for (unsigned i = 0; i < stage_count; ++i)
            std::printf("E4B2_CHECKED_INTERVAL sample=%u scope=%s cpu_ms=%.6f calls=%llu "
                        "hardware_wait=unmeasured\n", sample, names[i], intervals[i].ns / 1e6,
                        static_cast<unsigned long long>(intervals[i].calls));
    }
    int8_t *buffer(unsigned pool) {
        void *data = nullptr;
        size_t bytes = 0;
        ok(potal_paired_resident_buffer_view(session, static_cast<potal_resident_ws_buffer>(pool),
                                            &data, &bytes), "owned buffer");
        require(bytes == 65536 && uintptr_t(data) % 64 == 0, "bounded aligned DMA allocation");
        return static_cast<int8_t *>(data);
    }
    void load(unsigned pair, unsigned k, unsigned j) {
        auto &d = loads[pair];
        d = {pair, k, j, (k + 15) / 16, 0, 0, 71 + pair, 9, 144};
        auto *source = buffer(pair);
        for (unsigned row = 0; row < k; ++row)
            for (unsigned column = 0; column < j; ++column)
                source[row * 144 + column] = e4b2_weight(pair, 0, row, column);
        ok(timed(b_load_issue, [&] { return potal_paired_resident_load_issue(session, &d); }),
           "B-only DMA");
        potal_resident_ws_ref unpublished{};
        require(potal_paired_resident_get(session, pair, &unpublished) == POTAL_ERR_BUSY,
                "B issue is not publication");
        ok(timed(b_completion, [&] { return potal_paired_fence(session); }), "B completion");
        ok(timed(resident_handoff, [&] {
            return potal_paired_resident_get(session, pair, &residents[pair]);
        }), "published resident");
        std::fill_n(source, 65536, int8_t(93));
    }
    void run(unsigned pair, unsigned path, bool indexed, unsigned k_begin, unsigned k_rows,
             unsigned j_begin, unsigned columns, unsigned acc_tile, unsigned rows,
             const std::vector<std::vector<uint64_t>> &compact = {},
             bool first = true, bool last = true, unsigned epoch_k_end = 0) {
        std::vector<std::vector<uint64_t>> partials = compact;
        if (!path) {
            for (unsigned k = 0; k < k_rows; k += 16) {
                partials.emplace_back();
                for (unsigned x = k; x < std::min(k + 16, k_rows); ++x)
                    partials.back().push_back(k_begin + x);
            }
        }
        require(!partials.empty() && partials.size() <= 2, "bounded hardware fixture");
        potal_resident_ws_run d{};
        d.resident = residents[pair]; d.path = path; d.indexed = indexed;
        d.k_offset = k_begin; d.j_offset = j_begin; d.j_columns = columns; d.i_rows = rows;
        d.acc_tile = acc_tile; d.output_id = 100 + acc_tile;
        d.a_offset_tiles = path ? loads[pair].main_a_tiles : 0;
        d.first = first; d.last = last; d.scale_generation = generation++;
        d.sequence = first ? 0 : progress[path][acc_tile];
        std::vector<uint64_t> original;
        for (unsigned p = 0; p < partials.size(); ++p) {
            if (indexed) d.fragment_counts[p] = uint32_t(partials[p].size());
            original.insert(original.end(), partials[p].begin(), partials[p].end());
            if (p + 1 < partials.size())
                while (original.size() % 16) original.push_back(partials[p][0]);
        }
        d.k_rows = path ? uint32_t(original.size()) : k_rows;
        d.a_stride = ((d.k_rows + 15) / 16) * 16;
        if (indexed) {
            const potal_resident_ws_indices ix{residents[pair], uint32_t(original.size()), original.data()};
            ok(timed(index_setup, [&] {
                return potal_paired_resident_index_program(session, &ix, &d.index);
            }), "full resident Index");
        }
        std::vector<potal_resident_ws_fragment> manifest;
        for (const auto &partial : partials) {
            potal_resident_ws_fragment fragment{};
            fragment.count = uint32_t(partial.size());
            std::copy(partial.begin(), partial.end(), fragment.original_k);
            manifest.push_back(fragment);
        }
        for (unsigned j = 0; j < (columns + 15) / 16; ++j) {
            if (!first) {
                d.epochs[j] = epochs[path][acc_tile + j];
                continue;
            }
            potal_resident_ws_epoch e{};
            e.path = path; e.acc_tile = acc_tile + j; e.rows = rows;
            e.columns = std::min(16u, columns - j * 16);
            e.output_id = d.output_id; e.original_j = j_begin + j * 16;
            e.weight_id = loads[pair].weight_id; e.weight_generation = loads[pair].weight_generation;
            e.logical_k_begin = k_begin;
            e.logical_k_end = path ? partials.back().back() + 1 :
                (epoch_k_end ? epoch_k_end : k_begin + k_rows);
            if (path) { e.fragments = manifest.data(); e.fragment_count = manifest.size(); }
            ok(timed(epoch_setup, [&] {
                return potal_paired_resident_epoch_begin(session, &e, &d.epochs[j]);
            }), "immutable ACC epoch");
            epochs[path][acc_tile + j] = d.epochs[j];
        }
        auto *activation = buffer(2 + path);
        std::fill_n(activation, 65536, int8_t(0));
        for (unsigned i = 0; i < rows; ++i)
            for (unsigned p = 0; p < partials.size(); ++p)
                for (unsigned x = 0; x < partials[p].size(); ++x)
                    activation[i * d.a_stride + p * 16 + x] =
                        e4b2_activation(0, i, unsigned(partials[p][x]));
        std::vector<uint64_t> blocks;
        std::vector<uint32_t> carriers;
        for (const auto &partial : partials) {
            blocks.push_back(partial[0] / 32);
            for (unsigned j = 0; j < columns; ++j)
                carriers.push_back(e4b2_carrier(0, unsigned(partial[0] / 32), j_begin + j));
        }
        const potal_resident_ws_scales scales{residents[pair], 0, d.scale_generation,
            uint32_t(partials.size()), columns, j_begin, blocks.data(), carriers.data(), columns};
        ok(timed(scale_setup, [&] {
            return potal_paired_resident_scale_program(session, &scales, &d.scale);
        }), "P1 original-block Scale");
        if (measuring) {
            potal_resident_ws_program encoded{};
            ok(timed(binding_check, [&] {
                return potal_paired_resident_run_check(session, &d);
            }), "profile full binding validation");
            ok(timed(physical_encode, [&] {
                return potal_resident_ws_run_encode(&loads[pair], &d, activation, 65536, &encoded);
            }), "profile physical descriptor encode");
        }
        ok(timed(loop_issue, [&] { return potal_paired_resident_issue(session, &d); }),
           "checked resident generated LOOP");
        ok(timed(acc_completion, [&] { return potal_paired_fence(session); }),
           "physical compute completion");
        potal_resident_ws_readback rb{};
        rb.bank = path; rb.acc_tile = acc_tile; rb.rows = rows; rb.columns = columns;
        rb.stride = ((columns + 15) / 16) * 16;
        for (unsigned j = 0; j < (columns + 15) / 16; ++j) {
            rb.epochs[j] = d.epochs[j]; rb.sequences[j] = d.sequence + partials.size();
        }
        potal_resident_ws_ref receipt{};
        ok(timed(readback_issue, [&] {
            return potal_paired_resident_readback_issue(session, &rb, &receipt);
        }), "addressed raw INT32 readback");
        ok(timed(readback_completion, [&] { return potal_paired_fence(session); }),
           "physical readback completion");
        const int32_t *result = nullptr;
        size_t stride = 0;
        ok(timed(result_receipt, [&] {
            return potal_paired_resident_result(session, receipt, &result, &stride);
        }), "completed result receipt");
        for (unsigned i = 0; i < rows; ++i)
            for (unsigned j = 0; j < columns; ++j) {
                auto &prior = golden[path][acc_tile + j / 16][i * 16 + j % 16];
                int32_t expected = first ? 0 : prior;
                for (const auto &partial : partials) {
                    std::array<uint32_t, 16> source{};
                    std::transform(partial.begin(), partial.end(), source.begin(),
                                   [](uint64_t k) { return uint32_t(k); });
                    expected = e4b2_clamp(int64_t(expected) + e4b2_oracle_partial(pair, 0, 0,
                        i, j_begin + j, source.data(), unsigned(partial.size())));
                }
                if (result[i * stride + j] != expected) {
                    std::fprintf(stderr, "E4B2_CHECKED_INTEGER_FAIL pair=%u path=%u row=%u col=%u "
                                 "expected=%d actual=%d\n", pair, path, i, j, expected, result[i * stride + j]);
                    throw std::runtime_error("independent checked-hardware INT32 golden");
                }
                prior = expected;
                ++cells;
            }
        for (unsigned j = 0; j < (columns + 15) / 16; ++j)
            progress[path][acc_tile + j] = rb.sequences[j];
        for (unsigned p = 0; p < partials.size(); ++p)
            for (unsigned j = 0; j < (columns + 15) / 16; ++j)
                for (unsigned c = 0; c < std::min(16u, columns - j * 16); ++c)
                    ok(timed(scale_release, [&] {
                        return potal_paired_scale_release(session, p * ((columns + 15) / 16) + j,
                                                         d.scale_generation, c);
                    }), "completion-qualified Scale Release");
        ++runs;
        std::printf("E4B2_CHECKED_CASE PASS pair=%u path=%u indexed=%u K=%u J=%u "
                    "ACC=%u rows=%u columns=%u first=%u last=%u sequence=%llu\n",
                    pair, path, indexed, k_begin, j_begin,
                    path * 512 + acc_tile * 16, rows, columns, first, last,
                    static_cast<unsigned long long>(d.sequence));
    }
    void suite() {
        const auto opened = potal_paired_resident_query(session);
        require(!opened.software_only && opened.capability_matched, "actual hardware mode requires capability");
        ok(timed(prepare, [&] { return potal_paired_resident_prepare(session); }), "owned prepare");
        for (unsigned pair = 0; pair < 2; ++pair) {
            load(pair, 448, 144);
            run(pair, 0, false, 48, 16, 80, 16, 5, 16);
            run(pair, 0, false, 0, 16, 80, 16, 6, 16, {}, true, false, 48);
            run(pair, 0, false, 16, 16, 80, 16, 6, 16, {}, false, false);
            run(pair, 0, false, 32, 16, 80, 16, 6, 16, {}, false, true);
            run(pair, 0, true, 112, 32, 48, 32, 20, 16);
            run(pair, 1, true, 96, 3, 48, 32, 5, 16, {{96, 98, 126}});
            ok(timed(retirement, [&] { return potal_paired_resident_retire(session, residents[pair]); }),
               "retire full resident");
            load(pair, 443, 139);
            run(pair, 0, false, 432, 11, 128, 11, 27, 7);
            run(pair, 0, true, 432, 11, 128, 11, 28, 7);
            ok(timed(retirement, [&] { return potal_paired_resident_retire(session, residents[pair]); }),
               "retire tail resident");
        }
        const auto q = potal_paired_resident_query(session);
        require(!q.pending && !q.control_pending && !q.core.live_scale_slots &&
                !q.readable[0] && !q.readable[1] && !q.core.failed, "known-safe final hardware state");
        std::printf("E4B2_CHECKED_COUNTS commands=%llu drains=%llu B_MVIN=%llu runs=%llu cells=%llu\n",
                    static_cast<unsigned long long>(q.core.commands),
                    static_cast<unsigned long long>(q.core.drains),
                    static_cast<unsigned long long>(q.attempted_b_mvins),
                    static_cast<unsigned long long>(runs), static_cast<unsigned long long>(cells));
        ok(timed(close, [&] { return potal_paired_resident_close(&session); }),
           "known-safe hardware close");
    }
};
} // namespace

int main(int argc, char **argv) {
    unsigned repeat = 1;
    const bool probe_only = argc == 2 && std::strcmp(argv[1], "--probe-only") == 0;
    const bool measure = argc == 3 && std::strcmp(argv[1], "--repeat") == 0;
    if (measure) {
        errno = 0;
        char *end = nullptr;
        const unsigned long value = std::strtoul(argv[2], &end, 10);
        if (errno || !end || *end || value == 0 || value > 100) return 2;
        repeat = unsigned(value);
    } else if (argc != 1 && !probe_only) return 2;
#if defined(__riscv) && defined(__linux__)
    ggml::potal::issuer_owner owner;
    std::string why;
    potal_status startup = POTAL_OK, result = POTAL_OK;
    bool passed = false;
    const bool started = owner.start([&](const ggml::potal::owner_runners &) {
        const auto info = ggml::potal::get_issuer_info();
        if (!info.ready || info.cpu != 1)
            return ggml::potal::consume_outcome{true, POTAL_ERR_INVALID_ARGUMENT, "resident_affinity"};
        for (unsigned sample = 0; sample < (measure ? repeat + 1 : 1); ++sample) {
            hardware h;
            result = potal_paired_resident_open(nullptr, &h.session);
            if (result != POTAL_OK) {
                std::fprintf(stderr, "E4B2_CHECKED_UNSUPPORTED status=%s extension_issued=0\n",
                             potal_status_name(result));
                return ggml::potal::consume_outcome{false, result, "resident_capability"};
            }
            try {
                if (probe_only) {
                    ok(potal_paired_resident_close(&h.session), "capability-only close");
                    std::printf("E4B2_CHECKED_CAPABILITY_SUPPORTED extension_issued=0\n");
                    passed = true;
                    return ggml::potal::consume_outcome{false, POTAL_OK, "resident_capability"};
                }
                h.measuring = measure && sample != 0;
                const auto begin = measure && sample ? std::chrono::steady_clock::now() :
                    std::chrono::steady_clock::time_point{};
                h.suite();
                if (measure && sample) {
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count();
                    std::printf("E4B2_CHECKED_SAMPLE sample=%u cpu_observed_suite_ms=%.6f "
                                "includes_staging_validation_comparison=1 includes_case_reporting=1\n",
                                sample - 1, ms);
                    h.print_intervals(sample - 1);
                } else if (measure) std::printf("E4B2_CHECKED_WARMUP_PASS timing=0\n");
            } catch (const std::exception &error) {
                std::fprintf(stderr, "E4B2_CHECKED_HARDWARE_FAIL %s retained_session=%p\n",
                             error.what(), static_cast<void *>(h.session));
                result = POTAL_ERR_BAD_RESPONSE;
                return ggml::potal::consume_outcome{true, result, "resident_checked"};
            }
        }
        passed = true;
        return ggml::potal::consume_outcome{false, POTAL_OK, "resident_checked"};
    }, why, startup);
    owner.join();
    if (!started || !passed) return result == POTAL_ERR_UNSUPPORTED ? 2 : 1;
    if (!probe_only)
        std::printf("E4B2_CHECKED_HARDWARE_PASS repeats=%u CPU1_hart0=1 fault_injection=0\n", repeat);
    return 0;
#else
    (void)repeat;
    (void)probe_only;
    (void)measure;
    std::fprintf(stderr, "RISC-V Linux issuer_owner required; no CPU fallback.\n");
    return 2;
#endif
}
