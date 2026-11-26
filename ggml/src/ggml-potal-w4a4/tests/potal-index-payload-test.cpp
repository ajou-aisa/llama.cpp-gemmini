// Same-stripe Index storage sharing must not change any logical installation or command.
#include "support/paired-h2-emulator.hpp"
#include "support/w4a4-golden.hpp"
#include "ggml-cpp.h"
#include "ggml.h"
#include "matmul/dense.hpp"
#include "matmul/paired-auto.hpp"
#include "../../ggml-npu-common/quantization/activation/exsia/types.hpp"
#include "../../ggml-npu-common/residual/rmd/rmd-builder.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace ggml::potal;
namespace gold = w4a4_golden;
namespace rmd = ggml::npu::rmd;

namespace {

static_assert(POTAL_W4A4 == 1 && POTAL_HW_DIM == 32, "native W4 DIM32 regression");
constexpr size_t M = 3, N = 65, K = 384, LD = 96;
constexpr uint32_t blocks[] = {0, 1, 2, 3, 8, 9};

void require(bool ok, const std::string & why) {
    if (!ok) throw std::runtime_error(why);
}

struct fixture {
    ggml_context_ptr ctx;
    hp1_weight weight;
    gold::mxint4_weight golden;
    potal_tile_plan main;
    potal_paired_execution_plan shared, expanded;
    rmd::StripePacketHandle packet;
    residual::decoded_packet decoded;
    std::unique_ptr<int8_t, decltype(&std::free)> a{nullptr, std::free};
    std::vector<int8_t> logical;
    std::vector<int32_t> main_golden, residual_golden;
    std::vector<int64_t> correction_golden;

    fixture() {
        ggml_init_params ip{4 * 1024 * 1024, nullptr, false};
        ctx.reset(ggml_init(ip));
        require(bool(ctx), "ggml context");
        auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_MXINT4, K, N);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
        auto * y = ggml_mul_mat(ctx.get(), w, x);
        std::vector<float> values(N * K);
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k)
                values[n * K + k] = k / 32 == 4 && n % 5 == 0 ? 0.0f :
                    std::ldexp(float(int((3 * n + 5 * k) % 15) - 7), int(k / 32 % 4) - 4);
        ggml_quantize_chunk(GGML_TYPE_MXINT4, values.data(), w->data, 0, N, K, nullptr);
        require(potal_frozen_plan(M, N, K, 1, 2, 4, 1, &main) == POTAL_OK, "frozen Main");
        npu_matmul_args args;
        require(init_args(w, y, {M, N, K}, "qwen2", "index-payload",
                    ggml::npu::residual::ResidualRoute::ws_packet, main, args) &&
                weight.allocate(N, K) && prepare_weight(args, weight) == quants::weight::hp1::status::ok,
                "real MXINT4 preparation");
        std::string why;
        require(gold::decode_mxint4(static_cast<const uint8_t *>(w->data),
                    ggml_row_size(GGML_TYPE_MXINT4, K), N, K, golden, why), why);
        logical.resize(M * K);
        for (size_t i = 0; i < logical.size(); ++i) logical[i] = int(i * 5 % 16) - 8;
        a.reset(static_cast<int8_t *>(dma_alloc(32 * K / 2)));
        require(a && potal_int4_pack_rows(reinterpret_cast<uint8_t *>(a.get()), 32 * K / 2,
                    K / 2, logical.data(), K, M, K) == POTAL_OK, "packed A4");

        rmd::RmdStripeBuilder builder;
        builder.reset(0, 0, M, K, N, 4);
        for (size_t b = 0; b < 6; ++b)
            for (size_t k = 0; k < 1 + 2 * b; ++k)
                for (size_t row = 0; row < M; ++row)
                    require(builder.add_residual(row, blocks[b] * 32 + k,
                                row == 0 ? 17 : row == 1 ? -3 : 5), "radix-16 residual");
        packet = builder.finish();
        require(packet && residual::decode_packet(*packet, decoded) == rmd::RmdStatus::success,
                "real W4 packet decode");
        potal_paired_tile_plan tile;
        require(potal_paired_plan_stripe(main, M, N, K, &decoded, tile, shared, &why) ==
                    potal_paired_status::ok, why);
        require(tile.j_groups == 2 && tile.k_panels == 3 && tile.residual_windows == 3 &&
                    tile.panels[0].windows.size() == 2 && tile.panels[1].windows.empty() &&
                    tile.residual_empty_panels == 1 && tile.residual_filler_slots > 0 && tile.pad_J != 0,
                "fixture needs multiple J groups, tail, panels, windows, filler and empty panel");

        // Expand only storage and spans. Every op, dependency, header and counter is copied unchanged.
        expanded = shared;
        expanded.index_entries.clear();
        std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> spans;
        size_t installs = 0, writes = 0, unique = 0;
        for (const auto & op : shared.ops) {
            if (op.kind != potal_paired_op_kind::index_program) continue;
            const auto key = std::make_pair(op.panel, op.window);
            const auto span = std::make_pair(op.entry_begin, op.entry_end);
            if (op.j_group == 0) {
                require(spans.emplace(key, span).second, "one J0 span per window");
                require(op.entry_begin == unique, "contiguous unique storage");
                unique = op.entry_end;
            } else {
                require(spans.at(key) == span, "matching panel/window must alias J0");
            }
            auto & copy = expanded.ops[op.id];
            copy.entry_begin = expanded.index_entries.size();
            expanded.index_entries.insert(expanded.index_entries.end(),
                shared.index_entries.begin() + op.entry_begin, shared.index_entries.begin() + op.entry_end);
            copy.entry_end = expanded.index_entries.size();
            ++installs;
            writes += op.entry_end - op.entry_begin;
        }
        require(unique == 117 && shared.index_entries.size() == unique &&
                    expanded.index_entries.size() == 2 * unique && writes == 234 &&
                    shared.index_writes == writes && expanded.index_writes == writes &&
                    installs == 6 && shared.index_programs == installs && expanded.index_programs == installs,
                "half the payload storage, unchanged logical installations and writes");
        require(potal_validate_execution_plan(shared, &decoded, &why) == potal_paired_status::ok &&
                    potal_validate_execution_plan(expanded, &decoded, &why) == potal_paired_status::ok,
                "shared and public expanded plans both validate: " + why);

        main_golden.resize(M * N);
        for (size_t row = 0; row < M; ++row)
            gold::scu_row(logical.data() + row * K, golden, main_golden.data() + row * N);
        residual_golden.assign(decoded.rows.size() * N, 0);
        correction_golden.assign(M * N, 0);
        for (size_t row = 0; row < decoded.rows.size(); ++row) {
            for (size_t n = 0; n < N; ++n) {
                int32_t acc = 0;
                size_t compact = 0;
                for (const auto & run : decoded.runs) {
                    int64_t raw = 0;
                    for (uint16_t k : run.local_k)
                        raw += int64_t(decoded.digits[row * decoded.k_compact + compact++]) *
                            golden.q[n * K + run.block_id * 32 + k];
                    acc = gold::acc_add(acc, gold::scu(raw, golden.carrier[run.block_id * N + n]));
                }
                residual_golden[row * N + n] = acc;
                const auto & lane = decoded.rows[row];
                correction_golden[lane.source_row * N + n] += int64_t(acc) * (INT64_C(1) << (4 * lane.lane));
            }
        }
        require(std::any_of(main_golden.begin(), main_golden.end(), [](int32_t v) { return v != 0; }) &&
                    std::any_of(residual_golden.begin(), residual_golden.end(), [](int32_t v) { return v != 0; }),
                "fixture must execute nonzero Main and Residual math");
    }
};

void mutations(const fixture & f) {
    std::string why;
    auto bad = f.shared;
    // Still internally in range and original_k == panel.k_begin + resident_k, but not the packet's K.
    ++bad.index_entries.front().original_k;
    ++bad.index_entries.front().resident_k;
    require(potal_validate_execution_plan(bad, &f.decoded, &why) ==
                potal_paired_status::invalid_execution_plan, "shared-entry packet mutation accepted");
    bad = f.expanded;
    for (const auto & op : bad.ops)
        if (op.kind == potal_paired_op_kind::index_program && op.j_group == 1) {
            ++bad.index_entries[op.entry_begin].original_k;
            ++bad.index_entries[op.entry_begin].resident_k;
            break;
        }
    why.clear();
    require(potal_validate_execution_plan(bad, &f.decoded, &why) ==
                potal_paired_status::invalid_execution_plan, "later expanded op skipped packet validation");
    bad = f.shared;
    for (auto & op : bad.ops) {
        if (op.kind != potal_paired_op_kind::index_program || op.j_group != 1 || op.window != 0) continue;
        const size_t count = op.entry_end - op.entry_begin;
        ++op.entry_begin;
        op.entry_end = op.entry_begin + count;
        break;
    }
    why.clear();
    require(potal_validate_execution_plan(bad, &f.decoded, &why) ==
                potal_paired_status::invalid_execution_plan, "later J-group wrong range accepted");
    bad = f.shared;
    for (auto & op : bad.ops)
        if (op.kind == potal_paired_op_kind::index_program && op.j_group == 1) {
            op.entry_begin = bad.index_entries.size();
            op.entry_end = op.entry_begin + op.index_active_rows;
            break;
        }
    require(potal_validate_execution_plan(bad, &f.decoded, &why) ==
                potal_paired_status::invalid_execution_plan, "out-of-storage range accepted");
}

// Flat machine values allow exact IPC comparison without hashing or struct padding.
std::vector<uint64_t> execute(const fixture & f, const potal_paired_execution_plan & plan,
                             bool queue, potal_paired_stream_mode stream, bool prepared, bool refuse) {
    potal_test::h2_emulator emulator;
    std::unique_ptr<int32_t, decltype(&std::free)> cm(
        static_cast<int32_t *>(dma_alloc(32 * LD * sizeof(int32_t))), std::free);
    std::unique_ptr<int32_t, decltype(&std::free)> cr(
        static_cast<int32_t *>(dma_alloc(32 * LD * sizeof(int32_t))), std::free);
    require(cm && cr, "readback allocation");
    uint32_t generation = 0;
    potal_paired_request req;
    req.rows = M; req.n = N; req.k = K;
    req.main_plan = &f.main; req.weight = &f.weight;
    req.a = f.a.get(); req.lda = K; req.residual = &f.decoded;
    req.c_main = cm.get(); req.ldc_main = LD;
    req.c_residual = cr.get(); req.ldc_residual = LD;
    req.generation = &generation; req.plan = &plan;
    req.ktile_scale_reuse = true;
    req.stream = stream;
    req.stream_counter = stream == potal_paired_stream_mode::off ? nullptr : potal_test::h2_emulator::on_counter;
    potal_paired_prepared job;
    if (prepared) {
        auto copy = plan;
        const char * stage = nullptr;
        std::string why;
        require(potal_paired_prepare(req, std::move(copy), job, &stage, &why) == potal_paired_status::ok, why);
        require(job.ready && job.index_windows == plan.index_programs &&
                    job.queue_entries.size() == plan.index_entries.size(), "prepared Queue mirror and op budget");
        for (size_t i = 0; i < job.queue_entries.size(); ++i)
            require(job.queue_entries[i] == plan.index_entries[i].resident_k, "prepared Queue payload");
        req.plan = nullptr;
        req.prepared = &job;
    }
    const auto sink = emulator.sink();
    potal_paired_report report;
    const auto status = potal_paired_run_session(req, &sink, report);
    if (queue) {
        // W4 explicitly rejects Queue mode before admission or command issue.
        // Sharing prepared payloads must not accidentally enable that route.
        require(status == potal_paired_status::invalid_argument && report.hw_status == POTAL_ERR_UNSUPPORTED &&
                    report.stage && std::strcmp(report.stage, "index_queue") == 0 &&
                    !report.issued && !report.fatal && emulator.records.empty(),
                "W4 Queue rejection before any command");
        require(potal_index_queue_remaining() == (refuse ? 3u : 255u),
                "unsupported Queue request consumed admission credits");
        if (prepared) require(job.ready, "unsupported Queue request consumed prepared ownership");
        return {uint64_t(status), uint64_t(report.hw_status), potal_index_queue_remaining()};
    }
    require(status == potal_paired_status::ok && !report.fatal && report.issued && emulator.errors.empty(),
            std::string(report.stage ? report.stage : "run") + ": " + report.detail + emulator.errors);
    const auto & c = report.counts;
    require(emulator.main_loops == plan.main_gemms && emulator.residual_loops == plan.residual_gemms &&
                emulator.index_configs == plan.index_programs && emulator.index_writes == plan.index_writes,
            "emulator must perform every logical installation and GEMM");
    require(c.main_loop_ws == plan.main_gemms && c.residual_loop_ws == plan.residual_gemms &&
                c.queue_allocations == (queue ? plan.index_programs : 0) &&
                c.index_configs == (queue ? 0 : plan.index_programs) &&
                c.index_writes == (queue ? 0 : plan.index_writes) &&
                emulator.queue_allocations == c.queue_allocations && emulator.queue_applied == c.queue_allocations,
            "unchanged scalar writes / Queue installations");
    if (prepared) require(!job.ready, "prepared job consumed");
    if (stream != potal_paired_stream_mode::off) {
        // Counts from the fixture's six odd run lengths, not index_entries.size()/j_groups.
        require(c.residual_valid_bytes == f.decoded.rows.size() * 21 &&
                    c.residual_staging_bytes == f.decoded.rows.size() * 3 * 64 &&
                    c.residual_pitch_padding_bytes == f.decoded.rows.size() * 3 * 32 &&
                    c.residual_load_bytes_declared == f.decoded.rows.size() * (18 + 20 + 22) * 2,
                "stream packed valid/staging/pitch/load byte diagnostics");
    }
    std::vector<uint64_t> values;
    for (size_t row = 0; row < M; ++row)
        for (size_t n = 0; n < N; ++n) {
            require(cm.get()[row * LD + n] == f.main_golden[row * N + n], "Main INT32 golden");
            values.push_back(uint32_t(cm.get()[row * LD + n]));
        }
    for (size_t row = 0; row < f.decoded.rows.size(); ++row)
        for (size_t n = 0; n < N; ++n) {
            require(cr.get()[row * LD + n] == f.residual_golden[row * N + n], "Residual INT32 golden");
            values.push_back(uint32_t(cr.get()[row * LD + n]));
        }
    rmd::BlockScaledInt64Correction correction;
    require(residual::reconstruct_rows(*f.packet, f.decoded.rows, cr.get(), LD, N, correction) ==
                rmd::RmdStatus::success && correction.values == f.correction_golden, "INT64 reconstruction");
    ggml::npu::quants::act::exsia::StripeMetadataSnapshot metadata;
    metadata.theta = -3;
    std::vector<float> out(M * N, 123.0f);
    size_t nonzero = 0;
    require(residual::merge_main_residual_hp1(f.weight, 0, M, metadata, correction,
                {cm.get(), LD, 0.125}, out.data(), M, N, 1, &nonzero) == rmd::RmdStatus::success,
            "F32 merge");
    for (size_t i = 0; i < out.size(); ++i) {
        const float main = float(double(f.main_golden[i]) * 0.125 * double(f.golden.scale[i % N]));
        const float want = main + float(double(f.correction_golden[i]) * double(f.golden.scale[i % N]) * 0.125);
        require(std::memcmp(&out[i], &want, sizeof(float)) == 0, "F32 bitwise golden");
        uint32_t bits;
        std::memcpy(&bits, &out[i], sizeof(bits));
        values.push_back(bits);
    }
    values.insert(values.end(), {c.main_loop_ws, c.residual_loop_ws, c.index_configs, c.index_writes,
        c.queue_allocations, c.scale_writes, c.scale_releases, c.readbacks, c.readback_tiles,
        c.fences, c.drains, c.ops, c.residual_valid_bytes, c.residual_staging_bytes,
        c.residual_pitch_padding_bytes, c.residual_load_bytes_declared, emulator.records.size()});
    for (auto record : emulator.records) {
        // Only DMA pointer-valued fields vary between allocations. Keep null/non-null,
        // every header, extent, stride, Scale key, Queue ticket/generation, and dependency fence.
        if (record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB ||
            record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC) {
            record.rs1 = record.rs1 != 0;
            record.rs2 = record.rs2 != 0;
        } else if (record.funct == POTAL_FUNCT_MVOUT || record.funct == POTAL_FUNCT_MVIN) {
            record.rs1 = record.rs1 != 0;
        }
        values.insert(values.end(), {record.funct, record.xd, record.rs1, record.rs2});
    }
    return values;
}

bool transfer(int fd, void * data, size_t bytes, bool sending) {
    auto * p = static_cast<char *>(data);
    while (bytes) {
        const ssize_t n = sending ? write(fd, p, bytes) : read(fd, p, bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        bytes -= size_t(n);
    }
    return true;
}

bool reaped(pid_t pid) {
    int status = 0;
    pid_t done;
    do { done = waitpid(pid, &status, 0); } while (done < 0 && errno == EINTR);
    return done == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// Fork before any session: shared and expanded see identical process credits and
// device generations. No reset seam and no normalization of Queue commands.
void compare(const fixture & f, bool queue, potal_paired_stream_mode stream, bool prepared) {
    int pipefd[2];
    require(pipe(pipefd) == 0, "comparison pipe");
    const pid_t pid = fork();
    require(pid >= 0, "comparison fork");
    if (pid == 0) {
        close(pipefd[0]);
        try {
            auto values = execute(f, f.shared, queue, stream, prepared, false);
            size_t size = values.size();
            const bool ok = transfer(pipefd[1], &size, sizeof(size), true) &&
                transfer(pipefd[1], values.data(), size * sizeof(uint64_t), true);
            close(pipefd[1]);
            _exit(ok ? 0 : 1);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "shared: %s\n", e.what());
            _exit(1);
        }
    }
    close(pipefd[1]);
    try {
        const auto want = execute(f, f.expanded, queue, stream, prepared, false);
        size_t size = 0;
        require(transfer(pipefd[0], &size, sizeof(size), false) && size == want.size(), "snapshot extent");
        std::vector<uint64_t> got(size);
        require(transfer(pipefd[0], got.data(), size * sizeof(uint64_t), false), "shared snapshot");
        close(pipefd[0]);
        require(reaped(pid), "shared execution failed");
        require(got == want, "shared/expanded outputs, counters or encoded commands differ");
    } catch (...) {
        close(pipefd[0]);
        reaped(pid);
        throw;
    }
}

} // namespace

int main() {
    int failures = 0;
    const char * names[] = {
        "scalar supplied", "Queue supplied rejection", "scalar prepared", "Queue prepared rejection",
        "bounded scalar residual_valid_bytes", "bounded Queue rejection",
        "continuous scalar residual_valid_bytes", "continuous Queue rejection",
        "Queue supplied rejection preserves credits", "Queue prepared rejection preserves credits"
    };
    // Each case has a fresh process lifetime; production never exposes a test reset.
    for (int mode = 0; mode < 10; ++mode) {
        std::fflush(nullptr);
        const pid_t pid = fork();
        if (pid == 0) {
            try {
                setenv("POTAL_INDEX_QUEUE_FRESH", "1", 1);
                setenv("POTAL_INDEX_QUEUE_BUDGET", mode >= 8 ? "3" : "255", 1);
                setenv("GGML_POTAL_INDEX_QUEUE", (mode % 2 || mode >= 8) ? "ON" : "OFF", 1);
                setenv("POTAL_PAIRED_TIMING", "0", 1);
                for (const char * name : {"POTAL_PAIRED_PROFILE", "POTAL_RAW_TRACE", "POTAL_EARLY_TRACE",
                                         "POTAL_PAIRED_WINDOW_DIAG", "POTAL_GRAPH_CPU", "POTAL_STREAM_COUNTERS"})
                    unsetenv(name);
                fixture f;
                mutations(f);
                const bool queue = mode % 2;
                if (mode >= 8) {
                    const auto shared = execute(f, f.shared, true, potal_paired_stream_mode::off, mode == 9, true);
                    const auto expanded = execute(f, f.expanded, true, potal_paired_stream_mode::off, mode == 9, true);
                    require(shared == expanded, "Queue rejection differs between storage representations");
                } else {
                    const auto stream = mode < 4 ? potal_paired_stream_mode::off :
                        mode < 6 ? potal_paired_stream_mode::bounded : potal_paired_stream_mode::continuous;
                    compare(f, queue, stream, mode == 2 || mode == 3);
                }
                _exit(0);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "index-payload mode=%d: %s\n", mode, e.what());
                _exit(1);
            }
        }
        const bool ok = pid > 0 && reaped(pid);
        failures += !ok;
        std::printf("[%s] shared/expanded %s\n", ok ? "PASS" : "FAIL", names[mode]);
    }
    std::printf("potal-index-payload-test: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
