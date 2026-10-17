/*
 * PoTal GGML backend hardware smoke test (RISC-V Linux on FireSim,
 * FireSimQuadRocketSinglePoTalD16Config: one DIM16 Sat32 PoTal on hart 0).
 *
 * Every case builds a real GGML graph y = ggml_mul_mat(w, x) with I8 tensors
 * (w: weights ne = {K, 16}; x: activations ne = {K, 16}; y: F32 ne = {16, 16}),
 * selects the POTAL device explicitly and runs the graph on it. Hardware
 * evidence comes from the backend's counters (request handed to the issuer
 * thread, potal_gemm_ws_issue() and potal_fence() returning POTAL_OK). The
 * numerical check compares all 256 outputs with an int64 golden computed here
 * from the GGML definition y[m][n] = sum_k x[m][k] * w[n][k], independent of
 * the backend's packing code.
 *
 *   k16, k32  ggml_backend_graph_compute() on POTAL; patterns lcg (full signed
 *             range), transpose_probe (any swapped index changes the result),
 *             extremes (+/-128 * +/-127 products, largest sums)
 *   sched     ggml_backend_sched with POTAL + CPU: MUL_MAT placed on POTAL and
 *             run in hardware, ggml_scale(0.5) placed on CPU; exact output
 *
 * Options: --case=k16|k32|sched|all (default all), --verbose (GGML debug logs).
 * Markers on stdout (GGML logs interleaved), per hardware GEMM:
 *   GGML_BACKEND=POTAL, DISPATCH=HARDWARE, POTAL_GEMM_ISSUED, POTAL_FENCE_PASS,
 *   GGML_MUL_MAT_NUMERICAL_PASS; final "POTAL_GGML_SMOKE PASS|FAIL".
 * Exit status: 0 pass, 1 fail, 2 usage or environment error.
 *
 * Only the backend's issuer thread is pinned (to the hart-0 CPU); do not run
 * the program under taskset. Times are guest CLOCK_MONOTONIC, not FPGA cycles.
 */
#include "ggml-cpp.h"
#include "ggml-potal.h"

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int   M        = 16;
constexpr int   N        = 16;
constexpr float SENTINEL = -1.0e9f; // beyond any K <= 32 int8 dot product

bool g_verbose  = false;
int  g_failures = 0;
int  g_gemms    = 0; // hardware GEMMs the cases ran

void log_to_stdout(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_DEBUG && !g_verbose) {
        return;
    }
    std::fputs(text, stdout);
}

void fail(const std::string & what) {
    std::printf("FAIL %s\n", what.c_str());
    g_failures++;
}

ggml_backend_potal_stats stats_of(ggml_backend_t potal) {
    ggml_backend_potal_stats st = {};
    ggml_backend_potal_get_stats(potal, &st);
    return st;
}

long self_tid() {
    return static_cast<long>(syscall(SYS_gettid));
}

std::string self_affinity() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return "?";
    }
    std::string s = "{";
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &set)) {
            s += (s.size() > 1 ? "," : "") + std::to_string(c);
        }
    }
    return s + "}";
}

// Hart of a Linux CPU from /proc/cpuinfo, read here independently of the backend.
int hart_of_cpu(int cpu) {
    FILE * f = std::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    int  processor = -1;
    int  hart      = -1;
    while (hart == -1 && std::fgets(line, sizeof(line), f) != nullptr) {
        int v;
        if (std::sscanf(line, "processor : %d", &v) == 1) {
            processor = v;
        } else if (std::sscanf(line, "hart : %d", &v) == 1 && processor == cpu) {
            hart = v;
        }
    }
    std::fclose(f);
    return hart;
}

enum class pattern { lcg, transpose_probe, extremes };

const char * pattern_name(pattern p) {
    switch (p) {
        case pattern::lcg:             return "lcg";
        case pattern::transpose_probe: return "transpose_probe";
        case pattern::extremes:        return "extremes";
    }
    return "?";
}

// act[m][k] in GGML x storage order, wt[n][k] in GGML w storage order.
void make_inputs(pattern p, int k, uint32_t seed, std::vector<int8_t> & act, std::vector<int8_t> & wt) {
    act.assign(M * k, 0);
    wt.assign(N * k, 0);
    switch (p) {
        case pattern::lcg:
            {
                uint32_t s    = seed;
                auto     next = [&s] {
                    s = s * 1664525u + 1013904223u;
                    return static_cast<int8_t>(s >> 24);
                };
                for (int8_t & v : act) {
                    v = next();
                }
                for (int8_t & v : wt) {
                    v = next();
                }
                act[0] = -128; // both ends of the signed range at least once
                wt[0]  = -128;
                act[1] = 127;
                wt[1]  = -128;
                break;
            }
        case pattern::transpose_probe:
            // Activations distinct per position; weight row n holds +1 and -2 at
            // two n-dependent K positions ((3n+1)%K != (5n+2)%K for even K), so
            // y[m][n] = x[m][(3n+1)%K] - 2 x[m][(5n+2)%K].
            for (int m = 0; m < M; ++m) {
                for (int kk = 0; kk < k; ++kk) {
                    act[m * k + kk] = static_cast<int8_t>((m * k + kk) % 251 - 125);
                }
            }
            for (int n = 0; n < N; ++n) {
                wt[n * k + (3 * n + 1) % k] = 1;
                wt[n * k + (5 * n + 2) % k] = -2;
            }
            break;
        case pattern::extremes:
            for (int m = 0; m < M; ++m) {
                for (int kk = 0; kk < k; ++kk) {
                    act[m * k + kk] = (m + kk) % 2 ? -128 : 127;
                }
            }
            for (int n = 0; n < N; ++n) {
                for (int kk = 0; kk < k; ++kk) {
                    wt[n * k + kk] = n % 3 == 0 ? -128 : ((n + kk) % 2 ? 127 : -128);
                }
            }
            break;
    }
}

// y[m][n] = sum_k act[m][k] * wt[n][k] in int64: the GGML MUL_MAT definition.
std::vector<int64_t> golden(int k, const std::vector<int8_t> & act, const std::vector<int8_t> & wt) {
    std::vector<int64_t> y(M * N, 0);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int kk = 0; kk < k; ++kk) {
                y[m * N + n] += static_cast<int64_t>(act[m * k + kk]) * wt[n * k + kk];
            }
        }
    }
    return y;
}

// Markers for one hardware GEMM: from the backend counters and the output.
bool report(const std::string & name, ggml_backend_t potal, ggml_status status,
            const ggml_backend_potal_stats & before, const ggml_backend_potal_stats & after,
            const std::vector<float> & out, const std::vector<int64_t> & want, float scale) {
    bool ok = status == GGML_STATUS_SUCCESS;
    if (!ok) {
        fail(name + " graph compute status=" + std::to_string(static_cast<int>(status)));
    }
    if (ggml_backend_is_potal(potal)) {
        std::printf("GGML_BACKEND=%s case=%s\n", ggml_backend_name(potal), name.c_str());
    }

    const uint64_t nodes     = after.mul_mat_nodes - before.mul_mat_nodes;
    const uint64_t requests  = after.hw_requests - before.hw_requests;
    const uint64_t issued    = after.gemm_issued - before.gemm_issued;
    const uint64_t fenced    = after.gemm_fenced - before.gemm_fenced;
    const uint64_t completed = after.gemm_completed - before.gemm_completed;
    if (nodes == 1 && requests == 1 && issued == 1) {
        std::printf("DISPATCH=HARDWARE case=%s issuer_cpu=%d\n", name.c_str(), after.issuer_cpu);
        std::printf("POTAL_GEMM_ISSUED case=%s scale_slot=%u generation=%u issue_us=%.1f\n", name.c_str(),
                    after.last_scale_slot, after.last_generation, (after.issue_s - before.issue_s) * 1e6);
    } else {
        fail(name + " hardware dispatch: nodes=" + std::to_string(nodes) + " requests=" + std::to_string(requests) +
             " issued=" + std::to_string(issued));
        ok = false;
    }
    if (fenced == 1 && after.last_status == 0 /* POTAL_OK */) {
        std::printf("POTAL_FENCE_PASS case=%s fence_us=%.1f\n", name.c_str(), (after.fence_s - before.fence_s) * 1e6);
    } else {
        fail(name + " fence: fenced=" + std::to_string(fenced) + " last_status=" + std::to_string(after.last_status));
        ok = false;
    }
    if (completed != 1) {
        fail(name + " completed=" + std::to_string(completed));
        ok = false;
    }

    int mismatches = 0;
    int unwritten  = 0;
    int nans       = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = static_cast<float>(want[i]) * scale;
        if (out[i] == expected) {
            continue;
        }
        unwritten += out[i] == SENTINEL;
        nans += std::isnan(out[i]) ? 1 : 0;
        if (mismatches++ < 4) {
            std::printf("  %s y[m=%zu][n=%zu] expected=%.1f actual=%.1f\n", name.c_str(), i / N, i % N, expected,
                        out[i]);
        }
    }
    if (mismatches == 0 && status == GGML_STATUS_SUCCESS) {
        std::printf("GGML_MUL_MAT_NUMERICAL_PASS case=%s elements=%zu\n", name.c_str(), out.size());
    } else {
        fail(name + " numerical: mismatches=" + std::to_string(mismatches) + " unwritten=" +
             std::to_string(unwritten) + " nan=" + std::to_string(nans));
        ok = false;
    }
    std::printf("[POTAL_SMOKE] case=%s pack_us=%.1f stage_us=%.1f scale_us=%.1f issue_us=%.1f fence_us=%.1f "
                "unpack_us=%.1f (guest CLOCK_MONOTONIC, not FPGA cycles)\n",
                name.c_str(), (after.pack_s - before.pack_s) * 1e6, (after.stage_s - before.stage_s) * 1e6,
                (after.scale_s - before.scale_s) * 1e6, (after.issue_s - before.issue_s) * 1e6,
                (after.fence_s - before.fence_s) * 1e6, (after.unpack_s - before.unpack_s) * 1e6);
    return ok;
}

// ggml_backend_graph_compute() of y = ggml_mul_mat(w, x) on the POTAL backend.
void run_direct(ggml_backend_t potal, int k, pattern p, uint32_t seed) {
    const std::string name = "k" + std::to_string(k) + "/" + pattern_name(p);
    std::printf("[POTAL_SMOKE] case=%s M=%d N=%d K=%d\n", name.c_str(), M, N, k);
    std::vector<int8_t> act;
    std::vector<int8_t> wt;
    make_inputs(p, k, seed, act, wt);
    const std::vector<int64_t> want = golden(k, act, wt);

    const ggml_init_params params = { ggml_tensor_overhead() * 3 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr       ctx(ggml_init(params));
    ggml_tensor *          w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, k, N);
    ggml_tensor *          x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, k, M);
    ggml_tensor *          y = ggml_mul_mat(ctx.get(), w, x);
    ggml_set_name(w, "w");
    ggml_set_name(x, "x");
    ggml_set_name(y, "y");
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), potal));
    if (!buffer) {
        fail(name + " tensor allocation");
        return;
    }
    std::vector<float> out(M * N, SENTINEL);
    ggml_backend_tensor_set(w, wt.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(y, out.data(), 0, ggml_nbytes(y));

    const ggml_backend_potal_stats before = stats_of(potal);
    const ggml_status              status = ggml_backend_graph_compute(potal, gf);
    const ggml_backend_potal_stats after  = stats_of(potal);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    g_gemms++;
    report(name, potal, status, before, after, out, want, 1.0f);
}

// POTAL + CPU scheduler: MUL_MAT on POTAL (hardware), ggml_scale(0.5) on CPU.
void run_sched(ggml_backend_t potal, ggml_backend_t cpu, int k, uint32_t seed) {
    const std::string name = "sched/k" + std::to_string(k);
    std::printf("[POTAL_SMOKE] case=%s M=%d N=%d K=%d\n", name.c_str(), M, N, k);
    std::vector<int8_t> act;
    std::vector<int8_t> wt;
    make_inputs(pattern::lcg, k, seed, act, wt);
    const std::vector<int64_t> want = golden(k, act, wt);

    const ggml_init_params in_params = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context_ptr       inputs(ggml_init(in_params));
    ggml_tensor *          w = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I8, k, N);
    ggml_tensor *          x = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I8, k, M);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu));
    if (!buffer) {
        fail(name + " input allocation");
        return;
    }
    ggml_backend_tensor_set(w, wt.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act.data(), 0, ggml_nbytes(x));

    const ggml_init_params graph_params = { ggml_tensor_overhead() * 2 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr       graph(ggml_init(graph_params));
    ggml_tensor *          y = ggml_mul_mat(graph.get(), w, x);
    ggml_tensor *          z = ggml_scale(graph.get(), y, 0.5f);
    ggml_cgraph *          gf = ggml_new_graph(graph.get());
    ggml_build_forward_expand(gf, z);

    ggml_backend_t         backends[2] = { potal, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        fail(name + " ggml_backend_sched_alloc_graph");
        return;
    }
    ggml_backend_t on_y = ggml_backend_sched_get_tensor_backend(sched.get(), y);
    ggml_backend_t on_z = ggml_backend_sched_get_tensor_backend(sched.get(), z);
    std::printf("SCHED_ASSIGN case=%s mul_mat=%s scale=%s\n", name.c_str(), on_y ? ggml_backend_name(on_y) : "none",
                on_z ? ggml_backend_name(on_z) : "none");
    if (on_y != potal || on_z != cpu) {
        fail(name + " scheduler placement");
        return;
    }

    const ggml_backend_potal_stats before = stats_of(potal);
    const ggml_status              status = ggml_backend_sched_graph_compute(sched.get(), gf);
    const ggml_backend_potal_stats after  = stats_of(potal);
    std::vector<float>             out(M * N);
    ggml_backend_tensor_get(z, out.data(), 0, ggml_nbytes(z));
    g_gemms++;
    if (report(name, potal, status, before, after, out, want, 0.5f)) {
        std::printf("GGML_SCHED_POTAL_CPU_PASS case=%s\n", name.c_str());
    }
}

void usage(const char * argv0) {
    std::printf("usage: %s [--case=k16|k32|sched|all] [--verbose]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string which = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--case=", 7) == 0) {
            which = argv[i] + 7;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (which != "k16" && which != "k32" && which != "sched" && which != "all") {
        usage(argv[0]);
        return 2;
    }

    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_log_set(log_to_stdout, nullptr);

    const long        main_tid        = self_tid();
    const std::string affinity_before = self_affinity();
    std::printf("POTAL_GGML_SMOKE starting case=%s pid=%d main_tid=%ld main_affinity=%s\n", which.c_str(),
                (int) getpid(), main_tid, affinity_before.c_str());

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("POTAL");
    if (dev == nullptr) {
        std::printf("FAIL POTAL device not registered (build with GGML_POTAL=ON)\nPOTAL_GGML_SMOKE FAIL\n");
        return 2;
    }
    std::printf("[POTAL_SMOKE] device=%s description=\"%s\"\n", ggml_backend_dev_name(dev),
                ggml_backend_dev_description(dev));
    ggml_backend_ptr potal(ggml_backend_dev_init(dev, nullptr));
    if (!potal || !ggml_backend_is_potal(potal.get())) {
        std::printf("FAIL POTAL backend init\nPOTAL_GGML_SMOKE FAIL\n");
        return 2;
    }

    uint32_t      seed       = 1;
    const pattern patterns[] = { pattern::lcg, pattern::transpose_probe, pattern::extremes };
    for (int k : { 16, 32 }) {
        if (which == "all" || which == "k" + std::to_string(k)) {
            for (pattern p : patterns) {
                run_direct(potal.get(), k, p, seed++);
            }
        }
    }
    if (which == "all" || which == "sched") {
        ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (!cpu) {
            fail("CPU backend init");
        } else {
            run_sched(potal.get(), cpu.get(), 32, seed++);
        }
    }

    const ggml_backend_potal_stats st = stats_of(potal.get());
    std::printf("[POTAL_SMOKE] hw_gemm_count=%llu expected=%d failed=%llu requests=%llu\n",
                (unsigned long long) st.gemm_completed, g_gemms, (unsigned long long) st.gemm_failed,
                (unsigned long long) st.hw_requests);
    if (st.gemm_completed != static_cast<uint64_t>(g_gemms) || st.gemm_failed != 0) {
        fail("hardware GEMM count");
    }

    // The backend pinned its issuer thread to the CPU of hart 0; confirm with
    // /proc/cpuinfo here and check that the main thread kept its affinity.
    const int hart = hart_of_cpu(st.issuer_cpu);
    std::printf("POTAL_ISSUER ready=%d cpu=%d hart=%d tid=%ld main_tid=%ld\n", st.issuer_ready ? 1 : 0,
                st.issuer_cpu, hart, st.issuer_tid, main_tid);
    if (!st.issuer_ready || hart != 0 || st.issuer_tid == main_tid) {
        fail("issuer thread not on hart 0");
    }
    const std::string affinity_after = self_affinity();
    std::printf("[POTAL_SMOKE] main_affinity before=%s after=%s\n", affinity_before.c_str(), affinity_after.c_str());
    if (affinity_after != affinity_before) {
        fail("main thread affinity changed");
    }

    if (g_failures != 0) {
        std::printf("POTAL_GGML_SMOKE FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("POTAL_GGML_SMOKE PASS\n");
    return 0;
}
