#include "execution.hpp"

#include "ggml-impl.h"

#include "potal_hw_params.h"

#include <algorithm>
#include <cstring>

#if defined(__riscv)
#    if !defined(__linux__)
#        error "ggml-potal needs RISC-V Linux: its issuer thread pins itself to the hart-0 CPU"
#    endif
#    include <pthread.h>
#    include <sched.h>
#    include <sys/syscall.h>
#    include <unistd.h>

#    include <cerrno>
#    include <chrono>
#    include <condition_variable>
#    include <cstdio>
#    include <mutex>
#    include <string>
#    include <system_error>
#    include <thread>
#endif

static_assert(POTAL_HW_DIM == 16, "ggml-potal offloads the validated DIM16 GEMM only");

namespace ggml::potal {

namespace {

constexpr uint32_t DIM     = POTAL_HW_DIM;
constexpr uint32_t K_STAGE = 2 * DIM; // A row stride of potal-api-smoke and potal-multicore-smoke

// Scale slot of every backend GEMM. potal-api-smoke uses slots 0..41 and
// potal-multicore-smoke 128..129, so processes run back to back never share it.
constexpr uint32_t SCALE_SLOT = 192;

// DMA staging buffers for the whole process, in the validated smoke-test
// layout. Only the issuer thread touches them, and only while no GEMM is in
// flight; they are never freed, so an unfinished GEMM cannot hit reused memory.
alignas(POTAL_HW_MAX_BYTES) int8_t  stage_a[DIM * K_STAGE];
alignas(POTAL_HW_MAX_BYTES) int8_t  stage_b[K_STAGE * DIM];
alignas(POTAL_HW_MAX_BYTES) int32_t stage_c[DIM * DIM];

potal_gemm_ws make_request(uint32_t k, uint32_t generation) {
    potal_gemm_ws g    = {};
    g.a                = stage_a;
    g.b                = stage_b;
    g.c                = stage_c;
    g.lda              = K_STAGE;
    g.ldb              = DIM;
    g.ldc              = DIM;
    g.m                = DIM;
    g.n                = DIM;
    g.k                = k;
    g.scale_base       = SCALE_SLOT;
    g.scale_generation = generation;
    g.passes_per_block = k / DIM; // one scale block: the x1 carriers cover all of K
    g.flags            = 0;
    return g;
}

#if defined(__riscv)

using clock_type = std::chrono::steady_clock;

constexpr uint32_t CARRIER_X1 = 0; // shift carrier 0: the SCU multiplies by 1

// C is pre-filled with a value no x1 GEMM can produce (|sum| <= 32 * 128 * 128),
// so an element the DMA never wrote (e.g. through a stale accelerator TLB) is caught.
constexpr int32_t C_UNWRITTEN = INT32_MIN;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

// Linux CPU whose /proc/cpuinfo "hart" is 0, where the PoTal RoCC is attached.
bool find_hart0_cpu(int & cpu, std::string & err) {
    FILE * f = std::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        err = std::string("cannot open /proc/cpuinfo: ") + std::strerror(errno);
        return false;
    }
    char line[256];
    int  processor = -1;
    bool ok        = true;
    cpu            = -1;
    while (ok && std::fgets(line, sizeof(line), f) != nullptr) {
        int v;
        if (std::sscanf(line, "processor : %d", &v) == 1) {
            processor = v;
        } else if (std::sscanf(line, "hart : %d", &v) == 1 && v == 0) {
            ok  = processor >= 0 && cpu == -1;
            cpu = processor;
        }
    }
    std::fclose(f);
    if (!ok) {
        err = "malformed /proc/cpuinfo: hart 0 listed twice or before any processor line";
        return false;
    }
    if (cpu < 0) {
        err = "no Linux CPU reports hart 0 in /proc/cpuinfo";
        return false;
    }
    return true;
}

// Pin the calling thread to exactly `cpu`, then confirm the kernel's view:
// affinity == {cpu} and sched_getcpu() == cpu (as potal-multicore-smoke).
bool pin_self(int cpu, std::string & err) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc == 0) {
        cpu_set_t got;
        CPU_ZERO(&got);
        rc = pthread_getaffinity_np(pthread_self(), sizeof(got), &got);
        if (rc == 0 && (CPU_COUNT(&got) != 1 || !CPU_ISSET(cpu, &got))) {
            err = "affinity is not {" + std::to_string(cpu) + "} after pinning";
            return false;
        }
    }
    if (rc != 0) {
        err = "pinning to cpu " + std::to_string(cpu) + " failed: " + std::strerror(rc);
        return false;
    }
    const int running = sched_getcpu();
    if (running != cpu) {
        err = "sched_getcpu()=" + std::to_string(running) + " after pinning to cpu " + std::to_string(cpu);
        return false;
    }
    return true;
}

// Cleanup after a failure that happened before any GEMM was issued; the
// columns are idle, so their release cannot race with the accelerator.
void release_columns(uint32_t slot, uint32_t generation, uint32_t columns) {
    for (uint32_t col = 0; col < columns; ++col) {
        (void) potal_scale_release(slot, generation, col);
    }
}

class issuer {
  public:
    void start();
    gemm_report run(const gemm_job & job);
    issuer_info info();

  private:
    void        main_loop();
    bool        bring_up();
    gemm_report execute(const gemm_job & job);

    std::mutex              submit_mutex; // one request at a time
    std::mutex              mutex;        // guards the members below
    std::condition_variable cv;

    enum class state { starting, ready, failed };
    state            st      = state::starting;
    const gemm_job * pending = nullptr;
    bool             done    = false;
    gemm_report      result;
    std::string      failure; // set once, when st becomes failed
    potal_status     failure_status = POTAL_OK;

    // Owned by the issuer thread; published when st leaves `starting`.
    int      cpu        = -1;
    long     tid        = -1;
    uint32_t generation = 0;
};

void issuer::start() {
    try {
        std::thread(&issuer::main_loop, this).detach();
    } catch (const std::system_error & e) {
        std::lock_guard<std::mutex> lock(mutex);
        failure        = std::string("cannot start the issuer thread: ") + e.what();
        failure_status = POTAL_ERR_NO_HARDWARE;
        st             = state::failed;
        GGML_LOG_ERROR("[POTAL_ISSUER] FAIL %s\n", failure.c_str());
    }
}

gemm_report issuer::run(const gemm_job & job) {
    std::lock_guard<std::mutex>  one_request(submit_mutex);
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return st != state::starting; });
    if (st == state::failed) {
        gemm_report r;
        r.fatal  = true;
        r.status = failure_status;
        r.stage  = "issuer_disabled";
        r.detail = failure.c_str();
        return r;
    }
    pending = &job;
    done    = false;
    cv.notify_all();
    cv.wait(lock, [this] { return done; });
    pending = nullptr;
    return result;
}

issuer_info issuer::info() {
    std::lock_guard<std::mutex> lock(mutex);
    issuer_info i;
    if (st != state::starting) {
        i.ready = st == state::ready;
        i.cpu   = cpu;
        i.tid   = tid;
    }
    return i;
}

void issuer::main_loop() {
    const bool ready = bring_up();

    std::unique_lock<std::mutex> lock(mutex);
    st = ready ? state::ready : state::failed;
    cv.notify_all();
    while (st == state::ready) {
        cv.wait(lock, [this] { return pending != nullptr && !done; });
        const gemm_job * job = pending;
        lock.unlock();
        gemm_report r = execute(*job);
        lock.lock();
        if (r.fatal) {
            failure = r.status == POTAL_OK ? std::string(r.stage) :
                                             std::string(r.stage) + ": POTAL_" + potal_status_name(r.status);
            failure_status = r.status;
            st             = state::failed;
            r.detail       = failure.c_str();
            GGML_LOG_ERROR("[POTAL_ISSUER] disabled for this process after %s\n", failure.c_str());
        }
        result = r;
        done   = true;
        cv.notify_all();
    }
}

// Affinity before the first RoCC command, then the per-process TLB flush.
bool issuer::bring_up() {
    tid = static_cast<long>(syscall(SYS_gettid));
    (void) pthread_setname_np(pthread_self(), "potal-issuer");

    std::string err;
    int         hart0_cpu = -1;
    if (!find_hart0_cpu(hart0_cpu, err) || !pin_self(hart0_cpu, err)) {
        failure        = err;
        failure_status = POTAL_ERR_NO_HARDWARE;
        GGML_LOG_ERROR("[POTAL_ISSUER] FAIL tid=%ld: %s; no PoTal command issued\n", tid, err.c_str());
        return false;
    }
    cpu = hart0_cpu;
    GGML_LOG_INFO("[POTAL_ISSUER] tid=%ld cpu=%d hart=0 affinity={%d}\n", tid, cpu, cpu);

    // Drop accelerator TLB entries an earlier process left behind. FLUSH has
    // no response; the FENCE behind it completing means it was applied.
    const char * step = "tlb_flush";
    potal_status s    = potal_tlb_flush();
    if (s == POTAL_OK) {
        step = "tlb_flush_fence";
        s    = potal_fence();
    }
    if (s != POTAL_OK) {
        failure        = std::string(step) + ": POTAL_" + potal_status_name(s);
        failure_status = s;
        GGML_LOG_ERROR("[POTAL_TLB] FAIL %s\n", failure.c_str());
        return false;
    }
    GGML_LOG_INFO("[POTAL_TLB] flush PASS (FLUSH issued, FENCE=POTAL_OK)\n");
    return true;
}

// Identity-carrier GEMM: Scale WRITE -> LOOP_WS -> FENCE -> C readback -> Scale RELEASE.
gemm_report issuer::execute(const gemm_job & job) {
    gemm_report r;
    r.generation          = generation % POTAL_GENERATION_MAX + 1; // 1..255
    generation            = r.generation;
    const potal_gemm_ws g = make_request(job.k, r.generation);

    // Software checks first: nothing reaches the accelerator when they fail.
    r.status = potal_gemm_ws_scale_slot(&g, 0, 0, &r.scale_slot);
    if (r.status != POTAL_OK) {
        r.stage = "check";
        return r;
    }
    if (sched_getcpu() != cpu) {
        r.stage = "affinity"; // custom-3 would trap off hart 0
        r.fatal = true;
        return r;
    }

    auto t = clock_type::now();
    for (uint32_t m = 0; m < DIM; ++m) {
        std::memcpy(stage_a + m * K_STAGE, job.a + m * job.k, job.k);
    }
    std::memcpy(stage_b, job.b, job.k * DIM);
    std::fill(stage_c, stage_c + DIM * DIM, C_UNWRITTEN);
    r.stage_s += seconds_since(t);

    t                = clock_type::now();
    uint32_t written = 0;
    for (; written < DIM; ++written) {
        r.status = potal_scale_write(r.scale_slot, r.generation, written, CARRIER_X1);
        if (r.status != POTAL_OK) {
            break;
        }
    }
    r.scale_s += seconds_since(t);
    if (r.status != POTAL_OK) {
        release_columns(r.scale_slot, r.generation, written);
        r.stage = "scale_write";
        r.fatal = true;
        return r;
    }
    GGML_LOG_INFO("[POTAL_MATMUL] scale_base=%u generation=%u carrier=x1 columns=%u passes_per_block=%u "
                  "scale_write=POTAL_OK\n",
                  r.scale_slot, r.generation, DIM, g.passes_per_block);

    t         = clock_type::now();
    r.status  = potal_gemm_ws_issue(&g);
    r.issue_s = seconds_since(t);
    if (r.status != POTAL_OK) {
        // Refused before any GEMM command was sent (validated above, so ERR_BUSY).
        release_columns(r.scale_slot, r.generation, DIM);
        r.stage = "issue";
        r.fatal = true;
        return r;
    }
    r.issued = true;
    GGML_LOG_INFO("[POTAL_MATMUL] dispatch=HARDWARE issue_status=POTAL_OK K=%u issue_us=%.1f\n", job.k,
                  r.issue_s * 1e6);

    t         = clock_type::now();
    r.status  = potal_fence();
    r.fence_s = seconds_since(t);
    if (r.status != POTAL_OK) {
        // The GEMM may still run: leave the staging buffers and the scale slot alone.
        r.stage = "fence";
        r.fatal = true;
        return r;
    }
    r.fenced = true;
    GGML_LOG_INFO("[POTAL_MATMUL] fence_status=POTAL_OK fence_us=%.1f\n", r.fence_s * 1e6);

    t = clock_type::now();
    std::memcpy(job.c, stage_c, sizeof(stage_c));
    r.stage_s += seconds_since(t);

    t = clock_type::now();
    for (uint32_t col = 0; col < DIM && r.status == POTAL_OK; ++col) {
        r.status = potal_scale_release(r.scale_slot, r.generation, col);
    }
    r.scale_s += seconds_since(t);
    if (r.status != POTAL_OK) {
        r.stage = "scale_release";
        r.fatal = true;
        return r;
    }
    GGML_LOG_INFO("[POTAL_MATMUL] scale_release=POTAL_OK\n");

    const long unwritten = static_cast<long>(std::count(job.c, job.c + DIM * DIM, C_UNWRITTEN));
    if (unwritten != 0) {
        GGML_LOG_ERROR("[POTAL_MATMUL] %ld of %u C elements kept the pre-issue sentinel after FENCE\n", unwritten,
                       DIM * DIM);
        r.stage = "c_unwritten";
        r.fatal = true;
        return r;
    }
    r.ok = true;
    return r;
}

std::mutex g_issuer_mutex;
issuer *   g_issuer = nullptr; // created by the first request; leaked because its thread is detached

issuer & get_issuer() {
    std::lock_guard<std::mutex> lock(g_issuer_mutex);
    if (g_issuer == nullptr) {
        g_issuer = new issuer();
        g_issuer->start();
    }
    return *g_issuer;
}

#endif // __riscv

} // namespace

gemm_report run_gemm(const gemm_job & job) {
#if defined(__riscv)
    return get_issuer().run(job);
#else
    // No accelerator: the API validates the exact request the issuer would
    // send, then potal_gemm_ws_issue() reports POTAL_ERR_NO_HARDWARE.
    gemm_report         r;
    const potal_gemm_ws g = make_request(job.k, 1);
    r.status              = potal_gemm_ws_issue(&g);
    r.stage               = "issue";
    return r;
#endif
}

issuer_info get_issuer_info() {
#if defined(__riscv)
    std::lock_guard<std::mutex> lock(g_issuer_mutex);
    return g_issuer != nullptr ? g_issuer->info() : issuer_info{};
#else
    return {};
#endif
}

} // namespace ggml::potal
