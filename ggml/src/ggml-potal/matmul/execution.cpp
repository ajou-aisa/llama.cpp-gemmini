#include "execution.hpp"

#include "ggml-impl.h"
#include "log.hpp"
#include "stripe-api.hpp"

#include "potal_hw_params.h"

#include <gemmini/trace-context.hpp>

#include <algorithm>
#include <cstring>

#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

#if defined(__riscv)
#    if !defined(__linux__)
#        error "ggml-potal needs RISC-V Linux: its issuer thread pins itself to the hart-0 CPU"
#    endif
#    include <sys/syscall.h>
#    include <unistd.h>

#    include <cerrno>
#    include <chrono>
#    include <condition_variable>
#    include <cstdio>
#    include <functional>
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

// potal_tiled_matmul_auto() with the only request the PoTal path implements:
// WS, int8 A/B, int32 C, no D, no activation, identity scale factors.
tiled_report tiled_matmul(const tiled_job & job, potal_ws_session & session) {
    tiled_report r;
    const potal_ws_report before = session.report;
    session.report.status        = POTAL_OK;
    session.report.stage         = nullptr;
    const potal_scale_operand scales = { job.carriers, job.cstride, job.block_k };
    r.status = potal_tiled_matmul_auto(job.m, job.n, job.k,
                                       job.a, job.b, nullptr, job.c,
                                       job.lda, job.ldb, 0, job.ldc,
                                       POTAL_MVIN_SCALE_IDENTITY, POTAL_MVIN_SCALE_IDENTITY, 0,
                                       POTAL_NO_ACTIVATION, POTAL_ACC_SCALE_IDENTITY, 0,
                                       /* repeating_bias */ false,
                                       /* transpose_A */ false, /* transpose_B */ false,
                                       /* full_C */ true, /* low_D */ false,
                                       /* weightA */ 0,
                                       POTAL_WS,
                                       scales, session, &r.plan);
    r.work  = work_since(session.report, before);
    r.stage = session.report.stage;
    r.fatal = session.report.fatal;
    r.ok    = r.status == POTAL_OK;
    return r;
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

// Work run on the issuer thread. Returns true when it left the accelerator in
// an unknown state; `why` and `status` then describe it and the issuer
// disables itself for the rest of the process.
using issuer_task = std::function<bool(std::string & why, potal_status & status)>;

class issuer {
  public:
    void start();
    // false: the issuer is disabled and the task did not run.
    bool         run(const issuer_task & task);
    issuer_info  info();
    const char * failure_text(potal_status * status); // stable once the issuer is disabled

    // One request at a time: run() and a stripe stream hold this lock from
    // post() until wait_done() returned.
    std::unique_lock<std::mutex> acquire() { return std::unique_lock<std::mutex>(submit_mutex); }
    // With acquire() held: waits for the bring-up, then hands `task` over and
    // returns; the issuer keeps it alive until it ran. false: disabled.
    bool post(std::shared_ptr<const issuer_task> task);
    // With acquire() held, after a successful post(): waits until the task returned.
    void wait_done();

    // Issuer thread only (called from tasks).
    gemm_report    execute(const gemm_job & job);
    tiled_report   execute_tiled(const tiled_job & job);
    planned_result execute_planned(const potal_stripe_gemm_desc & desc);
    potal_paired_status execute_paired(const potal_paired_request & req, potal_paired_report & report);

  private:
    void main_loop();
    bool bring_up();

    std::mutex              submit_mutex; // one request at a time
    std::mutex              mutex;        // guards the members below
    std::condition_variable cv;

    enum class state { starting, ready, failed };
    state                              st      = state::starting;
    std::shared_ptr<const issuer_task> pending; // the request being run, owned
    bool                               done    = false;
    std::string                        failure; // set once, when st becomes failed
    potal_status                       failure_status = POTAL_OK;

    // Owned by the issuer thread; published when st leaves `starting`.
    int cpu = -1;
    long tid = -1;
    // Scale generations and work counters of every PoTal command of the
    // process: the I8 GEMM, the FULL tiled jobs and the planned stripe GEMMs.
    potal_stripe_session session = { { &potal_hw_ops(), 0, {} } };
};

void issuer::start() {
    try {
        std::thread(&issuer::main_loop, this).detach();
    } catch (const std::system_error & e) {
        std::lock_guard<std::mutex> lock(mutex);
        failure        = std::string("cannot start the issuer thread: ") + e.what();
        failure_status = POTAL_ERR_NO_HARDWARE;
        st             = state::failed;
        log::failure("potal.issuer", "[POTAL_ISSUER] FAIL %s", failure.c_str());
    }
}

bool issuer::run(const issuer_task & task) {
    std::unique_lock<std::mutex> one_request = acquire();
    if (!post(std::make_shared<const issuer_task>(task))) {
        return false;
    }
    wait_done();
    return true;
}

bool issuer::post(std::shared_ptr<const issuer_task> task) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return st != state::starting; });
    if (st == state::failed) {
        return false;
    }
    pending = std::move(task);
    done    = false;
    cv.notify_all();
    return true;
}

void issuer::wait_done() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return done; });
    pending.reset();
}

const char * issuer::failure_text(potal_status * status) {
    std::lock_guard<std::mutex> lock(mutex);
    *status = failure_status;
    return st == state::failed ? failure.c_str() : nullptr;
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
        const std::shared_ptr<const issuer_task> task = pending; // strong reference while it runs
        lock.unlock();
        std::string  why;
        potal_status status = POTAL_OK;
        const bool   fatal  = (*task)(why, status);
        lock.lock();
        if (fatal) {
            failure        = why;
            failure_status = status;
            st             = state::failed;
            log::failure("potal.issuer", "[POTAL_ISSUER] disabled for this process after %s", failure.c_str());
        }
        done = true;
        cv.notify_all();
    }
}

std::string failure_of(const char * stage, potal_status status) {
    return status == POTAL_OK ? std::string(stage) : std::string(stage) + ": POTAL_" + potal_status_name(status);
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
        log::failure("potal.issuer", "[POTAL_ISSUER] FAIL tid=%ld: %s; no PoTal command issued", tid, err.c_str());
        return false;
    }
    cpu = hart0_cpu;
    log::debug("potal.issuer", "[POTAL_ISSUER] tid=%ld cpu=%d hart=0 affinity={%d}", tid, cpu, cpu);

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
        log::failure("potal.tlb", "[POTAL_TLB] FAIL %s", failure.c_str());
        return false;
    }
    log::debug("potal.tlb", "[POTAL_TLB] flush PASS (FLUSH issued, FENCE=POTAL_OK)");
    return true;
}

// Identity-carrier GEMM: Scale WRITE -> LOOP_WS -> FENCE -> C readback -> Scale RELEASE.
gemm_report issuer::execute(const gemm_job & job) {
    gemm_report r;
    r.generation          = session.ws.generation % POTAL_GENERATION_MAX + 1; // 1..255
    session.ws.generation = r.generation;
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
    log::debug("potal.matmul",
               "[POTAL_MATMUL] scale_base=%u generation=%u carrier=x1 columns=%u passes_per_block=%u "
               "scale_write=POTAL_OK",
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
    log::debug("potal.matmul", "[POTAL_MATMUL] dispatch=HARDWARE issue_status=POTAL_OK K=%u issue_us=%.1f", job.k,
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
    log::debug("potal.matmul", "[POTAL_MATMUL] fence_status=POTAL_OK fence_us=%.1f", r.fence_s * 1e6);

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
    log::debug("potal.matmul", "[POTAL_MATMUL] scale_release=POTAL_OK");

    const long unwritten = static_cast<long>(std::count(job.c, job.c + DIM * DIM, C_UNWRITTEN));
    if (unwritten != 0) {
        log::failure("potal.matmul", "[POTAL_MATMUL] %ld of %u C elements kept the pre-issue sentinel after FENCE",
                     unwritten, DIM * DIM);
        r.stage = "c_unwritten";
        r.fatal = true;
        return r;
    }
    r.ok = true;
    return r;
}

// Tiled GEMM on the issuer: every invocation is Scale WRITE -> LOOP_WS -> FENCE -> RELEASE.
tiled_report issuer::execute_tiled(const tiled_job & job) {
    ggml::gemmini::trace::ScopedContext trace_scope(job.trace);
    tiled_report                        r;
    if (sched_getcpu() != cpu) {
        r.stage = "affinity"; // custom-3 would trap off hart 0
        r.fatal = true;
        return r;
    }
    // The staging buffers of this job are freshly allocated: their virtual
    // pages may have backed other physical pages in an earlier job. Drop every
    // accelerator translation before the first DMA. Every earlier job ended
    // with a FENCE, so no DMA is in flight while the TLB is flushed.
    r.status = potal_tlb_flush();
    if (r.status == POTAL_OK) {
        r.status = potal_fence();
    }
    if (r.status != POTAL_OK) {
        r.stage = "tlb_flush";
        r.fatal = true;
        return r;
    }
    session.ws.report.tlb_flushes++;
    r = tiled_matmul(job, session.ws);
    r.work.tlb_flushes = 1;
    return r;
}

// One planned stripe GEMM on the issuer: the hart-0 affinity check, then
// run_planned_stripe_gemm() on the issuer's session. A stripe GEMM may read
// buffers allocated since the previous GEMM (the run's slots, a residual's
// operands), so that step drops every accelerator translation before the
// GEMM's first DMA, and only for a descriptor its preflight accepted.
planned_result issuer::execute_planned(const potal_stripe_gemm_desc & desc) {
    if (sched_getcpu() != cpu) {
        planned_result r;
        r.stage = "affinity"; // custom-3 would trap off hart 0
        r.fatal = true;
        return r;
    }
    return run_planned_stripe_gemm(session, desc);
}

// One paired stripe on the issuer: the hart-0 affinity check, then its own paired
// session (potal_paired_run_session) with the issuer's scale generation counter, so
// the planned GEMMs and the paired stripes of the process share one generation
// sequence. Every earlier GEMM of the issuer ended with a FENCE, so the paired
// session opens drained; it flushes the TLB before its first DMA.
potal_paired_status issuer::execute_paired(const potal_paired_request & req, potal_paired_report & report) {
    // custom-3 would trap off hart 0: refused (fatal) inside the owner step.
    return run_paired_on_owner(req, &session.ws.generation, sched_getcpu() == cpu, report);
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
    issuer &    iss = get_issuer();
    gemm_report r;
    const bool  ran = iss.run([&](std::string & why, potal_status & status) {
        r = iss.execute(job);
        if (r.fatal) {
            why    = failure_of(r.stage, r.status);
            status = r.status;
        }
        return r.fatal;
    });
    if (!ran) {
        r        = gemm_report{};
        r.fatal  = true;
        r.stage  = "issuer_disabled";
        r.detail = iss.failure_text(&r.status);
    } else if (r.fatal) {
        potal_status ignored;
        r.detail = iss.failure_text(&ignored);
    }
    return r;
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

tiled_report run_tiled(const tiled_job & job) {
#if defined(__riscv)
    issuer &     iss = get_issuer();
    tiled_report r;
    const bool   ran = iss.run([&](std::string & why, potal_status & status) {
        r = iss.execute_tiled(job);
        if (r.fatal) {
            why    = failure_of(r.stage, r.status);
            status = r.status;
        }
        return r.fatal;
    });
    if (!ran) {
        r        = tiled_report{};
        r.fatal  = true;
        r.stage  = "issuer_disabled";
        r.detail = iss.failure_text(&r.status);
    } else if (r.fatal) {
        potal_status ignored;
        r.detail = iss.failure_text(&ignored);
    }
    return r;
#else
    // No accelerator: validation runs, the first hardware command reports
    // POTAL_ERR_NO_HARDWARE. A fresh session keeps host calls independent.
    potal_ws_session session = { &potal_hw_ops(), 0, {} };
    return tiled_matmul(job, session);
#endif
}

tiled_report run_tiled_in_session(const tiled_job & job, potal_ws_session & session) {
    return tiled_matmul(job, session);
}

issuer_info get_issuer_info() {
#if defined(__riscv)
    std::lock_guard<std::mutex> lock(g_issuer_mutex);
    return g_issuer != nullptr ? g_issuer->info() : issuer_info{};
#else
    return {};
#endif
}

// ---- stripe pipeline consumer ------------------------------------------------

bool stripe_owner::pin_producer(bool one_cpu, std::string & why) {
    (void) one_cpu;
    (void) why;
    return true;
}

void stripe_owner::unpin_producer() {}

struct producer_affinity::saved {
#if defined(__linux__)
    cpu_set_t mask; // the caller's mask before pin()
#endif
    bool pinned = false;
    int  cpu    = -1;
};

producer_affinity::producer_affinity() : saved_(std::make_unique<saved>()) {}

producer_affinity::~producer_affinity() {
    std::string why;
    if (!restore(why)) {
        log::failure("potal.issuer", "[POTAL_STRIPE] producer affinity not restored: %s", why.c_str());
    }
}

bool producer_affinity::pin(int exclude_cpu, bool one_cpu, std::string & why) {
#if defined(__linux__)
    if (saved_->pinned) {
        why = "producer already pinned";
        return false;
    }
    cpu_set_t current;
    CPU_ZERO(&current);
    int rc = pthread_getaffinity_np(pthread_self(), sizeof(current), &current);
    if (rc != 0) {
        why = std::string("reading the producer affinity failed: ") + std::strerror(rc);
        return false;
    }
    cpu_set_t narrowed;
    CPU_ZERO(&narrowed);
    int chosen = -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &current) && c != exclude_cpu && (!one_cpu || chosen < 0)) {
            CPU_SET(c, &narrowed);
            chosen = chosen < 0 ? c : chosen;
        }
    }
    if (chosen < 0) {
        why = "the caller's affinity mask has no CPU besides the issuer's cpu " + std::to_string(exclude_cpu) +
              ": producer and issuer cannot run side by side";
        return false;
    }
    rc = pthread_setaffinity_np(pthread_self(), sizeof(narrowed), &narrowed);
    if (rc != 0) {
        why = "narrowing the producer affinity failed: " + std::string(std::strerror(rc));
        return false;
    }
    saved_->mask   = current;
    saved_->pinned = true;
    saved_->cpu    = one_cpu ? chosen : -1;
    cpu_set_t got;
    CPU_ZERO(&got);
    rc = pthread_getaffinity_np(pthread_self(), sizeof(got), &got);
    if (rc != 0 || !CPU_EQUAL(&got, &narrowed)) {
        std::string ignored;
        (void) restore(ignored);
        why = "producer affinity differs from the requested mask after narrowing";
        return false;
    }
    return true;
#else
    (void) exclude_cpu;
    (void) one_cpu;
    why = "producer placement needs Linux thread affinity";
    return false;
#endif
}

bool producer_affinity::restore(std::string & why) {
#if defined(__linux__)
    if (!saved_->pinned) {
        return true;
    }
    saved_->pinned = false;
    saved_->cpu    = -1;
    const int rc   = pthread_setaffinity_np(pthread_self(), sizeof(saved_->mask), &saved_->mask);
    if (rc != 0) {
        why = std::string("restoring the producer affinity failed: ") + std::strerror(rc);
        return false;
    }
#else
    (void) why;
#endif
    return true;
}

int producer_affinity::cpu() const {
    return saved_->cpu;
}

struct issuer_owner::state {
#if defined(__riscv)
    issuer *                     iss    = nullptr;
    std::unique_lock<std::mutex> hold;          // the issuer's one-request lock, start() .. join()
    bool                         posted = false; // a consumer was handed over and not joined
#endif
    producer_affinity            producer;
};

issuer_owner::issuer_owner() : st_(std::make_unique<state>()) {}

issuer_owner::~issuer_owner() {
    unpin_producer();
    join();
}

bool issuer_owner::start(stripe_consumer consume, std::string & why, potal_status & status) {
#if defined(__riscv)
    if (st_->hold.owns_lock()) {
        why    = "stripe stream already started";
        status = POTAL_ERR_BUSY;
        return false;
    }
    issuer & iss = get_issuer();
    st_->iss     = &iss;
    st_->hold    = iss.acquire();
    // The issuer owns the consumer (and through it the run) until it returns.
    auto task = std::make_shared<const issuer_task>(
        [&iss, consume = std::move(consume)](std::string & task_why, potal_status & task_status) {
            owner_runners runners;
            runners.planned = [&iss](const potal_stripe_gemm_desc & desc) { return iss.execute_planned(desc); };
            runners.paired  = [&iss](const potal_paired_request & r, potal_paired_report & rep) {
                return iss.execute_paired(r, rep);
            };
            const consume_outcome o = consume(runners);
            if (o.fatal) {
                task_why    = failure_of(o.stage != nullptr ? o.stage : "stripe_pipeline", o.status);
                task_status = o.status;
            }
            return o.fatal;
        });
    if (!iss.post(std::move(task))) {
        const char * text = iss.failure_text(&status);
        why               = std::string("issuer disabled: ") + (text != nullptr ? text : "-");
        st_->hold.unlock();
        return false;
    }
    st_->posted = true;
    return true;
#else
    (void) consume;
    why    = "no PoTal hardware: this build is not for RISC-V";
    status = POTAL_ERR_NO_HARDWARE;
    return false;
#endif
}

void issuer_owner::join() {
#if defined(__riscv)
    if (st_->posted) {
        st_->iss->wait_done();
        st_->posted = false;
    }
    if (st_->hold.owns_lock()) {
        st_->hold.unlock();
    }
#endif
}

bool issuer_owner::pin_producer(bool one_cpu, std::string & why) {
#if defined(__riscv)
    const issuer_info i = st_->posted ? st_->iss->info() : issuer_info{};
    if (!i.ready || i.cpu < 0) {
        why = "the issuer is not running a stripe stream";
        return false;
    }
    return st_->producer.pin(i.cpu, one_cpu, why);
#else
    (void) one_cpu;
    why = "no issuer: this build is not for RISC-V";
    return false;
#endif
}

void issuer_owner::unpin_producer() {
    std::string why;
    if (!st_->producer.restore(why)) {
        log::failure("potal.issuer", "[POTAL_STRIPE] producer affinity not restored: %s", why.c_str());
    }
}

} // namespace ggml::potal
