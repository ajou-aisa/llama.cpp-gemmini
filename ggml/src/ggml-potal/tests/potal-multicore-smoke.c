/*
 * PoTal multicore smoke test (RISC-V Linux, DIM16 Sat32 SoC with one PoTal on hart 0).
 *
 * One issuer pthread owns the accelerator: it is pinned to the Linux CPU whose
 * /proc/cpuinfo "hart" is 0 and is the only thread that issues RoCC commands
 * (TLB FLUSH, SCALE_WRITE/RELEASE, GEMM, FENCE). CPU workers (pthreads, or an
 * OpenMP team with --mode=openmp) are each pinned to a different other CPU and
 * run a verifiable INT8 quantize + INT32 dot-product kernel while the issuer
 * runs DIM16 GEMMs.
 *
 * Evidence reported separately:
 *   cpu_multicore_verified  every worker ran on its own pinned non-PoTal CPU
 *                           (periodic sched_getcpu()) and its checksum matches
 *                           a serial recomputation
 *   potal_gemm_verified     every GEMM equals the int64 golden model
 *   overlap_observed        summed worker progress increased between the return
 *                           of GEMM issue and the return of its FENCE
 *
 * Options: --mode=pthread|openmp --workers=N --iterations=N --work-units=N --verbose
 * Worker count: --workers, else OMP_NUM_THREADS (openmp mode), else every
 * allowed online CPU except the PoTal CPU.
 * Exit status: 0 pass, 1 fail, 2 usage or environment error.
 * Log prefixes: [POTAL_MC] [POTAL_OMP] [POTAL_TLB] [POTAL_GEMM]; final line
 * "POTAL_MULTICORE_SMOKE PASS" or "POTAL_MULTICORE_SMOKE FAIL".
 */
#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "potal.h"
#include "potal_hw_params.h"

#if POTAL_HW_DIM != 16
#error "potal-multicore-smoke validates the DIM16 GEMM path only"
#endif

#define DIM          16
#define K_MAX        (2 * DIM)
#define SENTINEL     (-9999)
#define MAX_CPUS     64
#define SCALE_BASE   128u /* away from the slots potal-api-smoke uses */
#define UNIT_ELEMS   64   /* worker elements per work unit */
#define QBLOCK       32   /* quantization block (per-block absmax scale) */
#define CPU_SAMPLE   16   /* sched_getcpu() every CPU_SAMPLE units */

/* ---- Options and shared state ------------------------------------------- */

enum mode { MODE_PTHREAD, MODE_OPENMP };

static struct {
    enum mode mode;
    int       workers; /* 0 = derive */
    int       iterations;
    int       work_units;
    int       verbose;
} opt = { MODE_PTHREAD, 0, 4, 4000, 0 };

static int cpu_hart[MAX_CPUS]; /* -1 = unknown */
static int potal_cpu = -1;
static int worker_cpus[MAX_CPUS];
static int n_worker_cpus;
static int n_workers;

typedef struct worker_state {
    atomic_uint_fast64_t progress; /* completed work units */
    /* Written by the worker, read after join. */
    long     tid;
    int      omp_thread;           /* -1 in pthread mode */
    int      cpu;                  /* pinned CPU */
    int      ok;                   /* pinning verified */
    char     affinity[64];
    uint64_t checksum;
    uint64_t samples;              /* sched_getcpu() samples */
    uint64_t samples_on_cpu;       /* ... equal to the pinned CPU */
    int      foreign_cpu;          /* last unexpected CPU, -1 none */
    double   t_start, t_end;       /* seconds since t0 */
    double   cpu_time;             /* CLOCK_THREAD_CPUTIME_ID seconds */
} worker_state;

static worker_state workers[MAX_CPUS];

typedef struct gemm_record {
    int      k;
    double   t_issue, t_issued, t_fence;
    uint64_t prog_issue, prog_issued, prog_fence;
    int      ok;
} gemm_record;

static struct {
    long        tid;
    int         cpu;
    int         ok;           /* pinning verified, RoCC allowed */
    int         gemm_ok;      /* all GEMMs matched */
    char        affinity[64];
    char        error[160];
    double      cpu_time;
    gemm_record *gemms;
} issuer;

static struct timespec t0;
static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_int gate_count;
static atomic_int gate_abort;
static int gate_total;
static double t_gate;

/* ---- Utilities ---------------------------------------------------------- */

static void mc_log(const char *fmt, ...) {
    va_list ap;
    pthread_mutex_lock(&log_lock);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    pthread_mutex_unlock(&log_lock);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)(t.tv_sec - t0.tv_sec) + (double)(t.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double thread_cpu_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static long tid(void) {
    return (long)syscall(SYS_gettid);
}

static void mask_str(const cpu_set_t *set, char *buf, size_t len) {
    size_t n = 0;
    n += (size_t)snprintf(buf + n, len - n, "{");
    int first = 1;
    for (int c = 0; c < MAX_CPUS && n < len; ++c) {
        if (CPU_ISSET(c, set)) {
            n += (size_t)snprintf(buf + n, len - n, first ? "%d" : ",%d", c);
            first = 0;
        }
    }
    if (n < len) {
        snprintf(buf + n, len - n, "}");
    }
}

/*
 * Pin the calling thread to exactly `cpu`, then confirm the kernel's view:
 * pthread_getaffinity_np() == {cpu} and sched_getcpu() == cpu.
 */
static int pin_self(int cpu, char *affinity, size_t len, char *err, size_t err_len) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        snprintf(err, err_len, "pthread_setaffinity_np(cpu=%d) failed: %s", cpu, strerror(rc));
        return 0;
    }
    cpu_set_t got;
    CPU_ZERO(&got);
    rc = pthread_getaffinity_np(pthread_self(), sizeof(got), &got);
    if (rc != 0) {
        snprintf(err, err_len, "pthread_getaffinity_np failed: %s", strerror(rc));
        return 0;
    }
    mask_str(&got, affinity, len);
    if (CPU_COUNT(&got) != 1 || !CPU_ISSET(cpu, &got)) {
        snprintf(err, err_len, "affinity is %s, expected {%d}", affinity, cpu);
        return 0;
    }
    const int running = sched_getcpu();
    if (running != cpu) {
        snprintf(err, err_len, "sched_getcpu()=%d after pinning to %d", running, cpu);
        return 0;
    }
    return 1;
}

/*
 * One-shot start gate for the issuer and all workers (pthreads or OpenMP
 * threads). The last arrival records t_gate; acquire/release on gate_count
 * orders everything written before arrival. A participant that failed setup
 * arrives with ok == 0 so nobody waits forever.
 */
static int gate_arrive_and_wait(int ok) {
    if (!ok) {
        atomic_store_explicit(&gate_abort, 1, memory_order_release);
    }
    if (atomic_fetch_add_explicit(&gate_count, 1, memory_order_acq_rel) + 1 == gate_total) {
        t_gate = now();
    }
    while (atomic_load_explicit(&gate_count, memory_order_acquire) < gate_total) {
        sched_yield();
    }
    return !atomic_load_explicit(&gate_abort, memory_order_acquire);
}

static uint64_t total_progress(void) {
    uint64_t sum = 0;
    for (int w = 0; w < n_workers; ++w) {
        sum += atomic_load_explicit(&workers[w].progress, memory_order_acquire);
    }
    return sum;
}

/* ---- Topology ----------------------------------------------------------- */

/* Parse a sysfs CPU list such as "0-3" or "0,2-3". */
static int parse_cpu_list(const char *s, cpu_set_t *set) {
    CPU_ZERO(set);
    while (*s && *s != '\n') {
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) {
            return 0;
        }
        long b = a;
        if (*end == '-') {
            s = end + 1;
            b = strtol(s, &end, 10);
            if (end == s) {
                return 0;
            }
        }
        if (a < 0 || b >= MAX_CPUS || a > b) {
            return 0;
        }
        for (long c = a; c <= b; ++c) {
            CPU_SET((int)c, set);
        }
        s = *end == ',' ? end + 1 : end;
    }
    return 1;
}

static int discover_topology(void) {
    for (int c = 0; c < MAX_CPUS; ++c) {
        cpu_hart[c] = -1;
    }

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        mc_log("[POTAL_MC] FAIL topology: cannot open /proc/cpuinfo: %s\n", strerror(errno));
        return 0;
    }
    char line[256];
    int cur = -1;
    int hart_seen[MAX_CPUS] = {0};
    while (fgets(line, sizeof(line), f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) {
            if (v < 0 || v >= MAX_CPUS) {
                mc_log("[POTAL_MC] FAIL topology: processor %d out of range\n", v);
                fclose(f);
                return 0;
            }
            cur = v;
        } else if (sscanf(line, "hart : %d", &v) == 1) {
            if (cur < 0 || cpu_hart[cur] != -1 || v < 0 || v >= MAX_CPUS || hart_seen[v]) {
                mc_log("[POTAL_MC] FAIL topology: bad or duplicate hart line (cpu=%d hart=%d)\n", cur, v);
                fclose(f);
                return 0;
            }
            cpu_hart[cur] = v;
            hart_seen[v] = 1;
        }
    }
    fclose(f);

    cpu_set_t online;
    f = fopen("/sys/devices/system/cpu/online", "r");
    if (!f || !fgets(line, sizeof(line), f) || !parse_cpu_list(line, &online)) {
        mc_log("[POTAL_MC] FAIL topology: cannot read /sys/devices/system/cpu/online\n");
        if (f) {
            fclose(f);
        }
        return 0;
    }
    fclose(f);

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        mc_log("[POTAL_MC] FAIL topology: sched_getaffinity: %s\n", strerror(errno));
        return 0;
    }

    char buf[64];
    mask_str(&online, buf, sizeof(buf));
    mc_log("[POTAL_MC] online_cpus=%d %s\n", CPU_COUNT(&online), buf);
    mask_str(&allowed, buf, sizeof(buf));
    mc_log("[POTAL_MC] process_affinity=%s\n", buf);

    int ok = 1;
    mc_log("[POTAL_MC] topology:\n");
    for (int c = 0; c < MAX_CPUS; ++c) {
        if (!CPU_ISSET(c, &online)) {
            if (cpu_hart[c] != -1) {
                mc_log("  linux_cpu=%d hart=%d  ERROR: not online\n", c, cpu_hart[c]);
                ok = 0;
            }
            continue;
        }
        if (cpu_hart[c] == -1) {
            mc_log("  linux_cpu=%d hart=?  ERROR: no hart in /proc/cpuinfo\n", c);
            ok = 0;
            continue;
        }
        const int is_potal = cpu_hart[c] == 0;
        if (is_potal) {
            potal_cpu = c;
        }
        mc_log("  linux_cpu=%d hart=%d%s%s\n", c, cpu_hart[c], is_potal ? "  role=potal" : "",
               CPU_ISSET(c, &allowed) ? "" : "  (not in process affinity)");
    }
    if (!ok) {
        mc_log("[POTAL_MC] FAIL topology: incomplete CPU/hart map\n");
        return 0;
    }
    if (potal_cpu < 0) {
        mc_log("[POTAL_MC] FAIL topology: no online Linux CPU reports hart 0\n");
        return 0;
    }
    if (!CPU_ISSET(potal_cpu, &allowed)) {
        mc_log("[POTAL_MC] FAIL topology: hart0 CPU %d is excluded by the process affinity/cpuset\n",
               potal_cpu);
        return 0;
    }
    mc_log("[POTAL_MC] hart0_linux_cpu=%d\n", potal_cpu);

    n_worker_cpus = 0;
    for (int c = 0; c < MAX_CPUS; ++c) {
        if (c != potal_cpu && CPU_ISSET(c, &online) && CPU_ISSET(c, &allowed)) {
            worker_cpus[n_worker_cpus++] = c;
        }
    }
    mc_log("[POTAL_MC] worker_cpus_available=%d\n", n_worker_cpus);
    return 1;
}

/* ---- CPU worker kernel -------------------------------------------------- */

static uint32_t lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/*
 * One work unit: generate UNIT_ELEMS deterministic floats, quantize them to
 * INT8 with one absmax scale per QBLOCK, take an INT32 dot product with a
 * fixed INT8 weight vector, and fold the integers into a 64-bit hash.
 */
static uint64_t work_unit(int worker, uint32_t unit, uint64_t h) {
    float x[UNIT_ELEMS];
    int8_t q[UNIT_ELEMS];
    uint32_t s = 0x9e3779b9u ^ ((uint32_t)worker << 24) ^ unit;
    for (int i = 0; i < UNIT_ELEMS; ++i) {
        x[i] = (float)((int32_t)(lcg(&s) >> 8) - (1 << 23)) / (float)(1 << 16);
    }
    for (int b = 0; b < UNIT_ELEMS; b += QBLOCK) {
        float amax = 0.0f;
        for (int i = b; i < b + QBLOCK; ++i) {
            const float a = x[i] < 0 ? -x[i] : x[i];
            amax = a > amax ? a : amax;
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int32_t dot = 0;
        for (int i = b; i < b + QBLOCK; ++i) {
            const float v = x[i] * inv;
            q[i] = (int8_t)(v < 0 ? v - 0.5f : v + 0.5f);
            const int8_t w = (int8_t)((i * 29 + 7) & 0xff);
            dot += (int32_t)q[i] * (int32_t)w;
        }
        h = (h ^ (uint64_t)(uint32_t)dot) * 0x100000001b3ull;
        h = (h ^ (uint64_t)(uint8_t)q[b]) * 0x100000001b3ull;
    }
    return h;
}

static uint64_t reference_checksum(int worker, int units) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int u = 0; u < units; ++u) {
        h = work_unit(worker, (uint32_t)u, h);
    }
    return h;
}

/* Body shared by pthread and OpenMP workers; `idx` selects the CPU. */
static void worker_body(int idx, int omp_thread) {
    worker_state *ws = &workers[idx];
    char err[160] = "";
    ws->tid = tid();
    ws->omp_thread = omp_thread;
    ws->cpu = worker_cpus[idx];
    ws->foreign_cpu = -1;
    ws->ok = pin_self(ws->cpu, ws->affinity, sizeof(ws->affinity), err, sizeof(err));
    if (ws->ok && ws->cpu == potal_cpu) {
        snprintf(err, sizeof(err), "worker placed on the PoTal CPU %d", potal_cpu);
        ws->ok = 0;
    }
    const int running = sched_getcpu();
    if (omp_thread >= 0) {
        mc_log("[POTAL_MC] worker%d omp_thread=%d tid=%ld cpu=%d hart=%d affinity=%s%s%s\n", idx,
               omp_thread, ws->tid, running, running >= 0 && running < MAX_CPUS ? cpu_hart[running] : -1,
               ws->affinity, ws->ok ? "" : " FAIL ", err);
    } else {
        mc_log("[POTAL_MC] worker%d tid=%ld cpu=%d hart=%d affinity=%s%s%s\n", idx, ws->tid, running,
               running >= 0 && running < MAX_CPUS ? cpu_hart[running] : -1, ws->affinity,
               ws->ok ? "" : " FAIL ", err);
    }

    if (!gate_arrive_and_wait(ws->ok)) {
        return;
    }

    const double c0 = thread_cpu_time();
    ws->t_start = now();
    uint64_t h = 0xcbf29ce484222325ull;
    for (int u = 0; u < opt.work_units; ++u) {
        h = work_unit(idx, (uint32_t)u, h);
        atomic_store_explicit(&ws->progress, (uint64_t)u + 1, memory_order_release);
        if (u % CPU_SAMPLE == 0) {
            const int c = sched_getcpu();
            ws->samples++;
            if (c == ws->cpu) {
                ws->samples_on_cpu++;
            } else {
                ws->foreign_cpu = c;
            }
        }
        if (opt.verbose && opt.work_units >= 4 && (u + 1) % (opt.work_units / 4) == 0) {
            mc_log("[POTAL_MC] worker%d t=%.6f progress=%d/%d cpu=%d\n", idx, now(), u + 1,
                   opt.work_units, sched_getcpu());
        }
    }
    ws->checksum = h;
    ws->t_end = now();
    ws->cpu_time = thread_cpu_time() - c0;
}

static void *worker_thread(void *arg) {
    worker_body((int)(intptr_t)arg, -1);
    return NULL;
}

/* ---- PoTal issuer ------------------------------------------------------- */

static int8_t  A[DIM][K_MAX] __attribute__((aligned(POTAL_HW_MAX_BYTES)));
static int8_t  B[K_MAX][DIM] __attribute__((aligned(POTAL_HW_MAX_BYTES)));
static int32_t C[DIM][DIM]   __attribute__((aligned(POTAL_HW_MAX_BYTES)));

/* Golden model, same semantics as potal-api-smoke.c. */
static int32_t sat32(int64_t v) {
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
}

static int32_t scu(int32_t raw, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || raw == 0) {
        return 0;
    }
    if (carrier >= 32) {
        return raw > 0 ? INT32_MAX : INT32_MIN;
    }
    return sat32((int64_t)raw * ((int64_t)1 << carrier));
}

static uint32_t carrier_for(int block, int col) {
    static const uint32_t carriers[DIM] = {
        0, 1, POTAL_CARRIER_ZERO, 2, 5, 13, 20, 31,
        32, 32767, 3, POTAL_CARRIER_ZERO, 7, 0, 1, 4,
    };
    return carriers[(col + 5 * block) % DIM];
}

static int issuer_fail(const char *step, potal_status s) {
    snprintf(issuer.error, sizeof(issuer.error), "%s: %s", step, potal_status_name(s));
    mc_log("[POTAL_GEMM] FAIL %s status=%d(%s)\n", step, (int)s, potal_status_name(s));
    return 0;
}

static int check_issuer_cpu(const char *where) {
    const int c = sched_getcpu();
    if (c != issuer.cpu) {
        snprintf(issuer.error, sizeof(issuer.error), "issuer on cpu %d at %s", c, where);
        mc_log("[POTAL_MC] FAIL issuer migrated to cpu=%d at %s\n", c, where);
        return 0;
    }
    return 1;
}

static int run_one_gemm(int iter, gemm_record *r) {
    const int k = iter % 2 == 0 ? DIM : 2 * DIM;
    const uint32_t generation = (uint32_t)(iter % POTAL_GENERATION_MAX) + 1;
    const int k_tiles = k / DIM;
    potal_gemm_ws g = {
        .a = &A[0][0], .b = &B[0][0], .c = &C[0][0],
        .lda = K_MAX, .ldb = DIM, .ldc = DIM,
        .m = DIM, .n = DIM, .k = (uint32_t)k,
        .scale_base = SCALE_BASE, .scale_generation = generation, .passes_per_block = 1,
        .flags = 0,
    };
    r->k = k;

    /* DMA buffers are written only by this thread and only between fences. */
    for (int i = 0; i < DIM; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            A[i][kk] = (int8_t)(((i * 37 + kk * 11 + iter * 5) & 0xff) - 128);
        }
        for (int j = 0; j < DIM; ++j) {
            C[i][j] = SENTINEL;
        }
    }
    for (int kk = 0; kk < k; ++kk) {
        for (int j = 0; j < DIM; ++j) {
            B[kk][j] = (int8_t)(((kk * 23 + j * 59 + iter * 3) & 0xff) - 128);
        }
    }

    for (int b = 0; b < k_tiles; ++b) {
        uint32_t slot;
        potal_status s = potal_gemm_ws_scale_slot(&g, (uint32_t)b, 0, &slot);
        if (s != POTAL_OK) {
            return issuer_fail("scale slot", s);
        }
        for (int col = 0; col < DIM; ++col) {
            s = potal_scale_write(slot, generation, (uint32_t)col, carrier_for(b, col));
            if (s != POTAL_OK) {
                return issuer_fail("scale write", s);
            }
        }
    }
    if (!check_issuer_cpu("gemm issue")) {
        return 0;
    }

    r->prog_issue = total_progress();
    r->t_issue = now();
    potal_status s = potal_gemm_ws_issue(&g);
    r->t_issued = now();
    r->prog_issued = total_progress();
    if (s != POTAL_OK) {
        return issuer_fail("gemm issue", s);
    }
    s = potal_fence();
    r->t_fence = now();
    r->prog_fence = total_progress();
    if (s != POTAL_OK) {
        return issuer_fail("gemm fence", s);
    }

    int mismatches = 0;
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            int32_t acc = 0;
            for (int pass = 0; pass < k_tiles; ++pass) {
                int32_t raw = 0;
                for (int kk = pass * DIM; kk < (pass + 1) * DIM; ++kk) {
                    raw += (int32_t)A[i][kk] * (int32_t)B[kk][j];
                }
                const int32_t part = scu(raw, carrier_for(pass, j));
                acc = pass == 0 ? part : sat32((int64_t)acc + part);
            }
            if (C[i][j] != acc) {
                if (mismatches++ < 4) {
                    mc_log("[POTAL_GEMM] iter=%d C[%d][%d] expected=%ld actual=%ld\n", iter, i, j,
                           (long)acc, (long)C[i][j]);
                }
            }
        }
    }

    for (int b = 0; b < k_tiles; ++b) {
        uint32_t slot;
        (void)potal_gemm_ws_scale_slot(&g, (uint32_t)b, 0, &slot);
        for (int col = 0; col < DIM; ++col) {
            s = potal_scale_release(slot, generation, (uint32_t)col);
            if (s != POTAL_OK) {
                return issuer_fail("scale release", s);
            }
        }
    }

    r->ok = mismatches == 0;
    mc_log("[POTAL_GEMM] iter=%d K%d numerical comparison %s mismatches=%d issue_to_fence_us=%.1f "
           "worker_progress_during_gemm=%" PRIu64 "\n",
           iter, k, r->ok ? "PASS" : "FAIL", mismatches, (r->t_fence - r->t_issue) * 1e6,
           r->prog_fence - r->prog_issued);
    if (!r->ok) {
        snprintf(issuer.error, sizeof(issuer.error), "GEMM iter %d mismatches=%d", iter, mismatches);
    }
    return r->ok;
}

static void *issuer_thread(void *arg) {
    (void)arg;
    char err[160] = "";
    issuer.tid = tid();
    issuer.cpu = potal_cpu;
    /* Affinity is set and verified before any RoCC instruction. */
    issuer.ok = pin_self(potal_cpu, issuer.affinity, sizeof(issuer.affinity), err, sizeof(err));
    const int running = sched_getcpu();
    const int hart = running >= 0 && running < MAX_CPUS ? cpu_hart[running] : -1;
    if (issuer.ok && hart != 0) {
        snprintf(err, sizeof(err), "issuer cpu %d maps to hart %d, not 0", running, hart);
        issuer.ok = 0;
    }
    mc_log("[POTAL_MC] issuer tid=%ld cpu=%d hart=%d affinity=%s%s%s\n", issuer.tid, running, hart,
           issuer.affinity, issuer.ok ? "" : " FAIL ", err);
    if (!issuer.ok) {
        snprintf(issuer.error, sizeof(issuer.error), "%s", err);
        gate_arrive_and_wait(0);
        return NULL;
    }

    const double c0 = thread_cpu_time();
    potal_status s = potal_tlb_flush();
    if (s != POTAL_OK) {
        issuer.ok = issuer_fail("tlb flush", s);
        gate_arrive_and_wait(0);
        return NULL;
    }
    mc_log("[POTAL_TLB] flush issued\n");
    s = potal_fence();
    if (s != POTAL_OK) {
        issuer.ok = issuer_fail("tlb flush fence", s);
        gate_arrive_and_wait(0);
        return NULL;
    }
    mc_log("[POTAL_TLB] fence completed\n");

    if (!gate_arrive_and_wait(1)) {
        return NULL;
    }

    issuer.gemm_ok = 1;
    for (int it = 0; it < opt.iterations; ++it) {
        if (!run_one_gemm(it, &issuer.gemms[it])) {
            issuer.gemm_ok = 0;
            break;
        }
    }
    issuer.cpu_time = thread_cpu_time() - c0;
    return NULL;
}

/* ---- Worker pools ------------------------------------------------------- */

static int run_pthread_workers(pthread_t *threads) {
    for (int w = 0; w < n_workers; ++w) {
        const int rc = pthread_create(&threads[w], NULL, worker_thread, (void *)(intptr_t)w);
        if (rc != 0) {
            mc_log("[POTAL_MC] FAIL pthread_create worker%d: %s\n", w, strerror(rc));
            /* Release threads already waiting at the gate. */
            atomic_store_explicit(&gate_abort, 1, memory_order_release);
            atomic_fetch_add_explicit(&gate_count, n_workers - w, memory_order_acq_rel);
            for (int j = 0; j < w; ++j) {
                pthread_join(threads[j], NULL);
            }
            return 0;
        }
    }
    int ok = 1;
    for (int w = 0; w < n_workers; ++w) {
        const int rc = pthread_join(threads[w], NULL);
        if (rc != 0) {
            mc_log("[POTAL_MC] FAIL pthread_join worker%d: %s\n", w, strerror(rc));
            ok = 0;
        }
    }
    return ok;
}

#ifdef _OPENMP
static const char *proc_bind_name(omp_proc_bind_t b) {
    switch (b) {
        case omp_proc_bind_false:  return "false";
        case omp_proc_bind_true:   return "true";
        case omp_proc_bind_master: return "master";
        case omp_proc_bind_close:  return "close";
        case omp_proc_bind_spread: return "spread";
        default:                   return "unknown";
    }
}

static int run_openmp_workers(void) {
    const char *places = getenv("OMP_PLACES");
    const char *gomp_aff = getenv("GOMP_CPU_AFFINITY");
    mc_log("[POTAL_OMP] enabled=1\n");
    mc_log("[POTAL_OMP] _OPENMP=%d\n", _OPENMP);
    mc_log("[POTAL_OMP] OMP_NUM_THREADS=%s\n", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)");
    mc_log("[POTAL_OMP] requested_threads=%d\n", n_workers);
    mc_log("[POTAL_OMP] dynamic=%d\n", omp_get_dynamic());
    mc_log("[POTAL_OMP] proc_bind=%s\n", proc_bind_name(omp_get_proc_bind()));
    mc_log("[POTAL_OMP] places=%s num_places=%d\n", places ? places : "(unset)", omp_get_num_places());
    mc_log("[POTAL_OMP] GOMP_CPU_AFFINITY=%s\n", gomp_aff ? gomp_aff : "(unset)");
    if (places || gomp_aff || omp_get_proc_bind() != omp_proc_bind_false) {
        mc_log("[POTAL_OMP] WARN runtime binding is configured; explicit per-thread affinity overrides it\n");
    }

    /* The master thread joins the team; restore its affinity afterwards. */
    cpu_set_t saved;
    CPU_ZERO(&saved);
    pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved);

    atomic_int team = 0;
    atomic_int mismatch = 0;
#pragma omp parallel num_threads(n_workers)
    {
        const int t = omp_get_thread_num();
        const int size = omp_get_num_threads();
        if (t == 0) {
            atomic_store(&team, size);
        }
        if (size != n_workers) {
            /* Too small a team: never wait for workers that do not exist. */
            atomic_store(&mismatch, 1);
            if (t == 0) {
                atomic_store_explicit(&gate_abort, 1, memory_order_release);
                atomic_fetch_add_explicit(&gate_count, n_workers, memory_order_acq_rel);
            }
        } else {
            worker_body(t, t);
        }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved);

    mc_log("[POTAL_OMP] actual_threads=%d\n", atomic_load(&team));
    if (atomic_load(&mismatch)) {
        mc_log("[POTAL_OMP] FAIL team size %d != requested %d\n", atomic_load(&team), n_workers);
        return 0;
    }
    return 1;
}
#endif

/* ---- Main --------------------------------------------------------------- */

static int parse_int(const char *arg, const char *name, int *out, int min) {
    const size_t n = strlen(name);
    if (strncmp(arg, name, n) != 0 || arg[n] != '=') {
        return 0;
    }
    char *end;
    const long v = strtol(arg + n + 1, &end, 10);
    if (*end != '\0' || v < min || v > 100000000L) {
        fprintf(stderr, "invalid %s\n", arg);
        exit(2);
    }
    *out = (int)v;
    return 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--mode=pthread|openmp] [--workers=N] [--iterations=N] [--work-units=N] [--verbose]\n"
            "  --workers     CPU worker threads (default: OMP_NUM_THREADS in openmp mode, else all\n"
            "                allowed online CPUs except the hart0 CPU)\n"
            "  --iterations  PoTal GEMMs (alternating K16/K32, default 4)\n"
            "  --work-units  work units per worker (default 4000; raise to cover more GEMMs)\n",
            argv0);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--mode=pthread") == 0) {
            opt.mode = MODE_PTHREAD;
        } else if (strcmp(a, "--mode=openmp") == 0) {
            opt.mode = MODE_OPENMP;
        } else if (strcmp(a, "--verbose") == 0) {
            opt.verbose = 1;
        } else if (parse_int(a, "--workers", &opt.workers, 1) ||
                   parse_int(a, "--iterations", &opt.iterations, 1) ||
                   parse_int(a, "--work-units", &opt.work_units, 1)) {
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    mc_log("POTAL_MULTICORE_SMOKE START\n");
    mc_log("[POTAL_MC] hardware=DIM%u\n", (unsigned)potal_hw_dim());
    mc_log("[POTAL_MC] params_sha256=%s\n", potal_hw_params_sha256());
    mc_log("[POTAL_MC] mode=%s\n", opt.mode == MODE_OPENMP ? "openmp" : "pthread");
    mc_log("[POTAL_MC] main tid=%ld cpu=%d\n", tid(), sched_getcpu());

#ifndef _OPENMP
    if (opt.mode == MODE_OPENMP) {
        mc_log("[POTAL_OMP] enabled=0\n");
        mc_log("[POTAL_MC] FAIL --mode=openmp requested but built with POTAL_ENABLE_OPENMP=OFF\n");
        mc_log("POTAL_MULTICORE_SMOKE FAIL\n");
        return 2;
    }
#endif

    if (!discover_topology()) {
        mc_log("POTAL_MULTICORE_SMOKE FAIL\n");
        return 2;
    }

    n_workers = opt.workers;
#ifdef _OPENMP
    if (n_workers == 0 && opt.mode == MODE_OPENMP && getenv("OMP_NUM_THREADS")) {
        n_workers = omp_get_max_threads();
    }
#endif
    if (n_workers == 0) {
        n_workers = n_worker_cpus;
    }
    if (n_workers < 1 || n_workers > n_worker_cpus) {
        mc_log("[POTAL_MC] FAIL requested workers=%d but only %d non-PoTal CPUs are available\n",
               n_workers, n_worker_cpus);
        mc_log("POTAL_MULTICORE_SMOKE FAIL\n");
        return 2;
    }
    mc_log("[POTAL_MC] workers=%d iterations=%d work_units=%d\n", n_workers, opt.iterations,
           opt.work_units);

    issuer.gemms = calloc((size_t)opt.iterations, sizeof(gemm_record));
    pthread_t *threads = calloc((size_t)n_workers, sizeof(pthread_t));
    if (!issuer.gemms || !threads) {
        mc_log("[POTAL_MC] FAIL out of memory\n");
        mc_log("POTAL_MULTICORE_SMOKE FAIL\n");
        return 2;
    }
    gate_total = n_workers + 1;

    /* The main thread keeps its full affinity; every thread pins itself. */
    pthread_t issuer_tid;
    int rc = pthread_create(&issuer_tid, NULL, issuer_thread, NULL);
    if (rc != 0) {
        mc_log("[POTAL_MC] FAIL pthread_create issuer: %s\n", strerror(rc));
        mc_log("POTAL_MULTICORE_SMOKE FAIL\n");
        return 1;
    }

    int threads_ok;
#ifdef _OPENMP
    if (opt.mode == MODE_OPENMP) {
        threads_ok = run_openmp_workers();
    } else
#endif
    {
        threads_ok = run_pthread_workers(threads);
    }
    rc = pthread_join(issuer_tid, NULL);
    if (rc != 0) {
        mc_log("[POTAL_MC] FAIL pthread_join issuer: %s\n", strerror(rc));
        threads_ok = 0;
    }
    const int started = !atomic_load(&gate_abort);

    /* ---- Verification (all threads joined) ---------------------------- */
    int cpu_ok = threads_ok && started;
    int distinct = 1;
    for (int w = 0; w < n_workers; ++w) {
        worker_state *ws = &workers[w];
        const uint64_t progress = atomic_load(&ws->progress);
        int ok = ws->ok && progress == (uint64_t)opt.work_units && ws->samples > 0 &&
                 ws->samples_on_cpu == ws->samples && ws->cpu != potal_cpu;
        const uint64_t ref = started ? reference_checksum(w, opt.work_units) : 0;
        const int sum_ok = started && ws->checksum == ref;
        for (int v = 0; v < w; ++v) {
            if (workers[v].cpu == ws->cpu) {
                distinct = 0;
            }
        }
        mc_log("[POTAL_MC] worker%d progress=%" PRIu64 " checksum=0x%016" PRIx64 " ref=0x%016" PRIx64
               " checksum %s\n", w, progress, ws->checksum, ref, sum_ok ? "PASS" : "FAIL");
        mc_log("[POTAL_MC] worker%d cpu=%d hart=%d cpu_samples=%" PRIu64 "/%" PRIu64 " foreign_cpu=%d "
               "start=%.6f end=%.6f wall_s=%.6f thread_cpu_s=%.6f\n", w, ws->cpu, cpu_hart[ws->cpu],
               ws->samples_on_cpu, ws->samples, ws->foreign_cpu, ws->t_start, ws->t_end,
               ws->t_end - ws->t_start, ws->cpu_time);
        cpu_ok = cpu_ok && ok && sum_ok;
    }
    cpu_ok = cpu_ok && distinct;

    const int gemm_ok = started && issuer.ok && issuer.gemm_ok;
    int overlap = 0;
    double gemm_first = 0, gemm_last = 0;
    if (started) {
        for (int it = 0; it < opt.iterations; ++it) {
            const gemm_record *r = &issuer.gemms[it];
            if (r->t_fence == 0) {
                break;
            }
            if (it == 0) {
                gemm_first = r->t_issue;
            }
            gemm_last = r->t_fence;
            const uint64_t during = r->prog_fence - r->prog_issued;
            overlap |= r->ok && during > 0;
            mc_log("[POTAL_GEMM] iter=%d K%d issue=%.6f issued=%.6f fence=%.6f progress issue/issued/fence="
                   "%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n", it, r->k, r->t_issue, r->t_issued, r->t_fence,
                   r->prog_issue, r->prog_issued, r->prog_fence);
        }
    }

    mc_log("[POTAL_MC] summary gate=%.6f gemm_window=[%.6f,%.6f] issuer_cpu=%d issuer_thread_cpu_s=%.6f\n",
           t_gate, gemm_first, gemm_last, issuer.cpu, issuer.cpu_time);
    if (issuer.error[0]) {
        mc_log("[POTAL_MC] issuer error: %s\n", issuer.error);
    }
    mc_log("[POTAL_MC] cpu_multicore_verified=%d\n", cpu_ok);
    mc_log("[POTAL_MC] potal_gemm_verified=%d\n", gemm_ok);
    mc_log("[POTAL_MC] overlap_observed=%d\n", overlap);
    mc_log("%s\n", cpu_ok ? "CPU_MULTICORE_VERIFIED" : "FAIL CPU_MULTICORE_NOT_VERIFIED");
    mc_log("%s\n", gemm_ok ? "POTAL_GEMM_VERIFIED" : "FAIL POTAL_GEMM_NOT_VERIFIED");
    mc_log("%s\n", overlap ? "CPU_POTAL_OVERLAP_OBSERVED" : "CPU_POTAL_OVERLAP_NOT_OBSERVED");
    if (!overlap && gemm_ok && cpu_ok) {
        mc_log("[POTAL_MC] note: no worker progress inside any issue->FENCE window; "
               "raise --work-units and/or --iterations\n");
    }

    free(threads);
    free(issuer.gemms);
    const int pass = cpu_ok && gemm_ok;
    mc_log("POTAL_MULTICORE_SMOKE %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
