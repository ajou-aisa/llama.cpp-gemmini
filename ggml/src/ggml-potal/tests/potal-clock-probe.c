// Cross-hart clock probe for H3-8I-25P timeline validation.
// Prints machine-readable PROBE lines and one PROBE_VERDICT line. It measures, never assumes:
// per-CPU CLOCK_MONOTONIC resolution/cost, rdcycle/rdtime availability, the rdcycle-to-MONOTONIC
// slope, and for every ordered CPU pair a release/acquire ping-pong that checks a0<=b0<=b1<=a1 and
// bounds the rdcycle offset between the two harts.
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_CPUS 64
#define TIMEOUT_NS 5000000000LL

static int cpu_hart[MAX_CPUS];
static int cpus[MAX_CPUS];
static int ncpus;

static int64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static uint64_t read_cycle(void) {
#if defined(__riscv)
    uint64_t v;
    __asm__ volatile("rdcycle %0" : "=r"(v));
    return v;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t) hi << 32) | lo;
#else
    return 0;
#endif
}

#if defined(__riscv)
static uint64_t read_time_csr(void) {
    uint64_t v;
    __asm__ volatile("rdtime %0" : "=r"(v));
    return v;
}
#endif

static int pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) return -1;
    cpu_set_t got;
    CPU_ZERO(&got);
    if (pthread_getaffinity_np(pthread_self(), sizeof(got), &got) != 0) return -1;
    if (CPU_COUNT(&got) != 1 || !CPU_ISSET(cpu, &got)) return -1;
    return sched_getcpu() == cpu ? 0 : -1;
}

static void topology(void) {
    for (int i = 0; i < MAX_CPUS; ++i) cpu_hart[i] = -1;
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[256];
    int cpu = -1;
    while (f && fgets(line, sizeof line, f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) cpu = v;
        else if (sscanf(line, "hart : %d", &v) == 1 && cpu >= 0 && cpu < MAX_CPUS) cpu_hart[cpu] = v;
    }
    if (f) fclose(f);
    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof(set), &set);
    for (int c = 0; c < MAX_CPUS && ncpus < MAX_CPUS; ++c)
        if (CPU_ISSET(c, &set)) cpus[ncpus++] = c;
    char src[128] = "unknown";
    FILE *cs = fopen("/sys/devices/system/clocksource/clocksource0/current_clocksource", "r");
    if (cs && fgets(src, sizeof src, cs)) src[strcspn(src, "\n")] = 0;
    if (cs) fclose(cs);
    printf("PROBE topology cpus=%d clocksource=%s\n", ncpus, src);
    for (int i = 0; i < ncpus; ++i) printf("PROBE map cpu=%d hart=%d\n", cpus[i], cpu_hart[cpus[i]]);
}

static int rdtime_probe(void) {
#if defined(__riscv)
    pid_t pid = fork();
    if (pid == 0) {
        signal(SIGILL, SIG_DFL);
        volatile uint64_t a = read_time_csr();
        volatile uint64_t b = read_time_csr();
        _exit(b >= a ? 0 : 3);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) < 0) return -1;
    if (WIFSIGNALED(status)) return 0;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 1 : 0;
#else
    return 0;
#endif
}

static int cycle_probe(void) {
    pid_t pid = fork();
    if (pid == 0) {
        volatile uint64_t a = read_cycle();
        volatile uint64_t b = read_cycle();
        _exit(b >= a ? 0 : 3);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) < 0) return -1;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 1 : 0;
}

static int cmp64(const void *a, const void *b) {
    const int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
    return x < y ? -1 : x > y;
}

static void per_cpu(int cpu, int have_cycle, int64_t *res_out) {
    if (pin(cpu) != 0) {
        printf("PROBE cpu=%d pin=FAIL\n", cpu);
        *res_out = -1;
        return;
    }
    struct timespec res;
    clock_getres(CLOCK_MONOTONIC, &res);
    enum { N = 2000 };
    static int64_t d[N];
    int64_t prev = mono_ns(), zero = 0;
    const int64_t t0 = prev;
    for (int i = 0; i < N; ++i) {
        int64_t now = mono_ns();
        d[i] = now - prev;
        if (d[i] == 0) ++zero;
        prev = now;
    }
    const int64_t cost = (prev - t0) / N;
    int64_t minpos = INT64_MAX;
    for (int i = 0; i < N; ++i)
        if (d[i] > 0 && d[i] < minpos) minpos = d[i];
    qsort(d, N, sizeof d[0], cmp64);
    double slope = 0.0, resid = 0.0;
    if (have_cycle) {
        const int64_t m0 = mono_ns();
        const uint64_t c0 = read_cycle();
        int64_t mid_m = 0;
        uint64_t mid_c = 0;
        while (mono_ns() - m0 < 20000000LL) {
            if (!mid_m && mono_ns() - m0 >= 10000000LL) {
                mid_m = mono_ns();
                mid_c = read_cycle();
            }
        }
        const uint64_t c1 = read_cycle();
        const int64_t m1 = mono_ns();
        slope = (double) (c1 - c0) / (double) (m1 - m0);
        if (mid_m) resid = (double) (mid_c - c0) / slope - (double) (mid_m - m0);
    }
    printf("PROBE cpu=%d hart=%d res_ns=%ld read_cost_ns=%" PRId64 " zero_deltas=%" PRId64
           " min_pos_delta_ns=%" PRId64 " median_delta_ns=%" PRId64 " cycle_per_ns=%.6f fit_residual_ns=%.1f\n",
           cpu, cpu_hart[cpu], res.tv_nsec, cost, zero, minpos == INT64_MAX ? -1 : minpos, d[N / 2], slope, resid);
    *res_out = minpos == INT64_MAX ? res.tv_nsec : minpos;
}

struct pp {
    _Alignas(64) atomic_uint_fast64_t seq;
    _Alignas(64) atomic_uint_fast64_t ack;
    _Alignas(64) int a_cpu, b_cpu, rounds, b_ok;
    int64_t *b0, *b1;
    uint64_t *b0c, *b1c;
};

static void *pp_b(void *arg) {
    struct pp *p = arg;
    p->b_ok = pin(p->b_cpu) == 0;
    for (int i = 1; i <= p->rounds && p->b_ok; ++i) {
        const int64_t start = mono_ns();
        while (atomic_load_explicit(&p->seq, memory_order_acquire) != (uint64_t) i) {
            if (mono_ns() - start > TIMEOUT_NS) {
                p->b_ok = 0;
                return NULL;
            }
        }
        p->b0c[i - 1] = read_cycle();
        p->b0[i - 1] = mono_ns();
        p->b1[i - 1] = mono_ns();
        p->b1c[i - 1] = read_cycle();
        atomic_store_explicit(&p->ack, (uint64_t) i, memory_order_release);
    }
    return NULL;
}

static int pingpong(int a, int b, int rounds, int64_t tick, int have_cycle, int64_t *viol0_out, int64_t *viol1_out) {
    struct pp p;
    memset(&p, 0, sizeof p);
    p.a_cpu = a;
    p.b_cpu = b;
    p.rounds = rounds;
    int64_t *a0 = calloc(rounds, 8), *a1 = calloc(rounds, 8);
    uint64_t *a0c = calloc(rounds, 8), *a1c = calloc(rounds, 8);
    p.b0 = calloc(rounds, 8);
    p.b1 = calloc(rounds, 8);
    p.b0c = calloc(rounds, 8);
    p.b1c = calloc(rounds, 8);
    if (!a0 || !a1 || !a0c || !a1c || !p.b0 || !p.b1 || !p.b0c || !p.b1c) return -1;
    pthread_t th;
    if (pin(a) != 0 || pthread_create(&th, NULL, pp_b, &p) != 0) return -1;
    int ok = 1;
    for (int i = 1; i <= rounds && ok; ++i) {
        a0[i - 1] = mono_ns();
        a0c[i - 1] = read_cycle();
        atomic_store_explicit(&p.seq, (uint64_t) i, memory_order_release);
        const int64_t start = mono_ns();
        while (atomic_load_explicit(&p.ack, memory_order_acquire) != (uint64_t) i) {
            if (mono_ns() - start > TIMEOUT_NS) {
                ok = 0;
                break;
            }
        }
        a1c[i - 1] = read_cycle();
        a1[i - 1] = mono_ns();
    }
    pthread_join(th, NULL);
    ok = ok && p.b_ok;
    int64_t v0 = 0, v1 = 0, rt_min = INT64_MAX, rt_sum = 0;
    double lo[2] = {-1e300, -1e300}, hi[2] = {1e300, 1e300};
    for (int i = 0; ok && i < rounds; ++i) {
        if (!(a0[i] <= p.b0[i] && p.b0[i] <= p.b1[i] && p.b1[i] <= a1[i])) ++v0;
        if (!(a0[i] <= p.b0[i] + tick && p.b0[i] <= p.b1[i] + tick && p.b1[i] <= a1[i] + tick)) ++v1;
        const int64_t rt = a1[i] - a0[i];
        rt_sum += rt;
        if (rt < rt_min) rt_min = rt;
        if (have_cycle) {
            const int h = i < rounds / 2 ? 0 : 1;
            const double l = (double) (int64_t) (p.b1c[i] - a1c[i]), u = (double) (int64_t) (p.b0c[i] - a0c[i]);
            if (l > lo[h]) lo[h] = l;
            if (u < hi[h]) hi[h] = u;
        }
    }
    printf("PROBE pair a_cpu=%d a_hart=%d b_cpu=%d b_hart=%d rounds=%d ok=%d viol_strict=%" PRId64
           " viol_tick=%" PRId64 " tick_ns=%" PRId64 " rt_min_ns=%" PRId64 " rt_mean_ns=%" PRId64,
           a, cpu_hart[a], b, cpu_hart[b], rounds, ok, v0, v1, tick, rt_min == INT64_MAX ? -1 : rt_min,
           ok && rounds ? rt_sum / rounds : -1);
    if (have_cycle)
        printf(" cyc_off_early=[%.0f,%.0f] cyc_off_late=[%.0f,%.0f]", lo[0], hi[0], lo[1], hi[1]);
    printf("\n");
    *viol0_out = v0;
    *viol1_out = v1;
    free(a0); free(a1); free(a0c); free(a1c);
    free(p.b0); free(p.b1); free(p.b0c); free(p.b1c);
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    int rounds = 2000;
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], "--rounds=", 9) == 0) rounds = atoi(argv[i] + 9);
    if (rounds < 2) rounds = 2;
    setvbuf(stdout, NULL, _IOLBF, 0);
    topology();
    const int have_cycle = cycle_probe() == 1;
    const int have_time = rdtime_probe() == 1;
    printf("PROBE user_rdcycle=%d user_rdtime=%d\n", have_cycle, have_time);
    int64_t tick = 0;
    int pin_fail = 0;
    for (int i = 0; i < ncpus; ++i) {
        int64_t r = 0;
        per_cpu(cpus[i], have_cycle, &r);
        if (r < 0) ++pin_fail;
        else if (r > tick) tick = r;
    }
    int64_t strict = 0, tolerant = 0;
    int failed = 0;
    for (int i = 0; i < ncpus; ++i)
        for (int j = 0; j < ncpus; ++j) {
            if (i == j) continue;
            int64_t v0 = 0, v1 = 0;
            if (pingpong(cpus[i], cpus[j], rounds, tick, have_cycle, &v0, &v1) != 0) ++failed;
            strict += v0;
            tolerant += v1;
        }
    const int pass = !pin_fail && !failed && tolerant == 0;
    printf("PROBE_VERDICT %s cpus=%d pairs=%d rounds=%d tick_ns=%" PRId64 " viol_strict=%" PRId64
           " viol_tick=%" PRId64 " pair_failures=%d pin_failures=%d user_rdcycle=%d\n",
           pass ? "PASS" : "FAIL", ncpus, ncpus * (ncpus - 1), rounds, tick, strict, tolerant, failed, pin_fail,
           have_cycle);
    return pass ? 0 : 1;
}
