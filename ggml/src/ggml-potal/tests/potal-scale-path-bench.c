#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "potal.h"
#include "potal_hw_params.h"
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C11 caller; the tiny C++ clock bridge uses guest steady_clock. */
uint64_t potal_scale_bench_steady_ns(void);
enum { REPEATS = 7, BATCHES = 36, SLOTS = 256, COLUMNS = 16 };
_Static_assert(POTAL_HW_DIM == COLUMNS, "DIM16 only");
static int use_cycles;

static uint64_t cycles(void) {
#if defined(__riscv)
    uint64_t c;
    if (use_cycles) {
        __asm__ volatile ("rdcycle %0" : "=r"(c) :: "memory");
        return c;
    }
#endif
    return 0;
}

static void checked(potal_status status, const char *stage) {
    if (status != POTAL_OK) {
        fprintf(stderr, "POTAL_SCALE_BENCH FAIL stage=%s status=%d\n", stage, status);
        /* Do not continue after a failed command with unknown live entries. */
        exit(1);
    }
}

static potal_status sink(void *ctx, const potal_paired_cmd_record *record) {
    (void)ctx;
    (void)record;
    return POTAL_OK;
}

static int compare(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int pin_hart0(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    int processor = -1, cpu = -1, valid = f != NULL;
    char line[256];
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) processor = v;
        else if (sscanf(line, "hart : %d", &v) == 1 && v == 0) {
            if (processor < 0 || processor >= CPU_SETSIZE || cpu != -1) valid = 0;
            cpu = processor;
        }
    }
    valid = valid && !ferror(f) && cpu >= 0;
    fclose(f);
    if (!valid) return 0;
    cpu_set_t set, got;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) || sched_getaffinity(0, sizeof(got), &got) ||
        !CPU_EQUAL(&set, &got) || sched_getcpu() != cpu) return 0;
    printf("[POTAL_SCALE_BENCH] affinity_linux_cpu=%d physical_hart=0\n", cpu);
    return 1;
}

static void run(int route, int repetition, double result[2]) {
    potal_paired_session *session = NULL;
    const potal_paired_sink software = {sink, NULL};
    const char *name = route == 0 ? "serial" : route == 1 ? "paired" : "paired_software_only";
    uint64_t elapsed[2] = {0, 0}, ticks[2] = {0, 0};
    if (route) checked(potal_paired_session_open(route == 2 ? &software : NULL, &session), "open");
    for (unsigned batch = 0; batch < BATCHES; ++batch) {
        const unsigned generation = batch + 1;
        for (unsigned phase = 0; phase < 2; ++phase) {
            const uint64_t start = potal_scale_bench_steady_ns(), c0 = cycles();
            for (unsigned slot = 0; slot < SLOTS; ++slot) {
                for (unsigned col = 0; col < COLUMNS; ++col) {
                    potal_status status;
                    if (route) {
                        status = phase ? potal_paired_scale_release(session, slot, generation, col)
                                       : potal_paired_scale_write(session, slot, generation, col, 0);
                    } else {
                        status = phase ? potal_scale_release(slot, generation, col)
                                       : potal_scale_write(slot, generation, col, 0);
                    }
                    checked(status, phase ? "release" : "write");
                }
            }
            ticks[phase] += cycles() - c0;
            elapsed[phase] += potal_scale_bench_steady_ns() - start;
        }
        if (route) {
            const potal_paired_session_status q = potal_paired_session_query(session);
            if (q.live_scale_slots || q.failed || q.outstanding_loops || q.outstanding_readbacks ||
                q.commands != (uint64_t)(batch + 1) * SLOTS * COLUMNS * 2) {
                fprintf(stderr, "POTAL_SCALE_BENCH FAIL batch state\n");
                exit(1);
            }
        }
    }
    if (route) checked(potal_paired_session_close(session), "close");
    for (unsigned phase = 0; phase < 2; ++phase) {
        const unsigned commands = BATCHES * SLOTS * COLUMNS;
        result[phase] = (double)elapsed[phase] / commands;
        if (repetition >= 0) {
            printf("[POTAL_SCALE_BENCH] route=%s phase=%s commands=%u elapsed_ms=%.6f ns_per_command=%.6f "
                   "repetition=%d cycles_supported=%d cycles_per_command=%.6f\n",
                   name, phase ? "release" : "write", commands, elapsed[phase] / 1e6,
                   result[phase], repetition + 1, use_cycles, (double)ticks[phase] / commands);
        }
    }
}

int main(int argc, char **argv) {
    int software_only = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--software-only")) software_only = 1;
        else if (!strcmp(argv[i], "--cycles")) use_cycles = 1;
        else { fprintf(stderr, "usage: %s [--software-only] [--cycles]\n", argv[0]); return 2; }
    }
#if !defined(__riscv)
    if (!software_only || use_cycles) {
        fprintf(stderr, "host requires --software-only without --cycles\n");
        return 2;
    }
#endif
    if (getenv("POTAL_RAW_TRACE") != NULL) {
        fprintf(stderr, "unset POTAL_RAW_TRACE before measuring\n");
        return 2;
    }
    if (!software_only && !pin_hart0()) {
        fprintf(stderr, "cannot pin physical hart 0\n");
        return 1;
    }
    double samples[3][2][REPEATS], warmup[2];
    const int first = software_only ? 2 : 0;
    if (!software_only) checked(potal_fence(), "initial fence");
    for (int route = first; route < 3; ++route) run(route, -1, warmup);
    for (int rep = 0; rep < REPEATS; ++rep) {
        for (int j = first; j < 3; ++j) {
            const int route = software_only ? 2 : (rep + j) % 3;
            double values[2];
            run(route, rep, values);
            for (int phase = 0; phase < 2; ++phase) samples[route][phase][rep] = values[phase];
        }
    }
    for (int route = first; route < 3; ++route) {
        for (int phase = 0; phase < 2; ++phase) {
            double *v = samples[route][phase], deviations[REPEATS];
            qsort(v, REPEATS, sizeof(*v), compare);
            const double median = v[REPEATS / 2];
            for (int i = 0; i < REPEATS; ++i) deviations[i] = v[i] > median ? v[i] - median : median - v[i];
            qsort(deviations, REPEATS, sizeof(*deviations), compare);
            printf("[POTAL_SCALE_BENCH_SUMMARY] route=%s phase=%s repeats=%d median_ns=%.6f min_ns=%.6f "
                   "max_ns=%.6f mad_ns=%.6f\n", route == 0 ? "serial" : route == 1 ? "paired" : "paired_software_only",
                   phase ? "release" : "write", REPEATS, median, v[0], v[REPEATS - 1], deviations[REPEATS / 2]);
        }
    }
    puts("POTAL_SCALE_BENCH PASS no_gemms=1 live_entries=0");
    return 0;
}
