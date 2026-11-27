/*
 * CPU-only issuer benchmark. No FPGA access, target cycles or throughput claim.
 * Link this C11 source with the same optimized potal-api used for both paths.
 * The sink hashes actual commands dispatched by the scalar and window APIs;
 * it supplies only transport success, never an expected command stream.
 */
#include "potal.h"
#include "potal_hw_params.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct bench {
    potal_paired_session *session;
    potal_index_config cfg;
    uint32_t entries[POTAL_HW_INDEX_ENTRIES];
    uint64_t hash, commands;
} bench;

typedef struct measurement {
    double ns_per_window;
    uint64_t hash, commands;
} measurement;

static potal_status hash_command(void *ctx, const potal_paired_cmd_record *command) {
    bench *b = ctx;
    b->hash = (b->hash ^ command->rs1) * UINT64_C(1099511628211);
    b->hash = (b->hash ^ command->rs2) * UINT64_C(1099511628211);
    b->hash ^= ((uint64_t)command->funct << 1) | command->xd;
    ++b->commands;
    return POTAL_OK;
}

static measurement measure(bench *b, int window, uint32_t repetitions) {
    b->hash = UINT64_C(14695981039346656037);
    b->commands = 0;
    const clock_t start = clock();
    if (start == (clock_t)-1) {
        fputs("CPU clock unavailable\n", stderr);
        exit(1);
    }
    for (uint32_t iteration = 0; iteration < repetitions; ++iteration) {
        uint64_t written = 0;
        potal_status status;
        if (window) {
            status = potal_paired_index_window_issue(b->session, &b->cfg, b->entries,
                b->cfg.active_rows, &written);
        } else {
            status = potal_paired_index_config(b->session, &b->cfg);
            for (uint32_t i = 0; status == POTAL_OK && i < b->cfg.active_rows; ++i) {
                const potal_index_write entry = { i, b->entries[i] };
                status = potal_paired_index_write(b->session, &entry);
                if (status == POTAL_OK) ++written;
            }
        }
        if (status != POTAL_OK || written != b->cfg.active_rows) {
            fprintf(stderr, "issuer failed: %s written=%" PRIu64 "\n", potal_status_name(status), written);
            exit(1);
        }
    }
    const clock_t finish = clock();
    if (finish == (clock_t)-1 || finish < start) {
        fputs("CPU clock interval unavailable\n", stderr);
        exit(1);
    }
    const measurement result = {
        ((double)finish - (double)start) * 1e9 / CLOCKS_PER_SEC / repetitions, b->hash, b->commands
    };
    return result;
}

int main(void) {
    const uint32_t sizes[] = { 1, 7, 16, 32, POTAL_HW_INDEX_ENTRIES };
    const uint32_t repetitions = 100000;
    bench b = {0};
    const potal_paired_sink sink = { hash_command, &b, NULL };
    if (potal_paired_session_open(&sink, &b.session) != POTAL_OK) return 1;
    for (uint32_t i = 0; i < POTAL_HW_INDEX_ENTRIES; ++i)
        b.entries[i] = (i * 7 + 3) % (2 * POTAL_HW_DIM - 1);
    puts("CPU-only process CPU time; cheap command sink; no FPGA or target timing.");
    puts("rows repetitions scalar_ns/window window_ns/window scalar_ns/write window_ns/write commands hash");
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        b.cfg = (potal_index_config){ 0, POTAL_HW_BANK_ROWS, 1, 2 * POTAL_HW_DIM - 1, sizes[i] };
        /* Warm both actual API paths before measuring; balance order by size. */
        (void)measure(&b, 0, repetitions);
        (void)measure(&b, 1, repetitions);
        measurement scalar, window;
        if (i % 2 == 0) {
            scalar = measure(&b, 0, repetitions);
            window = measure(&b, 1, repetitions);
        } else {
            window = measure(&b, 1, repetitions);
            scalar = measure(&b, 0, repetitions);
        }
        const uint64_t expected = (uint64_t)repetitions * (sizes[i] + 1);
        if (scalar.commands != expected || window.commands != expected || scalar.hash != window.hash) {
            fputs("scalar/window dispatched command count or hash mismatch\n", stderr);
            return 1;
        }
        printf("%u %u %.2f %.2f %.2f %.2f %" PRIu64 " %016" PRIx64 "\n",
            sizes[i], repetitions, scalar.ns_per_window, window.ns_per_window,
            scalar.ns_per_window / sizes[i], window.ns_per_window / sizes[i], expected, window.hash);
    }
    return potal_paired_session_close(b.session) == POTAL_OK ? 0 : 1;
}
