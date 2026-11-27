#define _POSIX_C_SOURCE 200809L
/* R: sampled API partitions and real raw-command coverage, including the Scale
 * direct path and an Index invalid-entry prefix. Missing/bypassed raw hooks,
 * double-counted nested calls, or incorrect partition boundaries must fail.
 * Existing API tests do not inspect diagnostic attribution.
 */
#include "potal.h"
#include "potal_hw_params.h"
#include "../trace/issuer-profile.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

typedef struct {
    potal_paired_session * session;
    unsigned writes, releases, wrong_progress;
    uint32_t fail_funct;
} recorder;

static potal_status record(void * ctx, const potal_paired_cmd_record * command) {
    recorder * r = ctx;
    const potal_paired_session_status state = potal_paired_session_query(r->session);
    if (command->funct == POTAL_FUNCT_SCALE_WRITE) {
        ++r->writes;
        /* The direct window publishes this slot after its columns, not after
         * every raw command. Diagnostics must not select the scalar path.
         */
        r->wrong_progress += state.live_scale_slots != 0;
    }
    if (command->funct == POTAL_FUNCT_SCALE_RELEASE) {
        ++r->releases;
        r->wrong_progress += state.live_scale_slots != 1;
    }
    return command->funct == r->fail_funct ? POTAL_HW_INVALID_ADDRESS : POTAL_OK;
}

static int partitioned(const potal_issuer_scope_stats * s) {
    return s->ticks == s->pre_ticks + s->between_ticks + s->post_ticks + s->raw_ticks + s->fence_ticks &&
        s->instructions == s->pre_instructions + s->between_instructions +
            s->post_instructions + s->raw_instructions + s->fence_instructions;
}

int main(int argc, char ** argv) {
    const int enabled = argc == 1 || strcmp(argv[1], "--disabled") != 0;
    REQUIRE(setenv("POTAL_ISSUER_CPU_PROFILE", enabled ? "1" : "0", 1) == 0);
    REQUIRE(setenv("POTAL_ISSUER_CPU_SAMPLE_LOG2", "0", 1) == 0);
    recorder r = {0};
    const potal_paired_sink sink = {record, &r, NULL};
    REQUIRE(potal_paired_session_open(&sink, &r.session) == POTAL_OK);
    const potal_paired_scale_columns window = {0, 1, 1, 7, 1};
    const uint32_t values[7] = {0, 1, 2, 3, 4, 5, 6};
    const uint32_t * blocks[] = {values};
    uint64_t written = 0, released = 0;
    REQUIRE(potal_paired_scale_write_columns(r.session, &window, blocks, &written) == POTAL_OK);
    REQUIRE(potal_paired_scale_release_columns(r.session, &window, 0, &released) == POTAL_OK);
    REQUIRE(written == 7 && released == 7 && r.writes == 7 && r.releases == 7);
    REQUIRE(r.wrong_progress == 0);

    const potal_index_config config = {0, POTAL_HW_BANK_ROWS, 1, 32, 3};
    const uint32_t rows[] = {2, 32, 4}; /* fail after one successful write */
    written = 99;
    REQUIRE(potal_paired_index_window_issue(r.session, &config, rows, 3, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(written == 1);
    REQUIRE(potal_paired_session_query(r.session).index_transaction_written == 1);

    potal_issuer_profile_stats stats;
    potal_issuer_profile_snapshot(&stats);
    if (enabled) {
        REQUIRE(stats.scope[POTAL_ISSUER_SCALE_WINDOW].calls == 1);
        REQUIRE(stats.scope[POTAL_ISSUER_RELEASE_WINDOW].calls == 1);
        REQUIRE(stats.scope[POTAL_ISSUER_INDEX_WINDOW].calls == 1);
        REQUIRE(stats.scope[POTAL_ISSUER_INDEX_CONFIG].calls == 1);
        REQUIRE(stats.scope[POTAL_ISSUER_INDEX_WRITE].calls == 0);
        REQUIRE(stats.raw[POTAL_FUNCT_SCALE_WRITE].samples == 7);
        REQUIRE(stats.raw[POTAL_FUNCT_SCALE_RELEASE].samples == 7);
        REQUIRE(stats.raw[POTAL_FUNCT_INDEX_CONFIG].samples == 1);
        REQUIRE(stats.raw[POTAL_FUNCT_INDEX_WRITE].samples == 1);
        for (unsigned i = 0; i < POTAL_ISSUER_SCOPE_COUNT; ++i) {
            REQUIRE(partitioned(&stats.scope[i]));
        }
        for (unsigned i = 0; i < 128; ++i) REQUIRE(stats.raw[i].errors == 0);
    } else {
        for (unsigned i = 0; i < POTAL_ISSUER_SCOPE_COUNT; ++i) {
            REQUIRE(stats.scope[i].calls == 0 && stats.scope[i].samples == 0);
        }
        for (unsigned i = 0; i < 128; ++i) REQUIRE(stats.raw[i].samples == 0);
    }
    REQUIRE(potal_paired_session_cleanup(r.session) == POTAL_OK);
    REQUIRE(potal_paired_session_close(r.session) == POTAL_OK);
    r.fail_funct = POTAL_FUNCT_INDEX_CONFIG;
    REQUIRE(potal_paired_session_open(&sink, &r.session) == POTAL_OK);
    REQUIRE(potal_paired_index_config(r.session, &config) == POTAL_HW_INVALID_ADDRESS);
    potal_issuer_profile_snapshot(&stats);
    REQUIRE(stats.raw[POTAL_FUNCT_INDEX_CONFIG].errors == (enabled ? 1u : 0u));
    REQUIRE(potal_paired_session_cleanup(r.session) == POTAL_OK);
    /* Cleanup releases ownership; close still reports the original sticky error. */
    REQUIRE(potal_paired_session_close(r.session) == POTAL_HW_INVALID_ADDRESS);
    potal_issuer_profile_flush(stdout);
    puts("POTAL_ISSUER_PROFILE_TEST PASS");
    return 0;
}
