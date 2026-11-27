/* CPU-only command-boundary tests; link with potal-api and its generated headers.
 * R: exact scalar protocol, successful prefixes, publication, failure and drain
 * contracts. Dropped/reordered writes, eager validation/publication or lost live
 * work must fail. Existing scalar tests do not exercise the window entry point.
 */
#include "potal.h"
#include "potal_hw_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAPACITY (POTAL_HW_INDEX_ENTRIES + 16)
#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

typedef struct recording {
    potal_paired_session *session;
    potal_paired_cmd_record records[CAPACITY];
    potal_paired_session_status before[CAPACITY], after;
    size_t count, fail_at;
    potal_status injected, result, later, close;
    uint64_t written;
    int reenter;
} recording;

typedef struct scenario {
    const char *name;
    uint32_t rows, pair;
    int invalid;
    size_t fail_at; /* 1 = CONFIG, 2 = first WRITE */
    potal_status injected;
} scenario;

static potal_status record_command(void *ctx, const potal_paired_cmd_record *command) {
    recording *r = ctx;
    REQUIRE(r->count < CAPACITY);
    r->before[r->count] = potal_paired_session_query(r->session);
    r->records[r->count++] = *command;
    if (r->reenter) {
        r->reenter = 0;
        const potal_index_config cfg = { 0, POTAL_HW_BANK_ROWS, 1, 32, 1 };
        const uint32_t entry = 0;
        uint64_t written = 9;
        REQUIRE(potal_paired_index_window_issue(r->session, &cfg, &entry, 1, &written) == POTAL_ERR_BUSY);
        REQUIRE(written == 0);
    }
    return r->count == r->fail_at ? r->injected : POTAL_OK;
}

static int same_state(potal_paired_session_status a, potal_paired_session_status b) {
#define SAME(field) if (a.field != b.field) return 0
    SAME(failed); SAME(error); SAME(outstanding_loops); SAME(outstanding_readbacks);
    SAME(outstanding_acc_updates); SAME(index_transaction_open);
    SAME(index_transaction_pair); SAME(index_transaction_written);
    SAME(live_scale_slots); SAME(commands); SAME(drains);
    if ((a.error_stage == NULL) != (b.error_stage == NULL)) return 0;
    if (a.error_stage && strcmp(a.error_stage, b.error_stage)) return 0;
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p) {
        SAME(pair[p].resident_b); SAME(pair[p].b_sp_base); SAME(pair[p].b_j_tiles);
        SAME(pair[p].b_k_rows); SAME(pair[p].main_a_tiles); SAME(pair[p].index_valid);
        SAME(pair[p].index_active_rows); SAME(pair[p].outstanding);
    }
#undef SAME
    return 1;
}

static void execute(recording *r, const scenario *test, int window) {
    uint32_t entries[POTAL_HW_INDEX_ENTRIES];
    for (uint32_t i = 0; i < test->rows; ++i) entries[i] = (i * 7 + 3) % (2 * POTAL_HW_DIM - 1);
    if (test->invalid >= 0) entries[test->invalid] = 2 * POTAL_HW_DIM - 1;
    const potal_index_config cfg = {
        test->pair, (2 * test->pair + 1) * POTAL_HW_BANK_ROWS, 1,
        2 * POTAL_HW_DIM - 1, test->rows
    };
    const potal_paired_sink sink = { record_command, r, NULL };
    REQUIRE(potal_paired_session_open(&sink, &r->session) == POTAL_OK);
    /* A previously complete window must be invalidated by successful CONFIG. */
    potal_index_config seed = cfg;
    seed.active_rows = 1;
    const potal_index_write first = { 0, 0 };
    REQUIRE(potal_paired_index_config(r->session, &seed) == POTAL_OK);
    REQUIRE(potal_paired_index_write(r->session, &first) == POTAL_OK);
    REQUIRE(potal_paired_scale_write(r->session, 0, 7, 0, 0) == POTAL_OK);
    _Alignas(128) static int32_t output[POTAL_HW_DIM * POTAL_HW_DIM];
    const potal_acc_readback readback = { 0, 1, 1, 0, 0, output, POTAL_HW_DIM };
    REQUIRE(potal_paired_acc_readback_issue(r->session, &readback) == POTAL_OK);
    r->count = 0;
    r->fail_at = test->fail_at;
    r->injected = test->injected;
    if (window) {
        r->written = UINT64_MAX;
        r->result = potal_paired_index_window_issue(r->session, &cfg, entries, test->rows, &r->written);
    } else {
        r->result = potal_paired_index_config(r->session, &cfg);
        for (uint32_t i = 0; r->result == POTAL_OK && i < test->rows; ++i) {
            const potal_index_write entry = { i, entries[i] };
            r->result = potal_paired_index_write(r->session, &entry);
            if (r->result == POTAL_OK) ++r->written;
        }
    }
    r->after = potal_paired_session_query(r->session);
    REQUIRE(r->after.outstanding_readbacks == 1 && r->after.live_scale_slots == 1);
    REQUIRE(r->after.drains == 0);
    const uint64_t prefix = test->fail_at ? (test->fail_at > 1 ? test->fail_at - 2 : 0) :
        (test->invalid >= 0 ? (uint32_t)test->invalid : test->rows);
    REQUIRE(r->written == prefix);
    REQUIRE(r->result == (test->fail_at ? test->injected :
        (test->invalid >= 0 ? POTAL_ERR_INVALID_ARGUMENT : POTAL_OK)));
    const size_t attempts = test->fail_at ? test->fail_at : 1 + (size_t)prefix;
    REQUIRE(r->count == attempts);
    REQUIRE(r->records[0].funct == POTAL_FUNCT_INDEX_CONFIG && r->records[0].xd == 1);
    REQUIRE(r->records[0].rs1 == ((UINT64_C(1) << 32) | cfg.b_sp_base));
    REQUIRE(r->records[0].rs2 == (((uint64_t)test->rows << 32) | cfg.source_k_rows));
    for (size_t i = 1; i < attempts; ++i) {
        REQUIRE(r->records[i].funct == POTAL_FUNCT_INDEX_WRITE && r->records[i].xd == 1);
        REQUIRE(r->records[i].rs1 == i - 1 && r->records[i].rs2 == entries[i - 1]);
        REQUIRE(r->before[i].index_transaction_written == i - 1);
        REQUIRE(r->before[i].index_transaction_open && !r->before[i].pair[test->pair].index_valid);
    }
    REQUIRE(r->after.pair[test->pair].index_valid ==
        (test->fail_at == 1 || r->result == POTAL_OK));
    if (test->fail_at != 1) {
        REQUIRE(r->after.index_transaction_written == prefix);
        REQUIRE(r->after.index_transaction_open == (prefix != test->rows));
    }
    r->fail_at = 0;
    if (r->after.failed) {
        const potal_index_write next = { (uint32_t)prefix, 0 };
        r->later = potal_paired_index_write(r->session, &next);
        REQUIRE(r->later == test->injected && r->count == attempts);
        REQUIRE(potal_paired_session_cleanup(r->session) == POTAL_OK);
    } else {
        /* Validation is recoverable: resume precisely at the successful prefix. */
        for (uint32_t i = (uint32_t)prefix; i < test->rows; ++i) {
            const potal_index_write next = { i, 0 };
            REQUIRE(potal_paired_index_write(r->session, &next) == POTAL_OK);
        }
        REQUIRE(potal_paired_session_close(r->session) == POTAL_ERR_BUSY);
        REQUIRE(potal_paired_fence(r->session) == POTAL_OK);
        REQUIRE(potal_paired_scale_release(r->session, 0, 7, 0) == POTAL_OK);
    }
    const potal_paired_session_status drained = potal_paired_session_query(r->session);
    REQUIRE(!drained.outstanding_readbacks && !drained.live_scale_slots);
    r->close = potal_paired_session_close(r->session);
    REQUIRE(r->close == (r->after.failed ? test->injected : POTAL_OK));
    r->session = NULL;
}

static void rejects_bad_shape_without_commands(void) {
    recording r = {0};
    const potal_paired_sink sink = { record_command, &r, NULL };
    REQUIRE(potal_paired_session_open(&sink, &r.session) == POTAL_OK);
    potal_index_config cfg = { 0, POTAL_HW_BANK_ROWS, 1, 32, 1 };
    const uint32_t entry = 0;
    uint64_t written = 9;
    REQUIRE(potal_paired_index_window_issue(r.session, &cfg, &entry, 0, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(written == 0);
    REQUIRE(potal_paired_index_window_issue(r.session, NULL, &entry, 1, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(potal_paired_index_window_issue(r.session, &cfg, NULL, 1, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(potal_paired_index_window_issue(r.session, &cfg, &entry, 1, NULL) == POTAL_ERR_INVALID_ARGUMENT);
    cfg.pair = POTAL_PAIRS;
    REQUIRE(potal_paired_index_window_issue(r.session, &cfg, &entry, 1, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(r.count == 0 && written == 0);
    REQUIRE(potal_paired_session_close(r.session) == POTAL_OK);
}

/* R: managed admission and reentrant mutation retain scalar BUSY semantics. */
static void managed_admission_and_reentrancy(void) {
    recording r = {0};
    const potal_paired_sink sink = { record_command, &r, NULL };
    REQUIRE(potal_paired_session_open(&sink, &r.session) == POTAL_OK);
    _Alignas(128) static const int8_t weight[32 * 32 / 2] = {0};
    const potal_h2_weight w = { weight, 32, 32, 32, sizeof(weight) };
    REQUIRE(potal_paired_h2_begin(r.session, &w) == POTAL_OK);
    const potal_index_config cfg = { 0, POTAL_HW_BANK_ROWS, 1, 32, 1 };
    const uint32_t entry = 0;
    uint64_t written = 9;
    REQUIRE(potal_paired_index_window_issue(r.session, &cfg, &entry, 1, &written) == POTAL_ERR_INVALID_ARGUMENT);
    REQUIRE(written == 0 && r.count == 0);
    r.reenter = 1;
    REQUIRE(potal_paired_scale_write(r.session, 0, 7, 0, 0) == POTAL_OK);
    REQUIRE(!r.reenter && r.count == 1);
    REQUIRE(potal_paired_scale_release(r.session, 0, 7, 0) == POTAL_OK);
    REQUIRE(potal_paired_h2_close(&r.session) == POTAL_OK);
}

int main(void) {
    const scenario tests[] = {
        { "normal", 32, 0, -1, 0, POTAL_OK },
        { "tail", 7, 1, -1, 0, POTAL_OK },
        { "full", POTAL_HW_INDEX_ENTRIES, 0, -1, 0, POTAL_OK },
        { "invalid-first", 7, 0, 0, 0, POTAL_OK },
        { "invalid-middle", 7, 1, 3, 0, POTAL_OK },
        { "invalid-last", 7, 0, 6, 0, POTAL_OK },
        { "raw-config", 7, 1, -1, 1, POTAL_HW_INVALID_ADDRESS },
        { "raw-first", 7, 0, -1, 2, POTAL_HW_INVALID_ENCODING },
        { "raw-middle", 7, 1, -1, 5, POTAL_HW_BAD_COMMAND },
        { "raw-last", 7, 0, -1, 8, POTAL_ERR_BAD_RESPONSE },
    };
    rejects_bad_shape_without_commands();
    managed_admission_and_reentrancy();
    for (size_t t = 0; t < sizeof(tests) / sizeof(tests[0]); ++t) {
        recording *scalar = calloc(1, sizeof(*scalar)), *window = calloc(1, sizeof(*window));
        REQUIRE(scalar && window);
        execute(scalar, &tests[t], 0);
        execute(window, &tests[t], 1);
        REQUIRE(scalar->result == window->result && scalar->written == window->written);
        REQUIRE(same_state(scalar->after, window->after));
        REQUIRE(scalar->count == window->count && scalar->close == window->close);
        for (size_t i = 0; i < scalar->count; ++i) {
            const potal_paired_cmd_record a = scalar->records[i], b = window->records[i];
            REQUIRE(a.funct == b.funct && a.xd == b.xd && a.rs1 == b.rs1 && a.rs2 == b.rs2);
            REQUIRE(same_state(scalar->before[i], window->before[i]));
        }
        printf("PASS %s\n", tests[t].name);
        free(scalar);
        free(window);
    }
    puts("POTAL_INDEX_WINDOW_TEST PASS");
    return 0;
}
