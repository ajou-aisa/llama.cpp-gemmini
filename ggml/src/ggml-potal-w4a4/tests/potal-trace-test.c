#define _POSIX_C_SOURCE 200809L
#include "potal.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { CAPACITY = 16 };
static int dump;

static void require(int ok, const char *stage) {
    if (!ok) {
        fprintf(stdout, "POTAL_TRACE_TEST FAIL stage=%s\n", stage);
        exit(1);
    }
}

typedef struct record_sink {
    potal_paired_cmd_record records[CAPACITY];
    potal_status statuses[CAPACITY];
    unsigned count;
    int fail_write;
} record_sink;

static potal_status record_command(void *ctx, const potal_paired_cmd_record *record) {
    record_sink *s = ctx;
    require(s->count < CAPACITY, "record capacity");
    const potal_status status = s->fail_write && record->funct == POTAL_FUNCT_SCALE_WRITE
        ? POTAL_HW_SCALE_CONFLICT : POTAL_OK;
    s->fail_write = 0;
    s->records[s->count] = *record;
    s->statuses[s->count++] = status;
    if (dump) {
        printf("CMD funct=%u xd=%u rs1=%016" PRIx64 " rs2=%016" PRIx64 " status=%d\n",
               record->funct, record->xd, record->rs1, record->rs2, status);
    }
    return status;
}

static void check_trace(FILE *capture, const record_sink *sink, int enabled) {
    require(fflush(stderr) == 0 && fseek(capture, 0, SEEK_SET) == 0, "rewind trace");
    char line[256];
    for (unsigned i = 0; enabled && i < sink->count; ++i) {
        unsigned long long seq, rs1, rs2;
        unsigned funct, xd;
        int status;
        require(fgets(line, sizeof(line), capture) != NULL &&
                sscanf(line, "[POTAL_RAW] ENTER seq=%llu funct=%u xd=%u rs1=0x%llx rs2=0x%llx",
                       &seq, &funct, &xd, &rs1, &rs2) == 5, "parse ENTER");
        const potal_paired_cmd_record *r = &sink->records[i];
        require(seq == i + 1 && funct == r->funct && xd == r->xd &&
                rs1 == r->rs1 && rs2 == r->rs2, "ENTER command fields");
        require(fgets(line, sizeof(line), capture) != NULL &&
                sscanf(line, "[POTAL_RAW] EXIT seq=%llu funct=%u status=%d",
                       &seq, &funct, &status) == 3, "parse EXIT");
        require(seq == i + 1 && funct == r->funct && status == (int)sink->statuses[i],
                "EXIT status fields");
    }
    require(fgets(line, sizeof(line), capture) == NULL && !ferror(capture), "trace count");
}

static void run_case(const char *selection, int enabled, int fail) {
    record_sink records = {0};
    const potal_paired_sink sink = {record_command, &records};
    potal_paired_session *session = NULL;
    require((selection != NULL ? setenv("POTAL_RAW_TRACE", selection, 1) : unsetenv("POTAL_RAW_TRACE")) == 0,
            "select trace");
    FILE *capture = tmpfile();
    require(capture != NULL, "capture trace");
    require(fflush(stderr) == 0, "flush stderr");
    const int saved = dup(STDERR_FILENO);
    require(saved >= 0 && dup2(fileno(capture), STDERR_FILENO) >= 0, "redirect trace");
    require(potal_paired_session_open(&sink, &session) == POTAL_OK, "open");
    /* Opposite setting must not change tracing until the next session. */
    require(setenv("POTAL_RAW_TRACE", enabled ? "0" : "1", 1) == 0, "change active selection");
    require(potal_paired_tlb_flush(session) == POTAL_OK, "flush");
    for (unsigned col = 0; col < 2; ++col) {
        require(potal_paired_scale_write(session, 3, 7, col, col) == POTAL_OK, "write");
    }
    require(potal_paired_scale_write(session, 3, 7, 0, 0) == POTAL_ERR_CAPACITY, "duplicate");
    require(potal_paired_scale_release(session, 3, 8, 0) == POTAL_ERR_INVALID_ARGUMENT, "generation");
    require(potal_paired_session_close(session) == POTAL_ERR_BUSY, "live close");
    if (fail) {
        records.fail_write = 1;
        require(potal_paired_scale_write(session, 3, 7, 2, 2) == POTAL_HW_SCALE_CONFLICT, "hardware error");
        require(potal_paired_scale_release(session, 3, 7, 0) == POTAL_HW_SCALE_CONFLICT, "sticky error");
        require(potal_paired_session_cleanup(session) == POTAL_OK, "cleanup");
    } else {
        for (unsigned col = 0; col < 2; ++col) {
            require(potal_paired_scale_release(session, 3, 7, col) == POTAL_OK, "release");
        }
        require(potal_paired_scale_release(session, 3, 7, 0) == POTAL_ERR_INVALID_ARGUMENT, "release miss");
        require(potal_paired_fence(session) == POTAL_OK, "fence");
    }
    const potal_paired_session_status q = potal_paired_session_query(session);
    require(q.live_scale_slots == 0 && q.outstanding_loops == 0 && q.outstanding_readbacks == 0 &&
            q.commands == records.count && q.drains == (unsigned)(fail ? 0 : 3), "final state");
    require(q.failed == (unsigned)fail && (!fail || q.error == POTAL_HW_SCALE_CONFLICT), "failure state");
    require(potal_paired_session_close(session) == (fail ? POTAL_HW_SCALE_CONFLICT : POTAL_OK), "close");
    if (!dump) check_trace(capture, &records, enabled);
    require(fflush(stderr) == 0 && dup2(saved, STDERR_FILENO) >= 0, "restore stderr");
    require(close(saved) == 0 && fclose(capture) == 0, "close capture");
}

int main(int argc, char **argv) {
    dump = argc == 2 && strcmp(argv[1], "--dump") == 0;
    require(argc == 1 || dump, "arguments");
    const char *old = getenv("POTAL_RAW_TRACE");
    char *saved = old != NULL ? strdup(old) : NULL;
    require(old == NULL || saved != NULL, "save environment");
    run_case("0", 0, 0);
    run_case("1suffix", 1, 1);
    run_case("0", 0, 0);
    run_case("1", 1, 0);
    run_case(NULL, 0, 0);
    run_case("", 0, 0);
    run_case("true", 0, 0);
    if (saved != NULL) {
        require(setenv("POTAL_RAW_TRACE", saved, 1) == 0, "restore environment");
        free(saved);
    } else {
        require(unsetenv("POTAL_RAW_TRACE") == 0, "restore environment");
    }
    puts("POTAL_TRACE_TEST PASS");
    return 0;
}
