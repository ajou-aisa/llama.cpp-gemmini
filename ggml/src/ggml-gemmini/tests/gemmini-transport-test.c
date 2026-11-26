#include <stdio.h>
#include <string.h>

#include "gemmini.h"

static int failures = 0;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

static gemmini_command_t flush_command(void) {
    const gemmini_flush_desc_t d = { false };
    gemmini_command_t          c;
    gemmini_encode_flush(&d, &c);
    return c;
}

static gemmini_command_t counter_command(void) {
    const gemmini_counter_desc_t d = { GEMMINI_COUNTER_OP_READ, 1, 0 };
    gemmini_command_t            c;
    gemmini_encode_counter(&d, &c);
    return c;
}

static void test_order_and_drain(void) {
    gemmini_record_t   records[16];
    gemmini_recorder_t r;
    gemmini_recorder_init(&r, records, 16);
    gemmini_command_t       loop[GEMMINI_LOOP_WS_COMMANDS];
    gemmini_loop_ws_desc_t  d;
    memset(&d, 0, sizeof(d));
    d.i = d.j = d.k = 1;
    expect(gemmini_encode_loop_ws(&d, loop) == GEMMINI_OK, "encode loop");
    expect(gemmini_issue_all(&r.transport, loop, GEMMINI_LOOP_WS_COMMANDS) == GEMMINI_OK, "issue loop");
    expect(gemmini_drain(&r.transport) == GEMMINI_OK, "drain");
    expect(r.count == GEMMINI_LOOP_WS_COMMANDS + 1, "7 records");
    for (size_t i = 0; i < GEMMINI_LOOP_WS_COMMANDS; ++i) {
        expect(records[i].seq == i && records[i].kind == GEMMINI_RECORD_ISSUE, "issue order");
        expect(records[i].command.funct == loop[i].funct && records[i].command.rs1 == loop[i].rs1 &&
                   records[i].command.rs2 == loop[i].rs2,
               "recorded operands");
        expect(!records[i].has_response, "no response requested");
    }
    expect(records[6].kind == GEMMINI_RECORD_DRAIN && records[6].seq == 6, "drain recorded last");
}

static void test_responses(void) {
    gemmini_record_t   records[4];
    gemmini_recorder_t r;
    gemmini_recorder_init(&r, records, 4);
    r.response_value = 0xfedcba9876543210ull;
    const gemmini_command_t counter = counter_command();
    const gemmini_command_t flush   = flush_command();
    uint64_t                value   = 0;
    expect(gemmini_issue(&r.transport, &counter, NULL) == GEMMINI_ERR_RESPONSE, "xd=1 without response rejected");
    expect(gemmini_issue(&r.transport, &flush, &value) == GEMMINI_ERR_RESPONSE, "xd=0 with response rejected");
    expect(r.count == 0, "rejected commands are not issued");
    expect(gemmini_issue(&r.transport, &counter, &value) == GEMMINI_OK, "counter issued");
    expect(value == 0xfedcba9876543210ull, "full 64-bit response");
    expect(records[0].has_response, "response recorded");
    expect(gemmini_counter_read(&r.transport, 2, &value) == GEMMINI_OK && records[1].command.rs1 == (2u << 4),
           "counter helper");
}

static void test_errors(void) {
    gemmini_record_t   records[2];
    gemmini_recorder_t r;
    gemmini_recorder_init(&r, records, 2);
    r.fail_seq    = 1;
    r.fail_status = GEMMINI_ERR_TRANSPORT;
    const gemmini_command_t flush = flush_command();
    expect(gemmini_issue(&r.transport, &flush, NULL) == GEMMINI_OK, "seq 0 ok");
    expect(gemmini_issue(&r.transport, &flush, NULL) == GEMMINI_ERR_TRANSPORT, "injected error propagates");
    expect(gemmini_issue(&r.transport, &flush, NULL) == GEMMINI_ERR_TRANSPORT && r.overflowed, "capacity overflow fails");

    gemmini_command_t loop[GEMMINI_LOOP_WS_COMMANDS];
    gemmini_loop_ws_desc_t d;
    memset(&d, 0, sizeof(d));
    d.i = d.j = d.k = 1;
    gemmini_encode_loop_ws(&d, loop);
    gemmini_recorder_reset(&r);
    r.fail_seq = 0;
    expect(gemmini_issue_all(&r.transport, loop, GEMMINI_LOOP_WS_COMMANDS) == GEMMINI_ERR_TRANSPORT && r.count == 1,
           "issue_all stops at the first failure");

    gemmini_command_t unknown = flush;
    unknown.funct = 15;
    expect(gemmini_issue(&r.transport, &unknown, NULL) == GEMMINI_ERR_INVALID_ARGUMENT, "unsupported funct rejected");
    expect(gemmini_issue(NULL, &flush, NULL) == GEMMINI_ERR_INVALID_ARGUMENT, "NULL transport rejected");
}

static void test_rocc_host(void) {
    const gemmini_transport_t *rocc  = gemmini_rocc_transport();
    const gemmini_command_t    flush = flush_command();
#if defined(__riscv)
    expect(gemmini_rocc_available() == 1, "RoCC available on RISC-V");
    (void) rocc;
    (void) flush;
#else
    expect(gemmini_rocc_available() == 0, "no RoCC on this host");
    expect(gemmini_issue(rocc, &flush, NULL) == GEMMINI_ERR_NO_HARDWARE, "host RoCC issue fails explicitly");
    expect(gemmini_drain(rocc) == GEMMINI_ERR_NO_HARDWARE, "host RoCC drain fails explicitly");
#endif
}

int main(void) {
    test_order_and_drain();
    test_responses();
    test_errors();
    test_rocc_host();
    if (failures) {
        fprintf(stderr, "gemmini-transport-test: %d failure(s)\n", failures);
        return 1;
    }
    printf("GEMMINI_TRANSPORT_TEST PASS\n");
    return 0;
}
