/*
 * Gemmini command transport: the only place commands leave the software.
 *
 * issue() hands one command to the accelerator (or a model of it). It returns
 * when the command was accepted, never when its work is done. A command with
 * xd = 1 must be given a response pointer, one with xd = 0 must not: the
 * transport never invents a response for a command that has none.
 *
 * drain() is the Gemmini fence: on RoCC a CPU "fence", which Rocket stalls
 * until the accelerator is no longer busy. Gemmini returns no status for it
 * (unlike the PoTAL FENCE command).
 */
#ifndef GEMMINI_TRANSPORT_H
#define GEMMINI_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#include "gemmini_command.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemmini_transport {
    void *context;
    gemmini_status_t (*issue)(void *context, const gemmini_command_t *command, uint64_t *response);
    gemmini_status_t (*drain)(void *context);
} gemmini_transport_t;

/* Checks the xd/response pairing, then calls transport->issue. */
gemmini_status_t gemmini_issue(const gemmini_transport_t *transport, const gemmini_command_t *command,
                               uint64_t *response);
gemmini_status_t gemmini_issue_all(const gemmini_transport_t *transport, const gemmini_command_t *commands,
                                   size_t count);
gemmini_status_t gemmini_drain(const gemmini_transport_t *transport);

/*
 * RoCC custom-3 issue. On RISC-V it executes the instruction; on any other
 * target every call returns GEMMINI_ERR_NO_HARDWARE.
 */
const gemmini_transport_t *gemmini_rocc_transport(void);
int gemmini_rocc_available(void);

/*
 * Test transport: records commands and drains in order. It computes nothing
 * and models no hardware timing. Responses are the configured value.
 */
typedef enum gemmini_record_kind {
    GEMMINI_RECORD_ISSUE,
    GEMMINI_RECORD_DRAIN,
} gemmini_record_kind_t;

typedef struct gemmini_record {
    uint64_t              seq;
    gemmini_record_kind_t kind;
    gemmini_command_t     command; /* zero for drains */
    int                   has_response;
    gemmini_status_t      status;
} gemmini_record_t;

typedef struct gemmini_recorder {
    gemmini_transport_t transport;
    gemmini_record_t   *records;
    size_t              capacity;
    size_t              count;
    uint64_t            next_seq;
    uint64_t            response_value;
    uint64_t            fail_seq;    /* inject fail_status at this seq; UINT64_MAX: never */
    gemmini_status_t    fail_status;
    int                 overflowed;  /* a record did not fit; the issue failed */
} gemmini_recorder_t;

void gemmini_recorder_init(gemmini_recorder_t *recorder, gemmini_record_t *records, size_t capacity);
void gemmini_recorder_reset(gemmini_recorder_t *recorder);

#ifdef __cplusplus
}
#endif

#endif /* GEMMINI_TRANSPORT_H */
