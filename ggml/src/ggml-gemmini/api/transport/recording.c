#include "gemmini.h"

#include <string.h>

static gemmini_status_t record(gemmini_recorder_t *recorder, gemmini_record_kind_t kind,
                               const gemmini_command_t *command, uint64_t *response) {
    const uint64_t seq = recorder->next_seq++;
    if (recorder->count == recorder->capacity) {
        recorder->overflowed = 1;
        return GEMMINI_ERR_TRANSPORT;
    }
    gemmini_record_t *entry = &recorder->records[recorder->count++];
    memset(entry, 0, sizeof(*entry));
    entry->seq          = seq;
    entry->kind         = kind;
    entry->has_response = response != NULL;
    entry->status       = seq == recorder->fail_seq ? recorder->fail_status : GEMMINI_OK;
    if (command) {
        entry->command = *command;
    }
    if (entry->status == GEMMINI_OK && response) {
        *response = recorder->response_value;
    }
    return entry->status;
}

static gemmini_status_t recorder_issue(void *context, const gemmini_command_t *command, uint64_t *response) {
    return record((gemmini_recorder_t *) context, GEMMINI_RECORD_ISSUE, command, response);
}

static gemmini_status_t recorder_drain(void *context) {
    return record((gemmini_recorder_t *) context, GEMMINI_RECORD_DRAIN, NULL, NULL);
}

void gemmini_recorder_init(gemmini_recorder_t *recorder, gemmini_record_t *records, size_t capacity) {
    memset(recorder, 0, sizeof(*recorder));
    recorder->transport.context = recorder;
    recorder->transport.issue   = recorder_issue;
    recorder->transport.drain   = recorder_drain;
    recorder->records           = records;
    recorder->capacity          = capacity;
    recorder->fail_seq          = UINT64_MAX;
}

void gemmini_recorder_reset(gemmini_recorder_t *recorder) {
    recorder->count      = 0;
    recorder->next_seq   = 0;
    recorder->overflowed = 0;
}
