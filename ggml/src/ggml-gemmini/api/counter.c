#include "gemmini.h"

static gemmini_status_t counter_issue(const gemmini_transport_t *transport, gemmini_counter_op_t op, uint32_t index,
                                      uint32_t event, uint64_t *value) {
    const gemmini_counter_desc_t desc = { .op = op, .index = index, .event = event };
    gemmini_command_t            command;
    const gemmini_status_t       status = gemmini_encode_counter(&desc, &command);
    if (status != GEMMINI_OK) {
        return status;
    }
    uint64_t discarded;
    return gemmini_issue(transport, &command, value ? value : &discarded);
}

gemmini_status_t gemmini_counter_read(const gemmini_transport_t *transport, uint32_t index, uint64_t *value) {
    if (!value) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    return counter_issue(transport, GEMMINI_COUNTER_OP_READ, index, 0, value);
}

gemmini_status_t gemmini_counter_configure(const gemmini_transport_t *transport, uint32_t index, uint32_t event) {
    return counter_issue(transport, GEMMINI_COUNTER_OP_CONFIGURE, index, event, NULL);
}

gemmini_status_t gemmini_counter_snapshot_take(const gemmini_transport_t *transport) {
    return counter_issue(transport, GEMMINI_COUNTER_OP_SNAPSHOT_TAKE, 0, 0, NULL);
}

gemmini_status_t gemmini_counter_snapshot_reset(const gemmini_transport_t *transport) {
    return counter_issue(transport, GEMMINI_COUNTER_OP_SNAPSHOT_RESET, 0, 0, NULL);
}

gemmini_status_t gemmini_counter_reset(const gemmini_transport_t *transport) {
    return counter_issue(transport, GEMMINI_COUNTER_OP_RESET, 0, 0, NULL);
}
