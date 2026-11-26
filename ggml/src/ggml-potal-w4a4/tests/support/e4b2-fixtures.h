#ifndef POTAL_E4B2_FIXTURES_H
#define POTAL_E4B2_FIXTURES_H

#include <limits.h>
#include <stdint.h>

/* Original-tensor oracle, independent of command packing and physical rows. */
static inline int8_t e4b2_weight(unsigned pair, unsigned mode, unsigned k, unsigned j) {
    if (mode == 1) {
        if (k == 0 || k == 16) return 1;
        return k == 32 ? -1 : 0;
    }
    if (mode == 2) return k == 96 ? 1 : k == 112 ? -1 : 0;
    if (mode == 3) return k < 16 ? 127 : k == 16 ? -127 : 0;
    if (j % 4 == 0)
        return (int8_t)((int)(((k / 16) * 9 + j / 16 + pair * 47) % 251) - 125);
    return (int8_t)((int)((k * 3 + j * 5 + pair * 11) % 15) - 7);
}

static inline int8_t e4b2_activation(unsigned mode, unsigned i, unsigned k) {
    return mode ? 1 : (int8_t)((int)((i * 7 + k * 2) % 11) - 5);
}

static inline uint32_t e4b2_carrier(unsigned mode, unsigned block, unsigned j) {
    if (mode == 1 || mode == 2) return 31;
    if (mode == 3) return 24;
    if ((block + j) % 13 == 0) return UINT32_C(0x80000000);
    return (block + j) % 4;
}

static inline int32_t e4b2_clamp(int64_t value) {
    return value > INT32_MAX ? INT32_MAX :
        value < INT32_MIN ? INT32_MIN : (int32_t)value;
}

static inline int32_t e4b2_shift(int64_t dot, uint32_t carrier) {
    if (carrier == UINT32_C(0x80000000) || dot == 0) return 0;
    if (carrier >= 32) return dot > 0 ? INT32_MAX : INT32_MIN;
    return e4b2_clamp(dot * (INT64_C(1) << carrier));
}

static inline int32_t e4b2_oracle_partial(unsigned pair, unsigned weight_mode,
                                         unsigned arithmetic_mode, unsigned i, unsigned j,
                                         const uint32_t *original_k, unsigned count) {
    int64_t dot = 0;
    for (unsigned x = 0; x < count; ++x)
        dot += (int64_t)e4b2_activation(arithmetic_mode, i, original_k[x]) *
            e4b2_weight(pair, weight_mode, original_k[x], j);
    return e4b2_shift(dot, e4b2_carrier(arithmetic_mode, original_k[0] / 32, j));
}

#endif
