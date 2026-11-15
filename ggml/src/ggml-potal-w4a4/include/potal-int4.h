#ifndef POTAL_INT4_H
#define POTAL_INT4_H

#include <stddef.h>
#include <stdint.h>
#include "potal.h"

/* Transport ABI 1: adjacent signed nibbles, even column in the low nibble.
 * Scalar helpers require valid row storage, indices and signed INT4 values. */
static inline size_t potal_int4_row_bytes(size_t cols) {
    return cols / 2 + cols % 2;
}

static inline uint8_t potal_int4_pack_pair(int8_t lo, int8_t hi) {
    return (uint8_t)(((uint8_t)lo & 15u) | (((uint8_t)hi & 15u) << 4));
}

static inline int8_t potal_int4_get(const uint8_t *row, size_t col) {
    const unsigned u = (row[col / 2] >> (4 * (col % 2))) & 15u;
    return (int8_t)((int)(u ^ 8u) - 8);
}

static inline void potal_int4_set(uint8_t *row, size_t col, int8_t value) {
    const unsigned shift = 4 * (col % 2);
    row[col / 2] = (uint8_t)((row[col / 2] & ~(15u << shift)) |
                            (((uint8_t)value & 15u) << shift));
}

/* Validate the entire source before writing. Source padding is never read;
 * destination padding (including an odd final high nibble) is always zero. */
static inline potal_status potal_int4_pack_rows(
    uint8_t *dst, size_t dst_bytes, size_t dst_stride_bytes,
    const int8_t *src, size_t src_stride_elems, size_t rows, size_t cols) {
    if (!dst || !src || !rows || !cols || src_stride_elems < cols ||
        dst_stride_bytes < potal_int4_row_bytes(cols) ||
        rows > SIZE_MAX / dst_stride_bytes ||
        rows - 1 > (SIZE_MAX - cols) / src_stride_elems) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const size_t dst_span = rows * dst_stride_bytes;
    const size_t src_span = (rows - 1) * src_stride_elems + cols;
    const uintptr_t d = (uintptr_t)dst, s = (uintptr_t)src;
    if (dst_span > UINTPTR_MAX - d || src_span > UINTPTR_MAX - s ||
        (d < s + src_span && s < d + dst_span)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (dst_bytes < dst_span) {
        return POTAL_ERR_CAPACITY;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const int8_t digit = src[r * src_stride_elems + c];
            if (digit < -8 || digit > 7) return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        uint8_t *row = dst + r * dst_stride_bytes;
        for (size_t b = 0; b < dst_stride_bytes; ++b) row[b] = 0;
        for (size_t c = 0; c < cols; ++c) {
            potal_int4_set(row, c, src[r * src_stride_elems + c]);
        }
    }
    return POTAL_OK;
}

#endif /* POTAL_INT4_H */
