#pragma once

#include <stddef.h>
#include <string.h>

static inline void gemmini_get_layer(const char *tensor_name, char *out, size_t out_size) {
    if (out == NULL || out_size == 0) {
        return;
    }

    out[0] = '\0';

    if (tensor_name == NULL) {
        return;
    }

    const char *start = tensor_name;
    while (*start == '-') {
        ++start;
    }

    if (*start == '\0') {
        return;
    }

    const char *end = strchr(start, '-');
    size_t len = end ? (size_t)(end - start) : strlen(start);

    if (len >= out_size) {
        len = out_size - 1;
    }

    memcpy(out, start, len);
    out[len] = '\0';
}
