#pragma once

#include <stddef.h>
#include <string.h>

static inline void gemmini_copy_layer_part(char *out, size_t out_size, size_t *pos,
                                           const char *text, size_t length, int lowercase) {
    for (size_t i = 0; i < length && *pos + 1 < out_size; ++i) {
        char c = text[i];
        if (lowercase && c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
        out[(*pos)++] = c;
    }
    out[*pos] = '\0';
}

static inline void gemmini_get_layer(const char *tensor_name, char *out, size_t out_size) {
    if (out == NULL || out_size == 0) {
        return;
    }

    out[0] = '\0';
    if (tensor_name == NULL) {
        return;
    }

    if (strncmp(tensor_name, "blk.", 4) == 0) {
        size_t pos = 0;
        gemmini_copy_layer_part(out, out_size, &pos, tensor_name, strlen(tensor_name), 0);
        return;
    }

    const char *suffix = tensor_name;
    const char *digit_end = NULL;
    while ((suffix = strchr(suffix, '-')) != NULL) {
        const char *digit = suffix + 1;
        if (suffix != tensor_name && *digit >= '0' && *digit <= '9') {
            while (*digit >= '0' && *digit <= '9') {
                ++digit;
            }
            if (*digit == '\0' || *digit == ' ' || *digit == '(') {
                digit_end = digit;
                break;
            }
        }
        ++suffix;
    }

    if (digit_end != NULL) {
        const char *base_end = strpbrk(tensor_name, " (");
        if (base_end == NULL || base_end > suffix) {
            base_end = suffix;
        }
        size_t pos = 0;
        gemmini_copy_layer_part(out, out_size, &pos, "blk.", 4, 0);
        gemmini_copy_layer_part(out, out_size, &pos, suffix + 1,
                                (size_t) (digit_end - suffix - 1), 0);
        gemmini_copy_layer_part(out, out_size, &pos, ".", 1, 0);
        gemmini_copy_layer_part(out, out_size, &pos, tensor_name,
                                (size_t) (base_end - tensor_name), 1);
        return;
    }

    size_t pos = 0;
    gemmini_copy_layer_part(out, out_size, &pos, tensor_name, strlen(tensor_name), 0);
}
