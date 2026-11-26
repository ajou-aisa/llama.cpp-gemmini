#include "pack.hpp"

#include "ggml-backend.h"

#include <algorithm>
#include <limits>

namespace ggml::potal {

namespace {

// Unallocated (the scheduler places it in the POTAL default buffer type, which
// is host memory) or backed by a host buffer. Views are judged by their source.
bool in_host_memory(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

} // namespace

ggml_potal_reject check_mul_mat(const ggml_tensor * op, bool require_data, mul_mat_shape * shape) {
    // No raw GGML I8 route in this fork: reject before touching operand data.
    (void)op;
    (void)require_data;
    (void)shape;
    return GGML_POTAL_REJECT_TYPE;
}

void pack_a(const ggml_tensor * src1, const mul_mat_shape & s, int8_t * a) {
    const char * base = static_cast<const char *>(src1->data);
    for (int64_t m = 0; m < s.m; ++m) {
        for (int64_t k = 0; k < s.k; ++k) {
            a[m * s.k + k] = *reinterpret_cast<const int8_t *>(base + k * src1->nb[0] + m * src1->nb[1]);
        }
    }
}

void pack_b(const ggml_tensor * src0, const mul_mat_shape & s, int8_t * b) {
    const char * base = static_cast<const char *>(src0->data);
    for (int64_t k = 0; k < s.k; ++k) {
        for (int64_t n = 0; n < s.n; ++n) {
            b[k * s.n + n] = *reinterpret_cast<const int8_t *>(base + k * src0->nb[0] + n * src0->nb[1]);
        }
    }
}

void unpack_c(const int32_t * c, const mul_mat_shape & s, ggml_tensor * dst) {
    char * base = static_cast<char *>(dst->data);
    for (int64_t m = 0; m < s.m; ++m) {
        for (int64_t n = 0; n < s.n; ++n) {
            *reinterpret_cast<float *>(base + n * dst->nb[0] + m * dst->nb[1]) = static_cast<float>(c[m * s.n + n]);
        }
    }
}

void poison_dst(ggml_tensor * dst) {
    if (dst == nullptr || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !ggml_is_contiguous(dst) || !in_host_memory(dst)) {
        return;
    }
    float * out = static_cast<float *>(dst->data);
    std::fill(out, out + ggml_nelements(dst), std::numeric_limits<float>::quiet_NaN());
}

} // namespace ggml::potal
