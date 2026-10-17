#include "pack.hpp"

#include "ggml-backend.h"

#include <algorithm>
#include <limits>

namespace ggml::potal {

namespace {

bool is_2d(const ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1;
}

// Unallocated (the scheduler places it in the POTAL default buffer type, which
// is host memory) or backed by a host buffer. Views are judged by their source.
bool in_host_memory(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

} // namespace

ggml_potal_reject check_mul_mat(const ggml_tensor * op, bool require_data, mul_mat_shape * shape) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (op->op != GGML_OP_MUL_MAT || w == nullptr || x == nullptr) {
        return GGML_POTAL_REJECT_OP;
    }
    // op_params[0] is the precision hint, meaningless for an integer GEMM; any
    // other tag (e.g. a Gemmini attention role) asks for semantics this path lacks.
    for (size_t i = 1; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
        if (op->op_params[i] != 0) {
            return GGML_POTAL_REJECT_OP_PARAMS;
        }
    }
    if (w->type != GGML_TYPE_I8 || x->type != GGML_TYPE_I8 || op->type != GGML_TYPE_F32) {
        return GGML_POTAL_REJECT_TYPE;
    }
    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = x->ne[1];
    if ((k != TILE && k != K_MAX) || n != TILE || m != TILE || x->ne[0] != k ||
        op->ne[0] != n || op->ne[1] != m || !is_2d(w) || !is_2d(x) || !is_2d(op)) {
        return GGML_POTAL_REJECT_SHAPE;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op)) {
        return GGML_POTAL_REJECT_LAYOUT;
    }
    if (!in_host_memory(w) || !in_host_memory(x) || !in_host_memory(op) ||
        (require_data && (w->data == nullptr || x->data == nullptr || op->data == nullptr))) {
        return GGML_POTAL_REJECT_MEMORY;
    }
    if (shape != nullptr) {
        *shape = { m, n, k };
    }
    return GGML_POTAL_REJECT_NONE;
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
