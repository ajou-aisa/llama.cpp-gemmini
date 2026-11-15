#include "exec.hpp"
#include "../ggml-gemmini-buffer.hpp"
#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <cstring>

namespace {
const char * name(ggml_backend_t) { return "GEMMINI"; }
void free_backend(ggml_backend_t b) { ggml::gemmini::fp32::reset_weights(); delete b; }
ggml_status compute(ggml_backend_t, ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto * t = graph->nodes[i];
        switch (t->op) {
        case GGML_OP_MUL_MAT: ggml::gemmini::fp32::execute(t); break;
        case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE: case GGML_OP_PERMUTE: break;
        default: GGML_ABORT("Gemmini-FP32 unsupported assigned op '%s' (%s)", t->name, ggml_op_name(t->op));
        }
    }
    return GGML_STATUS_SUCCESS;
}
ggml_backend_i backend_i() {
    ggml_backend_i i{};
    i.get_name = name; i.free = free_backend; i.graph_compute = compute;
    return i;
}
ggml_guid_t guid() {
    static ggml_guid g = {0xf3, 0x28, 0xd8, 0x05, 0x73, 0x41, 0x48, 0x19, 0x80, 0x10, 1, 2, 3, 4, 5, 6};
    return &g;
}
const char * device_name(ggml_backend_dev_t) { return "GEMMINI"; }
const char * description(ggml_backend_dev_t) { return "FP32 Gemmini DIM8 WS (hart 0 issuer)"; }
void memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }
enum ggml_backend_dev_type type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
void props(ggml_backend_dev_t d, ggml_backend_dev_props * p) {
    *p = {};
    p->name = device_name(d); p->description = description(d); p->type = type(d);
    p->caps.host_buffer = true; p->caps.buffer_from_host_ptr = true;
}
ggml_backend_t init(ggml_backend_dev_t, const char *) { return ggml_backend_gemmini_init(); }
ggml_backend_buffer_type_t buft(ggml_backend_dev_t d) { return ggml::gemmini::gemmini_buffer_type(d); }
ggml_backend_buffer_t from_host(ggml_backend_dev_t d, void * ptr, size_t size, size_t) {
    return ggml::gemmini::gemmini_buffer_from_host_ptr(d, ptr, size);
}
bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return ggml_backend_buft_is_host(b); }
bool supports(ggml_backend_dev_t, const ggml_tensor * t) {
    if (t->op == GGML_OP_FLASH_ATTN_EXT) GGML_ABORT("Gemmini-FP32 flash attention bypasses QK/PV: '%s'", t->name);
    if (t->op == GGML_OP_MUL_MAT) {
        if (!ggml::gemmini::fp32::supported(t)) GGML_ABORT("Gemmini-FP32 unsupported mul_mat '%s'", t->name);
        return true;
    }
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE;
}
const char * reg_name(ggml_backend_reg_t) { return "GEMMINI"; }
size_t device_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t device(ggml_backend_reg_t reg, size_t n) {
    GGML_ASSERT(n == 0);
    static ggml_backend_device d = [] {
        ggml_backend_device d{};
        d.iface.get_name = device_name; d.iface.get_description = description;
        d.iface.get_memory = memory; d.iface.get_type = type; d.iface.get_props = props;
        d.iface.init_backend = init; d.iface.get_buffer_type = buft;
        d.iface.buffer_from_host_ptr = from_host; d.iface.supports_op = supports;
        d.iface.supports_buft = supports_buft;
        return d;
    }();
    d.reg = reg;
    return &d;
}
}
ggml_backend_t ggml_backend_gemmini_init() {
    return new ggml_backend{guid(), backend_i(), device(ggml_backend_gemmini_reg(), 0), nullptr};
}
ggml_backend_reg_t ggml_backend_gemmini_reg() {
    static ggml_backend_reg reg{GGML_BACKEND_API_VERSION, {reg_name, device_count, device, nullptr}, nullptr};
    return &reg;
}
bool ggml_backend_gemmini_get_attention_stats(ggml_backend_t, ggml_gemmini_attention_stats * out) {
    if (!out) return false;
    *out = {};
    const auto c = ggml::gemmini::fp32::counters();
    out->qk_nodes = c.qk; out->pv_nodes = c.pv;
    return true;
}
GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
