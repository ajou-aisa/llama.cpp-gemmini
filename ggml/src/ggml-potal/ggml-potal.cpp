// PoTal GGML backend: GGML_OP_MUL_MAT (I8 x I8 -> F32, M = N = 16, K = 16 or 32)
// through the PoTal RoCC API (potal.h). See README.md and ggml-potal.h.

#include "ggml-potal.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "matmul/execution.hpp"
#include "matmul/pack.hpp"
#include "potal.h"

#include <atomic>
#include <chrono>
#include <cstdio>

namespace {

using clock_type = std::chrono::steady_clock;

struct potal_context {
    ggml_backend_potal_stats stats = {};
};

// supports_op is a device call without a backend instance: device-wide counters.
std::atomic<uint64_t> g_supports_accepted{ 0 };
std::atomic<uint64_t> g_supports_rejected[GGML_POTAL_REJECT_COUNT];

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

bool is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

const char * type_name_of(const ggml_tensor * t) {
    return t != nullptr ? ggml_type_name(t->type) : "-";
}

ggml_guid_t potal_guid() {
    static ggml_guid guid = { 0x9e, 0x47, 0x2b, 0xd1, 0x5c, 0x08, 0x4a, 0x6f,
                              0xb3, 0x7d, 0xe2, 0x15, 0x60, 0xc9, 0x3a, 0x84 };
    return &guid;
}

// ---- backend ---------------------------------------------------------------

const char * potal_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "POTAL";
}

void potal_free(ggml_backend_t backend) {
    delete static_cast<potal_context *>(backend->context);
    delete backend;
}

// One validated MUL_MAT node: pack -> PoTal issuer (hardware) -> INT32 -> F32.
ggml_status potal_mul_mat(potal_context & ctx, ggml_tensor * node, const ggml::potal::mul_mat_shape & s) {
    using namespace ggml::potal;
    ggml_backend_potal_stats & st = ctx.stats;
    st.mul_mat_nodes++;
    GGML_LOG_INFO("[POTAL_BACKEND] node=%s op=MUL_MAT src0_type=%s src1_type=%s dst_type=%s M=%lld N=%lld K=%lld\n",
                  node->name, type_name_of(node->src[0]), type_name_of(node->src[1]), type_name_of(node),
                  (long long) s.m, (long long) s.n, (long long) s.k);

    int8_t  a[TILE * K_MAX];
    int8_t  b[K_MAX * TILE];
    int32_t c[TILE * TILE];

    auto t = clock_type::now();
    pack_a(node->src[1], s, a);
    pack_b(node->src[0], s, b);
    const double pack_s = seconds_since(t);
    st.pack_s += pack_s;
    GGML_LOG_INFO("[POTAL_MATMUL] pack_A_bytes=%lld pack_B_bytes=%lld pack_us=%.1f\n", (long long) (s.m * s.k),
                  (long long) (s.k * s.n), pack_s * 1e6);

    st.hw_requests++;
    const gemm_report r = run_gemm({ a, b, static_cast<uint32_t>(s.k), c });
    st.gemm_issued += r.issued ? 1 : 0;
    st.gemm_fenced += r.fenced ? 1 : 0;
    st.last_status     = r.status;
    st.last_scale_slot = r.scale_slot;
    st.last_generation = r.generation;
    st.stage_s += r.stage_s;
    st.scale_s += r.scale_s;
    st.issue_s += r.issue_s;
    st.fence_s += r.fence_s;
    if (!r.ok) {
        st.gemm_failed++;
        poison_dst(node);
        GGML_LOG_ERROR("[POTAL_MATMUL] node=%s dispatch=FAILED stage=%s status=POTAL_%s%s%s; dst set to NaN\n",
                       node->name, r.stage, potal_status_name(r.status), r.detail ? " detail=" : "",
                       r.detail ? r.detail : "");
        return GGML_STATUS_FAILED;
    }

    t = clock_type::now();
    unpack_c(c, s, node);
    const double unpack_s = seconds_since(t);
    st.unpack_s += unpack_s;
    st.gemm_completed++;
    GGML_LOG_INFO("[POTAL_MATMUL] output_type=F32 unpack_us=%.1f\n", unpack_s * 1e6);
    GGML_LOG_INFO("[POTAL_BACKEND] hw_gemm_count=%llu\n", (unsigned long long) st.gemm_completed);
    return GGML_STATUS_SUCCESS;
}

ggml_status potal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    potal_context & ctx = *static_cast<potal_context *>(backend->context);

    // Refuse the whole graph before any hardware work if one node is outside
    // the contract (only a direct caller can get here; the scheduler asks supports_op).
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (is_view_op(node->op)) {
            continue;
        }
        const ggml_potal_reject reason = node->op == GGML_OP_MUL_MAT ?
                                             ggml::potal::check_mul_mat(node, true, nullptr) :
                                             GGML_POTAL_REJECT_OP;
        if (reason != GGML_POTAL_REJECT_NONE) {
            ctx.stats.rejected_nodes++;
            ggml::potal::poison_dst(node);
            GGML_LOG_ERROR("[POTAL_BACKEND] node=%s op=%s rejected at compute (%s); nothing issued\n", node->name,
                           ggml_op_desc(node), ggml_potal_reject_name(reason));
            return GGML_STATUS_FAILED;
        }
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue; // view ops: metadata only, the data is already in place
        }
        ggml::potal::mul_mat_shape s = {};
        ggml::potal::check_mul_mat(node, true, &s); // passed in the loop above
        const ggml_status status = potal_mul_mat(ctx, node, s);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i potal_backend_i = {
    /* .get_name                = */ potal_get_name,
    /* .free                    = */ potal_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ potal_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
};

// ---- device ----------------------------------------------------------------

const char * potal_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "POTAL";
}

const char * potal_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    static char description[192];
    static const int written = std::snprintf(
        description, sizeof(description),
        "PoTal A8W8 WS Sat32 DIM%u RoCC custom-3 on hart 0: MUL_MAT I8 x I8 -> F32 (INT32 ACC, x1 carriers), "
        "M=N=16, K=16|32",
        (unsigned) potal_hw_dim());
    GGML_UNUSED(written);
    return description;
}

void potal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

enum ggml_backend_dev_type potal_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void potal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = potal_device_get_name(dev);
    props->description = potal_device_get_description(dev);
    props->type        = potal_device_get_type(dev);
    potal_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

ggml_backend_t potal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_potal_init();
}

// Tensors stay in ordinary host memory; the backend packs them into its own
// DMA staging buffers, so the CPU buffer type is the right default.
ggml_backend_buffer_type_t potal_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

bool potal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_potal_reject reason = ggml::potal::check_op(op);
    if (reason == GGML_POTAL_REJECT_NONE) {
        g_supports_accepted.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_supports_rejected[reason].fetch_add(1, std::memory_order_relaxed);
    }
    if (op->op == GGML_OP_MUL_MAT) {
        GGML_LOG_DEBUG("[POTAL_BACKEND] supports_op op=MUL_MAT name=%s result=%d reason=%s src0_type=%s "
                       "src1_type=%s ne0=%lld ne1=%lld K=%lld\n",
                       op->name, reason == GGML_POTAL_REJECT_NONE, ggml_potal_reject_name(reason),
                       type_name_of(op->src[0]), type_name_of(op->src[1]), (long long) op->ne[0],
                       (long long) op->ne[1], op->src[0] ? (long long) op->src[0]->ne[0] : -1LL);
    }
    return reason == GGML_POTAL_REJECT_NONE;
}

bool potal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

const ggml_backend_device_i potal_device_i = {
    /* .get_name             = */ potal_device_get_name,
    /* .get_description      = */ potal_device_get_description,
    /* .get_memory           = */ potal_device_get_memory,
    /* .get_type             = */ potal_device_get_type,
    /* .get_props            = */ potal_device_get_props,
    /* .init_backend         = */ potal_device_init_backend,
    /* .get_buffer_type      = */ potal_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ potal_device_supports_op,
    /* .supports_buft        = */ potal_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---- backend registry ------------------------------------------------------

const char * potal_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "POTAL";
}

size_t potal_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t potal_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ potal_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

const ggml_backend_reg_i potal_reg_i = {
    /* .get_name         = */ potal_reg_get_name,
    /* .get_device_count = */ potal_reg_get_device_count,
    /* .get_device       = */ potal_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

} // namespace

ggml_backend_t ggml_backend_potal_init(void) {
    GGML_LOG_INFO("[POTAL_BACKEND] device_init dim=%u precision=A8W8 params_sha256=%s\n", (unsigned) potal_hw_dim(),
                  potal_hw_params_sha256());
    return new ggml_backend{
        /* .guid      = */ potal_guid(),
        /* .interface = */ potal_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_potal_reg(), 0),
        /* .context   = */ new potal_context,
    };
}

bool ggml_backend_is_potal(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, potal_guid());
}

ggml_backend_reg_t ggml_backend_potal_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ potal_reg_i,
        /* .context     = */ nullptr,
    };
    static const bool logged = [] {
        GGML_LOG_INFO("[POTAL_BACKEND] registered name=POTAL dim=%u params_sha256=%s\n", (unsigned) potal_hw_dim(),
                      potal_hw_params_sha256());
        return true;
    }();
    GGML_UNUSED(logged);
    return &reg;
}

const char * ggml_potal_reject_name(enum ggml_potal_reject reason) {
    switch (reason) {
        case GGML_POTAL_REJECT_NONE:      return "none";
        case GGML_POTAL_REJECT_OP:        return "op";
        case GGML_POTAL_REJECT_OP_PARAMS: return "op_params";
        case GGML_POTAL_REJECT_TYPE:      return "type";
        case GGML_POTAL_REJECT_SHAPE:     return "shape";
        case GGML_POTAL_REJECT_LAYOUT:    return "layout";
        case GGML_POTAL_REJECT_MEMORY:    return "memory";
        case GGML_POTAL_REJECT_COUNT:     break;
    }
    return "unknown";
}

bool ggml_backend_potal_get_stats(ggml_backend_t backend, ggml_backend_potal_stats * stats) {
    if (!ggml_backend_is_potal(backend) || stats == nullptr) {
        return false;
    }
    *stats                      = static_cast<const potal_context *>(backend->context)->stats;
    stats->supports_op_accepted = g_supports_accepted.load(std::memory_order_relaxed);
    for (int r = 0; r < GGML_POTAL_REJECT_COUNT; ++r) {
        stats->supports_op_rejected[r] = g_supports_rejected[r].load(std::memory_order_relaxed);
    }
    const ggml::potal::issuer_info issuer = ggml::potal::get_issuer_info();
    stats->issuer_ready                   = issuer.ready;
    stats->issuer_cpu                     = issuer.cpu;
    stats->issuer_tid                     = issuer.tid;
    return true;
}
