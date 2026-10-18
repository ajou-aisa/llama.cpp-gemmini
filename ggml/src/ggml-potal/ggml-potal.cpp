// PoTal GGML backend: backend, device and registry. Op validation and dispatch
// live in ops.cpp (GGML_OP_MUL_MAT through the PoTal RoCC API, potal.h).
// See README.md and ggml-potal.h.

#include "ggml-potal.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-potal-buffer.hpp"
#include "log.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/pack.hpp"
#include "ops.hpp"
#include "potal.h"

#include <gemmini/log.hpp>
#include <gemmini/semantic.h>
#include <gemmini/trace-context.hpp>

#include <atomic>
#include <cstdio>
#include <memory>

namespace {

using potal_context = ggml::potal::backend_state;

// supports_op is a device call without a backend instance: device-wide counters.
std::atomic<uint64_t> g_supports_accepted{ 0 };
std::atomic<uint64_t> g_supports_rejected[GGML_POTAL_REJECT_COUNT];

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
    // End-of-run evidence for tools (llama-cli) that do not read the stats API,
    // written before the stats and the weight cache are destroyed.
    const ggml_backend_potal_stats & st    = static_cast<const potal_context *>(backend->context)->stats;
    const auto                       cache = static_cast<const potal_context *>(backend->context)->weights.snapshot();
    ggml::potal::log::debug(
        "potal.summary",
        "[POTAL] summary i8_completed=%llu i8_failed=%llu hp1_nodes=%llu hp1_completed=%llu "
        "hp1_failed=%llu prefill_shaped=%llu decode_shaped=%llu route_tensor_block=%llu "
        "route_token_block=%llu route_stripe_block=%llu route_exsia_block=%llu route_block_block=%llu "
        "tiled_jobs=%llu loop_ws_issued=%llu validated_api=%llu fences=%llu scale_writes=%llu "
        "scale_releases=%llu tlb_flushes=%llu zero_carriers=%llu shift_carriers=%llu "
        "compensation=%s residual_payloads=%llu compensation_executed=%llu residual_events=%llu "
        "supports_op_mul_mat_rejected_answers=%llu act_quant_s=%.3f weight_pack_s=%.3f scale_s=%.3f "
        "issue_s=%.3f fence_s=%.3f route_wall_s=%.3f compensation_s=%.3f weight_cache_hits=%llu "
        "weight_cache_misses=%llu weight_cache_bypasses=%llu weight_cache_uncached=%llu "
        "weight_cache_entries=%llu weight_cache_bytes=%llu weight_validations=%llu weight_packs=%llu "
        "weight_pack_bytes=%llu weight_alloc_s=%.3f weight_lookup_s=%.6f stripe_pipeline_nodes=%llu "
        "stripe_pipeline_failed=%llu stripes_accepted=%llu stripes_semantic_completed=%llu "
        "max_outstanding_semantic=%llu producer_capacity_waits=%llu main_logical_gemms=%llu main_loop_ws=%llu "
        "main_fences=%llu residual_nonempty_packets=%llu residual_logical_gemms=%llu residual_loop_ws=%llu "
        "residual_fences=%llu residual_compact_k=%llu residual_execution_k=%llu active_lane_rows=%llu "
        "radix_reconstructions=%llu merges_completed=%llu corrections_nonzero=%llu pipeline_wall_s=%.3f",
        (unsigned long long) st.gemm_completed, (unsigned long long) st.gemm_failed,
        (unsigned long long) st.hp1_nodes, (unsigned long long) st.hp1_completed,
        (unsigned long long) st.hp1_failed, (unsigned long long) st.prefill_shaped_nodes,
        (unsigned long long) st.decode_shaped_nodes,
        (unsigned long long) st.route_completed[GGML_POTAL_ROUTE_TENSOR_BLOCK],
        (unsigned long long) st.route_completed[GGML_POTAL_ROUTE_TOKEN_BLOCK],
        (unsigned long long) st.route_completed[GGML_POTAL_ROUTE_STRIPE_BLOCK],
        (unsigned long long) st.route_completed[GGML_POTAL_ROUTE_EXSIA_BLOCK],
        (unsigned long long) st.route_completed[GGML_POTAL_ROUTE_BLOCK_BLOCK],
        (unsigned long long) st.tiled_jobs, (unsigned long long) st.tiled_invocations,
        (unsigned long long) st.tiled_validated, (unsigned long long) st.tiled_fences,
        (unsigned long long) st.scale_writes, (unsigned long long) st.scale_releases,
        (unsigned long long) st.tlb_flushes, (unsigned long long) st.zero_carriers,
        (unsigned long long) st.shift_carriers, ggml::potal::options::compensation_name,
        (unsigned long long) st.residual_payloads, (unsigned long long) st.compensation_executed,
        (unsigned long long) st.residual_events,
        (unsigned long long) ggml::potal::supports_mul_mat_rejected(), st.act_quant_s, st.weight_pack_s,
        st.scale_s, st.issue_s, st.fence_s, st.route_wall_s, st.compensation_s,
        (unsigned long long) st.weight_cache_hits, (unsigned long long) st.weight_cache_misses,
        (unsigned long long) st.weight_cache_bypasses, (unsigned long long) st.weight_cache_uncached,
        (unsigned long long) cache.entries, (unsigned long long) cache.bytes,
        (unsigned long long) st.weight_validations, (unsigned long long) st.weight_packs,
        (unsigned long long) st.weight_pack_bytes, st.weight_alloc_s, st.weight_lookup_s,
        (unsigned long long) st.stripe_pipeline_nodes, (unsigned long long) st.stripe_pipeline_failed,
        (unsigned long long) st.stripes_accepted, (unsigned long long) st.stripes_semantic_completed,
        (unsigned long long) st.max_outstanding_semantic, (unsigned long long) st.producer_capacity_waits,
        (unsigned long long) st.main_logical_gemms, (unsigned long long) st.main_loop_ws,
        (unsigned long long) st.main_fences, (unsigned long long) st.residual_nonempty_packets,
        (unsigned long long) st.residual_logical_gemms, (unsigned long long) st.residual_loop_ws,
        (unsigned long long) st.residual_fences, (unsigned long long) st.residual_compact_k,
        (unsigned long long) st.residual_execution_k, (unsigned long long) st.residual_active_lane_rows,
        (unsigned long long) st.radix_reconstructions, (unsigned long long) st.merges_completed,
        (unsigned long long) st.corrections_nonzero, st.pipeline_wall_s);
    // Debug records are flushed per record; drain the buffered cycle log too.
    (void) ggml::gemmini::log::cycle.flush();
    delete static_cast<potal_context *>(backend->context);
    delete backend;
}

ggml_status potal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    potal_context & ctx = *static_cast<potal_context *>(backend->context);

    // Refuse the whole graph before any hardware work if one node is outside
    // the contract (only a direct caller can get here; the scheduler asks supports_op).
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor *           node   = cgraph->nodes[i];
        const ggml_potal_reject reason = ggml::potal::check_compute(node);
        if (reason != GGML_POTAL_REJECT_NONE) {
            ctx.stats.rejected_nodes++;
            ggml::potal::poison_dst(node);
            ggml::potal::log::failure("potal.backend",
                                      "[POTAL_BACKEND] node=%s op=%s rejected at compute (%s); nothing issued",
                                      node->name, ggml_op_desc(node), ggml_potal_reject_name(reason));
            return GGML_STATUS_FAILED;
        }
    }

    // Same per-operator trace context as the Gemmini backend: the issuer binds
    // it while it runs this node's jobs.
    const gemmini_trace_context graph_trace = gemmini_trace_capture();
    const uint64_t graph_id = gemmini_trace_reserve_ids(static_cast<uint64_t>(cgraph->n_nodes) + 1);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue; // view ops: metadata only, the data is already in place
        }
#if LOG_CYCLE || CYCLE_SIM
        // The node's semantic identity (phase, graph occurrence, ordinal) for the
        // records below, bound as the Gemmini and CPU backends do; one issuing
        // worker. The C entry turns a node outside the captured graph into a
        // failed run (the session records why) instead of an exception.
        const std::unique_ptr<void, void (*)(void *)> semantic_scope(gemmini_semantic_enter(node, 1),
                                                                     gemmini_semantic_exit);
        if (!semantic_scope) {
            ggml::potal::log::failure("potal.backend",
                                      "[POTAL_BACKEND] node=%s semantic context unavailable; nothing issued",
                                      node->name);
            return GGML_STATUS_FAILED;
        }
#endif
        ggml::gemmini::trace::ScopedContext operator_context(
            gemmini_trace_operator(graph_trace, graph_id, graph_id + 1 + static_cast<uint64_t>(i),
                                   static_cast<uint64_t>(i), ggml_op_name(node->op), 0, 1, 1));
        const ggml_status status = ggml::potal::compute_mul_mat(ctx, node, i); // passed the loop above
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
    static char description[256];
    static const int written = std::snprintf(
        description, sizeof(description),
        "PoTal A8W8 WS Sat32 DIM%u RoCC custom-3 on hart 0: MUL_MAT I8 x I8 -> F32 (x1 carriers, M=N=16, "
        "K=16|32); F32 x Q8_HP1 -> F32 (%s act quant, HP1 SCU carriers, potal_tiled_matmul_auto, "
        "compensation %s)",
        (unsigned) potal_hw_dim(), ggml::potal::options::activation_quant_name(),
        ggml::potal::options::compensation_name);
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
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

ggml_backend_t potal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    ggml_backend_t backend = ggml_backend_potal_init();
    // llama passes the model architecture, as to the Gemmini backend (layer names).
    static_cast<potal_context *>(backend->context)->model_arch = params ? params : "";
    return backend;
}

// Tensors stay in ordinary host memory; the backend packs them into its own
// DMA staging buffers. The POTAL buffer type is that host memory under its own
// identity: llama.cpp lists an ACCEL device as a weight buffer candidate only
// when its default buffer type is not the CPU one (make_cpu_buft_list).
ggml_backend_buffer_type_t potal_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml::potal::potal_buffer_type(dev);
}

// The model loader's mmap path: the buffer aliases the mapping, which the
// loader keeps alive for the life of the model.
ggml_backend_buffer_t potal_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size,
                                                        size_t max_tensor_size) {
    GGML_UNUSED(max_tensor_size);
    return ggml::potal::potal_buffer_from_host_ptr(dev, ptr, size);
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
        ggml::potal::note_supports_mul_mat(op, reason); // records rejections
        if (reason == GGML_POTAL_REJECT_NONE) {
            ggml::potal::log::debug("potal.supports_op",
                                    "[POTAL_BACKEND] supports_op op=MUL_MAT name=%s result=1 reason=none src0_type=%s "
                                    "src1_type=%s ne0=%lld ne1=%lld K=%lld",
                                    op->name, type_name_of(op->src[0]), type_name_of(op->src[1]),
                                    (long long) op->ne[0], (long long) op->ne[1],
                                    op->src[0] ? (long long) op->src[0]->ne[0] : -1LL);
        }
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
    /* .buffer_from_host_ptr = */ potal_device_buffer_from_host_ptr,
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
    // Registration runs while a tool still parses its arguments, before it
    // routes the logs (--gemmini-debug-log): its record is written here, on the
    // first backend instance, so it lands in the same debug JSONL as the rest.
    static const bool registered = [] {
        ggml::potal::log::debug("potal.backend", "[POTAL_BACKEND] registered name=POTAL dim=%u params_sha256=%s",
                                (unsigned) potal_hw_dim(), potal_hw_params_sha256());
        return true;
    }();
    GGML_UNUSED(registered);
    ggml::potal::log::debug("potal.backend", "[POTAL_BACKEND] device_init dim=%u precision=A8W8 params_sha256=%s",
                            (unsigned) potal_hw_dim(), potal_hw_params_sha256());
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
    stats->supports_mul_mat_rejected      = ggml::potal::supports_mul_mat_rejected();
    const auto cache                      = static_cast<const potal_context *>(backend->context)->weights.snapshot();
    stats->weight_cache_entries           = cache.entries;
    stats->weight_cache_bytes             = cache.bytes;
    stats->weight_cache_budget            = cache.budget;
    return true;
}

void ggml_backend_potal_clear_weight_cache(ggml_backend_t backend) {
    if (ggml_backend_is_potal(backend)) {
        static_cast<potal_context *>(backend->context)->weights.clear();
    }
}
