// PoTal GGML backend: backend, device and registry. Op validation and dispatch
// live in ops.cpp (GGML_OP_MUL_MAT through the PoTal RoCC API, potal.h).
// See README.md and ggml-potal.h.

#include "ggml-potal.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-potal-buffer.hpp"
#include "log.hpp"
#include "matmul/attention.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/pack.hpp"
#include "ops.hpp"
#include "potal.h"
#include "trace/timeline.hpp"

#include <gemmini/log.hpp>
#include <gemmini/semantic.h>
#include <gemmini/trace-context.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

using potal_context = ggml::potal::backend_state;

// supports_op is a device call without a backend instance: device-wide counters.
std::atomic<uint64_t> g_supports_accepted{ 0 };
std::atomic<uint64_t> g_supports_rejected[GGML_POTAL_REJECT_COUNT];

const char * type_name_of(const ggml_tensor * t) {
    return t != nullptr ? ggml_type_name(t->type) : "-";
}

// potal_graph_compute calls in the process (graph index of the node dump).
std::atomic<uint64_t> g_graph_count{ 0 };

// Live backends in creation order (ggml_backend_potal_get_last_stats).
std::mutex                  g_live_mu;
std::vector<ggml_backend_t> g_live;

// ---- node dump (GGML_POTAL_DUMP_DIR / _GRAPHS / _NODES, read once) ---------

struct dump_config {
    bool                     on = false;  // directory set and at least one node pattern
    std::string              dir;
    std::vector<uint64_t>    graphs;  // empty: every graph
    std::vector<std::string> nodes;
};

std::vector<std::string> split_list(const char * v) {
    std::vector<std::string> out;
    if (v == nullptr) {
        return out;
    }
    std::string cur;
    for (const char * p = v;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
            if (*p == '\0') {
                break;
            }
        } else {
            cur += *p;
        }
    }
    return out;
}

const dump_config & dump_cfg() {
    static const dump_config c = [] {
        dump_config  d;
        const char * dir = std::getenv("GGML_POTAL_DUMP_DIR");
        if (dir == nullptr || *dir == '\0') {
            return d;
        }
        d.dir   = dir;
        d.nodes = split_list(std::getenv("GGML_POTAL_DUMP_NODES"));
        for (const std::string & g : split_list(std::getenv("GGML_POTAL_DUMP_GRAPHS"))) {
            d.graphs.push_back(std::strtoull(g.c_str(), nullptr, 10));
        }
        d.on = !d.nodes.empty();
        return d;
    }();
    return c;
}

const char * dump_kind(const ggml_tensor * node) {
    switch (ggml::potal::attention::get_role(node)) {
        case ggml::potal::attention::role::qk: return "qk";
        case ggml::potal::attention::role::pv: return "pv";
        case ggml::potal::attention::role::none: break;
    }
    return "linear";
}

std::string sanitize(const char * s) {
    std::string out;
    for (const char * p = s; *p != '\0'; ++p) {
        const char c = *p;
        out += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_') ? c : '_';
    }
    return out.empty() ? std::string("_") : out;
}

// Element bytes of a tensor whose type has one element per block (F32, F16, I8 ...).
bool dumpable(const ggml_tensor * t) {
    return t != nullptr && t->data != nullptr && ggml_blck_size(t->type) == 1;
}

size_t dump_bytes(const ggml_tensor * t) {
    return dumpable(t) ? static_cast<size_t>(ggml_nelements(t)) * ggml_type_size(t->type) : 0;
}

// Row-major (ne order) element bytes, following nb (views, permuted K/V).
void write_elements(FILE * f, const ggml_tensor * t) {
    if (!dumpable(t)) {
        return;
    }
    const size_t es   = ggml_type_size(t->type);
    const char * base = static_cast<const char *>(t->data);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const char * row = base + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                if (t->nb[0] == es) {
                    std::fwrite(row, es, static_cast<size_t>(t->ne[0]), f);
                } else {
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                        std::fwrite(row + i0 * t->nb[0], es, 1, f);
                    }
                }
            }
        }
    }
}

std::string shape_of(const char * tag, const ggml_tensor * t) {
    char b[256];
    if (t == nullptr) {
        std::snprintf(b, sizeof(b), " %s_ne=- %s_nb=-", tag, tag);
    } else {
        std::snprintf(b, sizeof(b), " %s_ne=%lld,%lld,%lld,%lld %s_nb=%zu,%zu,%zu,%zu", tag, (long long) t->ne[0],
                      (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], tag, t->nb[0], t->nb[1],
                      t->nb[2], t->nb[3]);
    }
    return b;
}

// One completed node: header line, then src1, dst and (F16/F32 only) src0 elements.
void dump_node(const dump_config & c, uint64_t graph, int index, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const char *        kind = dump_kind(node);
    bool                hit  = false;
    for (const std::string & p : c.nodes) {
        hit = hit || std::strstr(node->name, p.c_str()) != nullptr ||
              (src0 != nullptr && std::strstr(src0->name, p.c_str()) != nullptr) ||
              std::strstr(kind, p.c_str()) != nullptr;
    }
    if (!hit) {
        return;
    }
    const bool        with_src0 = src0 != nullptr && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    const std::string path      = c.dir + "/g" + std::to_string(graph) + "_n" + std::to_string(index) + "_" +
                             sanitize(node->name) + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "[POTAL_DUMP] cannot write %s: %s\n", path.c_str(), std::strerror(errno));
        return;
    }
    std::string h = "POTAL_DUMP v1 kind=" + std::string(kind) + " graph=" + std::to_string(graph) +
                    " node_index=" + std::to_string(index) + " node=" + sanitize(node->name) +
                    " src0=" + (src0 ? sanitize(src0->name) : "-") + " types=" + type_name_of(src0) + "," +
                    type_name_of(src1) + "," + ggml_type_name(node->type) + shape_of("src0", src0) +
                    shape_of("src1", src1) + shape_of("dst", node) + " sections=src1:" +
                    std::to_string(dump_bytes(src1)) + ",dst:" + std::to_string(dump_bytes(node));
    if (with_src0) {
        h += ",src0:" + std::to_string(dump_bytes(src0));
    }
    h += "\n";
    std::fputs(h.c_str(), f);
    write_elements(f, src1);
    write_elements(f, node);
    if (with_src0) {
        write_elements(f, src0);
    }
    const bool err = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || err) {
        std::fprintf(stderr, "[POTAL_DUMP] writing %s failed\n", path.c_str());
    }
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
        "radix_reconstructions=%llu merges_completed=%llu corrections_nonzero=%llu pipeline_wall_s=%.3f "
        "paired_stripes=%llu paired_fallback_stripes=%llu paired_main_loop_ws=%llu paired_residual_loop_ws=%llu "
        "paired_index_configs=%llu paired_index_writes=%llu paired_readbacks=%llu paired_readback_tiles=%llu "
        "paired_fences=%llu paired_drains=%llu",
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
        (unsigned long long) st.corrections_nonzero, st.pipeline_wall_s,
        (unsigned long long) st.paired_stripes, (unsigned long long) st.paired_fallback_stripes,
        (unsigned long long) st.paired_main_loop_ws, (unsigned long long) st.paired_residual_loop_ws,
        (unsigned long long) st.paired_index_configs, (unsigned long long) st.paired_index_writes,
        (unsigned long long) st.paired_readbacks, (unsigned long long) st.paired_readback_tiles,
        (unsigned long long) st.paired_fences, (unsigned long long) st.paired_drains);
    // Debug records are flushed per record; drain the buffered cycle log too.
    if (st.attention_qk_nodes || st.attention_pv_nodes) {
        ggml::potal::log::debug("potal.attention",
            "[POTAL_ATTENTION] summary qk_nodes=%llu pv_nodes=%llu completed=%llu failed=%llu "
            "limb_jobs=%llu residual_jobs=%llu loop_ws=%llu separate_acc_qk_jobs=%llu bias_mvins=%llu",
            (unsigned long long)st.attention_qk_nodes, (unsigned long long)st.attention_pv_nodes,
            (unsigned long long)st.attention_completed, (unsigned long long)st.attention_failed,
            (unsigned long long)st.attention_limb_jobs, (unsigned long long)st.attention_residual_jobs,
            (unsigned long long)st.attention_loop_ws, (unsigned long long)st.attention_qk_limb_jobs,
            (unsigned long long)st.attention_bias_mvins);
    }
    (void) ggml::gemmini::log::cycle.flush();
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        g_live.erase(std::remove(g_live.begin(), g_live.end(), backend), g_live.end());
    }
    delete static_cast<potal_context *>(backend->context);
    delete backend;
}

ggml_status potal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    potal_context & ctx = *static_cast<potal_context *>(backend->context);
    const uint64_t      graph_index = g_graph_count.fetch_add(1, std::memory_order_relaxed);
    const dump_config & dump        = dump_cfg();
    // Strict issuer (H3-8I-25S): hart 0's CPU runs only the issuer, so the calling thread
    // (graph walk, node preparation, output commit) leaves it for this call.
    ggml::potal::issuer_cpu_exclusion caller_off_issuer;
    {
        bool roles_invalid = false, strict_invalid = false;
        const bool roles = ggml::potal::hart_roles_requested(roles_invalid);
        if (ggml::potal::strict_issuer_requested(strict_invalid) && roles) {
            std::string why;
            if (!caller_off_issuer.begin(why)) {
                ggml::potal::log::failure("potal.backend", "[POTAL_BACKEND] strict issuer: %s; nothing issued",
                                          why.c_str());
                return GGML_STATUS_FAILED;
            }
        }
    }
    // Graph-level timeline spans on the calling thread (inert unless POTAL_TIMELINE is set).
    if (::potal::timeline::enabled()) ::potal::timeline::register_thread(::potal::timeline::role::caller, -1, -1);
    ::potal::timeline::scoped_span gc_total(::potal::timeline::active(), ::potal::timeline::stage::gc_total, 0, 0,
                                            ::potal::timeline::no_id, ::potal::timeline::no_id);
    ::potal::timeline::scoped_span gc_setup(::potal::timeline::active(), ::potal::timeline::stage::gc_setup, 0, 0,
                                            ::potal::timeline::no_id, ::potal::timeline::no_id);

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
    gc_setup.close();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue; // view ops: metadata only, the data is already in place
        }
        ::potal::timeline::scoped_span node_setup(::potal::timeline::active(), ::potal::timeline::stage::gc_node_setup,
                                                  0, 0, ::potal::timeline::no_id, uint32_t(i));
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
        node_setup.close();
        const ggml_status status = ggml::potal::compute_mul_mat(ctx, node, i); // passed the loop above
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        if (dump.on && (dump.graphs.empty() ||
                        std::find(dump.graphs.begin(), dump.graphs.end(), graph_index) != dump.graphs.end())) {
            dump_node(dump, graph_index, i, node);
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
        "PoTal A8W8 WS Sat32 DIM%u RoCC custom-3 on hart 0: MUL_MAT I8 x I8 -> F32 (x1 carriers, M=N=%u, "
        "K=%u|%u); F32 x Q8_HP1 -> F32 (%s act quant, HP1 SCU carriers, potal_tiled_matmul_auto, "
        "compensation %s)",
        (unsigned) potal_hw_dim(), (unsigned) potal_hw_dim(), (unsigned) potal_hw_dim(),
        (unsigned) (2 * potal_hw_dim()), ggml::potal::options::activation_quant_name(),
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
    ggml_backend_t backend = new ggml_backend{
        /* .guid      = */ potal_guid(),
        /* .interface = */ potal_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_potal_reg(), 0),
        /* .context   = */ new potal_context,
    };
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        g_live.push_back(backend);
    }
    return backend;
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

bool ggml_backend_potal_get_last_stats(ggml_backend_potal_stats * out) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    return !g_live.empty() && ggml_backend_potal_get_stats(g_live.back(), out);
}

uint64_t ggml_backend_potal_graph_count(void) {
    return g_graph_count.load(std::memory_order_relaxed);
}

void ggml_backend_potal_clear_weight_cache(ggml_backend_t backend) {
    if (ggml_backend_is_potal(backend)) {
        static_cast<potal_context *>(backend->context)->weights.clear();
    }
}
