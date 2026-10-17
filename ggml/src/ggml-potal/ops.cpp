#include "ops.hpp"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "matmul/dense.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/pack.hpp"
#include "potal.h"

// Existing Gemmini logging/cycle system (ggml-gemmini-utils); no gemmini.h.
#include <gemmini/cycle_reader.hpp>
#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>

#include <atomic>
#include <chrono>

namespace ggml::potal {

namespace {

using clock_type = std::chrono::steady_clock;
namespace glog   = ggml::gemmini::log;
namespace gcycle = ggml::gemmini::cycle;
namespace hp1    = quants::weight::hp1;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

const char * type_name_of(const ggml_tensor * t) {
    return t != nullptr ? ggml_type_name(t->type) : "-";
}

bool in_host_memory(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
}

bool is_hp1_mul_mat(const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_Q8_HP1;
}

// One stage interval in the cycle log (target cycles under LOG_CYCLE; 0 otherwise).
void cycle_stage(const std::string & layer, const char * op, uint64_t start, uint64_t end) {
#if LOG_CYCLE
    glog::cycle(layer.c_str(), op, start, end);
#else
    (void) layer;
    (void) op;
    (void) start;
    (void) end;
#endif
}

// ---- validated I8 x I8 single GEMM (W4-4 contract, unchanged) ----------------

ggml_status mul_mat_i8(backend_state & state, ggml_tensor * node, const mul_mat_shape & s) {
    ggml_backend_potal_stats & st = state.stats;
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

// ---- F32 x Q8_HP1 tiled GEMM (W4-5) ------------------------------------------

ggml_status hp1_fail(ggml_backend_potal_stats & st, ggml_tensor * node, const std::string & layer, const char * stage,
                     const char * detail) {
    st.hp1_failed++;
    poison_dst(node);
    GGML_LOG_ERROR("[POTAL] node=%s layer=%s dispatch=FAILED stage=%s %s; dst set to NaN\n", node->name,
                   layer.c_str(), stage, detail);
#if LOG_DEBUG
    glog::debug(layer.c_str(), "[POTAL] node=%s failure stage=%s %s", node->name, stage, detail);
#endif
    return GGML_STATUS_FAILED;
}

ggml_status mul_mat_hp1(backend_state & state, ggml_tensor * node, int node_index, const hp1_shape & s) {
    ggml_backend_potal_stats & st = state.stats;
    const ggml_tensor *        w  = node->src[0];
    const ggml_tensor *        x  = node->src[1];
    st.hp1_nodes++;
    const std::string layer = ggml::gemmini::types::resolve_matmul_layer(state.model_arch, w->name, x->name,
                                                                         node->name);
#if LOG_DEBUG
    glog::debug(layer.c_str(),
                "[POTAL] backend=POTAL node=%s node_index=%d model_arch=%s weight=%s weight_type=%s "
                "activation_quant=%s compensation=%s M=%lld N=%lld K=%lld",
                node->name, node_index, state.model_arch.c_str(), w->name, options::weight_type,
                options::activation_quant, options::compensation_name, (long long) s.m, (long long) s.n,
                (long long) s.k);
#else
    (void) node_index;
#endif

    hp1_operands      ops;
    ggml_potal_args_t args;
    if (!ops.allocate(s) ||
        !init_args(w, x, node, s, state.model_arch.c_str(), layer.c_str(), args)) {
        return hp1_fail(st, node, layer, "alloc", "out of memory for the DMA staging buffers");
    }

    // 1. Activation: quants::quantize_activation (GGML_GEMMINI_ACTIVATION_QUANT=TENSOR).
    uint64_t c0               = gcycle::read();
    auto     t                = clock_type::now();
    float    activation_scale = 0.0f;
    const bool aq             = quantize_activations(x, args, ops, &activation_scale);
    st.act_quant_s += seconds_since(t);
    uint64_t c1 = gcycle::read();
    cycle_stage(layer, "potal.act_quant_tensor", c0, c1);
    if (!aq) {
        return hp1_fail(st, node, layer, "act_quant", "quantize_activation failed (no finite value)");
    }
#if LOG_DEBUG
    glog::debug(layer.c_str(), "[POTAL] activation_quant=TENSOR activation_scale=%.9g", (double) activation_scale);
#endif

    // 2. Weight: HP1 blocks validated and unpacked to B, carriers, channel scales.
    c0 = gcycle::read();
    t  = clock_type::now();
    hp1::stats                          ws;
    quants::wreader::WeightReaderStatus reader = quants::wreader::WeightReaderStatus::Success;
    const hp1::status                   wq     = unpack_weights(args, ops, &ws, &reader);
    st.weight_pack_s += seconds_since(t);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.weight_pack_hp1", c0, c1);
    if (wq != hp1::status::ok) {
        const std::string why = std::string(hp1::status_name(wq)) + " reader=" +
                                quants::wreader::weight_reader_status_name(reader);
        return hp1_fail(st, node, layer, "weight_hp1", why.c_str());
    }
    st.zero_carriers += ws.zero_blocks;
#if LOG_DEBUG
    const uint64_t nonzero = ws.blocks - ws.zero_blocks;
    glog::debug(layer.c_str(),
                "[POTAL] weight_type=Q8_HP1 blocks=%llu zero_carrier_count=%llu zero_rows=%llu "
                "exponent_min=%d exponent_max=%d exponent_mean=%.3f channel_scale_min=%.9g channel_scale_max=%.9g",
                (unsigned long long) ws.blocks, (unsigned long long) ws.zero_blocks,
                (unsigned long long) ws.zero_rows, ws.min_m, ws.max_m, nonzero ? ws.sum_m / nonzero : 0.0,
                (double) ws.min_channel_scale, (double) ws.max_channel_scale);
#endif

    // 3. PoTal: tiled GEMM on the hart-0 issuer.
    const tiled_job job = {
        (size_t) s.m, (size_t) s.n, (size_t) s.k,
        ops.a, ops.lda, ops.b, ops.ldb,
        ops.carriers, ops.cstride, hp1::block_k,
        ops.c, ops.ldc,
    };
    c0 = gcycle::read();
    const tiled_report r = run_tiled(job);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.tiled_gemm", c0, c1);
    st.hw_requests++;
    st.tiled_invocations += r.work.invocations;
    st.tiled_validated += r.work.validated_gemms;
    st.tiled_fences += r.work.fences;
    st.scale_writes += r.work.scale_writes;
    st.scale_releases += r.work.scale_releases;
    st.scale_s += r.work.scale_s;
    st.issue_s += r.work.issue_s;
    st.fence_s += r.work.fence_s;
    st.last_status     = r.status;
    st.last_generation = r.work.last_generation;
#if LOG_DEBUG
    glog::debug(layer.c_str(),
                "[POTAL] M=%lld N=%lld K=%lld tile_I=%zu tile_J=%zu tile_K=%zu I0=%zu J0=%zu K0=%zu "
                "pad_I=%zu pad_J=%zu pad_K=%zu spad_rows=%zu spad_rows_available=%zu acc_rows=%zu "
                "acc_rows_available=%zu scale_slots=%u scale_slots_available=%u passes_per_block=%u "
                "generation=%u gemm_issue_count=%llu validated_api_issues=%llu fence_ok=%llu scale_writes=%llu "
                "scale_releases=%llu status=POTAL_%s",
                (long long) s.m, (long long) s.n, (long long) s.k, r.plan.tile_I, r.plan.tile_J, r.plan.tile_K,
                r.plan.I0, r.plan.J0, r.plan.K0, r.plan.pad_I, r.plan.pad_J, r.plan.pad_K, r.plan.spad_rows,
                r.plan.max_spad_rows, r.plan.acc_rows, r.plan.max_acc_rows, r.plan.scale_slots,
                options::scale_window_slots, r.plan.passes_per_block, r.work.last_generation,
                (unsigned long long) r.work.invocations, (unsigned long long) r.work.validated_gemms,
                (unsigned long long) r.work.fences, (unsigned long long) r.work.scale_writes,
                (unsigned long long) r.work.scale_releases, potal_status_name(r.status));
#endif
    if (!r.ok) {
        if (r.fatal) {
            ops.abandon(); // the accelerator may still reach these buffers
        }
        const std::string detail = std::string("status=POTAL_") + potal_status_name(r.status) +
                                   " step=" + (r.stage ? r.stage : "-") + (r.fatal ? " fatal=1" : " fatal=0") +
                                   (r.detail ? std::string(" issuer=") + r.detail : std::string());
        return hp1_fail(st, node, layer, "potal_gemm", detail.c_str());
    }

    // 4. F32: C x activation_scale x channel_scale (the HP1 exponent was applied by the SCU).
    c0 = gcycle::read();
    t  = clock_type::now();
    reconstruct(ops, args, activation_scale);
    st.reconstruct_s += seconds_since(t);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.output_reconstruct", c0, c1);
    st.hp1_completed++;
#if LOG_DEBUG
    glog::debug(layer.c_str(), "[POTAL] node=%s dispatch=HARDWARE fence=POTAL_OK completed=%llu", node->name,
                (unsigned long long) st.hp1_completed);
#endif
    return GGML_STATUS_SUCCESS;
}

// Device-wide: supports_op has no backend instance.
std::atomic<uint64_t> g_mul_mat_rejected{ 0 };

} // namespace

bool is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

ggml_potal_reject check_op(const ggml_tensor * op) {
    if (is_view_op(op->op)) {
        // Metadata only: graph_compute leaves the data where it is.
        return in_host_memory(op) ? GGML_POTAL_REJECT_NONE : GGML_POTAL_REJECT_MEMORY;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return GGML_POTAL_REJECT_OP;
    }
    return is_hp1_mul_mat(op) ? check_mul_mat_hp1(op, false, nullptr) : check_mul_mat(op, false, nullptr);
}

ggml_potal_reject check_compute(const ggml_tensor * node) {
    if (is_view_op(node->op)) {
        return GGML_POTAL_REJECT_NONE;
    }
    if (node->op != GGML_OP_MUL_MAT) {
        return GGML_POTAL_REJECT_OP;
    }
    return is_hp1_mul_mat(node) ? check_mul_mat_hp1(node, true, nullptr) : check_mul_mat(node, true, nullptr);
}

ggml_status compute_mul_mat(backend_state & state, ggml_tensor * node, int node_index) {
    if (is_hp1_mul_mat(node)) {
        hp1_shape s = {};
        check_mul_mat_hp1(node, true, &s); // passed check_compute
        return mul_mat_hp1(state, node, node_index, s);
    }
    mul_mat_shape s = {};
    check_mul_mat(node, true, &s);
    return mul_mat_i8(state, node, s);
}

void note_supports_mul_mat(const ggml_tensor * op, ggml_potal_reject reason) {
    if (reason == GGML_POTAL_REJECT_NONE) {
        return;
    }
    g_mul_mat_rejected.fetch_add(1, std::memory_order_relaxed);
#if LOG_DEBUG
    glog::debug("potal.supports_op",
                "[POTAL] cpu_fallback op=MUL_MAT node=%s reason=%s src0=%s src0_type=%s src1_type=%s ne0=%lld "
                "ne1=%lld K=%lld",
                op->name, ggml_potal_reject_name(reason), op->src[0] ? op->src[0]->name : "-",
                type_name_of(op->src[0]), type_name_of(op->src[1]), (long long) op->ne[0], (long long) op->ne[1],
                op->src[0] ? (long long) op->src[0]->ne[0] : -1LL);
#else
    (void) op;
#endif
}

uint64_t supports_mul_mat_rejected() {
    return g_mul_mat_rejected.load(std::memory_order_relaxed);
}

} // namespace ggml::potal
