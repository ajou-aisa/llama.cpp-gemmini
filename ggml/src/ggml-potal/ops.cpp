#include "ops.hpp"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "log.hpp"
#include "matmul/dense.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/pack.hpp"
#include "potal.h"
#include "potal_hw_params.h"

#include "ggml-gemmini/matmul/options.hpp" // resolve_matmul_options (shared Gemmini source)

// Existing Gemmini logging/cycle/trace system (ggml-gemmini-utils); no gemmini.h.
#include <gemmini/cycle_reader.hpp>
#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#include <gemmini/trace-context.hpp>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>

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

// Inference identity of the running node, as the Gemmini backend's records
// carry it: phase/decode from the performance context, ids from the operator
// trace context bound by potal_graph_compute. phase=none outside an inference.
std::string trace_fields() {
    namespace perf                = ggml::gemmini::performance;
    const gemmini_trace_context t = gemmini_trace_capture();
    const perf::Context         p = ggml::gemmini::trace::inference_context(t);
    // The trace context has no decode ordinal; the published context has it
    // while the same operation is current.
    const perf::Context g      = perf::capture_context();
    const bool          decode = p.request_id && p.phase == perf::Phase::decode;
    const bool          known  = decode && g.request_id == p.request_id && g.operation_id == p.operation_id;
    const std::string   decode_ordinal = known ? std::to_string(g.decode_ordinal) : "null"; // prefill: null
    char                buf[256];
    std::snprintf(buf, sizeof(buf),
                  "phase=%s decode_ordinal=%s request_id=%" PRIu64 " operation_id=%" PRIu64 " graph_id=%" PRIu64
                  " operator_id=%" PRIu64 " node_ordinal=%" PRIu64,
                  p.request_id ? perf::phase_name(p.phase) : "none", decode_ordinal.c_str(), p.request_id,
                  p.operation_id, t.graph_id, t.operator_id, t.node_id);
    return buf;
}

// The work ran on PoTal, which exposes no cycle counter: the operation gets an
// NPU record that says so (the summary then reports POTAL/... cycles=n/a with
// this reason, not "no records"). The host stages of the node (quantization,
// packing, reconstruction) are not CPU-resource samples, so the phase's CPU
// work wall is marked incomplete rather than silently low.
void record_unmeasured_npu() {
    ggml::gemmini::performance::record_npu("POTAL", "dense", "work_total_cycles", 0, false,
                                           "unavailable_device_counter");
    ggml::gemmini::performance::incomplete_cpu_wall("potal_host_cpu_stage_coverage_incomplete");
}

// ---- validated I8 x I8 single GEMM (W4-4 contract, unchanged) ----------------

ggml_status mul_mat_i8(backend_state & state, ggml_tensor * node, const mul_mat_shape & s) {
    ggml_backend_potal_stats & st = state.stats;
    st.mul_mat_nodes++;
    log::debug("potal.backend",
               "[POTAL_BACKEND] node=%s op=MUL_MAT backend=POTAL src0_type=%s src1_type=%s dst_type=%s M=%lld N=%lld "
               "K=%lld %s",
               node->name, type_name_of(node->src[0]), type_name_of(node->src[1]), type_name_of(node),
               (long long) s.m, (long long) s.n, (long long) s.k, trace_fields().c_str());

    int8_t  a[TILE * K_MAX];
    int8_t  b[K_MAX * TILE];
    int32_t c[TILE * TILE];

    auto t = clock_type::now();
    pack_a(node->src[1], s, a);
    pack_b(node->src[0], s, b);
    const double pack_s = seconds_since(t);
    st.pack_s += pack_s;
    log::debug("potal.matmul", "[POTAL_MATMUL] pack_A_bytes=%lld pack_B_bytes=%lld pack_us=%.1f",
               (long long) (s.m * s.k), (long long) (s.k * s.n), pack_s * 1e6);

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
        log::failure("potal.matmul",
                     "[POTAL_MATMUL] node=%s dispatch=FAILED stage=%s status=POTAL_%s%s%s; dst set to NaN", node->name, r.stage, potal_status_name(r.status), r.detail ? " detail=" : "",
                     r.detail ? r.detail : "");
        return GGML_STATUS_FAILED;
    }

    t = clock_type::now();
    unpack_c(c, s, node);
    const double unpack_s = seconds_since(t);
    st.unpack_s += unpack_s;
    st.gemm_completed++;
    log::debug("potal.matmul", "[POTAL_MATMUL] output_type=F32 unpack_us=%.1f", unpack_s * 1e6);
    log::debug("potal.backend", "[POTAL_BACKEND] node=%s dispatch=HARDWARE hw_gemm_count=%llu", node->name,
               (unsigned long long) st.gemm_completed);
    record_unmeasured_npu();
    return GGML_STATUS_SUCCESS;
}

// ---- F32 x Q8_HP1 routed, tiled GEMM ---------------------------------------

ggml_status hp1_fail(ggml_backend_potal_stats & st, ggml_tensor * node, const std::string & layer, const char * stage,
                     const char * detail) {
    st.hp1_failed++;
    poison_dst(node);
    log::failure(layer.c_str(), "[POTAL] node=%s layer=%s dispatch=FAILED stage=%s %s; dst set to NaN", node->name,
                 layer.c_str(), stage, detail);
    return GGML_STATUS_FAILED;
}

const char * rmd_backend_name(ggml::gemmini::RmdBackend b) {
    return b == ggml::gemmini::RmdBackend::cpu_direct ? "cpu_direct" : "gemmini_ws_compact";
}

ggml_status mul_mat_hp1(backend_state & state, ggml_tensor * node, int node_index, const hp1_shape & s) {
    ggml_backend_potal_stats & st = state.stats;
    const ggml_tensor *        w  = node->src[0];
    const ggml_tensor *        x  = node->src[1];
    st.hp1_nodes++;
    (s.m == 1 ? st.decode_shaped_nodes : st.prefill_shaped_nodes)++;
    const std::string layer = ggml::gemmini::types::resolve_matmul_layer(state.model_arch, w->name, x->name,
                                                                         node->name);
    ggml::gemmini::trace::CpuStage node_stage(layer.c_str(), "potal.mul_mat_hp1",
                                              ggml::gemmini::trace::CpuStage::Scope::envelope);

    // 0. Matmul options as the Gemmini backend resolves them (build default, GEMMINI_* environment).
    const auto resolved = ggml::gemmini::resolve_matmul_options();
    if (!resolved.ok()) {
        return hp1_fail(st, node, layer, "options", "invalid matmul options (GEMMINI_MATMUL_MODE / GEMMINI_RMD_BACKEND)");
    }
    if (resolved.options.mode != ggml::gemmini::MatmulInvocationMode::full) {
        return hp1_fail(st, node, layer, "options", "stripe_pipeline is not executed by PoTal (FULL only)");
    }
    const ggml::gemmini::RmdBackend rmd_backend = resolved.options.rmd_backend;
    if (options::compensation && rmd_backend != ggml::gemmini::RmdBackend::cpu_direct) {
        return hp1_fail(st, node, layer, "options",
                        "compensation=ON with rmd_backend=gemmini_ws_compact: PoTal has no WS residual executor");
    }
#if LOG_DEBUG
    log::debug(layer.c_str(),
                "[POTAL] config backend=POTAL operation=MUL_MAT node=%s node_index=%d model_arch=%s layer=%s "
                "weight=%s weight_type=%s weight_granularity=%s activation_quant=%s compensation=%s rmd_backend=%s "
                "M=%lld N=%lld K=%lld rows=%s %s params_sha256=%s",
                node->name, node_index, state.model_arch.empty() ? "unknown" : state.model_arch.c_str(),
                layer.c_str(), w->name, options::weight_type,
                options::weight_granularity, options::activation_quant_name(), options::compensation_name,
                rmd_backend_name(rmd_backend), (long long) s.m, (long long) s.n, (long long) s.k,
                s.m == 1 ? "decode_shaped(M=1)" : "prefill_shaped(M>1)", trace_fields().c_str(),
                potal_hw_params_sha256());
#else
    (void) node_index;
#endif

    // Tile plan first: the quantizers read the stripe geometry from it.
    potal_tile_plan plan;
    const potal_status plan_status = potal_tile_select((size_t) s.m, (size_t) s.n, (size_t) s.k,
                                                       quants::weight::hp1::block_k / POTAL_HW_DIM, &plan);
    if (plan_status != POTAL_OK) {
        const std::string why = std::string("potal_tile_select status=POTAL_") + potal_status_name(plan_status);
        return hp1_fail(st, node, layer, "tile_select", why.c_str());
    }

    ggml_gemmini_args_t args;
    const auto          residual_route = rmd_backend == ggml::gemmini::RmdBackend::cpu_direct
                                             ? ggml::gemmini::residual::ResidualRoute::cpu_direct
                                             : ggml::gemmini::residual::ResidualRoute::ws_packet;
    if (!init_args(w, node, s, state.model_arch.c_str(), layer.c_str(), residual_route, plan, args)) {
        return hp1_fail(st, node, layer, "alloc", "out of memory for the activation codes");
    }

    // 1. Weight: the prepared B / carriers / channel scales of w (weight_cache.hpp).
    //    A hit reuses them; otherwise the HP1 blocks are validated and unpacked.
    uint64_t           c0     = gcycle::read();
    auto               t      = clock_type::now();
    weight_lookup      wl     = state.weights.get(w, args);
    const double       wl_s   = seconds_since(t);
    uint64_t           c1     = gcycle::read();
    const bool         reused = wl.outcome == weight_cache_outcome::hit;
    if (reused) {
        st.weight_cache_hits++;
        st.weight_lookup_s += wl_s;
    } else {
        cycle_stage(layer, "potal.weight_pack_hp1", c0, c1);
        st.weight_alloc_s += wl.alloc_s;
        st.weight_pack_s += wl.prepare_s;
        st.weight_validations += wl.alloc_failed ? 0 : 1;
        st.weight_cache_misses += wl.outcome == weight_cache_outcome::miss ? 1 : 0;
        st.weight_cache_bypasses += wl.outcome == weight_cache_outcome::bypass ? 1 : 0;
        st.weight_cache_uncached += wl.outcome == weight_cache_outcome::uncached ? 1 : 0;
    }
    if (wl.alloc_failed) {
        return hp1_fail(st, node, layer, "alloc", "out of memory for the weight operands");
    }
    if (wl.status != hp1::status::ok) {
        const std::string why = std::string(hp1::status_name(wl.status)) + " reader=" +
                                ggml::gemmini::quants::wreader::weight_reader_status_name(wl.reader);
        return hp1_fail(st, node, layer, "weight_hp1", why.c_str());
    }
    const hp1_weight & weight = *wl.weight;
    if (!reused) {
        st.weight_packs++;
        st.weight_pack_bytes += weight.bytes();
    }
    const hp1::stats & ws = weight.stats;
    st.zero_carriers += ws.zero_blocks;
    st.shift_carriers += ws.shift_blocks;
#if LOG_DEBUG
    const auto cache = state.weights.snapshot();
    log::debug(layer.c_str(),
                "[POTAL] weight=%s weight_cache=%s validation=%s pack=%s prepare_ms=%.3f alloc_ms=%.3f "
                "lookup_ms=%.6f weight_bytes=%zu cache_entries=%llu cache_bytes=%llu cache_budget=%llu",
                w->name, weight_cache_outcome_name(wl.outcome), reused ? "skipped" : "performed",
                reused ? "skipped" : "performed", wl.prepare_s * 1e3, wl.alloc_s * 1e3, reused ? wl_s * 1e3 : 0.0,
                weight.bytes(), (unsigned long long) cache.entries, (unsigned long long) cache.bytes,
                (unsigned long long) cache.budget);
#endif

    // Per-call operands: activation codes and the accumulator; B/carriers/scale point at `weight`.
    hp1_operands ops;
    if (!ops.allocate(s, weight)) {
        return hp1_fail(st, node, layer, "alloc", "out of memory for the DMA staging buffers");
    }

    // 2. Activation: ggml::gemmini::quants::quantize_activation (configured algorithm).
    c0            = gcycle::read();
    t             = clock_type::now();
    const bool aq = quantize_activations(x, args, ops);
    st.act_quant_s += seconds_since(t);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.act_quant", c0, c1);
    if (!aq) {
        return hp1_fail(st, node, layer, "act_quant", "quantize_activation failed");
    }
    route_plan route;
    if (!select_route(args, route)) {
        return hp1_fail(st, node, layer, "route", "no activation metadata after quantization");
    }
#if LOG_DEBUG
    const uint64_t nonzero = ws.blocks - ws.zero_blocks;
    log::debug(layer.c_str(),
                "[POTAL] route=%s activation_meta=%s dispatcher=potal_tiled_matmul_auto weight_type=Q8_HP1 "
                "blocks=%llu zero_carrier_count=%llu shift_carrier_count=%llu zero_rows=%llu exponent_min=%d "
                "exponent_max=%d exponent_mean=%.3f channel_scale_min=%.9g channel_scale_max=%.9g",
                route.name, act_route_name(route.activation), (unsigned long long) ws.blocks,
                (unsigned long long) ws.zero_blocks, (unsigned long long) ws.shift_blocks,
                (unsigned long long) ws.zero_rows, ws.min_m, ws.max_m, nonzero ? ws.sum_m / nonzero : 0.0,
                (double) ws.min_channel_scale, (double) ws.max_channel_scale);
#endif

    // 3. PoTal: potal_tiled_matmul_auto on the hart-0 issuer, then F32 reconstruction.
    c0 = gcycle::read();
    t  = clock_type::now();
    const route_result r = execute_route(route, args, ops, [](const tiled_job & job) { return run_tiled(job); });
    // Caller-side interval: submission, issuer wait and reconstruction, not SCU time.
    st.route_wall_s += seconds_since(t);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.route_call", c0, c1);
    st.hw_requests += r.jobs;
    st.tiled_jobs += r.jobs;
    st.tiled_invocations += r.work.invocations;
    st.tiled_validated += r.work.validated_gemms;
    st.tiled_fences += r.work.fences;
    st.scale_writes += r.work.scale_writes;
    st.scale_releases += r.work.scale_releases;
    st.tlb_flushes += r.work.tlb_flushes;
    st.scale_s += r.work.scale_s;
    st.issue_s += r.work.issue_s;
    st.fence_s += r.work.fence_s;
    st.last_status     = r.last.status;
    st.last_generation = r.work.last_generation;
#if LOG_DEBUG
    const potal_tile_plan & p = r.last.plan;
    log::debug(layer.c_str(),
                "[POTAL] route=%s M=%lld N=%lld K=%lld jobs=%llu k_groups=%zu tile_I=%zu tile_J=%zu tile_K=%zu "
                "I0=%zu J0=%zu K0=%zu pad_I=%zu pad_J=%zu pad_K=%zu spad_rows=%zu spad_rows_available=%zu "
                "acc_rows=%zu acc_rows_available=%zu scale_slots=%u scale_slots_available=%u passes_per_block=%u "
                "generation=%u tlb_flushes=%llu loop_ws_issued=%llu validated_api_issues=%llu fence_ok=%llu "
                "scale_writes=%llu scale_releases=%llu status=POTAL_%s",
                route.name, (long long) s.m, (long long) s.n, (long long) s.k, (unsigned long long) r.jobs,
                r.k_groups, p.tile_I, p.tile_J, p.tile_K, p.I0, p.J0, p.K0, p.pad_I, p.pad_J, p.pad_K, p.spad_rows,
                p.max_spad_rows, p.acc_rows, p.max_acc_rows, p.scale_slots, options::scale_window_slots,
                p.passes_per_block, r.work.last_generation, (unsigned long long) r.work.tlb_flushes,
                (unsigned long long) r.work.invocations, (unsigned long long) r.work.validated_gemms,
                (unsigned long long) r.work.fences, (unsigned long long) r.work.scale_writes,
                (unsigned long long) r.work.scale_releases, potal_status_name(r.last.status));
#endif
    if (!r.ok) {
        if (r.fatal) {
            // The accelerator may still reach these buffers: leak them, the cached weight included.
            ops.abandon();
            if (wl.transient) {
                wl.transient->abandon();
            } else {
                state.weights.abandon(&weight);
            }
        }
        const std::string detail = std::string("route=") + route.name + " status=POTAL_" +
                                   potal_status_name(r.last.status) + " step=" +
                                   (r.last.stage ? r.last.stage : (r.stage ? r.stage : "-")) +
                                   (r.fatal ? " fatal=1" : " fatal=0") +
                                   (r.last.detail ? std::string(" issuer=") + r.last.detail : std::string());
        return hp1_fail(st, node, layer, r.stage ? r.stage : "potal_gemm", detail.c_str());
    }
    st.route_completed[static_cast<int>(route.activation)]++;

    // 4. Error compensation: whatever residual the quantizer actually captured.
    c0 = gcycle::read();
    t  = clock_type::now();
    const compensation_result comp = apply_compensation(args);
    st.compensation_s += seconds_since(t);
    c1 = gcycle::read();
    cycle_stage(layer, "potal.compensation", c0, c1);
    st.residual_payloads += comp.payloads;
    st.compensation_executed += comp.executed;
    st.residual_events += comp.residuals;
#if LOG_DEBUG
    log::debug(layer.c_str(),
                "[POTAL] compensation=%s rmd_backend=%s residual_payloads=%zu corrections_merged=%zu "
                "residual_events=%zu nonzero_corrections=%zu status=%s",
                options::compensation_name, rmd_backend_name(rmd_backend), comp.payloads, comp.executed,
                comp.residuals, comp.nonzero, comp.ok ? "ok" : "failed");
#endif
    if (!comp.ok) {
        st.compensation_failed++;
        return hp1_fail(st, node, layer, "compensation", comp.detail ? comp.detail : "residual correction failed");
    }

    st.hp1_completed++;
    record_unmeasured_npu();
#if LOG_DEBUG
    log::debug(layer.c_str(), "[POTAL] node=%s route=%s dispatch=HARDWARE fence=POTAL_OK completed=%llu",
                node->name, route.name, (unsigned long long) st.hp1_completed);
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
    log::debug("potal.supports_op",
                "[POTAL] supports_op_reject op=MUL_MAT node=%s reason=%s src0=%s src0_type=%s src1_type=%s ne0=%lld "
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
