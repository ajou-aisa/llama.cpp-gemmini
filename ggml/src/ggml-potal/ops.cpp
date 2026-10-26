#include "ops.hpp"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "log.hpp"
#include "matmul/attention.hpp"
#include "matmul/dense.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/pack.hpp"
#include "matmul/stripe.hpp"
#include "potal.h"
#include "potal_hw_params.h"

#include "ggml-gemmini/matmul/options.hpp" // resolve_matmul_options (shared Gemmini source)

// Existing Gemmini logging/cycle/trace system (ggml-gemmini-utils); no gemmini.h.
#include <gemmini/cycle_reader.hpp>
#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#include <gemmini/trace-context.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>

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

using log::trace_fields;

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

ggml_status mul_mat_attention(backend_state & state, ggml_tensor * node) {
    auto & stats = state.stats;
    (attention::get_role(node) == attention::role::qk ? stats.attention_qk_nodes : stats.attention_pv_nodes)++;
    const auto start = clock_type::now();
    const attention::result result = attention::execute_hardware(node);
    stats.attention_wall_s += seconds_since(start);
    stats.attention_limb_jobs += result.jobs;
    stats.attention_residual_jobs += result.residual_jobs;
    stats.attention_loop_ws += result.loops;
    stats.attention_qk_limb_jobs += result.qk_limb_jobs;
    stats.attention_pv_corrections += result.correction_jobs;
    stats.attention_bias_mvins += result.bias_updates;
    stats.hw_requests += result.jobs;
    stats.scale_writes += result.scale_writes;
    stats.scale_releases += result.scale_releases;
    stats.tlb_flushes += result.tlb_flushes;
    stats.scale_s += result.scale_s;
    stats.issue_s += result.issue_s;
    stats.fence_s += result.fence_s;
    stats.last_status = result.status;
    if (result.status != POTAL_OK) {
        ++stats.attention_failed;
        // Attention accepts strided outputs; poison logical elements only.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (int64_t b = 0; b < node->ne[3]; ++b)
            for (int64_t h = 0; h < node->ne[2]; ++h)
                for (int64_t i = 0; i < node->ne[1]; ++i)
                    for (int64_t j = 0; j < node->ne[0]; ++j) {
                        auto * p = static_cast<char *>(node->data) +
                                   b * node->nb[3] + h * node->nb[2] + i * node->nb[1] + j * node->nb[0];
                        std::memcpy(p, &nan, sizeof(nan));
                    }
        log::failure("potal.attention", "[POTAL_ATTENTION] node=%s status=%s stage=%s fatal=%d",
                     node->name, potal_status_name(result.status), result.stage ? result.stage : "-", result.fatal);
        return GGML_STATUS_FAILED;
    }
    ++stats.attention_completed;
    log::debug("potal.attention", "[POTAL_ATTENTION] node=%s role=%s dispatch=HARDWARE limb_jobs=%llu "
               "loop_ws=%llu bias_mvins=%llu",
               node->name, attention::get_role(node) == attention::role::qk ? "QK" : "PV",
               (unsigned long long)result.jobs, (unsigned long long)result.loops,
               (unsigned long long)result.bias_updates);
    ggml::gemmini::performance::record_npu("POTAL", "attention", "work_total_cycles", 0, false,
                                          "unavailable_device_counter");
    return GGML_STATUS_SUCCESS;
}

// ---- F32 x Q8_HP1 routed, tiled GEMM ---------------------------------------

} // namespace

bool options::paired_fences_conservative(bool * invalid) {
    *invalid           = false;
    const char * value = std::getenv("GGML_POTAL_PAIRED_FENCES");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    std::string v(value);
    for (char & ch : v) {
        ch = (char) std::toupper((unsigned char) ch);
    }
    if (v == "RTL") {
        return false;
    }
    if (v == "CONSERVATIVE") {
        return true;
    }
    *invalid = true;
    return false;
}

bool options::paired_auto_requested(bool * invalid) {
    *invalid           = false;
    const char * value = std::getenv("GGML_POTAL_PAIRED_AUTO");
    if (value == nullptr || *value == '\0') {
        return paired_auto_default;
    }
    std::string v(value);
    for (char & ch : v) {
        ch = (char) std::toupper((unsigned char) ch);
    }
    if (v == "ON" || v == "1") {
        return true;
    }
    if (v == "OFF" || v == "0") {
        return false;
    }
    *invalid = true;
    return false;
}

namespace {

ggml_status hp1_fail(ggml_backend_potal_stats & st, ggml_tensor * node, const std::string & layer, const char * stage,
                     const char * detail) {
    st.hp1_failed++;
    poison_dst(node);
    log::failure(layer.c_str(), "[POTAL] node=%s layer=%s dispatch=FAILED stage=%s %s; dst set to NaN", node->name,
                 layer.c_str(), stage, detail);
    return GGML_STATUS_FAILED;
}

// STRIPE_PIPELINE (matmul/stripe.hpp): ExSIA hands every folding-committed
// stripe to the hart-0 issuer at once; Main and Residual GEMMs on PoTal,
// reconstruction and merge on the CPU, dst written once on success.
ggml_status mul_mat_hp1_pipeline(backend_state & state, ggml_tensor * node, const std::string & layer,
                                 ggml_gemmini_args_t & args, weight_lookup & wl, const potal_tile_plan & plan,
                                 size_t job_capacity) {
    ggml_backend_potal_stats & st     = state.stats;
    const hp1_weight &         weight = *wl.weight;

    // Paired route: requested (build default / GGML_POTAL_PAIRED_AUTO) and eligible
    // (compensation ON: the ws_packet residual exists). Compensation OFF keeps the
    // serial route; an invalid request fails the node (an options failure, like the
    // matmul options: before the node counts as a pipelined node), never picks a route.
    bool       paired_invalid = false, fences_invalid = false;
    const bool paired_request = options::paired_auto_requested(&paired_invalid);
    const bool conservative   = options::paired_fences_conservative(&fences_invalid);
    if (paired_invalid) {
        return hp1_fail(st, node, layer, "options", "GGML_POTAL_PAIRED_AUTO must be ON, OFF, 1 or 0");
    }
    if (fences_invalid) {
        return hp1_fail(st, node, layer, "options", "GGML_POTAL_PAIRED_FENCES must be RTL or CONSERVATIVE");
    }
    st.stripe_pipeline_nodes++;

    pipeline_config cfg;
    cfg.job_capacity = job_capacity;
    cfg.residual     = options::compensation;
    cfg.paired       = paired_request && options::compensation;
    cfg.paired_conservative_fences = cfg.paired && conservative;
    cfg.layer        = layer;
    issuer_owner owner;
    const uint64_t c0 = gcycle::read();
    const auto     t  = clock_type::now();
    const pipeline_report r = run_stripe_pipeline(node->src[1], args, weight, plan, cfg, owner);
    // Caller-side interval: quantization, the overlapped issuer work and the
    // final drain; not SCU or array time.
    st.pipeline_wall_s += seconds_since(t);
    cycle_stage(layer, "potal.stripe_pipeline", c0, gcycle::read());

    // The run's counters, merged once on this thread after the run ended.
    const pipeline_counts & c = r.counts;
    st.hw_requests += c.main_logical_gemms + c.residual_logical_gemms;
    st.tiled_jobs += c.main_logical_gemms + c.residual_logical_gemms;
    st.tiled_invocations += c.main_loop_ws + c.residual_loop_ws;
    st.tiled_validated += c.main_validated + c.residual_validated;
    st.tiled_fences += c.main_fences + c.residual_fences;
    st.scale_writes += c.scale_writes;
    st.scale_releases += c.scale_releases;
    st.tlb_flushes += c.tlb_flushes;
    st.paired_stripes += c.paired_stripes;
    st.paired_fallback_stripes += c.paired_fallback_stripes;
    st.paired_main_loop_ws += c.paired_main_loop_ws;
    st.paired_residual_loop_ws += c.paired_residual_loop_ws;
    st.paired_index_configs += c.paired_index_configs;
    st.paired_index_writes += c.paired_index_writes;
    st.paired_readbacks += c.paired_readbacks;
    st.paired_readback_tiles += c.paired_readback_tiles;
    st.paired_fences += c.paired_fences;
    st.paired_drains += c.paired_drains;
    st.act_quant_s += c.act_quant_s;
    st.scale_s += c.scale_s;
    st.issue_s += c.issue_s;
    st.fence_s += c.fence_s;
    st.last_status     = r.status;
    st.last_generation = c.last_generation != 0 ? c.last_generation : st.last_generation;
    st.residual_payloads += c.residual_nonempty_packets;
    st.compensation_executed += c.merges_completed;
    st.stripes_accepted += c.stripes_accepted;
    st.stripes_semantic_completed += c.stripes_semantic_completed;
    st.max_outstanding_semantic = std::max(st.max_outstanding_semantic, c.max_outstanding_semantic);
    st.producer_capacity_waits += c.producer_capacity_waits;
    st.main_logical_gemms += c.main_logical_gemms;
    st.main_loop_ws += c.main_loop_ws;
    st.main_fences += c.main_fences;
    st.residual_nonempty_packets += c.residual_nonempty_packets;
    st.residual_logical_gemms += c.residual_logical_gemms;
    st.residual_loop_ws += c.residual_loop_ws;
    st.residual_fences += c.residual_fences;
    st.residual_compact_k += c.residual_compact_k;
    st.residual_execution_k += c.residual_execution_k;
    st.residual_active_lane_rows += c.residual_active_lane_rows;
    st.radix_reconstructions += c.radix_reconstructions;
    st.merges_completed += c.merges_completed;
    st.corrections_nonzero += c.corrections_nonzero;
    st.last_stripe_count = r.expected_stripes;
#if LOG_DEBUG
    log::debug(layer.c_str(),
                "[POTAL] route=ExSIA+block matmul_mode=STRIPE_PIPELINE residual_executor=POTAL/%s node=%s "
                "run_id=%s stripe_count=%zu job_capacity=%zu tile_I=%zu tile_J=%zu tile_K=%zu stripes_accepted=%llu "
                "stripes_semantic_completed=%llu max_outstanding_semantic=%llu producer_capacity_waits=%llu "
                "main_logical_gemms=%llu main_loop_ws=%llu main_fences=%llu residual_nonempty_packets=%llu "
                "residual_logical_gemms=%llu residual_loop_ws=%llu residual_fences=%llu residual_compact_k=%llu "
                "residual_execution_k=%llu active_lane_rows=%llu radix_reconstructions=%llu merges_completed=%llu "
                "corrections_nonzero=%llu scale_writes=%llu scale_releases=%llu tlb_flushes=%llu generation=%u "
                "paired=%s paired_fences=%s paired_stripes=%llu paired_fallback_stripes=%llu paired_main_loop_ws=%llu "
                "paired_residual_loop_ws=%llu paired_index_configs=%llu paired_index_writes=%llu "
                "paired_readbacks=%llu paired_readback_tiles=%llu paired_fences=%llu paired_drains=%llu status=%s",
                options::residual_executor_name(options::MatmulInvocationMode::stripe_pipeline), node->name,
                r.run_bound ? std::to_string(r.run_id).c_str() : "-", r.expected_stripes, job_capacity,
                plan.tile_I, plan.tile_J, plan.tile_K, (unsigned long long) c.stripes_accepted,
                (unsigned long long) c.stripes_semantic_completed, (unsigned long long) c.max_outstanding_semantic,
                (unsigned long long) c.producer_capacity_waits, (unsigned long long) c.main_logical_gemms,
                (unsigned long long) c.main_loop_ws, (unsigned long long) c.main_fences,
                (unsigned long long) c.residual_nonempty_packets, (unsigned long long) c.residual_logical_gemms,
                (unsigned long long) c.residual_loop_ws, (unsigned long long) c.residual_fences,
                (unsigned long long) c.residual_compact_k, (unsigned long long) c.residual_execution_k,
                (unsigned long long) c.residual_active_lane_rows, (unsigned long long) c.radix_reconstructions,
                (unsigned long long) c.merges_completed, (unsigned long long) c.corrections_nonzero,
                (unsigned long long) c.scale_writes, (unsigned long long) c.scale_releases,
                (unsigned long long) c.tlb_flushes, c.last_generation, cfg.paired ? "ON" : "OFF",
                cfg.paired_conservative_fences ? "CONSERVATIVE" : "RTL",
                (unsigned long long) c.paired_stripes, (unsigned long long) c.paired_fallback_stripes,
                (unsigned long long) c.paired_main_loop_ws, (unsigned long long) c.paired_residual_loop_ws,
                (unsigned long long) c.paired_index_configs, (unsigned long long) c.paired_index_writes,
                (unsigned long long) c.paired_readbacks, (unsigned long long) c.paired_readback_tiles,
                (unsigned long long) c.paired_fences, (unsigned long long) c.paired_drains,
                r.ok ? "committed" : "failed");
#endif
    if (!r.ok) {
        st.stripe_pipeline_failed++;
        if (r.fatal) {
            // The accelerator may still reach the weight: leak it, the cached entry included.
            if (wl.transient) {
                wl.transient->abandon();
            } else {
                state.weights.abandon(&weight);
            }
        }
        const std::string detail = std::string("route=ExSIA+block matmul_mode=STRIPE_PIPELINE status=POTAL_") +
                                   potal_status_name(r.status) + (r.fatal ? " fatal=1" : " fatal=0") +
                                   " detail=" + (r.detail.empty() ? std::string("-") : r.detail);
        return hp1_fail(st, node, layer, r.stage != nullptr ? r.stage : "stripe_pipeline", detail.c_str());
    }
    st.route_completed[static_cast<int>(act_route::exsia)]++;
    st.hp1_completed++;
    record_unmeasured_npu();
#if LOG_DEBUG
    log::debug(layer.c_str(), "[POTAL] node=%s route=ExSIA+block dispatch=HARDWARE matmul_mode=STRIPE_PIPELINE "
                              "fence=POTAL_OK completed=%llu",
                node->name, (unsigned long long) st.hp1_completed);
#endif
    return GGML_STATUS_SUCCESS;
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
        return hp1_fail(st, node, layer, "options",
                        "invalid matmul options (GEMMINI_MATMUL_MODE / GEMMINI_RMD_BACKEND / "
                        "GEMMINI_STRIPE_JOB_CAPACITY)");
    }
    const options::MatmulInvocationMode mode        = resolved.options.mode;
    const ggml::gemmini::RmdBackend     rmd_backend = resolved.options.rmd_backend;
    if (const char * why = options::unsupported_combination(mode, rmd_backend, resolved.options.job_capacity)) {
        return hp1_fail(st, node, layer, "options", why);
    }
#if LOG_DEBUG
    log::debug(layer.c_str(),
                "[POTAL] config backend=POTAL operation=MUL_MAT node=%s node_index=%d model_arch=%s layer=%s "
                "weight=%s weight_type=%s weight_granularity=%s activation_quant=%s compensation=%s rmd_backend=%s "
                "matmul_mode=%s residual_executor=POTAL/%s job_capacity=%zu "
                "M=%lld N=%lld K=%lld rows=%s %s params_sha256=%s",
                node->name, node_index, state.model_arch.empty() ? "unknown" : state.model_arch.c_str(),
                layer.c_str(), w->name, options::weight_type,
                options::weight_granularity, options::activation_quant_name(), options::compensation_name,
                options::rmd_backend_name(rmd_backend), options::matmul_mode_name(mode),
                options::residual_executor_name(mode), resolved.options.job_capacity, (long long) s.m,
                (long long) s.n, (long long) s.k, s.m == 1 ? "decode_shaped(M=1)" : "prefill_shaped(M>1)",
                trace_fields().c_str(), potal_hw_params_sha256());
#else
    (void) node_index;
#endif

    // Tile plan first: the quantizers read the stripe geometry from it. In
    // STRIPE_PIPELINE this is the frozen Main plan of every stripe.
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

    if (mode == options::MatmulInvocationMode::stripe_pipeline) {
        return mul_mat_hp1_pipeline(state, node, layer, args, wl, plan, resolved.options.job_capacity);
    }

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
                options::compensation_name, options::rmd_backend_name(rmd_backend), comp.payloads, comp.executed,
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
    if (attention::get_role(op) != attention::role::none) {
        return attention::check(op, false);
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
    if (attention::get_role(node) != attention::role::none) {
        return attention::check(node, true);
    }
    return is_hp1_mul_mat(node) ? check_mul_mat_hp1(node, true, nullptr) : check_mul_mat(node, true, nullptr);
}

ggml_status compute_mul_mat(backend_state & state, ggml_tensor * node, int node_index) {
    if (attention::get_role(node) != attention::role::none) {
        return mul_mat_attention(state, node);
    }
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
