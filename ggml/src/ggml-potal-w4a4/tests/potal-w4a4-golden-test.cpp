// W4A4 native golden (H3-8I-25AQ W4A4): the fork's four-bit lowering against the independent
// golden of the existing ggml-gemmini A4W4 algorithm (support/w4a4-golden.hpp).
//
//   X  ExSIA cross-check: ggml-gemmini's own quantize_activation (compiled at 4/4) vs the golden:
//      Main digits, residual r and stripe theta, every element
//   D  dense MXINT4 x F32 through run_stripe_pipeline, paired K-tile route on the H2 emulator:
//      Main INT32 and Residual INT32 per (lane, row) captured at the paired runner, the set of
//      residual lane rows, and the committed F32 output bitwise
//   A  QK / PV: attention::prepare_* + reference INT64 and stream scale vs the golden, and
//      attention::execute on the RTL-semantics emulator F32 output bitwise
// Inputs are seeded random plus boundaries: K/M/N tails, SCU carriers >= 32 (saturation), zero
// weight blocks, the hardware-only weight code -8, activations that saturate INT32 (the
// INT32_MAX residual exception), K codes clamped at 119, P exactly 0 and 1, GQA broadcast.
// Prints W4A4_GOLDEN PASS cases=<n> only when every case matched.

#include "support/paired-h2-emulator.hpp"
#include "support/stripe-owner.hpp"
#include "support/w4a4-golden.hpp"

#include "ggml-cpp.h"
#include "ggml-gemmini.h"
#include "ggml.h"
#include "matmul/attention.hpp"
#include "matmul/attention-residual.hpp"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/paired-auto.hpp"
#include "matmul/stripe-api.hpp"
#include "matmul/stripe.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "quants/act/exsia/types.hpp"
#include "quants/act/quantize.hpp"
#include "residual/rmd/rmd-types.hpp"
#include "residual/rmd/rmd-bitmap-builder.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <variant>
#include <vector>

using namespace ggml::potal;
namespace gold = w4a4_golden;
namespace att  = ggml::potal::attention;

namespace {

int cases = 0, failed = 0;

void verdict(const char * group, const std::string & name, bool ok, const std::string & detail) {
    ++cases;
    failed += !ok;
    std::printf("W4A4_CASE %s %s %s %s\n", group, name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

// Independent accounting from accepted sink commands, not production counters
// or the executor's predicted loop totals. Fixed DIM32 includes all tile padding.
struct issue_observer {
    potal_paired_sink downstream;
    uint32_t operation = POTAL_ISSUE_LINEAR;
    uint64_t bounds = 0;
    potal_issue_counters expected{};

    static potal_status command(void * ctx, const potal_paired_cmd_record * rec) {
        auto & self = *static_cast<issue_observer *>(ctx);
        if (rec->funct == POTAL_FUNCT_LOOP_WS && rec->xd != 1)
            return POTAL_HW_INVALID_ENCODING;
        const auto status = self.downstream.cmd(self.downstream.ctx, rec);
        if (status != POTAL_OK) return status;
        if (rec->funct == POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS) self.bounds = rec->rs2;
        if (rec->funct == POTAL_FUNCT_LOOP_WS) {
            const bool residual = (rec->rs1 >> 62) & 1;
            const uint64_t macs = (self.bounds & 65535) * ((self.bounds >> 16) & 65535) *
                ((self.bounds >> 32) & 65535) * UINT64_C(32768);
            auto & e = self.expected;
            if (self.operation == POTAL_ISSUE_ATTN_QK) {
                ++e.w4_attn_qk_gemm_issues;
                e.w4_attn_qk_residual_gemm_issues += residual;
            } else if (self.operation == POTAL_ISSUE_ATTN_PV) {
                ++e.w4_attn_pv_gemm_issues;
                e.w4_attn_pv_residual_gemm_issues += residual;
            } else if (residual) {
                ++e.w4_residual_gemm_issues;
                e.w4_residual_macs_padded += macs;
            } else {
                ++e.w4_main_gemm_issues;
                e.w4_main_macs_padded += macs;
            }
        }
        return status;
    }

    potal_paired_sink sink() { return {command, this, nullptr}; }

    bool check(const char * name, const potal_issue_counters & before) const {
        potal_issue_counters after;
        potal_issue_counters_read(&after);
        bool ok = true;
#define CHECK_ISSUE(f) ok = (after.f - before.f == expected.f) && ok;
        POTAL_ISSUE_COUNTER_FIELDS(CHECK_ISSUE)
#undef CHECK_ISSUE
        ok = ok && (expected.w4_main_gemm_issues || expected.w4_attn_qk_gemm_issues ||
                    expected.w4_attn_pv_gemm_issues);
        std::printf("W4A4_COUNTERS %s %s main=%" PRIu64 " residual=%" PRIu64
                    " main_macs_padded=%" PRIu64 " residual_macs_padded=%" PRIu64
                    " qk=%" PRIu64 " pv=%" PRIu64 " qk_jobs=%" PRIu64 " pv_jobs=%" PRIu64 "\n",
                    name, ok ? "PASS" : "FAIL", after.w4_main_gemm_issues - before.w4_main_gemm_issues,
                    after.w4_residual_gemm_issues - before.w4_residual_gemm_issues,
                    after.w4_main_macs_padded - before.w4_main_macs_padded,
                    after.w4_residual_macs_padded - before.w4_residual_macs_padded,
                    after.w4_attn_qk_gemm_issues - before.w4_attn_qk_gemm_issues,
                    after.w4_attn_pv_gemm_issues - before.w4_attn_pv_gemm_issues,
                    after.w4_attn_qk_digit_jobs - before.w4_attn_qk_digit_jobs,
                    after.w4_attn_pv_digit_jobs - before.w4_attn_pv_digit_jobs);
        return ok;
    }
};

#if defined(GGML_GEMMINI_EXSIA_STRIPE_ANCHOR)
constexpr int anchor = GGML_GEMMINI_EXSIA_STRIPE_ANCHOR;
#else
constexpr int anchor = 2;
#endif

// ---- dense -------------------------------------------------------------------------------------

struct dense_spec {
    const char * name;
    size_t       M, N, K;
    uint32_t     seed;
    bool         boundary;
};

struct dense_node {
    ggml_context_ptr ctx;
    ggml_tensor *    w = nullptr, * x = nullptr, * y = nullptr;
};

bool make_dense(const dense_spec & c, dense_node & nd) {
    ggml_init_params ip = { (c.M * c.K + c.N * c.K + c.M * c.N) * 8 + 4 * 1024 * 1024, nullptr, false };
    nd.ctx.reset(ggml_init(ip));
    if (!nd.ctx) return false;
    nd.w = ggml_new_tensor_2d(nd.ctx.get(), GGML_TYPE_MXINT4, (int64_t) c.K, (int64_t) c.N);
    nd.x = ggml_new_tensor_2d(nd.ctx.get(), GGML_TYPE_F32, (int64_t) c.K, (int64_t) c.M);
    nd.y = ggml_mul_mat(nd.ctx.get(), nd.w, nd.x);
    std::mt19937                    rng(c.seed);
    std::normal_distribution<float> nd01(0.0f, 1.0f);
    std::vector<float>              wf(c.N * c.K);
    const float mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
    for (size_t n = 0; n < c.N; ++n)
        for (size_t k = 0; k < c.K; ++k) {
            float m = mag[(k / 32 + n) % 4];
            if (c.boundary) {
                if (k / 32 == 1 && n % 3 == 0) m = 0.0f;           // zero block: m = INT16_MIN
                if (k / 32 == 2 && n % 4 == 1) m = 4.0e3f;         // carrier >= 32 against 1e-7 blocks
                if (k / 32 == 0 && n % 4 == 1) m = 1.0e-7f;
            }
            wf[n * c.K + k] = nd01(rng) * m;
        }
    ggml_quantize_chunk(GGML_TYPE_MXINT4, wf.data(), nd.w->data, 0, (int64_t) c.N, (int64_t) c.K, nullptr);
    if (c.boundary) {
        // Hardware-domain code -8 (GGUF encoders emit [-7,7]): low nibble 0 of a nonzero block.
        const size_t row_bytes = ggml_row_size(GGML_TYPE_MXINT4, (int64_t) c.K);
        for (size_t n = 2; n < c.N; n += 5) {
            uint8_t * blk = static_cast<uint8_t *>(nd.w->data) + n * row_bytes + 3 * gold::MXINT4_BLOCK_BYTES;
            int16_t   m;
            std::memcpy(&m, blk + 16, 2);
            if (m != gold::NEG) blk[n % 16] = (uint8_t) (blk[n % 16] & 0xF0);
        }
    }
    float * v = static_cast<float *>(nd.x->data);
    for (size_t m = 0; m < c.M; ++m) {
        for (size_t k = 0; k < c.K; ++k) v[m * c.K + k] = nd01(rng) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
        if (m % 2 == 0) v[m * c.K + (m * 7) % c.K] = 60.0f * (m % 4 ? -1.0f : 1.0f);
        if (c.boundary && m % 5 == 1)
            for (size_t k = 32; k < 64 && k < c.K; ++k) v[m * c.K + k] = (k % 2 ? -1.0f : 1.0f) * 4.0e10f; // saturates INT32
        if (c.boundary && m % 7 == 3) v[m * c.K + (m * 13) % c.K] = 2.5f * 1024.0f; // exact tie after scaling
    }
    return true;
}

// ggml-gemmini's own ExSIA at 4/4 on args from the fork's init_args.
void exsia_cross_check(const dense_spec & c, const dense_node & nd, const potal_tile_plan & plan,
                       const gold::dense_result & g) {
    namespace exsia = ggml::gemmini::quants::act::exsia;
    namespace grmd  = ggml::gemmini::rmd;
    const hp1_shape     s = { (int64_t) c.M, (int64_t) c.N, (int64_t) c.K };
    ggml_gemmini_args_t args;
    if (!init_args(nd.w, nd.y, s, "qwen2", "w4a4-golden-exsia", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                   args) ||
        !ggml::gemmini::quants::quantize_activation(nd.x, args)) {
        verdict("X", c.name, false, "reason=gemmini_quantize_activation_failed");
        return;
    }
    const auto * meta = std::get_if<exsia::Meta>(&args.act_quant.storage());
    if (meta == nullptr || args.A == nullptr) {
        verdict("X", c.name, false, "reason=no_exsia_meta");
        return;
    }
    std::vector<int64_t> res(c.M * c.K, 0);
    bool                 read_ok = true;
    for (const auto & ph : meta->rmd_packets) {
        const grmd::StripePacket & p = *ph;
        for (const grmd::BlockDescriptor & b : p.blocks)
            for (uint8_t pos = 0; pos < b.active_lane_count; ++pos) {
                int64_t place = 1;
                for (uint8_t l = 0; l < b.lane_ids[pos]; ++l) place *= gold::RADIX;
                for (size_t row = 0; row < p.row_count; ++row)
                    for (size_t k = 0; k < b.compact_k_count; ++k) {
                        int32_t d = 0;
                        read_ok = read_ok && grmd::read_packet_digit(p, b, pos, row, k, d) == grmd::RmdStatus::success;
                        const size_t ok_ = (size_t) b.block_id * grmd::kBlockSize + p.k_indices[b.k_index_offset + k];
                        res[(p.row_begin + row) * c.K + ok_] += (int64_t) d * place;
                    }
            }
    }
    size_t code_mis = 0, res_mis = 0, th_mis = meta->theta.size() != g.theta.size();
    for (size_t i = 0; i < c.M * c.K; ++i) {
        code_mis += args.A[i] != g.codes[i];
        res_mis += res[i] != g.residual[i];
    }
    for (size_t i = 0; i < std::min(meta->theta.size(), g.theta.size()); ++i) th_mis += meta->theta[i] != g.theta[i];
    char b[320];
    std::snprintf(b, sizeof(b),
                  "rho=%d stripes=%zu outliers=%zu residual_nnz=%zu max_lanes=%zu int32_max_exceptions=%zu "
                  "code_mismatch=%zu residual_mismatch=%zu theta_mismatch=%zu packet_read=%d",
                  (int) meta->rho, g.theta.size(), g.outliers, g.residual_nnz, g.max_lanes, g.int32_max_exceptions,
                  code_mis, res_mis, th_mis, (int) read_ok);
    verdict("X", c.name,
            read_ok && meta->rho == gold::DENSE_RHO && !code_mis && !res_mis && !th_mis &&
                (!c.boundary || g.int32_max_exceptions > 0),
            b);
}

void dense_case(const dense_spec & c) {
    dense_node nd;
    potal_tile_plan plan;
    if (!make_dense(c, nd) || potal_tile_select(c.M, c.N, c.K, 32 / POTAL_HW_DIM, &plan) != POTAL_OK) {
        verdict("D", c.name, false, "reason=fixture_or_tile_select");
        return;
    }
    gold::mxint4_weight gw;
    gold::dense_result  g;
    std::string         why;
    if (!gold::decode_mxint4(static_cast<const uint8_t *>(nd.w->data), nd.w->nb[1], c.N, c.K, gw, why) ||
        !gold::dense(static_cast<const float *>(nd.x->data), c.M, c.K, gw, plan.tile_I * POTAL_HW_DIM, anchor, g, why)) {
        verdict("D", c.name, false, "reason=golden_" + why);
        return;
    }
    exsia_cross_check(c, nd, plan, g);

    const hp1_shape     s = { (int64_t) c.M, (int64_t) c.N, (int64_t) c.K };
    ggml_gemmini_args_t args;
    hp1_weight          weight;
    if (!init_args(nd.w, nd.y, s, "qwen2", "w4a4-golden", ggml::gemmini::residual::ResidualRoute::ws_packet, plan, args) ||
        !weight.allocate((int64_t) c.N, (int64_t) c.K) || prepare_weight(args, weight) != quants::weight::hp1::status::ok) {
        verdict("D", c.name, false, "reason=fork_init_args_or_prepare_weight");
        return;
    }
    float * dst = static_cast<float *>(nd.y->data);
    const uint32_t sentinel_bits = 0x7fc0beefu;
    for (size_t i = 0; i < c.M * c.N; ++i) std::memcpy(dst + i, &sentinel_bits, 4);

    pipeline_config cfg;
    cfg.job_capacity             = 1;
    cfg.residual                 = options::compensation;
    cfg.paired                   = true;
    cfg.paired_ktile_scale_reuse = POTAL_HW_DIM == 32;
    cfg.layer                    = "w4a4-golden";
    cfg.backend                  = "EMULATED";

    std::mutex                    mu;
    std::vector<int32_t>          main_c(c.M * c.N, 0);
    std::vector<uint8_t>          main_seen(c.M, 0);
    std::map<std::pair<uint8_t, uint32_t>, std::vector<int32_t>> lane_c;
    size_t                        calls = 0, capture_errors = 0;
    potal_test::emu_session       es;
    potal_test::h2_emulator       h2;
    issue_observer                observer{h2.sink()};
    potal_issue_counters          issue_before;
    potal_issue_counters_read(&issue_before);
    uint32_t                      generation = 0;
    paired_runner runner = [&](const potal_paired_request & req, potal_paired_report & rep) {
        potal_paired_request q = req;
        q.generation           = &generation;
        const potal_paired_sink sink = observer.sink();
        const potal_paired_status st = potal_paired_run_session(q, &sink, rep);
        std::lock_guard<std::mutex> lock(mu);
        ++calls;
        if (st != potal_paired_status::ok) return st;
        if (req.row_begin + req.rows > c.M || req.n != c.N) { ++capture_errors; return st; }
        for (size_t m = 0; m < req.rows; ++m) {
            main_seen[req.row_begin + m] = 1;
            std::memcpy(main_c.data() + (req.row_begin + m) * c.N, req.c_main + m * req.ldc_main, c.N * sizeof(int32_t));
        }
        const auto & rows = rep.plan.residual_rows;
        for (size_t i = 0; i < rows.size(); ++i) {
            const std::pair<uint8_t, uint32_t> key = { rows[i].lane, (uint32_t) (req.row_begin + rows[i].source_row) };
            if (req.c_residual == nullptr || lane_c.count(key)) { ++capture_errors; continue; }
            lane_c[key].assign(req.c_residual + i * req.ldc_residual, req.c_residual + i * req.ldc_residual + c.N);
        }
        return st;
    };
    potal_test::thread_owner owner([&es](const potal_stripe_gemm_desc & d) { return es.run(d); }, runner);
    const pipeline_report    report = run_stripe_pipeline(nd.x, args, weight, plan, cfg, owner);
    owner.join();
    const bool counters_ok = observer.check(c.name, issue_before);

    size_t main_mis = 0, unseen = 0, lane_mis = 0, lane_val_mis = 0, out_mis = 0;
    for (size_t m = 0; m < c.M; ++m) unseen += !main_seen[m];
    for (size_t i = 0; i < c.M * c.N; ++i) main_mis += main_c[i] != g.main_c[i];
    std::set<std::pair<uint8_t, uint32_t>> fork_keys, gold_keys;
    for (const auto & kv : lane_c) fork_keys.insert(kv.first);
    for (const auto & kv : g.lane_c) gold_keys.insert(kv.first);
    lane_mis = fork_keys != gold_keys;
    for (const auto & kv : lane_c) {
        const auto it = g.lane_c.find(kv.first);
        if (it == g.lane_c.end()) continue;
        for (size_t n = 0; n < c.N; ++n) lane_val_mis += kv.second[n] != it->second[n];
    }
    double max_abs = 0;
    for (size_t i = 0; i < c.M * c.N; ++i) {
        out_mis += !same_bits(dst[i], g.out[i]);
        max_abs = std::max(max_abs, std::fabs((double) dst[i] - (double) g.out[i]));
    }
    const pipeline_counts & pc = report.counts;
    const bool ok = counters_ok && report.ok && !report.fatal && pc.paired_fallback_stripes == 0 && pc.paired_stripes == g.theta.size() &&
                    h2.errors.empty() && !h2.scale_miss && !capture_errors && !unseen && !main_mis && !lane_mis &&
                    !lane_val_mis && !out_mis;
    char b[640];
    std::snprintf(b, sizeof(b),
                  "M=%zu N=%zu K=%zu stripes=%zu stripe_rows=%zu paired=%" PRIu64 " fallback=%" PRIu64 " calls=%zu "
                  "main_int32_mismatch=%zu rows_unseen=%zu lane_rows fork=%zu golden=%zu set_mismatch=%zu "
                  "residual_int32_mismatch=%zu f32_differing=%zu/%zu max_abs=%.6g capture_errors=%zu h2_errors='%s' "
                  "stage=%s detail='%s'",
                  c.M, c.N, c.K, g.theta.size(), plan.tile_I * POTAL_HW_DIM, (uint64_t) pc.paired_stripes,
                  (uint64_t) pc.paired_fallback_stripes, calls, main_mis, unseen, fork_keys.size(), gold_keys.size(),
                  lane_mis, lane_val_mis, out_mis, c.M * c.N, max_abs, capture_errors, h2.errors.c_str(),
                  report.stage ? report.stage : "-", report.detail.c_str());
    verdict("D", c.name, ok, b);
}

// ---- attention ---------------------------------------------------------------------------------

struct host_tensor {
    ggml_tensor          t{};
    std::vector<uint8_t> bytes;
    void shape(ggml_type type, std::initializer_list<int64_t> ne) {
        t      = ggml_tensor{};
        t.type = type;
        int i  = 0;
        for (int64_t v : ne) t.ne[i++] = v;
        for (; i < 4; ++i) t.ne[i] = 1;
        t.nb[0] = ggml_type_size(type);
        for (int d = 1; d < 4; ++d) t.nb[d] = t.nb[d - 1] * (size_t) t.ne[d - 1];
        bytes.assign(t.nb[3] * (size_t) t.ne[3], 0);
        t.data = bytes.data();
    }
    size_t count() const { return (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]); }
    float  get(size_t i) const {
        if (t.type == GGML_TYPE_F16) { ggml_fp16_t h; std::memcpy(&h, bytes.data() + 2 * i, 2); return ggml_fp16_to_fp32(h); }
        float v; std::memcpy(&v, bytes.data() + 4 * i, 4); return v;
    }
    void set(size_t i, float v) {
        if (t.type == GGML_TYPE_F16) { const ggml_fp16_t h = ggml_fp32_to_fp16(v); std::memcpy(bytes.data() + 2 * i, &h, 2); }
        else std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
};

struct att_spec {
    const char * name;
    att::role    r;
    int64_t      D, KV, T, HK, HQ;
    uint32_t     seed;
    bool         boundary;
    int64_t      batches = 1;
};

void transfer_boundaries() {
    host_tensor keys, queries;
    keys.shape(GGML_TYPE_F32, {1, 1, 1, 1});
    queries.shape(GGML_TYPE_F32, {1, 129, 2, 1});
    keys.set(0, 1);
    for (size_t i = 0; i < queries.count(); ++i) queries.set(i, 2);
    queries.set(128, 65536);
    queries.set(257, 131072);
    std::vector<int> shifts;
    verdict("TRANSFER", "prefix-excludes-tail",
            att::prepare_qk_transfer(keys.t, queries.t, 0, 0, shifts) == POTAL_OK && shifts == std::vector<int>{-1},
            "rows >= 128 do not select the channel exponent");

    att::stationary weight;
    weight.n = 777;
    keys.set(0, std::numeric_limits<float>::denorm_min());
    shifts = {16};
    verdict("TRANSFER", "underflow-rejected",
            att::prepare_stationary(keys.t, 0, 0, att::role::qk, weight, &shifts) == POTAL_ERR_CAPACITY && weight.n == 777,
            "lossy scaling must not publish prepared weights");
    shifts = {std::numeric_limits<int>::min()};
    verdict("TRANSFER", "invalid-shift-rejected",
            att::prepare_stationary(keys.t, 0, 0, att::role::qk, weight, &shifts) == POTAL_ERR_INVALID_ARGUMENT,
            "validate before negating an externally supplied shift");

    att::stream activation;
    activation.rows = 777;
    queries.set(0, std::numeric_limits<float>::max());
    shifts = {16};
    verdict("TRANSFER", "overflow-rejected",
            att::prepare_stream(queries.t, 0, 0, 0, 1, att::role::qk, activation,
                                att::default_pv_algorithm, &shifts) == POTAL_ERR_CAPACITY && activation.rows == 777,
            "lossy scaling must not publish prepared queries");
}

void attention_case(const att_spec & c) {
    const bool qk = c.r == att::role::qk;
    const char * policy = std::getenv("POTAL_QK_POLICY");
    const bool residual = qk && policy && std::strcmp(policy, "legacy") != 0;
    const bool transfer = residual && std::strcmp(policy, "residual-transfer") == 0;
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    host_tensor a, b, out;
    if (qk) {
        a.shape(GGML_TYPE_F16, { c.D, c.KV, c.HK, c.batches });
        b.shape(GGML_TYPE_F32, { c.D, c.T, c.HQ, c.batches });
        for (size_t i = 0; i < a.count(); ++i) a.set(i, nd(rng) * 2.0f);
        for (size_t i = 0; i < b.count(); ++i) b.set(i, nd(rng) * 1.5f);
        if (c.boundary) {
            for (size_t i = 0; i < a.count(); i += 9) a.set(i, (i % 2 ? -1.0f : 1.0f) * 1.99f); // K code 127 -> 119
            for (size_t row = 0; row < (size_t) (c.T * c.HQ * c.batches); row += 3)                     // Q outlier block
                for (int64_t k = 32; k < 64 && k < c.D; ++k) b.set(row * c.D + k, nd(rng) * 6.0e4f);
        }
        out.shape(GGML_TYPE_F32, { c.KV, c.T, c.HQ, c.batches });
    } else {
        a.shape(GGML_TYPE_F16, { c.KV, c.D, c.HK, c.batches });
        b.shape(GGML_TYPE_F32, { c.KV, c.T, c.HQ, c.batches });
        for (size_t i = 0; i < a.count(); ++i) a.set(i, nd(rng) * 0.5f * ((i / 32) % 3 == 2 ? 300.0f : 1.0f));
        for (size_t row = 0; row < (size_t) (c.T * c.HQ * c.batches); ++row) {
            std::vector<double> e(c.KV);
            double sum = 0;
            for (int64_t k = 0; k < c.KV; ++k) sum += e[k] = std::exp(2.0 * nd(rng));
            for (int64_t k = 0; k < c.KV; ++k) b.set(row * c.KV + k, (float) (e[k] / sum));
            if (c.boundary && row % 4 == 1)
                for (int64_t k = 0; k < c.KV; ++k) b.set(row * c.KV + k, k == (int64_t) (row % c.KV) ? 1.0f : 0.0f);
        }
        out.shape(GGML_TYPE_F32, { c.D, c.T, c.HQ, c.batches });
    }
    out.t.op     = GGML_OP_MUL_MAT;
    out.t.src[0] = &a.t;
    out.t.src[1] = &b.t;
    ggml_gemmini_set_attention_role(&out.t, qk ? GGML_GEMMINI_ATTENTION_QK : GGML_GEMMINI_ATTENTION_PV);

    const size_t rows = (size_t) c.T, n = (size_t) a.t.ne[1], K = (size_t) a.t.ne[0];
    const size_t hr = (size_t) (c.HQ / c.HK);
    potal_tile_plan plan;
    if (potal_tile_select(rows, n, (K + 31) / 32 * 32, att::block_k / POTAL_HW_DIM, &plan) != POTAL_OK) {
        verdict(qk ? "QK" : "PV", c.name, false, "reason=tile_select");
        return;
    }
    const size_t stripe_rows = plan.tile_I * POTAL_HW_DIM;
    std::vector<float> want(out.count(), 0.0f);
    size_t stripes = 0, int_mis = 0, scale_mis = 0, prep_fail = 0, max_lanes = 0;
    size_t base_mis = 0;
    int    fork_scale = 0, golden_scale = 0;
    std::string why;
    for (size_t batch = 0; batch < size_t(c.batches); ++batch) {
    for (size_t ah = 0; ah < (size_t) c.HK; ++ah) {
        std::vector<int> shifts(K, 0);
        if (transfer) {
            for (size_t k = 0; k < K; ++k) {
                float maximum = 0;
                for (size_t head = ah * hr; head < (ah + 1) * hr; ++head)
                    for (size_t row = 0; row < std::min<size_t>(128, rows); ++row)
                        maximum = std::max(maximum, std::fabs(b.get(((batch * c.HQ + head) * rows + row) * K + k)));
                shifts[k] = maximum == 0 ? 0 : -std::clamp(std::ilogb(maximum), -16, 16);
            }
            std::vector<int> actual;
            verdict("TRANSFER", c.name,
                    att::prepare_qk_transfer(a.t, b.t, ah, batch, actual) == POTAL_OK && actual == shifts,
                    "independent channel maxima across grouped query heads");
        }
        gold::att_stationary gs;
        if (!gold::att_prepare_stationary([&](size_t kk, size_t j) { return std::ldexp(a.get(((batch * c.HK + ah) * n + j) * K + kk), -shifts[kk]); }, K, n, qk, gs, why)) {
            verdict(qk ? "QK" : "PV", c.name, false, "reason=golden_" + why);
            return;
        }
        att::stationary fw;
        const bool fw_ok = att::prepare_stationary(a.t, ah, batch, c.r, fw, residual ? &shifts : nullptr) == POTAL_OK;
        for (size_t head = ah * hr; head < (ah + 1) * hr; ++head)
            for (size_t begin = 0; begin < rows; begin += stripe_rows) {
                ++stripes;
                const size_t r = std::min(stripe_rows, rows - begin);
                gold::att_stream     st;
                std::vector<int64_t> gi;
                std::vector<float>   gf;
                if (!gold::att_prepare_stream([&](size_t i, size_t kk) { return std::ldexp(b.get(((batch * c.HQ + head) * rows + begin + i) * K + kk), shifts[kk]); },
                                              r, K, qk, anchor, st, why, residual) ||
                    !gold::att_multiply(st, gs, gi, gf, why)) {
                    verdict(qk ? "QK" : "PV", c.name, false, "reason=golden_" + why);
                    return;
                }
                max_lanes = std::max(max_lanes, st.upper.size());
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < n; ++j) want[((batch * c.HQ + head) * rows + begin + i) * n + j] = gf[i * n + j];
                att::stream          fs;
                std::vector<int64_t> fi;
                if (!fw_ok || att::prepare_stream(b.t, head, batch, begin, r, c.r, fs,
                                                 att::default_pv_algorithm, residual ? &shifts : nullptr) != POTAL_OK ||
                    att::reference(fs, fw, c.r, fi) != POTAL_OK || fi.size() != gi.size()) {
                    ++prep_fail;
                    continue;
                }
                if (fs.scale != st.scale && scale_mis == 0) { fork_scale = fs.scale; golden_scale = st.scale; }
                scale_mis += fs.scale != st.scale;
                for (size_t j = 0; j < n; ++j) base_mis += fw.base[j] != gs.base[j];
                for (size_t i = 0; i < gi.size(); ++i) int_mis += fi[i] != gi[i];
            }
    }
    }
    // attention::execute takes the scalar-Index paired runner (contract section 5), on the H2 emulator.
    potal_test::h2_emulator h2;
    issue_observer observer{h2.sink(), qk ? POTAL_ISSUE_ATTN_QK : POTAL_ISSUE_ATTN_PV};
    potal_issue_counters issue_before;
    potal_issue_counters_read(&issue_before);
    uint32_t                generation = 0;
    const paired_runner     run = [&](const potal_paired_request & req, potal_paired_report & rep) {
        potal_paired_request q = req;
        q.generation           = &generation;
        const potal_paired_sink sink = observer.sink();
        return potal_paired_run_session(q, &sink, rep);
    };
    const auto res = att::execute(&out.t, run);
    if (qk) observer.expected.w4_attn_qk_digit_jobs = res.jobs;
    else observer.expected.w4_attn_pv_digit_jobs = res.jobs;
    const bool counters_ok = observer.check(c.name, issue_before);
    size_t diff = 0;
    double max_abs = 0;
    const float * got = reinterpret_cast<const float *>(out.bytes.data());
    for (size_t i = 0; i < want.size(); ++i) {
        diff += !same_bits(got[i], want[i]);
        max_abs = std::max(max_abs, std::fabs((double) got[i] - (double) want[i]));
    }
    const bool ok = counters_ok && res.status == POTAL_OK && !res.fatal && !prep_fail && !int_mis && !scale_mis && !base_mis && !diff;
    char b2[512];
    std::snprintf(b2, sizeof(b2),
                  "D=%" PRId64 " KV=%" PRId64 " T=%" PRId64 " heads=%" PRId64 "/%" PRId64 " stripes=%zu stripe_rows=%zu "
                  "upper_lanes=%zu execute=%d stage=%s jobs=%" PRIu64 " prepare_fail=%zu int64_mismatch=%zu "
                  "scale_mismatch=%zu(first fork=%d golden=%d) base_mismatch=%zu f32_differing=%zu/%zu max_abs=%.6g",
                  c.D, c.KV, c.T, c.HQ, c.HK, stripes, stripe_rows, max_lanes, (int) res.status,
                  res.stage ? res.stage : "-", res.jobs, prep_fail, int_mis, scale_mis, fork_scale, golden_scale, base_mis, diff,
                  want.size(), max_abs);
    verdict(qk ? "QK" : "PV", c.name, ok, b2);

    std::fill(out.bytes.begin(), out.bytes.end(), 0x5a);
    observer.expected = {};
    potal_issue_counters_read(&issue_before);
    potal_test::thread_owner owner({}, run);
    const auto strict = att::execute_strict(&out.t, owner, {{-1,-1,-1,-1,-1}}, false);
    if (qk) observer.expected.w4_attn_qk_digit_jobs = strict.jobs;
    else observer.expected.w4_attn_pv_digit_jobs = strict.jobs;
    bool strict_ok = strict.status == POTAL_OK && !strict.fatal &&
        strict.loops == res.loops && strict.fences == res.fences &&
        strict.scale_writes == res.scale_writes && strict.scale_releases == res.scale_releases &&
        observer.check(c.name, issue_before);
    for (size_t i = 0; i < want.size(); ++i) strict_ok = strict_ok && same_bits(out.get(i), want[i]);
    verdict("STRICT", c.name, strict_ok, "independent golden, GQA/batch/stripe reuse and unchanged commands");

    if (c.T == 1) {
        std::fill(out.bytes.begin(), out.bytes.end(), 0x5a);
        const auto untouched = out.bytes;
        h2.fail_funct = POTAL_FUNCT_FENCE;
        h2.fail_status = POTAL_ERR_BAD_RESPONSE;
        potal_test::thread_owner failing({}, run);
        const auto failure = att::execute_strict(&out.t, failing, {{-1,-1,-1,-1,-1}}, false);
        verdict("FENCE", c.name, failure.status != POTAL_OK && failure.fatal && out.bytes == untouched,
                "failed hardware fence preserves public output");
    }
}

void test_main_merge() {
    namespace grmd = ggml::gemmini::rmd;
    namespace exsia_ns = ggml::gemmini::quants::act::exsia;
    const auto check = [](const std::string & name, bool ok) { verdict("MERGE", name, ok, ""); };

    constexpr size_t cols = 7, rows = 3, ldc = 11, stride = 19;
    hp1_weight weight;
    if (!weight.allocate(cols, 32)) {
        check("fused merge weight allocation", false);
        return;
    }
    const float scales[] = {0.0f, -0.0f, 0.003f, 1e-30f, 1e30f, -0.125f, 1.0f};
    std::copy_n(scales, cols, weight.scale);
    std::vector<int32_t> main(rows * ldc, 0);
    const int32_t codes[] = {INT32_MIN, INT32_MAX, -16777217, 16777217, -1, 0, 1};
    grmd::BlockScaledInt64Correction corr;
    corr.values.resize(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            main[r * ldc + c] = codes[(r + c) % cols];
            corr.values[r * cols + c] = -int64_t(main[r * ldc + c]) + int64_t(c % 3);
        }
    }
    bool equal = true;
    for (int theta : {-140, -20, 0, 10, 100}) {
        const std::vector<float> untouched(5 * stride, 123.25f);
        auto reference = untouched;
        auto fused = untouched;
        exsia_ns::StripeMetadataSnapshot meta;
        meta.theta = theta;
        const double scale = std::ldexp(1.0f, theta);
        for (size_t r = 0; r < rows; ++r) {
            reconstruct_row(main.data() + r * ldc, cols, scale, weight.scale,
                            reference.data() + (r + 1) * stride, 2);
        }
        size_t want_count = SIZE_MAX, got_count = SIZE_MAX;
        const auto want = residual::merge_prepared_hp1(weight, 1, rows + 1, meta, corr,
            reference.data(), 5, stride, 2, &want_count);
        const auto got = residual::merge_main_residual_hp1(weight, 1, rows + 1, meta, corr,
            {main.data(), ldc, scale}, fused.data(), 5, stride, 2, &got_count);
        equal = equal && want == got && want_count == got_count &&
            std::memcmp(fused.data(), got == grmd::RmdStatus::success ? reference.data() : untouched.data(),
                        fused.size() * sizeof(float)) == 0;
    }
    check("fused Main/residual merge matches separate float rounding, strides and overflow", equal);

    exsia_ns::StripeMetadataSnapshot meta;
    meta.theta = 100;
    std::fill_n(weight.scale, cols, 1.0f);
    std::fill(main.begin(), main.end(), 0);
    std::fill(corr.values.begin(), corr.values.end(), 1);
    corr.values.back() = INT64_MAX;
    const std::vector<float> untouched(5 * stride, 123.25f);
    auto out = untouched;
    size_t count = SIZE_MAX;
    const auto late = residual::merge_main_residual_hp1(weight, 1, rows + 1, meta, corr,
        {main.data(), ldc, std::ldexp(1.0f, 100)}, out.data(), 5, stride, 2, &count);
    check("fused merge late overflow leaves output and count untouched",
          late == grmd::RmdStatus::overflow && out == untouched && count == SIZE_MAX);
    const auto bad_stride = residual::merge_main_residual_hp1(weight, 1, rows + 1, meta, corr,
        {main.data(), cols - 1, 1.0}, out.data(), 5, stride, 2, &count);
    check("fused merge rejects short Main stride before reading",
          bad_stride == grmd::RmdStatus::invalid_arguments && out == untouched && count == SIZE_MAX);
}

void test_upper_decode() {
    namespace rmd = ggml::gemmini::rmd;
    bool equal = true;
    const int32_t edges[] = {INT32_MIN, INT32_MAX, -129, -128, -9, -8, -1, 0, 1, 7, 8, 127, 128};
    for (auto role : {att::role::qk, att::role::pv})
    for (bool zero : {false, true})
    for (size_t rows : {1u, 3u, 33u})
    for (size_t k : {1u, 31u, 32u, 33u, 257u}) {
        att::stream a;
        a.rows = rows + 1; a.k = k; a.kp = (k + 31) / 32 * 32; a.limb_count = 1;
        a.codes.assign(a.rows * a.kp, 0);
        a.limbs[0].assign(a.codes.size(), 0);
        for (size_t row = 0; row < a.rows; ++row) for (size_t col = 0; col < k; ++col) {
            const size_t index = row * a.kp + col;
            const int32_t value = zero ? 0 : role == att::role::qk ? edges[(row * k + col) % 13] : (row * k + col) % 256;
            a.codes[index] = value;
            a.limbs[0][index] = int8_t(ggml::gemmini::lowdigit(uint64_t(int64_t(value) - (role == att::role::pv ? 8 : 0)), 4));
        }
        std::vector<uint64_t> selected((rows * a.kp + 63) / 64, UINT64_MAX);
        rmd::RmdBitmapBuilder builder;
        builder.reset(0, 0, rows, k, 37, 4, selected, a.kp);
        for (size_t row = 0; row < rows; ++row) for (size_t col = 0; col < k; ++col) {
            const size_t index = (row + 1) * a.kp + col;
            const int32_t upper = int32_t((int64_t(a.codes[index]) - a.limbs[0][index] -
                                          (role == att::role::pv ? 8 : 0)) / 16);
            equal = builder.emit(row, col, upper) && equal;
        }
        residual::decoded_packet reference;
        if (!builder.empty()) {
            auto packet = builder.finish();
            equal = packet && residual::decode_packet(*packet, reference) == rmd::RmdStatus::success && equal;
        }
        const auto got = att::detail::decode_upper(a, 1, rows, role);
        equal = equal && got->k_compact == reference.k_compact && got->digits == reference.digits &&
                got->rows.size() == reference.rows.size() && got->runs.size() == reference.runs.size();
        if (!equal) break;
        for (size_t i = 0; i < got->rows.size(); ++i)
            equal = equal && got->rows[i].lane == reference.rows[i].lane &&
                    got->rows[i].source_row == reference.rows[i].source_row;
        for (size_t i = 0; i < got->runs.size(); ++i)
            equal = equal && got->runs[i].block_id == reference.runs[i].block_id &&
                    got->runs[i].local_k == reference.runs[i].local_k;
    }
    verdict("UPPER", "direct residual rows match packet decode including INT32 boundaries", equal, "");
}

// One case in a child process: a crash in the lowering under test is a FAIL of that case only.
const char * only_case = nullptr; // argv[1]: run just this case, in-process (for a debugger)

template <class Fn> void isolated(const char * group, const char * name, Fn fn) {
    if (only_case != nullptr) {
        if (std::strcmp(only_case, name) == 0) fn();
        return;
    }
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        cases = failed = 0;
        fn();
        std::fflush(stdout);
        _exit(cases << 4 | failed);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
        char b[96];
        std::snprintf(b, sizeof(b), "reason=child_%s signal=%d", pid < 0 ? "fork_failed" : "crashed",
                      pid > 0 && WIFSIGNALED(status) ? WTERMSIG(status) : 0);
        verdict(group, name, false, b);
        return;
    }
    cases += WEXITSTATUS(status) >> 4;
    failed += WEXITSTATUS(status) & 15;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    // Initialize ggml's FP16 lookup table before constructing the host tensors.
    ggml_context * context = ggml_init({4096, nullptr, true});
    if (context == nullptr) return 1;
    ggml_free(context);
    only_case = argc > 1 ? argv[1] : nullptr;
    std::printf("W4A4_CONFIG activation_bits=%d weight_bits=%d dim=%d anchor=%d sigma=%d rmd=%d exsia=%d "
#if defined(POTAL_INPUT_BITS)
                "potal_input_bits=%d "
#endif
                "params_sha256=%s\n",
                GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_WEIGHT_BITS, (int) POTAL_HW_DIM, anchor,
                (int) GGML_GEMMINI_EXSIA_SIGMA, (int) options::compensation,
                (int) (options::activation_quant == options::ActivationQuantAlgo::EXSIA),
#if defined(POTAL_INPUT_BITS)
                (int) POTAL_INPUT_BITS,
#endif
                potal_hw_params_sha256());
    const bool w4 = GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4 && POTAL_HW_DIM == 32 &&
                    anchor == 3 && GGML_GEMMINI_EXSIA_SIGMA == 2 && options::compensation &&
                    options::activation_quant == options::ActivationQuantAlgo::EXSIA;
    if (!w4) {
        std::printf("W4A4_GOLDEN FAIL reason=not_a_w4a4_dim32_top3_exsia_rmd_build\n");
        return 1;
    }
    const dense_spec dense_cases[] = {
        { "dense-random", 33, 40, 160, 11, false },
        { "dense-boundary", 37, 45, 192, 12, true },
        { "dense-qwen-k896", 32, 64, 896, 13, true },
    };
    for (const auto & c : dense_cases) isolated("D", c.name, [&] { dense_case(c); });
    isolated("MERGE", "main-residual", [] { test_main_merge(); });
    isolated("UPPER", "direct-upper", [] { test_upper_decode(); });
    isolated("TRANSFER", "transfer-boundaries", [] { transfer_boundaries(); });
    const att_spec att_cases[] = {
        { "qk-random", att::role::qk, 64, 5, 3, 2, 4, 21, false },
        { "qk-boundary-tails", att::role::qk, 64, 37, 33, 2, 14, 22, true },
        { "pv-random", att::role::pv, 64, 5, 3, 2, 4, 23, false },
        { "qk-decode-batches", att::role::qk, 64, 37, 1, 2, 4, 25, true, 2 },
        { "qk-prefix-tail", att::role::qk, 33, 3, 129, 1, 2, 27, false },
        { "pv-decode-batches", att::role::pv, 64, 37, 1, 2, 4, 26, true, 2 },
        { "pv-boundary-tails", att::role::pv, 64, 70, 33, 2, 14, 24, true },
    };
    for (const auto & c : att_cases) isolated(c.r == att::role::qk ? "QK" : "PV", c.name, [&] { attention_case(c); });
    std::printf(failed ? "W4A4_GOLDEN FAIL cases=%d failed=%d\n" : "W4A4_GOLDEN PASS cases=%d\n", cases, failed);
    return failed ? 1 : 0;
}
