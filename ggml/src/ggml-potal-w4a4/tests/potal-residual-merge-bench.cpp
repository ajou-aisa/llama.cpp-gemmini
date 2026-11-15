// Benchmark (host and target, not a test): one stripe's residual merge with
// the prepared HP1 merge (residual::merge_prepared_hp1) against the Gemmini
// checked merge the pipeline ran before it (support/gemmini-merge.hpp), on the
// same real correction: ExSIA (RMD ON) quantizes the activation rows, the
// first stripe's packet goes through build_request, the Residual GEMM on the
// RTL-semantics emulator (run_planned_stripe_gemm) and the radix
// reconstruction. Only the merges are timed (median and mean of --reps, each
// on a fresh copy of the same output); the Gemmini reader's storage
// validations per merge come from its GGML_GEMMINI_TESTING counters; both
// merges must give the same bits. Host CPU times, not FireSim/NPU numbers.
//
//   potal-residual-merge-bench [--reps R] [--rows M] [NxK ...]
//   default shapes 320x128 2304x768 50257x768, M = 80 (one GPT-2 stripe)

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "log.hpp"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/stripe-api.hpp"
#include "matmul/tiling.hpp"
#include "potal_hw_params.h"
#include "residual/stripe-residual.hpp"
#include "support/gemmini-merge.hpp"
#include "support/stripe-owner.hpp"

#include "quants/act/exsia/types.hpp"
#include "quants/common/weight_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace ggml::potal;
namespace grmd     = ggml::gemmini::rmd;
namespace exsia_ns = ggml::gemmini::quants::act::exsia;
namespace wreader  = ggml::gemmini::quants::wreader;

namespace {

constexpr size_t D = POTAL_HW_DIM;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct stats {
    double median = 0, mean = 0;
};

stats summarize(std::vector<double> v) {
    stats s;
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    for (double x : v) {
        s.mean += x;
    }
    s.mean /= (double) v.size();
    return s;
}

// One shape: false when no correction could be prepared (the line says why).
bool run_shape(size_t N, size_t K, size_t M, int reps) {
    const auto fail = [&](const char * why) {
        std::printf("shape=%zux%zu rows=%zu NOT MEASURED: %s\n", N, K, M, why);
        return false;
    };
    ggml_init_params ip = { (M * K + N * K) * 8 + M * N * 4 + 16 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) {
        return fail("no memory for the tensors");
    }
    ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, (int64_t) K, (int64_t) N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (int64_t) K, (int64_t) M);
    ggml_tensor * y = ggml_mul_mat(ctx.get(), w, x);
    {
        std::mt19937                    rng((uint32_t) (N * 31 + K));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              wf(N * K);
        const float                     mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                wf[n * K + k] = nd(rng) * mag[(k / 32 + n) % 4];
            }
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, (int64_t) N, (int64_t) K, nullptr);
        float * v = static_cast<float *>(x->data);
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                v[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
            }
            if (m % 2 == 0) {
                v[m * K + (m * 7) % K] = 400.0f * (m % 4 ? -1.0f : 1.0f); // outliers: residual packets
            }
        }
    }
    const hp1_shape     s = { (int64_t) M, (int64_t) N, (int64_t) K };
    potal_tile_plan     plan;
    ggml_gemmini_args_t args;
    hp1_weight          weight;
    hp1_operands        ops;
    if (potal_tile_select(M, N, K, 32 / D, &plan) != POTAL_OK ||
        !init_args(w, y, s, "gpt2", "merge-bench", ggml::gemmini::residual::ResidualRoute::ws_packet, plan, args) ||
        !weight.allocate((int64_t) N, (int64_t) K) ||
        prepare_weight(args, weight) != quants::weight::hp1::status::ok || !ops.allocate(s, weight) ||
        !quantize_activations(x, args, ops)) {
        return fail("weight preparation or ExSIA quantization failed");
    }
    const auto * meta = std::get_if<exsia_ns::Meta>(&args.act_quant.storage());
    if (meta == nullptr || meta->rmd_packets.empty() || meta->rmd_packets.front() == nullptr) {
        return fail("no residual packet (needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA, GGML_GEMMINI_ENABLE_RMD=ON)");
    }
    const grmd::StripePacket & packet = *meta->rmd_packets.front();
    const size_t               row_begin = packet.row_begin, row_end = packet.row_begin + packet.row_count;
    exsia_ns::StripeMetadataSnapshot snap;
    snap.theta = meta->theta.at(packet.stripe_id);

    // The stripe's real correction: production request, emulated Residual GEMM, reconstruction.
    residual::request       req;
    potal_test::emu_session es;
    if (residual::build_request(packet, weight, req) != grmd::RmdStatus::success) {
        return fail("build_request failed");
    }
    potal_stripe_gemm_desc d = {};
    d.work_kind              = POTAL_STRIPE_WORK_RESIDUAL;
    d.m                      = (uint32_t) req.m;
    d.n                      = (uint32_t) req.n;
    d.k                      = (uint32_t) req.k_execution;
    d.a                      = req.a;
    d.b                      = req.b;
    d.c                      = req.c;
    d.lda                    = req.lda;
    d.ldb                    = req.ldb;
    d.ldc                    = req.ldc;
    d.carriers               = req.carriers;
    d.carrier_stride         = req.cstride;
    d.scale_block_k          = 32;
    d.tile_i                 = (uint32_t) req.plan.tile_I;
    d.tile_j                 = (uint32_t) req.plan.tile_J;
    d.tile_k                 = (uint32_t) req.plan.tile_K;
    grmd::BlockScaledInt64Correction correction;
    if (!es.run(d).ok || residual::reconstruct(packet, req, correction) != grmd::RmdStatus::success) {
        return fail("Residual GEMM or reconstruction failed");
    }
    size_t nonzero_values = 0;
    for (int64_t v : correction.values) {
        nonzero_values += v != 0;
    }

    std::vector<float> base(M * N);
    for (size_t i = 0; i < base.size(); ++i) {
        base[i] = 0.25f * (float) ((int) (i % 29) - 14);
    }
    ggml_gemmini_args_t view;

    // Bits and reader validations of one merge each.
    std::vector<float> before = base, after = base;
    potal_test::init_merge_args(args, before.data(), N, view);
    size_t                           nz_before = 0, nz_after = 0;
    grmd::BlockScaledInt64Correction copy      = correction;
    wreader::test_reset_weight_reader_counters();
    const grmd::RmdStatus st_before = potal_test::gemmini_merge(view, row_begin, row_end, meta->run_id.value_or(0),
                                                                snap, std::move(copy), &nz_before);
    const size_t validations_before = wreader::test_weight_reader_storage_validations();
    wreader::test_reset_weight_reader_counters();
    const grmd::RmdStatus st_after = residual::merge_prepared_hp1(weight, row_begin, row_end, snap, correction,
                                                                  after.data(), M, N, 1, &nz_after);
    const size_t validations_after = wreader::test_weight_reader_storage_validations();
    const bool   bitwise = st_before == grmd::RmdStatus::success && st_after == st_before && nz_after == nz_before &&
                         std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0;

    // Timing: each repetition merges into a fresh copy of the same output.
    std::vector<double> t_before, t_after;
    std::vector<float>  out;
    for (int r = 0; r < reps; ++r) {
        out  = base;
        copy = correction;
        potal_test::init_merge_args(args, out.data(), N, view);
        size_t     nz = 0;
        const auto t0 = std::chrono::steady_clock::now();
        (void) potal_test::gemmini_merge(view, row_begin, row_end, meta->run_id.value_or(0), snap, std::move(copy),
                                         &nz);
        t_before.push_back(ms_since(t0));
        out           = base;
        const auto t1 = std::chrono::steady_clock::now();
        (void) residual::merge_prepared_hp1(weight, row_begin, row_end, snap, correction, out.data(), M, N, 1, &nz);
        t_after.push_back(ms_since(t1));
    }
    const stats  b             = summarize(t_before);
    const stats  a             = summarize(t_after);
    const size_t weight_bytes  = ggml_row_size(GGML_TYPE_Q8_HP1, (int64_t) K) * N; // the GGUF blocks
    std::printf("shape=%zux%zu stripe=%zu rows=%zu..%zu packet_blocks=%zu residual_lane_rows=%zu "
                "correction_nonzero=%zu reps=%d merge_before_ms=%.3f (mean %.3f) merge_after_ms=%.3f (mean %.3f) "
                "speedup=%.1fx weight_validation_calls_before=%zu weight_validation_calls_after=%zu "
                "weight_bytes_revalidated_before=%zu weight_bytes_revalidated_after=%zu bitwise=%s\n",
                N, K, packet.stripe_id, row_begin, row_end, packet.blocks.size(), req.m, nonzero_values, reps,
                b.median, b.mean, a.median, a.mean, a.median > 0 ? b.median / a.median : 0.0, validations_before,
                validations_after, validations_before * weight_bytes, validations_after * weight_bytes,
                bitwise ? "yes" : "NO");
    return bitwise;
}

} // namespace

int main(int argc, char ** argv) {
    int                                    reps = 20;
    size_t                                 rows = 80;
    std::vector<std::pair<size_t, size_t>> shapes;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) {
            reps = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--rows" && i + 1 < argc) {
            rows = (size_t) std::max(1, std::atoi(argv[++i]));
        } else {
            size_t n = 0, k = 0;
            if (std::sscanf(a.c_str(), "%zux%zu", &n, &k) != 2 || n == 0 || k == 0 || k % 32 != 0) {
                std::fprintf(stderr, "usage: %s [--reps R] [--rows M] [NxK ...] (K a multiple of 32)\n", argv[0]);
                return 2;
            }
            shapes.push_back({ n, k });
        }
    }
    if (shapes.empty()) {
        shapes = { { 320, 128 }, { 2304, 768 }, { 50257, 768 } };
    }
    ggml::potal::log::ensure_outputs(); // Gemmini debug/cycle JSONL, not stdout
    std::printf("potal-residual-merge-bench DIM=%u activation=%s compensation=%s (host CPU times, not NPU cycles)\n",
                (unsigned) POTAL_HW_DIM, options::activation_quant_name(), options::compensation_name);
    bool all = true;
    for (const auto & shape : shapes) {
        all = run_shape(shape.first, shape.second, rows, reps) && all;
    }
    return all ? 0 : 1;
}
