// Host benchmark of the Q8_HP1 weight preparation (validation + unpacking into
// the PoTal B / carrier / channel_scale operands). No hardware is used.
//
//   potal-weight-prep-bench [--gguf model.gguf] [--reps N] [--steps N]
//
// Per weight (synthetic, or a real tensor of the GGUF opened read-only):
//   stage_*   the work of one per-call preparation, split as far as the code
//             allows (each stage re-runs that work on its own, so the stages
//             overlap the total; see the column notes in the output)
//   prepare   one full per-call preparation: staging allocation + unpack
//   cache     the backend path (weight_cache) on a copy of the weight in a
//             GGML_BACKEND_BUFFER_USAGE_WEIGHTS CPU buffer: cold MISS, warm
//             HIT, and `--steps` decode-step lookups
// Reader call counts come from the Gemmini reader's GGML_GEMMINI_TESTING counters.

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "gguf.h"
#include "matmul/dense.hpp"
#include "matmul/weight_cache.hpp"
#include "quants/common/weight_reader.hpp"
#include "quants/common/weight_route.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ggml::potal;
namespace hp1     = ggml::potal::quants::weight::hp1;
namespace wreader = ggml::gemmini::quants::wreader;
namespace wroute  = ggml::gemmini::quants::wroute;

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

ggml_tensor * synthetic_weight(ggml_context * ctx, int64_t K, int64_t N, std::mt19937 & rng) {
    ggml_tensor *                   w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_HP1, K, N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              f(N * K);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const float mag[] = { 1.0f, 8.0f, 0.125f, 64.0f };
            f[n * K + k]      = nd(rng) * 0.02f * mag[(k / 32 + n) % 4];
        }
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, f.data(), w->data, 0, N, K, nullptr);
    return w;
}

struct bench_case {
    const char * name;
    int64_t      m, n, k;
};

// Cold / warm / repeated lookups of the backend weight cache for w.
void bench_cache(ggml_backend_t cpu, const char * source, const bench_case & c, ggml_tensor * w, int reps, int steps,
                 const ggml_tensor * dst, const potal_tile_plan & plan) {
    // Copy of w in a weights buffer, as llama.cpp loads model tensors.
    ggml_init_params        ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr        ctx(ggml_init(ip));
    ggml_tensor *           tw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, w->ne[0], w->ne[1]);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(tw, w->data, 0, ggml_nbytes(w));

    const hp1_shape     s = { c.m, c.n, c.k };
    ggml_gemmini_args_t args;
    if (!init_args(tw, const_cast<ggml_tensor *>(dst), s, "gpt2", "bench",
                   ggml::gemmini::residual::ResidualRoute::cpu_direct, plan, args)) {
        std::printf("BENCH_FAIL case=%s init_args\n", c.name);
        return;
    }
    std::vector<double> t_cold, t_warm, t_steps;
    size_t              cold_validations = 0, cold_code_reads = 0, warm_validations = 0, warm_code_reads = 0;
    bool                identical = true, all_hits = true;
    weight_cache::counters end{};
    for (int r = 0; r < reps; ++r) {
        weight_cache cache(size_t(4096) << 20);
        wreader::test_reset_weight_reader_counters();
        auto          t    = clock_type::now();
        weight_lookup cold = cache.get(tw, args);
        t_cold.push_back(ms_since(t));
        cold_validations = wreader::test_weight_reader_storage_validations();
        cold_code_reads  = wreader::test_weight_reader_code_address_resolutions();
        if (cold.outcome != weight_cache_outcome::miss) {
            std::printf("BENCH_FAIL case=%s cold=%s\n", c.name, weight_cache_outcome_name(cold.outcome));
            return;
        }
        wreader::test_reset_weight_reader_counters();
        t                  = clock_type::now();
        weight_lookup warm = cache.get(tw, args);
        t_warm.push_back(ms_since(t));
        warm_validations = wreader::test_weight_reader_storage_validations();
        warm_code_reads  = wreader::test_weight_reader_code_address_resolutions();
        all_hits         = all_hits && warm.outcome == weight_cache_outcome::hit && warm.weight == cold.weight;
        t                = clock_type::now();
        for (int i = 0; i < steps; ++i) {
            all_hits = all_hits && cache.get(tw, args).outcome == weight_cache_outcome::hit;
        }
        t_steps.push_back(ms_since(t));
        if (r == 0) {
            // Cached operands == the per-call unpacking of the same weight.
            hp1_operands o;
            hp1::stats   ws;
            identical = o.allocate(s) && unpack_weights(args, o, &ws) == hp1::status::ok &&
                        std::memcmp(o.own.b, cold.weight->b, s.k * o.ldb) == 0 &&
                        std::memcmp(o.own.carriers, cold.weight->carriers,
                                    (size_t) s.k / hp1::block_k * o.cstride * sizeof(uint32_t)) == 0 &&
                        std::memcmp(o.own.scale, cold.weight->scale, o.n_pad * sizeof(float)) == 0;
        }
        end = cache.snapshot();
    }
    std::printf("BENCH_CACHE source=%s case=%s reps=%d steps=%d cold_ms=%.3f warm_ms=%.6f steps_total_ms=%.6f "
                "cold_storage_validations=%zu cold_code_reads=%zu warm_storage_validations=%zu warm_code_reads=%zu "
                "packs_after_warm=%llu hits=%llu entries=%llu cache_bytes=%llu identical=%d all_hits=%d\n",
                source, c.name, reps, steps, median(t_cold), median(t_warm), median(t_steps), cold_validations,
                cold_code_reads, warm_validations, warm_code_reads, (unsigned long long) (end.misses + end.bypasses +
                                                                                         end.uncached),
                (unsigned long long) end.hits, (unsigned long long) end.entries, (unsigned long long) end.bytes,
                identical ? 1 : 0, all_hits ? 1 : 0);
}

void bench(const char * source, const bench_case & c, ggml_tensor * w, int reps, ggml_backend_t cpu, int steps) {
    const hp1_shape s = { c.m, c.n, c.k };
    potal_tile_plan plan;
    if (potal_tile_select((size_t) s.m, (size_t) s.n, (size_t) s.k, hp1::block_k / POTAL_HW_DIM, &plan) != POTAL_OK) {
        std::printf("BENCH_FAIL case=%s tile_select\n", c.name);
        return;
    }
    ggml_init_params ip  = { (size_t) (s.m * s.n * 4 + 4096), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, s.n, s.m);

    std::vector<double> t_alloc, t_plan, t_validate, t_carrier, t_reorder, t_prepare;
    size_t              plan_validations = 0, unpack_validations = 0, unpack_code_reads = 0;
    for (int r = 0; r < reps; ++r) {
        ggml_gemmini_args_t args;
        if (!init_args(w, dst, s, "gpt2", "bench", ggml::gemmini::residual::ResidualRoute::cpu_direct, plan, args)) {
            std::printf("BENCH_FAIL case=%s init_args\n", c.name);
            return;
        }

        // Full per-call preparation, as mul_mat_hp1 runs it per MUL_MAT.
        auto t = clock_type::now();
        {
            hp1_operands o;
            const bool   ok = o.allocate(s);
            t_alloc.push_back(ms_since(t));
            wreader::test_reset_weight_reader_counters();
            hp1::stats ws;
            if (!ok || unpack_weights(args, o, &ws) != hp1::status::ok) {
                std::printf("BENCH_FAIL case=%s unpack\n", c.name);
                return;
            }
            unpack_validations = wreader::test_weight_reader_storage_validations();
            unpack_code_reads  = wreader::test_weight_reader_code_address_resolutions();
        }
        t_prepare.push_back(ms_since(t));

        // Stages of unpack, re-run separately on the same weight.
        wreader::test_reset_weight_reader_counters();
        t                                 = clock_type::now();
        const wroute::WeightRoutePlan wplan =
            wroute::resolve_weight_route_plan(args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
        t_plan.push_back(ms_since(t));
        plan_validations = wreader::test_weight_reader_storage_validations();
        t                = clock_type::now();
        const bool valid = wreader::validate(args, wplan) == wreader::WeightReaderStatus::Success;
        t_validate.push_back(ms_since(t));
        if (!wplan.valid || !valid) {
            std::printf("BENCH_FAIL case=%s plan\n", c.name);
            return;
        }
        const size_t          nb = args.q8_hp1_blocks_per_row;
        std::vector<uint32_t> carriers(nb * (size_t) s.n);
        std::vector<int8_t>   b((size_t) s.k * (size_t) s.n);
        t = clock_type::now();
        for (size_t n = 0; n < (size_t) s.n; ++n) {
            (void) wreader::read_column_scale_validated(args, wplan, n, 0);
            for (size_t blk = 0; blk < nb; ++blk) {
                carriers[blk * (size_t) s.n + n] = wreader::read_hp1_carrier_validated(args, wplan, n, blk).carrier;
            }
        }
        t_carrier.push_back(ms_since(t));
        t = clock_type::now();
        for (size_t n = 0; n < (size_t) s.n; ++n) {
            for (size_t k = 0; k < (size_t) s.k; ++k) {
                b[k * (size_t) s.n + n] = static_cast<int8_t>(wreader::read_code_validated(args, wplan, n, k).value);
            }
        }
        t_reorder.push_back(ms_since(t));
    }

    hp1_operands probe;
    probe.allocate(s);
    const size_t blocks        = (size_t) s.n * (size_t) s.k / hp1::block_k;
    const size_t packed_bytes  = (size_t) s.k * probe.ldb;
    const size_t carrier_bytes = (size_t) s.k / hp1::block_k * probe.cstride * sizeof(uint32_t);
    const size_t scale_bytes   = probe.n_pad * sizeof(float);
    std::printf("BENCH_STAGES source=%s case=%s M=%lld N=%lld K=%lld reps=%d elements=%zu blocks=%zu "
                "weight_bytes_read=%zu packed_weight_bytes=%zu carrier_bytes=%zu channel_scale_bytes=%zu "
                "prepare_ms=%.3f alloc_zero_ms=%.3f plan_resolve_ms=%.3f validate_pass_ms=%.3f "
                "carrier_scale_extract_ms=%.3f code_reorder_ms=%.3f plan_storage_validations=%zu "
                "unpack_storage_validations=%zu unpack_code_reads=%zu\n",
                source, c.name, (long long) s.m, (long long) s.n, (long long) s.k, reps,
                (size_t) s.n * (size_t) s.k, blocks, blocks * sizeof(block_q8_hp1), packed_bytes, carrier_bytes,
                scale_bytes, median(t_prepare), median(t_alloc), median(t_plan), median(t_validate),
                median(t_carrier), median(t_reorder), plan_validations, unpack_validations, unpack_code_reads);
    bench_cache(cpu, source, c, w, reps, steps, dst, plan);
}

} // namespace

int main(int argc, char ** argv) {
    const char * gguf_path = nullptr;
    int          reps      = 5;
    int          steps     = 32;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::max(1, std::atoi(argv[++i]));
        }
    }

    const bench_case cases[] = {
        { "16x16x32", 16, 16, 32 },        { "5x40x96", 5, 40, 96 },
        { "37x100x640", 37, 100, 640 },    { "3x160x4096", 3, 160, 4096 },
        { "1x300x768", 1, 300, 768 },      { "7x2304x768", 7, 2304, 768 },
        { "1x50257x768", 1, 50257, 768 },
    };
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    std::mt19937     rng(20261018);
    for (const bench_case & c : cases) {
        ggml_init_params ip = { (size_t) (c.n * c.k * 2 + 4096), nullptr, false };
        ggml_context_ptr ctx(ggml_init(ip));
        bench("synthetic", c, synthetic_weight(ctx.get(), c.k, c.n, rng), reps, cpu.get(), steps);
    }

    if (gguf_path != nullptr) {
        // Read-only: gguf loads the tensor data into a private ggml context.
        ggml_context *   raw = nullptr;
        gguf_init_params gp  = { false, &raw };
        gguf_context *   g   = gguf_init_from_file(gguf_path, gp);
        if (g == nullptr || raw == nullptr) {
            std::printf("BENCH_FAIL gguf open %s\n", gguf_path);
            return 1;
        }
        ggml_context_ptr ctx(raw);
        const struct {
            const char * tensor;
            bench_case   c;
        } real[] = {
            { "blk.0.attn_qkv.weight", { "7x2304x768", 7, 2304, 768 } },
            { "token_embd.weight", { "1x50257x768", 1, 50257, 768 } },
        };
        for (const auto & r : real) {
            ggml_tensor * w = ggml_get_tensor(raw, r.tensor);
            if (w == nullptr || w->type != GGML_TYPE_Q8_HP1 || w->ne[0] != r.c.k || w->ne[1] != r.c.n) {
                std::printf("BENCH_FAIL gguf tensor %s\n", r.tensor);
                continue;
            }
            bench(r.tensor, r.c, w, reps, cpu.get(), steps);
        }
        gguf_free(g);
    }
    return 0;
}
