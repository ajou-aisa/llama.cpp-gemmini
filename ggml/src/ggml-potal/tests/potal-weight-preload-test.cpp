#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "llama.h"
#include "matmul/dense.hpp"
#include "matmul/weight_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}

struct budget_env {
    const char * name = "GGML_POTAL_WEIGHT_CACHE_MB";
    bool present = std::getenv(name) != nullptr;
    std::string original = present ? std::getenv(name) : "";
    ~budget_env() {
        if (present) setenv(name, original.c_str(), 1);
        else unsetenv(name);
    }
};

#ifdef POTAL_W4A4
using block_type = block_q4_hp1;
constexpr ggml_type weight_type = GGML_TYPE_MXINT4;
constexpr size_t packing = 2;
#else
using block_type = block_q8_hp1;
constexpr ggml_type weight_type = GGML_TYPE_Q8_HP1;
constexpr size_t packing = 1;
#endif

struct fixture {
    // Backend-owned prepared weights must be freed before the source buffer.
    ggml_context_ptr weights;
    ggml_backend_buffer_ptr buffer;
    ggml_context_ptr graph_context;
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    ggml_backend_ptr potal{ggml_backend_potal_init()};
    ggml_backend_sched_ptr scheduler;
    ggml_tensor * w[2] = {};
    ggml_tensor * output = nullptr;

    fixture(int64_t n = 40, int64_t k = 64,
            ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        weights.reset(ggml_init({1u << 20, nullptr, true}));
        if (!weights || !cpu || !potal) throw std::runtime_error("fixture initialization failed");
        for (auto & tensor : w) tensor = ggml_new_tensor_2d(weights.get(), weight_type, k, n);
        buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()));
        if (!buffer) throw std::runtime_error("weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), usage);
        std::vector<block_type> blocks(size_t(n * k / 32));
        for (auto & b : blocks) {
            b.m = 0;
            b.channel_scale = 0.5f;
#ifdef POTAL_W4A4
            std::memset(b.qs, 0x99, sizeof(b.qs));
#else
            std::memset(b.qs, 1, sizeof(b.qs));
#endif
        }
        for (auto * tensor : w) ggml_backend_tensor_set(tensor, blocks.data(), 0, ggml_nbytes(tensor));
        ggml_backend_t backends[] = {potal.get(), cpu.get()};
        scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true));
        if (!scheduler) throw std::runtime_error("scheduler initialization failed");
    }

    ggml_cgraph * graph(unsigned index, int64_t rows) {
        graph_context.reset(ggml_init({1u << 20, nullptr, true}));
        if (!graph_context) throw std::runtime_error("graph allocation failed");
        auto * a = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, w[index]->ne[0], rows);
        ggml_set_input(a);
        output = ggml_mul_mat(graph_context.get(), w[index], a);
        ggml_set_name(output, "preload-test");
        auto * g = ggml_new_graph_custom(graph_context.get(), 32, false);
        ggml_build_forward_expand(g, output);
        return g;
    }

    bool reserve(unsigned index, int64_t rows, bool on_cpu = false) {
        auto * g = graph(index, rows);
        ggml_backend_sched_set_tensor_backend(scheduler.get(), output, on_cpu ? cpu.get() : potal.get());
        return ggml_backend_sched_reserve(scheduler.get(), g);
    }

    ggml_backend_potal_stats stats() const {
        ggml_backend_potal_stats s{};
        if (!ggml_backend_potal_get_stats(potal.get(), &s)) throw std::runtime_error("stats unavailable");
        return s;
    }
};

void test_reserve() {
    fixture f;
    const auto graph_count = ggml_backend_potal_graph_count();
    check("prefill reservation prepares assigned static weight", f.reserve(0, 8) && f.stats().weight_cache_entries == 1);
    check("decode-only weight is included", f.reserve(1, 1) && f.stats().weight_cache_entries == 2);
    const auto bytes = f.stats().weight_cache_bytes;
    check("repeated prefill reservation is idempotent", f.reserve(0, 8) && f.stats().weight_cache_bytes == bytes);
    const auto s = f.stats();
    check("preparation does not execute or count inference work",
          ggml_backend_potal_graph_count() == graph_count && s.hw_requests == 0 &&
          s.scale_writes == 0 && s.mul_mat_nodes == 0 && s.weight_packs == 0 &&
          s.weight_validations == 0 && s.weight_cache_hits == 0 && s.weight_cache_misses == 0);
    ggml_backend_potal_clear_weight_cache(f.potal.get());
    check("clear before first compute permits fresh preparation", f.reserve(0, 1) && f.stats().weight_cache_entries == 1);
    ggml_context_ptr empty(ggml_init({1u << 20, nullptr, true}));
    auto * g = ggml_new_graph_custom(empty.get(), 16, false);
    check("empty first compute needs no hardware", ggml_backend_graph_compute(f.potal.get(), g) == GGML_STATUS_SUCCESS);
    ggml_backend_potal_clear_weight_cache(f.potal.get());
    check("reserve after compute does not reopen preload", f.reserve(0, 8) && f.stats().weight_cache_entries == 0);
}

void test_selection_and_failure() {
    {
        fixture f;
        check("CPU-assigned weight is not prepared by PoTal", f.reserve(0, 8, true) && f.stats().weight_cache_entries == 0);
    }
    {
        fixture f(40, 64, GGML_BACKEND_BUFFER_USAGE_ANY);
        check("mutable compute-buffer weight is not preloaded", f.reserve(0, 8) && f.stats().weight_cache_entries == 0);
    }
    {
        fixture f;
        auto * blocks = static_cast<block_type *>(f.w[0]->data);
        blocks[0].padding[0] = 1;
        check("invalid metadata refuses reservation without publishing an entry",
              !f.reserve(0, 1) && f.stats().weight_cache_entries == 0 && f.stats().hw_requests == 0);
        check("failed preparation resets scheduler assignments",
              ggml_backend_sched_get_tensor_backend(f.scheduler.get(), f.output) == nullptr);
        blocks[0].padding[0] = 0;
        check("repaired metadata is checked again on next reservation",
              f.reserve(0, 1) && f.stats().weight_cache_entries == 1);
    }
}

void test_cache_contract() {
    using namespace ggml::potal;
    fixture f;
    ggml_gemmini_args_t args;
    check("weight-only arguments initialize", init_weight_args(f.w[0], args));
    weight_cache eager, lazy;
    check("direct static preparation succeeds", eager.prepare(f.w[0]) == GGML_STATUS_SUCCESS);
    const auto prepared = eager.get(f.w[0], args);
    const auto ordinary = lazy.get(f.w[0], args);
    bool same = prepared.weight && ordinary.weight;
    if (same) {
        const auto & a = *prepared.weight;
        const auto & b = *ordinary.weight;
        same = a.bytes() == b.bytes() && a.ldb == b.ldb && a.cstride == b.cstride &&
               std::memcmp(a.b, b.b, a.k * a.ldb / packing) == 0 &&
               std::memcmp(a.carriers, b.carriers, a.k / quants::weight::hp1::block_k * a.cstride * sizeof(uint32_t)) == 0 &&
               std::memcmp(a.scale, b.scale, a.n_pad * sizeof(float)) == 0;
    }
    check("eager and lazy prepared operands match byte-for-byte",
          same && prepared.outcome == weight_cache_outcome::hit && ordinary.outcome == weight_cache_outcome::miss);
    if (!prepared.weight) return;
    weight_cache exact(prepared.weight->bytes());
    check("explicit exact capacity permits one weight", exact.prepare(f.w[0]) == GGML_STATUS_SUCCESS);
    auto * bad = static_cast<block_type *>(f.w[1]->data);
    bad[0].padding[0] = 1;
    check("capacity is refused before unpacking an over-budget weight",
          exact.prepare(f.w[1]) == GGML_STATUS_ALLOC_FAILED && exact.snapshot().entries == 1);
    bad[0].padding[0] = 0;
    check("failed second entry preserves the first",
          exact.prepare(f.w[0]) == GGML_STATUS_SUCCESS && exact.snapshot().entries == 1);
    weight_cache off(0);
    check("explicit zero opts out of preparation", off.prepare(f.w[0]) == GGML_STATUS_SUCCESS && off.snapshot().entries == 0);
}

void test_environment() {
    setenv("GGML_POTAL_WEIGHT_CACHE_MB", "0", 1);
    {
        fixture f;
        check("environment zero disables eager preparation", f.reserve(0, 1) && f.stats().weight_cache_entries == 0);
    }
    setenv("GGML_POTAL_WEIGHT_CACHE_MB", "1", 1);
    {
        fixture f(1024, 4096);
        check("explicit insufficient budget fails initialization",
              !f.reserve(0, 1) && f.stats().weight_cache_entries == 0 &&
              f.stats().weight_cache_budget == (1u << 20));
    }
    for (const char * value : {"", "invalid"}) {
        setenv("GGML_POTAL_WEIGHT_CACHE_MB", value, 1);
        fixture f;
        check("empty or malformed budget retains the legacy default",
              f.reserve(0, 1) && f.stats().weight_cache_budget == (uint64_t(1024) << 20));
    }
    unsetenv("GGML_POTAL_WEIGHT_CACHE_MB");
}

int prepare_model(const char * path) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn = false;
    const auto start = std::chrono::steady_clock::now();
    llama_context * context = llama_init_from_model(model, cp);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ggml_backend_potal_stats s{};
    const bool ok = context && ggml_backend_potal_get_last_stats(&s) &&
                    s.weight_cache_entries > 0 && s.hw_requests == 0 && s.mul_mat_nodes == 0 &&
                    s.weight_packs == 0 && s.weight_validations == 0;
    std::printf("POTAL_MODEL_PRELOAD %s entries=%llu bytes=%llu budget=%llu context_host_s=%.6f hw_requests=%llu\n",
                ok ? "PASS" : "FAIL", (unsigned long long) s.weight_cache_entries,
                (unsigned long long) s.weight_cache_bytes, (unsigned long long) s.weight_cache_budget,
                seconds, (unsigned long long) s.hw_requests);
    if (context) llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    budget_env restore;
    if (argc == 3 && std::strcmp(argv[1], "--model") == 0) return prepare_model(argv[2]);
    if (argc != 1) return 2;
    unsetenv("GGML_POTAL_WEIGHT_CACHE_MB");
    try {
        test_reserve();
        test_selection_and_failure();
        test_cache_contract();
        test_environment();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "preload test exception: %s\n", e.what());
        ++failures;
    }
    std::printf("POTAL_WEIGHT_PRELOAD_TEST %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
