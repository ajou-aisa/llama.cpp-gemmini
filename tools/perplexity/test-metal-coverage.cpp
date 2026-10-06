#include "perplexity-metal.hpp"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

struct callback_state {
    int completed = 0;
};

bool previous_callback(ggml_tensor * tensor, bool ask, void * user) {
    const bool interested = tensor->op == GGML_OP_MUL_MAT && ggml_is_quantized(tensor->src[0]->type);
    if (ask) return interested;
    if (interested) ++static_cast<callback_state *>(user)->completed;
    return true;
}

bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

bool run_graph(ggml_backend_t metal, ggml_backend_t cpu, bool partial_cpu) {
    constexpr int64_t k = 64, n = 4, m = 2;
    const auto type = GGML_GEMMINI_WEIGHT_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
    context_ptr context(ggml_init({ggml_graph_overhead_custom(64, false) + 32 * ggml_tensor_overhead(),
                                  nullptr, true}), ggml_free);
    ggml_backend_t backends[] = {metal, cpu};
    scheduler_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 2, 64, false, false),
                            ggml_backend_sched_free);
    if (!check(context && scheduler, "allocate graph metadata and scheduler")) return false;
    auto * graph = ggml_new_graph_custom(context.get(), 64, false);
    auto * activation = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, m);
    ggml_set_name(activation, "coverage.activation");
    ggml_set_input(activation);
    ggml_backend_sched_set_tensor_backend(scheduler.get(), activation, cpu);
    ggml_tensor * weights[2];
    ggml_tensor * outputs[2];
    for (int i = 0; i < 2; ++i) {
        const auto backend = partial_cpu && i == 1 ? cpu : metal;
        weights[i] = ggml_new_tensor_2d(context.get(), type, k, n);
        ggml_set_name(weights[i], i == 0 ? "coverage.weight.0" : "coverage.weight.1");
        outputs[i] = ggml_mul_mat(context.get(), weights[i], activation);
        ggml_set_output(outputs[i]);
        ggml_backend_sched_set_tensor_backend(scheduler.get(), weights[i], backend);
        ggml_backend_sched_set_tensor_backend(scheduler.get(), outputs[i], backend);
        ggml_build_forward_expand(graph, outputs[i]);
    }
    if (!check(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "allocate graph tensors")) return false;
    std::vector<float> source(k * n), inputs(k * m);
    for (size_t i = 0; i < source.size(); ++i) source[i] = (int(i % 15) - 7) * 0.25f;
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (int(i % 9) - 4) * 0.125f;
    std::vector<uint8_t> packed(ggml_row_size(type, k) * n);
    ggml_quantize_chunk(type, source.data(), packed.data(), 0, n, k, nullptr);
    for (auto * weight : weights) ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(activation, inputs.data(), 0, inputs.size() * sizeof(float));
    callback_state previous;
    common_params params;
    params.n_gpu_layers = -1;
    params.warmup = false;
    params.cb_eval = previous_callback;
    params.cb_eval_user_data = &previous;
    perplexity_metal_guard guard;
    guard.install(params);
    ggml_backend_sched_set_eval_callback(scheduler.get(), params.cb_eval, params.cb_eval_user_data);
    const auto status = ggml_backend_sched_graph_compute(scheduler.get(), graph);
    const bool verified = guard.finish(status == GGML_STATUS_SUCCESS);
    bool ok = check(status == GGML_STATUS_SUCCESS, "real mixed-backend graph executes successfully");
    ok &= check(previous.completed == 2, "previous evaluation callback receives both completed matmuls");
    ok &= check(verified == !partial_cpu, "coverage rejects an actual CPU matmul and accepts full Metal");
    for (auto * output : outputs) {
        std::vector<float> values(n * m);
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        for (float value : values) ok &= check(std::isfinite(value), "computed output is finite");
    }
    return ok;
}

} // namespace

int main() {
    llama_backend_init();
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_ptr metal(ggml_backend_init_by_name("Metal", nullptr), ggml_backend_free);
    backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
    if (!metal || !cpu) {
        std::fprintf(stderr, "SKIP: Metal and CPU backends are required\n");
        return 77;
    }
    bool ok = run_graph(metal.get(), cpu.get(), false);
    ok &= run_graph(metal.get(), cpu.get(), true);
    common_params params;
    callback_state previous;
    params.n_gpu_layers = 0;
    params.cb_eval = previous_callback;
    params.cb_eval_user_data = &previous;
    perplexity_metal_guard guard;
    guard.install(params);
    ok &= check(!guard.active && params.cb_eval == previous_callback && params.cb_eval_user_data == &previous &&
                guard.finish(true), "intentional CPU execution preserves the existing callback");
    if (ok) std::puts("PASS: Metal PPL coverage, real CPU fallback rejection, and callback chaining");
    return ok ? 0 : 1;
}
