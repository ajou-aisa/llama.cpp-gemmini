#pragma once

#include "common.h"
#include "ggml-metal-cpu-exact.h"
#include "ggml-metal-cpu-exact-int.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

struct perplexity_metal_cpu_exact_guard {
    ggml_backend_sched_eval_callback previous = nullptr;
    void * previous_data = nullptr;
    uint64_t float_start = 0, int_start = 0, before = 0;
    uint64_t observed = 0, verified = 0;
    uint64_t observed_q6_heads = 0, cpu_q6_head_matmuls = 0;
    bool active = false, failed = false;
    bool allow_q6_head = false;

    static uint64_t calls() {
        return ggml_metal_cpu_exact_gpu_calls() + ggml_metal_cpu_exact_int_launches();
    }

    void install(common_params & params) {
        active = ggml_metal_cpu_exact_int_enabled();
        if (!active) return;
        const char * q6_head = std::getenv("GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD");
        allow_q6_head = q6_head && std::strcmp(q6_head, "1") == 0;
        if (params.n_gpu_layers != 0 || params.warmup || params.ppl_stride != 0 ||
            params.hellaswag || params.winogrande || params.multiple_choice || params.kl_divergence) {
            throw std::runtime_error("CPU-exact Metal PPL requires -ngl 0 --no-warmup and standard PPL");
        }
        previous = params.cb_eval;
        previous_data = params.cb_eval_user_data;
        params.cb_eval = observe;
        params.cb_eval_user_data = this;
        float_start = ggml_metal_cpu_exact_gpu_calls();
        int_start = ggml_metal_cpu_exact_int_launches();
    }

    static bool observe(ggml_tensor * tensor, bool ask, void * user) {
        auto & guard = *static_cast<perplexity_metal_cpu_exact_guard *>(user);
        const bool target = tensor->op == GGML_OP_MUL_MAT && tensor->src[0] &&
                            ggml_is_quantized(tensor->src[0]->type);
        const bool prior = guard.previous ? guard.previous(tensor, ask, guard.previous_data) : !ask;
        if (target) {
            const bool q6_head = guard.allow_q6_head && tensor->src[0]->type == GGML_TYPE_Q6_K &&
                std::strcmp(tensor->name, "result_output") == 0 &&
                (std::strcmp(tensor->src[0]->name, "output.weight") == 0 ||
                 std::strcmp(tensor->src[0]->name, "token_embd.weight") == 0);
            if (ask) {
                guard.before = calls();
                if (q6_head) {
                    ++guard.observed_q6_heads;
                    return true;
                }
                ++guard.observed;
                const bool hp1 = std::strcmp(EVALUATION_EXACT_COMPUTE, "INT") == 0 &&
                                 std::strcmp(EVALUATION_EXACT_ACTIVATION, "EXSIA") == 0;
                const ggml_type expected = hp1
                    ? (EVALUATION_EXACT_BITS == 4 ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1)
                    : (EVALUATION_EXACT_BITS == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
                guard.failed = guard.failed || tensor->src[0]->type != expected;
                return true;
            }
            if (q6_head) {
                if (calls() == guard.before) ++guard.cpu_q6_head_matmuls;
                else guard.failed = true;
            }
            else if (calls() > guard.before) ++guard.verified;
            else guard.failed = true;
        }
        guard.failed = guard.failed || (!ask && !prior);
        return ask ? prior : (prior && !guard.failed);
    }

    bool finish(bool complete, int tokens) const {
        if (!active) return true;
        const uint64_t floats = ggml_metal_cpu_exact_gpu_calls() - float_start;
        const uint64_t ints = ggml_metal_cpu_exact_int_launches() - int_start;
        const bool ok = complete && tokens > 0 && !failed && observed > 0 &&
                        observed == verified && floats + ints > 0 &&
                        observed_q6_heads == cpu_q6_head_matmuls &&
                        (!allow_q6_head || cpu_q6_head_matmuls > 0);
        std::fprintf(stderr,
            "METAL_CPU_EXACT_PROOF {\"schema\":\"metal-cpu-exact-ppl\",\"version\":1,"
            "\"complete\":%s,\"graph\":\"original_cpu\",\"compute\":\"%s\","
            "\"activation\":\"%s\",\"bits\":%d,\"dim\":%d,\"scored_tokens\":%d,"
            "\"observed_matmuls\":%llu,\"verified_matmuls\":%llu,"
            "\"cpu_q6_head_matmuls\":%llu,"
            "\"float_gpu_calls\":%llu,\"integer_gpu_launches\":%llu}\n",
            ok ? "true" : "false", EVALUATION_EXACT_COMPUTE, EVALUATION_EXACT_ACTIVATION,
            EVALUATION_EXACT_BITS, EVALUATION_EXACT_DIM, tokens,
            (unsigned long long) observed, (unsigned long long) verified,
            (unsigned long long) cpu_q6_head_matmuls,
            (unsigned long long) floats, (unsigned long long) ints);
        return ok;
    }
};
