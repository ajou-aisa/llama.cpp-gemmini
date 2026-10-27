#pragma once
#include "common.h"
#include "../../ggml/src/ggml-cuda/cpu-exact/ggml-cuda-rtn.h"
#include <cstdio>
#include <stdexcept>

struct perplexity_cuda_rtn_guard {
    ggml_backend_sched_eval_callback previous = nullptr;
    void * previous_data = nullptr;
    uint64_t observed = 0, verified = 0, before = 0, dense_start = 0, attention_start = 0;
    uint64_t hp1_q4 = 0, hp1_q8 = 0;
    bool active = false, failed = false;
    static uint64_t calls() { return ggml_cuda_rtn_dense_calls() + ggml_cuda_rtn_attention_calls(); }
    void install(common_params & params) {
        active = ggml_cuda_rtn_enabled();
        if (!active) return;
        if (params.n_gpu_layers != 0 || params.warmup || params.ppl_stride != 0)
            throw std::runtime_error("CUDA RTN PPL requires -ngl 0 --no-warmup and standard PPL");
        previous = params.cb_eval; previous_data = params.cb_eval_user_data;
        params.cb_eval = observe; params.cb_eval_user_data = this;
        dense_start = ggml_cuda_rtn_dense_calls(); attention_start = ggml_cuda_rtn_attention_calls();
    }
    static bool observe(ggml_tensor * t, bool ask, void * user) {
        auto & g = *static_cast<perplexity_cuda_rtn_guard *>(user);
        bool prior = g.previous ? g.previous(t, ask, g.previous_data) : !ask;
        if (t->op == GGML_OP_MUL_MAT) {
            if (ask && t->src[0]) {
                g.hp1_q4 += t->src[0]->type == GGML_TYPE_Q4_HP1;
                g.hp1_q8 += t->src[0]->type == GGML_TYPE_Q8_HP1;
            }
            if (ask) { ++g.observed; g.before = calls(); return true; }
            if (calls() == g.before + 1) ++g.verified;
            else { g.failed = true; std::fprintf(stderr,"CUDA RTN CPU fallback: %s\n",t->name); }
        }
        g.failed = g.failed || (!ask && !prior);
        return ask ? prior : prior && !g.failed;
    }
    bool finish(bool complete, int tokens) const {
        if (!active) return true;
        auto dense = ggml_cuda_rtn_dense_calls() - dense_start, attention = ggml_cuda_rtn_attention_calls() - attention_start;
        bool ok = complete && tokens > 0 && !failed && observed > 0 && observed == verified && dense > 0 && attention > 0;
        std::fprintf(stderr,"CUDA_RTN_PROOF {\"complete\":%s,\"activation_bits\":%d,\"attention_bits\":16,\"block_size\":32,\"compensation\":false,\"scored_tokens\":%d,\"observed_matmuls\":%llu,\"verified_matmuls\":%llu,\"dense_gpu_calls\":%llu,\"attention_gpu_calls\":%llu}\n",
            ok?"true":"false",ggml_cuda_rtn_activation_bits(),tokens,(unsigned long long)observed,(unsigned long long)verified,(unsigned long long)dense,(unsigned long long)attention);
        if (hp1_q4 || hp1_q8)
            std::fprintf(stderr,"RTN_WA_HP1_PROOF {\"activation\":\"BLOCK_RTN\",\"block_size\":32,\"outlier_selection\":false,\"stripe_folding\":false,\"residual\":false,\"q4_hp1_matmuls\":%llu,\"q8_hp1_matmuls\":%llu}\n",
                         (unsigned long long)hp1_q4,(unsigned long long)hp1_q8);
        ggml_cuda_rtn_clear();
        return ok;
    }
};
