#pragma once
#include "common.h"
#include "../../ggml/src/ggml-cuda/cpu-exact/ggml-cuda-rtn.h"
#include <cstdio>
#include <stdexcept>

struct perplexity_cuda_rtn_guard {
    ggml_backend_sched_eval_callback previous = nullptr;
    void * previous_data = nullptr;
    uint64_t observed = 0, verified = 0, before = 0, dense_start = 0, attention_start = 0;
    uint64_t mxint4 = 0, mxint8 = 0;
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
                g.mxint4 += t->src[0]->type == GGML_TYPE_MXINT4;
                g.mxint8 += t->src[0]->type == GGML_TYPE_MXINT8;
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
        int activation_bits = ggml_cuda_rtn_activation_bits();
        if (activation_bits == 16 && (mxint4 || mxint8)) {
            ok = ok && !(mxint4 && mxint8);
            activation_bits = mxint4 ? 4 : 8;
        }
        std::fprintf(stderr,"CUDA_RTN_PROOF {\"complete\":%s,\"activation_bits\":%d,\"attention_bits\":16,\"block_size\":32,\"compensation\":false,\"scored_tokens\":%d,\"observed_matmuls\":%llu,\"verified_matmuls\":%llu,\"dense_gpu_calls\":%llu,\"attention_gpu_calls\":%llu}\n",
            ok?"true":"false",activation_bits,tokens,(unsigned long long)observed,(unsigned long long)verified,(unsigned long long)dense,(unsigned long long)attention);
        if (mxint4 || mxint8)
            std::fprintf(stderr,"MXINT_WA_PROOF {\"activation\":\"MXINT_POT\",\"rounding\":\"ties_away\",\"block_size\":32,\"outlier_selection\":false,\"stripe_folding\":false,\"residual\":false,\"mxint4_matmuls\":%llu,\"mxint8_matmuls\":%llu}\n",
                         (unsigned long long)mxint4,(unsigned long long)mxint8);
        ggml_cuda_rtn_clear();
        return ok;
    }
};
