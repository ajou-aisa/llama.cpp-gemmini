#pragma once

#include "common.h"
#include "../eval/evaluation-metal.hpp"

struct perplexity_metal_guard {
    evaluation_metal state;
    ggml_backend_sched_eval_callback previous = nullptr;
    void * previous_data = nullptr;
    bool active = false;
    bool warmup = false;
    bool callback_stopped = false;
    int evaluated_tokens = 0;

    void install(common_params & params) {
        if (params.n_gpu_layers == 0) return;
        state.initialize();
        state.begin();
        previous = params.cb_eval;
        previous_data = params.cb_eval_user_data;
        warmup = params.warmup;
        params.cb_eval = observe;
        params.cb_eval_user_data = this;
        active = true;
    }

    static bool observe(ggml_tensor * tensor, bool ask, void * user) {
        auto & guard = *static_cast<perplexity_metal_guard *>(user);
        evaluation_metal::observe(tensor, ask, &guard.state);
        const bool proceed = guard.previous ? guard.previous(tensor, ask, guard.previous_data) : !ask;
        guard.callback_stopped = guard.callback_stopped || (!ask && !proceed);
        return proceed;
    }

    bool finish(bool complete) const {
        if (!active) return true;
        auto proof = state.evidence(complete && !callback_stopped);
        proof["source_role"] = "metal_quantized_ppl";
        proof["coverage_includes_warmup"] = warmup;
        proof["previous_callback_chained"] = previous != nullptr;
        proof["ppl_evaluated_tokens"] = evaluated_tokens;
        std::fprintf(stderr, "METAL_QUANTIZED_PROOF %s\n", proof.dump().c_str());
        return proof.at("placement_verified").get<bool>();
    }
};
