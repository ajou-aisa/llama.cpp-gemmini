#pragma once

#include "common.h"
#include "evaluation-lifecycle.h"
#include <gemmini/semantic.hpp>
#include <gemmini/cycle_sim_log.hpp>
#include <memory>
#include <optional>

class evaluation_trace {
public:
    evaluation_trace(const common_params & params, const std::string & model_identity,
                     size_t prompt_count, size_t generated_count, bool forced_cost_only = false);
    void phase(const std::string & kind, const std::vector<llama_token> & tokens,
               std::optional<uint64_t> decode_index = {});
    int decode(llama_context * ctx, llama_batch batch);
    void sample_complete(uint64_t index, llama_token token);
    void forced_complete(uint64_t index, llama_token token);
    void finish(bool success);
private:
    std::shared_ptr<ggml::gemmini::semantic::Session> semantic_;
    std::unique_ptr<evaluation_lifecycle> lifecycle_;
#if CYCLE_SIM
    std::shared_ptr<ggml::gemmini::cycle_sim::Session> target_;
    ggml::gemmini::cycle_sim::Context context_;
#endif
};
