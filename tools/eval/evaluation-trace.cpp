#include "evaluation-trace.h"
#include "json.hpp"
#include <gemmini/log.hpp>
#include <gemmini/log.h>
#include <stdexcept>

namespace semantic = ggml::gemmini::semantic;
namespace log = ggml::gemmini::log;

evaluation_trace::evaluation_trace(const common_params & params, const std::string & model_identity,
                                 size_t prompt_count, size_t generated_count, bool forced_cost_only) {
    if (forced_cost_only && (CYCLE_SIM || !LOG_CYCLE || !semantic::compiled_cpu_only_build() ||
                            params.n_gpu_layers != 0 || prompt_count != 256 || generated_count != 128))
        throw std::invalid_argument("forced cost-only trace requires CPU-only LOG_CYCLE build and 256+128 inputs");
#if LOG_CYCLE
    if (!log::cycle.set_output_path(GEMMINI_LOG_DEFAULT_CYCLE_PATH, true))
        throw std::runtime_error("evaluation: cannot create CPU cycle log");
    log::cycle.set_buffered(true);
#endif
#if LOG_CYCLE || CYCLE_SIM
    const auto cpu = [](const cpu_params & value) {
        std::string mask;
        for (bool bit : value.cpumask) mask += bit ? '1' : '0';
        return nlohmann::json{{"threads", value.n_threads}, {"poll", value.poll},
            {"priority", value.priority}, {"strict_cpu", value.strict_cpu},
            {"mask_valid", value.mask_valid}, {"mask", mask}};
    };
    const nlohmann::json workload = {{"model", model_identity}, {"prompt_tokens", prompt_count},
        {"generated_tokens", generated_count}, {"context_tokens", params.n_ctx},
        {"batch_tokens", params.n_batch}, {"microbatch_tokens", params.n_ubatch},
        {"cpu", cpu(params.cpuparams)}, {"cpu_batch", cpu(params.cpuparams_batch)},
        {"flash_attention", params.flash_attn}, {"kv_type_k", ggml_type_name(params.cache_type_k)},
        {"kv_type_v", ggml_type_name(params.cache_type_v)}, {"seed", params.sampling.seed},
        {"numa", params.numa}, {"build_target", LLAMA_BUILD_TARGET}};
    const bool cpu_only = semantic::compiled_cpu_only_build();
    const auto source = CYCLE_SIM ? semantic::Source::PotalCollection :
        cpu_only ? semantic::Source::FullCpu : semantic::Source::Unspecified;
    const nlohmann::json producer = {{"cycle_sim", CYCLE_SIM}, {"log_cycle", LOG_CYCLE},
        {"cpu_only_build", cpu_only}, {"git_commit", LLAMA_COMMIT}, {"warmup_excluded", true},
        {"reserve_measure_graphs_excluded", true}, {"application_runner", "native-evaluation-v3"},
        {"execution_kind", forced_cost_only ? "FORCED_CPU_COST_ONLY" : generated_count ? "FREE_GENERATION" : "METRIC_PREFILL"},
        {"trajectory_source", forced_cost_only ? "POTAL" : "SELF_SAMPLED"}};
    semantic_ = semantic::Session::start(source, workload.dump(), producer.dump(), cpu_only);
    if (generated_count && semantic_)
        lifecycle_ = std::make_unique<evaluation_lifecycle>(CYCLE_SIM ? "potal_collection" : "full_cpu",
                                                           LLAMA_COMMIT, generated_count, forced_cost_only);
#else
    (void) params; (void) model_identity; (void) prompt_count; (void) generated_count;
#endif
#if CYCLE_SIM
    namespace cycle_sim = ggml::gemmini::cycle_sim;
    if (generated_count) {
        target_ = cycle_sim::Session::start(cycle_sim::compiled_run_info(model_identity, prompt_count, generated_count));
        target_->set_policy_query({ggml_backend_dev_by_name("GEMMINI"), [](void * device, const void * node) {
            return device && ggml_backend_dev_supports_op(static_cast<ggml_backend_dev_t>(device),
                                                         static_cast<const ggml_tensor *>(node));
        }});
    }
#endif
}

void evaluation_trace::phase(const std::string & kind, const std::vector<llama_token> & tokens,
                             std::optional<uint64_t> decode_index) {
    if (semantic_) semantic_->phase(kind, decode_index, tokens.data(), tokens.size());
    if (lifecycle_) lifecycle_->phase(kind, decode_index, semantic_->completed_graph_count());
#if CYCLE_SIM
    if (target_) context_ = target_->phase(kind, decode_index, tokens.size());
#endif
}

int evaluation_trace::decode(llama_context * ctx, llama_batch batch) {
    if (lifecycle_) lifecycle_->begin_dispatch(semantic_->completed_graph_count());
#if CYCLE_SIM
    ggml::gemmini::cycle_sim::ScopedContext scope(context_);
#endif
    const int status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    if (semantic_) semantic_->ensure_healthy();
    if (lifecycle_) lifecycle_->end_dispatch(semantic_->completed_graph_count(), status);
#if CYCLE_SIM
    if (target_) target_->ensure_healthy();
#endif
    return status;
}

void evaluation_trace::sample_complete(uint64_t index, llama_token token) {
    if (lifecycle_) lifecycle_->sample(index, token);
}

void evaluation_trace::forced_complete(uint64_t index, llama_token token) {
    if (lifecycle_) lifecycle_->forced_token(index, token);
}

void evaluation_trace::finish(bool success) {
    if (lifecycle_) {
        const auto & events = lifecycle_->finish(success, semantic_->completed_graph_count());
        const auto path = log::resolve_output_path("log/execution-lifecycle.jsonl");
        if (path.empty() || !log::prepare_output_parent(path)) throw std::runtime_error("unsafe lifecycle path");
        std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.string().c_str(), "wx"), std::fclose);
        if (!file) throw std::runtime_error("cannot exclusively create execution lifecycle");
        for (const auto & event : events) {
            const auto line = event.dump() + '\n';
            if (std::fwrite(line.data(), 1, line.size(), file.get()) != line.size())
                throw std::runtime_error("execution lifecycle write failed");
        }
        if (std::fflush(file.get())) throw std::runtime_error("execution lifecycle flush failed");
    }
    if (semantic_) semantic_->finish(success);
#if CYCLE_SIM
    if (target_) target_->finish(success);
#endif
#if LOG_CYCLE
    if (!gemmini_log_cycle_flush() || !log::cycle.healthy())
        throw std::runtime_error("evaluation: CPU log flush failed");
#endif
}
