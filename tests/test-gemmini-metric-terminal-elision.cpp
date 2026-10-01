// Metric-only terminal lm_head: a METRIC_PREFILL_256 session observes the lm_head completely while its main and
// residual GEMMs, their reconstruction and its output (the logits) are never executed or written. Block layers,
// other outputs, full-mode sessions and runs without a session are unaffected.
#include <ggml-backend.h>
#include <ggml-gemmini.h>
#include <ggml.h>

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"

#include <gemmini/evaluation_metrics.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

namespace evaluation = ggml::gemmini::evaluation;
namespace adapter = ggml::gemmini::im2p_adapter;

constexpr int64_t I = DIM + 1, J = 17, K = 64;  // two K32 blocks
constexpr float sentinel = 12345.0f;

bool check(bool condition, const char *what) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

std::string read(const std::filesystem::path &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}

size_t count(const std::string &text, const std::string &needle) {
    size_t found = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++found;
    return found;
}

struct Outcome {
    adapter::TestCounters counters;
    std::vector<float> output;
    uint64_t elisions = 0;
    std::string activation, residual, scale;
};

// One mul_mat of the llama backend: `weight` x activations with outliers (so residual packets exist) into `output`,
// under a metric session that runs the terminal lm_head `terminal_mode` ("" = no session at all).
Outcome run(const std::filesystem::path &directory, const char *name, const char *weight_name,
            const char *output_name, const std::string &terminal_mode) {
    std::vector<float> activations(I * K, 0.5f);
    for (int64_t row = 0; row + 1 < I; ++row) {
        activations[row * K] = 256.0f;
        activations[row * K + 1] = 8.0f;
        activations[row * K + 40] = -17.0f;
    }
#if GGML_GEMMINI_WEIGHT_BITS == 4
    std::vector<block_q4_hp1> weights(J * (K / 32));
    for (size_t index = 0; index < weights.size(); ++index) {
        weights[index].channel_scale = 0.5f;
        weights[index].m = static_cast<int16_t>(index % 3);
        std::fill(std::begin(weights[index].qs), std::end(weights[index].qs), uint8_t{0x99});
    }
    const ggml_type type = GGML_TYPE_Q4_HP1;
#else
    std::vector<block_q8_hp1> weights(J * (K / 32));
    for (size_t index = 0; index < weights.size(); ++index) {
        weights[index].channel_scale = 0.5f;
        weights[index].m = static_cast<int16_t>(index % 3);
        std::fill(std::begin(weights[index].qs), std::end(weights[index].qs), int8_t{1});
    }
    const ggml_type type = GGML_TYPE_Q8_HP1;
#endif
    Outcome outcome;
    std::shared_ptr<evaluation::Session> session;
    evaluation::Config config;
    if (!terminal_mode.empty()) {
        config.run_id = "terminal";  // the same identity in every run: equal observations give equal bytes
        config.workload_id = "METRIC_PREFILL_256";
        config.manifest_sha256 = std::string(64, 'e');
        config.activation_path = (directory / (std::string(name) + "-activation.jsonl")).string();
        config.residual_path = (directory / (std::string(name) + "-residual.jsonl")).string();
        config.scale_path = (directory / (std::string(name) + "-scale.jsonl")).string();
        config.scale_aggregate = true;
        config.terminal_lm_head_metrics_only = terminal_mode == "metrics-only";
        session = evaluation::Session::start(config);
        session->chunk(0);
    }
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0), "llama");
    ggml_init_params params = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context *context = ggml_init(params);
    ggml_tensor *weight = ggml_new_tensor_2d(context, type, K, J);
    ggml_tensor *activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, I);
    ggml_set_name(weight, weight_name);
    ggml_set_name(activation, "result_norm");
    ggml_tensor *output = ggml_mul_mat(context, weight, activation);
    ggml_set_name(output, output_name);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(weight, weights.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(activation, activations.data(), 0, ggml_nbytes(activation));
    std::vector<float> initial(I * J, sentinel);
    ggml_backend_tensor_set(output, initial.data(), 0, ggml_nbytes(output));
    ggml_cgraph *graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    adapter::test_reset();
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    outcome.counters = adapter::test_counters();
    outcome.output.resize(I * J);
    ggml_backend_tensor_get(output, outcome.output.data(), 0, ggml_nbytes(output));
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    if (!computed) std::fprintf(stderr, "FAIL: graph compute %s\n", name);
    if (session) {
        outcome.elisions = session->terminal_lm_head_elisions();
        session->finish(true);
        session.reset();
        outcome.activation = read(config.activation_path);
        outcome.residual = read(config.residual_path);
        outcome.scale = read(config.scale_path);
    }
    return outcome;
}

bool untouched(const std::vector<float> &values) {
    for (const float value : values)
        if (value != sentinel) return false;
    return true;
}

bool numerically_executed(const Outcome &outcome) {
    return outcome.counters.full == 1 && outcome.counters.fence == 1 && outcome.counters.residual_executions > 0 &&
           outcome.counters.commit == 1 && !untouched(outcome.output) && outcome.elisions == 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    const Outcome full = run(directory, "lm-head-full", "output.weight", "result_output", "full");
    const Outcome elided = run(directory, "lm-head-metrics-only", "output.weight", "result_output", "metrics-only");
    bool ok = check(numerically_executed(full), "full mode executes the lm_head GEMMs and writes its output");
    ok = check(elided.counters.full == 0 && elided.counters.fence == 0, "metrics-only lm_head runs no main GEMM") && ok;
    ok = check(elided.counters.residual_executions == 0 && elided.counters.rmd_dot_calls == 0,
               "metrics-only lm_head runs no residual GEMM") && ok;
    ok = check(elided.counters.commit == 0 && untouched(elided.output),
               "metrics-only lm_head writes no output (logits not materialized)") && ok;
    ok = check(elided.elisions == 1, "metrics-only lm_head elision is recorded once") && ok;
    // Every observation exists and is byte-identical to the fully computed lm_head.
    const size_t stripes = count(full.residual, "\"kind\":\"MAIN_STRIPE\"");
    ok = check(count(full.activation, "\"kind\":\"COUNTS\"") == 1 && stripes > 0 &&
               count(full.residual, "\"kind\":\"RADIX_STRIPE\"") == stripes &&
               count(full.residual, "\"kind\":\"COMPACT_WORK\"") > 0 &&
               count(full.scale, "\"work_type\":\"DENSE\"") == 1 && count(full.scale, "\"work_type\":\"RESIDUAL\"") == 1,
               "full lm_head emits activation, residual and both SCU observations") && ok;
    ok = check(elided.activation == full.activation, "metrics-only lm_head activation records are identical") && ok;
    ok = check(elided.residual == full.residual, "metrics-only lm_head residual records are identical") && ok;
    ok = check(elided.scale == full.scale, "metrics-only lm_head SCU records are identical") && ok;
    // Never elided: a block layer, another output, a full-mode session, no session.
    ok = check(numerically_executed(run(directory, "block", "blk.0.attn_q.weight", "Qcur-0", "metrics-only")),
               "metrics-only session executes block layers") && ok;
    ok = check(numerically_executed(run(directory, "other-output", "output.weight", "logits_scratch", "metrics-only")),
               "metrics-only session executes an lm_head that is not the graph output") && ok;
    const Outcome unobserved = run(directory, "no-session", "output.weight", "result_output", "");
    ok = check(unobserved.counters.full == 1 && unobserved.counters.commit == 1 && !untouched(unobserved.output),
               "without a metric session the lm_head executes") && ok;
    ok = check(unobserved.output == full.output, "an observed full lm_head computes the same output") && ok;
    if (ok) std::puts("METRIC_TERMINAL_ELISION observed_full=PASS numerical_skipped=PASS guards=PASS");
    return ok ? 0 : 1;
}
