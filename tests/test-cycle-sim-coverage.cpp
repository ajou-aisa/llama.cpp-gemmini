#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <gemmini/cycle_sim_log.hpp>
#include "../src/llama-semantic-graph.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <unistd.h>

namespace sim = ggml::gemmini::cycle_sim;
namespace semantic = ggml::gemmini::semantic;

int main() {
    char directory[] = "/tmp/im2p-cycle-coverage-XXXXXX";
    assert(mkdtemp(directory));
    assert(setenv("GEMMINI_LOG_DIR", directory, 1) == 0);
    auto session = sim::Session::start(sim::compiled_run_info("cpu-coverage", 1, 1));
    auto semantic_session = semantic::Session::start(semantic::Source::PotalCollection,
        "{\"fixture\":\"cpu-coverage\",\"threads\":2}", "{\"backend\":\"CPU\"}", false);
    struct Policy {
        ggml_backend_dev_t device;
        size_t queries = 0;
    } policy{ggml_backend_dev_by_name("GEMMINI")};
    assert(policy.device);
    session->set_policy_query({&policy, [](void *opaque, const void *node) {
        auto &production = *static_cast<Policy *>(opaque);
        ++production.queries;
        return ggml_backend_dev_supports_op(production.device, static_cast<const ggml_tensor *>(node));
    }});
    auto backend = ggml_backend_cpu_init();
    assert(backend);
    ggml_backend_cpu_set_n_threads(backend, 2);
    for (int phase = 0; phase < 2; ++phase) {
        const int32_t token = phase + 1;
        semantic_session->phase(phase ? "decode" : "prefill",
                               phase ? std::optional<uint64_t>(0) : std::nullopt, &token, 1);
        auto context = session->phase(phase ? "decode" : "prefill",
                                     phase ? std::optional<uint64_t>(0) : std::nullopt, 1);
        sim::ScopedContext scope(context);
        auto memory = ggml_init({2 * 1024 * 1024, nullptr, true});
        assert(memory);
        auto weights = ggml_new_tensor_2d(memory, GGML_TYPE_F32, 4, 2);
        auto activation = ggml_new_tensor_2d(memory, GGML_TYPE_F32, 4, 1);
        auto result = ggml_mul_mat(memory, weights, activation);
        ggml_set_name(result, "same-layer");
        auto graph = ggml_new_graph(memory);
        ggml_build_forward_expand(graph, result);
        semantic::capture_graph(graph);
        const auto original_identity = semantic::context_for(result);
        assert(original_identity->identity.node_ordinal == 0);
        assert(original_identity->identity.graph_occurrence == 0);
        assert(original_identity->identity.phase_kind == (phase ? "decode" : "prefill"));
        auto buffer = ggml_backend_alloc_ctx_tensors(memory, backend);
        assert(buffer);
        const float w[] = {1, 2, 3, 4, 2, 2, 2, 2};
        const float a[] = {1, 1, 1, 1};
        ggml_backend_tensor_set(weights, w, 0, sizeof(w));
        ggml_backend_tensor_set(activation, a, 0, sizeof(a));
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        float values[2]{};
        ggml_backend_tensor_get(result, values, 0, sizeof(values));
        assert(values[0] == 10 && values[1] == 8);
        ggml_backend_buffer_free(buffer);
        ggml_free(memory);
    }
    session->finish();
    semantic_session->finish(true);
    assert(policy.queries == 2);
    ggml_backend_free(backend);
    std::ifstream input(std::filesystem::path(directory) / "npu-cycle-trace.jsonl");
    assert(input);
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    const auto count = [&text](const std::string &field, const std::string &value) {
        const std::regex pattern("\"" + field + "\"\\s*:\\s*\"" + value + "\"");
        return std::distance(std::sregex_iterator(text.begin(), text.end(), pattern), std::sregex_iterator());
    };
    assert(count("kind", "TARGET_OPERATION") == 2);
    assert(count("selected_target", "ORDINARY_CPU") == 2);
    assert(count("actual_backend", "CPU") == 2);
    assert(count("semantic_phase_kind", "prefill") == 1);
    assert(count("semantic_phase_kind", "decode") == 1);
    assert(count("kind", "CPU_INTERVAL") == 0);
    assert(count("kind", "NPU_WORK") == 0);
    std::ifstream semantic_input(std::filesystem::path(directory) / "semantic-graph.jsonl");
    assert(semantic_input);
    const std::string manifest{std::istreambuf_iterator<char>(semantic_input), {}};
    assert(manifest.find("\"expected_node_count\":2,\"executed_node_count\":2") != std::string::npos);
    std::cout << "CPU_TARGET_OPERATION_COVERAGE_PASS " << directory << '\n';
}
