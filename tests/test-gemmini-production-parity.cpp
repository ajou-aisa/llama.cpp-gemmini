#define main optrace_dispatch_fixture_main
#include "test-gemmini-optrace-dispatch.cpp"
#undef main
#include "gemmini-parity-output.hpp"
#include "accepted_work_observer.h"
#include "../src/llama-semantic-graph.hpp"
#include <fstream>

namespace {
std::vector<im2p_accepted_work_observation_t> observed;
bool failed_observer = false;
void observe(void *, const im2p_accepted_work_observation_t *record) {
    try { observed.push_back(*record); } catch (...) { failed_observer = true; }
}
void save_observations(const std::filesystem::path &path) {
    std::ofstream out(path);
    for (const auto &record : observed) {
        const auto &g = record.geometry;
        out << "{\"event\":" << record.event << ",\"activation_bits\":" << g.activation_bits
            << ",\"weight_bits\":" << g.weight_bits << ",\"dim\":" << g.dim
            << ",\"parent_m\":" << g.m << ",\"m\":" << g.row_count
            << ",\"n\":" << g.n << ",\"k\":" << g.k
            << ",\"tile_i_count\":" << g.tile_i_count << ",\"tile_j_count\":" << g.tile_j_count
            << ",\"tile_k_count\":" << g.tile_k_count << ",\"row_begin\":" << g.row_begin
            << ",\"row_count\":" << g.row_count << ",\"stripe_id\":" << g.stripe_id
            << ",\"host_slot\":" << record.host_slot
            << ",\"activation_stride_bytes\":" << record.activation_host_stride
            << ",\"weight_stride_bytes\":" << record.weight_host_stride
            << ",\"output_stride_bytes\":" << record.output_host_stride
            << ",\"scale_stride_bytes\":" << record.scale_host_stride
            << ",\"explicit_geometry\":" << record.explicit_geometry
            << ",\"rmd_raw\":" << record.rmd_raw << "}\n";
    }
    out.close();
    require(bool(out) && !failed_observer, "RTL observation output");
}
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "full|pipeline|residual|large-k required");
        const std::string mode = argv[1];
        require(mode == "full" || mode == "pipeline" || mode == "residual" || mode == "large-k",
                "invalid parity mode");
        const char *directory = std::getenv("GEMMINI_LOG_DIR");
        require(directory && *directory, "parity directory required");
        require(std::string(im2p_sim_implementation()) == "gemmini-hp1-integrated-v1",
                "parity reference must use the integrated RTL runtime");
        Fixture fixture(mode == "pipeline" ? 129 : 1, mode == "pipeline" ? 129 : 3,
                        mode == "pipeline" ? 96 : mode == "large-k" ? 3072 : 32);
        namespace semantic = ggml::gemmini::semantic;
        auto metadata = semantic::Session::start(semantic::Source::PotalCollection,
            "{\"fixture\":\"production-dispatch\"}", "{\"purpose\":\"RTL_PRODUCTION_PARITY_REFERENCE\"}", false);
        std::vector<int32_t> tokens(fixture.args.I, 7);
        metadata->phase("prefill", std::nullopt, tokens.data(), tokens.size());
        std::unique_ptr<ggml_context, decltype(&ggml_free)> memory(
            ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
        require(bool(memory), "semantic fixture graph allocation");
        auto *weights = ggml_new_tensor_2d(memory.get(), GGML_GEMMINI_WEIGHT_BITS == 4
            ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1, fixture.args.K, fixture.args.J);
        auto *activation = ggml_new_tensor_2d(memory.get(), GGML_TYPE_F32, fixture.args.K, fixture.args.I);
        auto *node = ggml_mul_mat(memory.get(), weights, activation);
        ggml_set_name(node, fixture.args.matmul_layer.c_str());
        auto *graph = ggml_new_graph(memory.get());
        ggml_build_forward_expand(graph, node);
        semantic::capture_graph(graph);
        trace::RunInfo info;
        info.model = "production-parity-fixture";
        info.activation_bits = GGML_GEMMINI_ACTIVATION_BITS;
        info.weight_bits = GGML_GEMMINI_WEIGHT_BITS;
        info.dim = DIM;
        info.profile = "a" + std::to_string(info.activation_bits) + "w" +
            std::to_string(info.weight_bits) + "-d" + std::to_string(DIM) + "-hp1";
        info.backend = "IM2P_SIM/GEMMINI_HP1";
        info.mode = mode == "pipeline" ? "STRIPE_PIPELINE" : "FULL";
        info.residual_enabled = GGML_GEMMINI_ENABLE_RMD != 0;
        info.hardware_contract_sha256 = std::string(64, '0');
        info.runtime_manifest_sha256 = std::string(64, '0');
        for (const char *name : {"IM2P.sim", "llama.cpp-gemmini", "headers"}) {
            info.source_commits[name] = std::string(40, '0');
            info.source_worktree_sha256[name] = std::string(64, '0');
        }
        const auto trace_path = (std::filesystem::path(directory) / "production-optrace.jsonl").string();
        auto session = trace::Session::start(trace_path.c_str(), info);
        fixture.args.optrace_context = std::make_shared<const trace::Context>(
            session->phase("prefill", std::nullopt, fixture.args.I));
        im2p_test_set_work_observer(observe, nullptr);
        if (mode == "residual") {
            rmd::RmdStripeBuilder builder;
            builder.reset(0, 0, fixture.args.I, fixture.args.K, fixture.args.J, GGML_GEMMINI_ACTIVATION_BITS);
            for (size_t k = 0; k < 31; ++k) builder.add_residual(0, k, 1);
            const auto packet = builder.finish();
            require(bool(packet), "residual packet");
            std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(im2p_sim_create(), im2p_sim_destroy);
            require(bool(sim), "RTL runtime");
            rmd::CompressedOutput actual;
            rmd::RmdExecutionMetrics metrics{};
            const auto status = rmd::execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet, actual, &metrics);
            if (status != rmd::RmdStatus::success)
                throw std::runtime_error(std::string("production RMD dispatch: ") + rmd::rmd_status_message(status));
            require(metrics.im2p_dot_calls == 1, "K31 one logical production dot");
            save_gemmini_parity_output(actual.values, static_cast<uint64_t>(actual.domain), actual.j_padded);
        } else {
            const auto completion = mode == "pipeline"
                ? ggml::gemmini::im2p_adapter::run_stripe_pipeline(fixture.args)
                : ggml::gemmini::im2p_adapter::run_full(fixture.args);
            require(completion.result.ok(), completion.result.message);
            save_gemmini_parity_output(fixture.output, 0, fixture.args.J);
        }
        im2p_test_set_work_observer(nullptr, nullptr);
        session->finish();
        metadata->execution(node, "IM2P_SIM/GEMMINI_HP1", "TARGET_NPU", true);
        metadata->finish(true);
        save_observations(std::filesystem::path(directory) / "rtl-observations.jsonl");
        std::printf("RTL_PRODUCTION_PARITY_PASS mode=%s\n", mode.c_str());
        return 0;
    } catch (const std::exception &error) {
        im2p_test_set_work_observer(nullptr, nullptr);
        std::fprintf(stderr, "RTL_PRODUCTION_PARITY_FAIL %s\n", error.what());
        return 1;
    }
}
