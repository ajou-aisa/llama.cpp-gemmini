#define main optrace_dispatch_fixture_main
#include "test-gemmini-optrace-dispatch.cpp"
#undef main
#include <gemmini/cycle_sim_log.hpp>
#include <gemmini/log.hpp>
#include <im2p_cpu_functional.hpp>
#include <im2p_cycle_sim.hpp>
#include "../src/llama-semantic-graph.hpp"
#include "gemmini-parity-output.hpp"

namespace {
struct Observation {
    im2p_production_geometry_v1_t geometry;
    uint64_t activation_stride, weight_stride, output_stride, scale_stride, work_context;
    uint32_t block_size, vector_op, output_domain;
};
std::vector<Observation> observations;
std::vector<std::vector<im2p_compact_run_t>> observed_runs;
bool observation_failed = false;
bool inject_log_failure = false;
std::unique_ptr<FileLimit> injected_limit;
size_t publication_count = 0;
bool published_values_seen = false;
void fail_float_publication(void *opaque) {
    auto &fixture = *static_cast<Fixture *>(opaque);
    ++publication_count;
    published_values_seen = std::all_of(fixture.output.begin(), fixture.output.end(),
        [&](float value) { return value == float(fixture.args.K); });
    injected_limit = std::make_unique<FileLimit>("1");
}
void fail_residual_publication(void *opaque) {
    const auto *values = std::get_if<rmd::BlockScaledInt64Correction>(
        static_cast<rmd::Correction *>(opaque));
    ++publication_count;
    published_values_seen = values && std::all_of(values->values.begin(), values->values.end(),
        [](auto value) { return value == 31; });
    injected_limit = std::make_unique<FileLimit>("1");
}
void observe(void *, const im2p_matmul_desc_t &d,
             const im2p_production_geometry_v1_t &g,
             const im2p_compact_runs_t *runs) noexcept {
    try {
        observations.push_back({g, d.activation_row_stride_bytes, d.weight_row_stride_bytes,
            d.output_row_stride * sizeof(int32_t), d.scale_row_stride, d.work_context,
            static_cast<uint32_t>(d.block_size), d.vector_op, d.output_domain});
        if (runs) {
            if (runs->version != IM2P_COMPACT_RUNS_VERSION || !runs->run_count)
                observation_failed = true;
            else
                observed_runs.emplace_back(runs->runs, runs->runs + runs->run_count);
        }
        if (inject_log_failure && !injected_limit)
            injected_limit = std::make_unique<FileLimit>("1");
    } catch (...) { observation_failed = true; }
}
void print_observation(const Observation &o) {
    const auto &g = o.geometry;
    std::printf("{\"observation\":\"CPU_FUNCTIONAL_DESCRIPTOR\",\"activation_bits\":%u,"
        "\"weight_bits\":%u,\"dim\":%u,\"parent_m\":%llu,\"m\":%llu,\"n\":%llu,\"k\":%llu,"
        "\"tile_i_count\":%llu,\"tile_j_count\":%llu,\"tile_k_count\":%llu,"
        "\"row_begin\":%llu,\"row_count\":%llu,\"stripe_id\":%llu,"
        "\"activation_stride_bytes\":%llu,\"weight_stride_bytes\":%llu,"
        "\"output_stride_bytes\":%llu,\"scale_stride_elements\":%llu,"
        "\"work_context\":%llu,\"block_size\":%u,\"vector_op\":%u,\"output_domain\":%u}\n",
        g.activation_bits, g.weight_bits, g.dim,
        static_cast<unsigned long long>(g.m), static_cast<unsigned long long>(g.row_count),
        static_cast<unsigned long long>(g.n), static_cast<unsigned long long>(g.k),
        static_cast<unsigned long long>(g.tile_i_count), static_cast<unsigned long long>(g.tile_j_count),
        static_cast<unsigned long long>(g.tile_k_count), static_cast<unsigned long long>(g.row_begin),
        static_cast<unsigned long long>(g.row_count), static_cast<unsigned long long>(g.stripe_id),
        static_cast<unsigned long long>(o.activation_stride), static_cast<unsigned long long>(o.weight_stride),
        static_cast<unsigned long long>(o.output_stride), static_cast<unsigned long long>(o.scale_stride),
        static_cast<unsigned long long>(o.work_context), o.block_size, o.vector_op, o.output_domain);
}
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "dispatch mode required");
        const std::string requested = argv[1];
        require(std::string(im2p_sim_implementation()) == "CPU_FUNCTIONAL",
                "collection must use the CPU-functional engine");
        inject_log_failure = requested == "fail-full" || requested == "fail-residual";
        const bool fail_publication = requested == "fail-publish-full" || requested == "fail-publish-residual";
        const std::string mode = fail_publication ? requested.substr(13)
            : inject_log_failure ? requested.substr(5) : requested;
        require(mode == "full" || mode == "pipeline" || mode == "residual" ||
                mode == "residual-runs" ||
                mode == "large-k", "invalid mode");
#if LOG_CYCLE
        require(ggml::gemmini::log::cycle.set_output_path("log/cycle-log.jsonl"),
                "existing CPU cycle-log path");
#endif
        Fixture fixture(mode == "pipeline" ? 129 : 1,
                        mode == "pipeline" ? 129 : 3,
                        mode == "pipeline" ? 96 : mode == "large-k" ? 3072 :
                        mode == "residual-runs" ? 128 : 32);
        namespace cycle = ggml::gemmini::cycle_sim;
        namespace semantic = ggml::gemmini::semantic;
        auto metadata = semantic::Session::start(semantic::Source::PotalCollection,
            "{\"fixture\":\"production-dispatch\"}", "{\"execution_kind\":\"CPU_FUNCTIONAL\"}", false);
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
        auto session = cycle::Session::start(cycle::compiled_run_info(
            "official-cpu-functional-dispatch-fixture", fixture.args.I, 1));
        const auto phase = session->phase("prefill", std::nullopt, fixture.args.I);
        cycle::Operation operation;
        operation.layer = fixture.args.matmul_layer;
        operation.actual_backend = "Gemmini";
        operation.activation_type = "F32";
        operation.weight_type = GGML_GEMMINI_WEIGHT_BITS == 4 ? "Q4_HP1" : "Q8_HP1";
        operation.m = fixture.args.I; operation.n = fixture.args.J; operation.k = fixture.args.K;
        operation.target_eligible = true;
        operation.semantic_context = semantic::context_for(node);
        fixture.args.cycle_sim_context = session->register_operation(node, operation, phase);
        cycle::ScopedContext scoped_context(fixture.args.cycle_sim_context);
        im2p::cpu_functional::set_dispatch_observer(observe, nullptr);
        if (fail_publication && mode == "full") {
            im2p::gemmini::cycle_sim::set_publication_observer(fail_float_publication, &fixture);
            im2p::gemmini::Options options{};
            options.production_geometry = true;
            auto run = im2p::gemmini::execute(&fixture.args, im2p::gemmini::Mode::full, options);
            require(run.status.ok() && bool(run.run), "publication FULL start");
            const auto result = im2p::gemmini::fence(*run.run);
            injected_limit.reset();
            require(!result.status.ok(), "post-copy FULL provenance failure status");
            require(std::all_of(fixture.output.begin(), fixture.output.end(), [](float value) { return value == -1; }),
                    "post-copy FULL failure changed caller bytes");
        } else if (mode == "residual" || mode == "residual-runs") {
            rmd::RmdStripeBuilder builder;
            builder.reset(0, 0, fixture.args.I, fixture.args.K, fixture.args.J,
                          GGML_GEMMINI_ACTIVATION_BITS);
            for (size_t k = 0; k < (mode == "residual-runs" ? 12 : 31); ++k)
                builder.add_residual(0, k, 1);
            if (mode == "residual-runs")
                for (size_t k = 0; k < 10; ++k)
                    builder.add_residual(0, 3 * rmd::kBlockSize + k, 1);
            const auto packet = builder.finish();
            require(bool(packet), "residual packet");
            std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(
                im2p_sim_create(), im2p_sim_destroy);
            require(bool(sim), "CPU-functional engine");
            rmd::Correction expected;
            const rmd::Correction sentinel =
                rmd::BlockScaledInt64Correction{{-777, -777, -777}};
            rmd::Correction actual = sentinel;
            if (fail_publication)
                im2p::gemmini::cycle_sim::set_publication_observer(
                    fail_residual_publication, &actual);
            rmd::RmdExecutionMetrics metrics{};
            require(rmd::execute_rmd_stripe_reference(fixture.args, *packet, expected) ==
                    rmd::RmdStatus::success, "residual reference");
            const auto result = rmd::execute_rmd_stripe_im2p(sim.get(), fixture.args, *packet,
                                                           actual, &metrics);
            injected_limit.reset();
            require((inject_log_failure || fail_publication)
                                       ? result == rmd::RmdStatus::execution_failed
                                       : result == rmd::RmdStatus::success, "residual target dispatch");
            if (inject_log_failure || fail_publication) {
                const auto *actual_values =
                    std::get_if<rmd::BlockScaledInt64Correction>(&actual);
                const auto *sentinel_values =
                    std::get_if<rmd::BlockScaledInt64Correction>(&sentinel);
                require(actual_values && sentinel_values &&
                            actual_values->values == sentinel_values->values &&
                            metrics.im2p_dot_calls == 0,
                        "failed residual provenance published output or metrics");
            } else {
                const auto *expected_values =
                    std::get_if<rmd::BlockScaledInt64Correction>(&expected);
                const auto *actual_values =
                    std::get_if<rmd::BlockScaledInt64Correction>(&actual);
                require(expected_values && actual_values &&
                            actual_values->values == expected_values->values,
                        "residual numerical equality");
                require(metrics.im2p_dot_calls == 1,
                        "one residual packet must remain one logical NPU GEMM");
                save_gemmini_parity_output(actual_values->values,
                                            IM2P_OUTPUT_SCU_FINAL, fixture.args.J);
            }
        } else {
            const auto result = mode == "pipeline"
                ? ggml::gemmini::im2p_adapter::run_stripe_pipeline(fixture.args)
                : ggml::gemmini::im2p_adapter::run_full(fixture.args);
            injected_limit.reset();
            require(inject_log_failure ? !result.result.ok() : result.result.ok(), result.result.message);
            require(std::all_of(fixture.output.begin(), fixture.output.end(),
                [&fixture](float value) {
                    return value == (inject_log_failure ? -1.0f : float(fixture.args.K));
                }),
                "dense numerical equality");
            if (!inject_log_failure) save_gemmini_parity_output(fixture.output, 0, fixture.args.J);
        }
        im2p::cpu_functional::set_dispatch_observer(nullptr, nullptr);
        im2p::gemmini::cycle_sim::set_publication_observer(nullptr, nullptr);
        const auto expected_count = mode == "pipeline"
            ? (fixture.args.I + fixture.args.activation_rows_per_stripe - 1) /
                fixture.args.activation_rows_per_stripe : 1;
        require(!observation_failed && observations.size() == expected_count,
                "independent descriptor observer count");
        if ((mode == "residual" || mode == "residual-runs") &&
            !inject_log_failure && !fail_publication)
            require(observed_runs.size() == 1 &&
                    observed_runs[0].size() == (mode == "residual-runs" ? 2 : 1) &&
                    observed_runs[0][0].compact_k_count ==
                        (mode == "residual-runs" ? 12 : 31) &&
                    (mode != "residual-runs" ||
                     (observed_runs[0][1].original_block_id == 3 &&
                      observed_runs[0][1].compact_k_begin == 12 &&
                      observed_runs[0][1].compact_k_count == 10)),
                    "run-aware residual observer lost original-block metadata");
        for (const auto &observation : observations) print_observation(observation);
        if (inject_log_failure || fail_publication) {
            bool failed = false;
            try { session->ensure_healthy(); } catch (...) { failed = true; }
            require(failed, "failed provenance did not poison run");
            if (fail_publication) {
                require(publication_count == 1 && published_values_seen, "fault did not occur after public output copy");
                std::printf("POST_COPY_ROLLBACK_PASS public_write_seen=true\n");
            }
            std::printf("CPU_FUNCTIONAL_ROLLBACK_PASS mode=%s completed_descriptors=%zu\n",
                        mode.c_str(), observations.size());
            return 0;
        }
        session->ensure_healthy();
        session->finish_operation(fixture.args.cycle_sim_context);
        session->finish();
        metadata->execution(node, "Gemmini", "TARGET_NPU", true);
        metadata->finish(true);
        std::printf("CPU_FUNCTIONAL_DISPATCH_PASS mode=%s numerical=PASS actual_rtl_acceptance=NOT_APPLICABLE\n",
                    mode.c_str());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "CPU_FUNCTIONAL_DISPATCH_FAIL %s\n", error.what());
        return 1;
    }
}
