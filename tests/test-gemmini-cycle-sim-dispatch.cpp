#define main optrace_dispatch_fixture_main
#include "test-gemmini-optrace-dispatch.cpp"
#undef main
#include <gemmini/cycle_sim_log.hpp>
#include <gemmini/log.hpp>
#include <im2p_cpu_functional.hpp>
#include <im2p_cycle_sim.hpp>
#include "quants/act/exsia/exsia.hpp"
#include "../tools/eval/evaluation-lifecycle.h"
#include "../src/llama-semantic-graph.hpp"
#include "gemmini-parity-output.hpp"
#include <cstdlib>

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
        require(mode == "full" || mode == "pipeline" || mode == "exsia-pipeline" ||
                mode == "exsia-rmd-pipeline" || mode == "residual" ||
                mode == "residual-runs" ||
                mode == "large-k", "invalid mode");
#if LOG_CYCLE
        require(ggml::gemmini::log::cycle.set_output_path("log/cycle-log.jsonl"),
                "existing CPU cycle-log path");
#endif
        const bool exsia_pipeline = mode == "exsia-pipeline" || mode == "exsia-rmd-pipeline";
        Fixture fixture(exsia_pipeline ? 161 : mode == "pipeline" ? 129 : 1,
                        exsia_pipeline ? 161 : mode == "pipeline" ? 129 : 3,
                        (mode == "pipeline" || exsia_pipeline) ? 96 : mode == "large-k" ? 3072 :
                        mode == "residual-runs" ? 128 : 32);
        namespace cycle = ggml::gemmini::cycle_sim;
        namespace semantic = ggml::gemmini::semantic;
        const auto run_info = cycle::compiled_run_info("native-production-dispatch-fixture",
                                                       fixture.args.I, exsia_pipeline ? 1 : 0);
        const char *source_commit = std::getenv("IM2P_FIXTURE_SOURCE_COMMIT");
        const char *source_manifest_sha256 = std::getenv("IM2P_FIXTURE_SOURCE_MANIFEST_SHA256");
        require(!exsia_pipeline || (source_commit && *source_commit &&
                source_manifest_sha256 && std::char_traits<char>::length(source_manifest_sha256) == 64),
                "fixture source commit and manifest SHA256 required");
        const nlohmann::json workload = {
            {"model", run_info.model}, {"prompt_tokens", fixture.args.I},
            {"generated_tokens", exsia_pipeline ? 1 : 0}, {"context_tokens", nullptr},
            {"batch_tokens", nullptr}, {"microbatch_tokens", nullptr},
            {"cpu", {{"kind", "NOT_APPLICABLE_NATIVE_FIXTURE"}}},
            {"cpu_batch", {{"kind", "NOT_APPLICABLE_NATIVE_FIXTURE"}}},
            {"flash_attention", false}, {"kv_type_k", nullptr}, {"kv_type_v", nullptr},
            {"seed", nullptr}, {"numa", nullptr},
            {"build_target", "native-dispatch-fixture/" + run_info.profile}};
        const nlohmann::json producer = {
            {"cycle_sim", true}, {"log_cycle", LOG_CYCLE != 0}, {"cpu_only_build", false},
            {"git_commit", source_commit ? source_commit : "unbound-test-fixture"},
            {"source_manifest_sha256", source_manifest_sha256 ? source_manifest_sha256 : "unbound-test-fixture"},
            {"hardware_contract_sha256", run_info.hardware_contract_sha256},
            {"warmup_excluded", true}, {"reserve_measure_graphs_excluded", true},
            {"application_runner", "native-production-dispatch-fixture"},
            {"execution_kind", "CPU_FUNCTIONAL_FIXTURE"}, {"trajectory_source", "FIXTURE_TOKEN"}};
        auto metadata = semantic::Session::start(semantic::Source::PotalCollection,
            workload.dump(), producer.dump(), false);
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
        auto session = cycle::Session::start(run_info);
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
        } else if (exsia_pipeline) {
            auto &producer_meta = fixture.args.act_quant.storage().emplace<
                ggml::gemmini::quants::act::exsia::Meta>();
            std::vector<float> input(fixture.args.I * fixture.args.K, 1.0f);
            if (mode == "exsia-rmd-pipeline") {
                for (size_t row = 0; row < fixture.args.activation_rows_per_stripe; ++row) {
                    input[row * fixture.args.K] = 256.0f;
                    input[row * fixture.args.K + 1] = 8.0f;
                    input[row * fixture.args.K + 40] = -17.0f;
                }
            }
            ggml_tensor activation_tensor{};
            activation_tensor.type = GGML_TYPE_F32;
            activation_tensor.data = input.data();
            auto started = ggml::gemmini::im2p_adapter::start_exsia_stripe_pipeline(fixture.args);
            require(started.result.ok() && started.pipeline &&
                    started.pipeline->install_sink().ok(), "ExSIA pipeline start");
            ggml::gemmini::quants::act::exsia::ExSIA exsia;
            exsia.set_execution_mode(ggml::gemmini::quants::act::exsia::ExSIAState::ExecutionMode::Sequential);
            const bool quantized = exsia.run(producer_meta, &activation_tensor, fixture.args,
                                            fixture.args.exsia_stripe_ready_sink);
            const auto result = started.pipeline->finish(quantized);
            require(result.result.ok(), result.result.message);
            require(std::all_of(fixture.output.begin(), fixture.output.end(),
                [&, column = size_t{0}](float value) mutable {
                    const size_t row = column++ / fixture.args.J;
                    return mode == "exsia-rmd-pipeline" &&
                        row < fixture.args.activation_rows_per_stripe
                        ? value > 0.0f && value != -1.0f
                        : value == float(fixture.args.K);
                }), "ExSIA numerical publication");
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
        const auto expected_count = mode == "pipeline" || exsia_pipeline
            ? (fixture.args.I + fixture.args.activation_rows_per_stripe - 1) /
                fixture.args.activation_rows_per_stripe : 1;
        const auto dense_observation_count = std::count_if(observations.begin(), observations.end(),
            [](const auto &observation) {
                return observation.geometry.scope == IM2P_GEOMETRY_STRIPE;
            });
        require(!observation_failed &&
                (mode == "exsia-rmd-pipeline"
                    ? dense_observation_count == expected_count && observations.size() > expected_count
                    : observations.size() == expected_count),
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
        if (mode == "pipeline" || exsia_pipeline) {
            const auto parents = session->producer_parents();
            require(parents.size() == 1 && parents[0].geometry.scope == IM2P_GEOMETRY_STREAM &&
                    parents[0].geometry.m == fixture.args.I &&
                    parents[0].work_ids == parents[0].fence_required_work_ids &&
                    parents[0].work_ids.size() >= expected_count,
                    "pipeline parent final geometry and fence work set");
            if (mode == "exsia-rmd-pipeline") {
                require(!parents[0].residual_bindings.empty(), "run-aware residual child work missing");
                for (const auto &binding : parents[0].residual_bindings)
                    require(binding.dense_parent_id == parents[0].parent_id &&
                            binding.child_parent_id != binding.dense_parent_id &&
                            binding.stripe_id < expected_count &&
                            binding.row_begin < binding.row_end &&
                            std::find(parents[0].work_ids.begin(), parents[0].work_ids.end(),
                                      binding.work_id) != parents[0].work_ids.end(),
                            "residual work lacks exact dense stripe and fence binding");
            }
            const auto ownership = session->producer_events();
            const auto count = [&](cycle::ProducerEventKind kind) {
                return std::count_if(ownership.begin(), ownership.end(), [&](const auto &event) {
                    return event.event.kind == kind;
                });
            };
            require(count(cycle::ProducerEventKind::FrontendQueueEnqueue) == expected_count &&
                    count(cycle::ProducerEventKind::FrontendQueueDequeue) == expected_count &&
                    count(cycle::ProducerEventKind::StreamWorkAccepted) == expected_count &&
                    count(cycle::ProducerEventKind::StreamWorkCompleted) == expected_count &&
                    count(cycle::ProducerEventKind::FrontendCapacityAcquire) == expected_count &&
                    count(cycle::ProducerEventKind::FrontendCapacityRelease) == expected_count,
                    "pipeline producer ownership transitions");
            if (exsia_pipeline)
                require(count(cycle::ProducerEventKind::ExsiaWorkspaceAcquire) == expected_count &&
                        count(cycle::ProducerEventKind::ActivationRowsCommit) == expected_count &&
                        count(cycle::ProducerEventKind::ResidualPacketSeal) == expected_count &&
                        count(cycle::ProducerEventKind::ExsiaWorkspaceRelease) == expected_count &&
                        count(cycle::ProducerEventKind::ResidualCallbackCompleted) == expected_count,
                        "ExSIA producer workspace and residual ownership transitions");
            if (mode == "exsia-rmd-pipeline") {
                const auto released = std::find_if(ownership.begin(), ownership.end(), [](const auto &event) {
                    return event.event.kind == cycle::ProducerEventKind::FrontendCapacityRelease &&
                        event.event.stripe_id == 0;
                });
                require(released != ownership.end() && released->event.rmd_packet &&
                        released->required_work_ids.size() > 1 &&
                        released->required_call_ids.size() == 1 &&
                        count(cycle::ProducerEventKind::ResidualHostMergeCompleted) > 0,
                        "residual capacity release requires actual compact work and merge call");
            }
            if (expected_count >= 3 && exsia_pipeline) {
                const auto first_release = std::find_if(ownership.begin(), ownership.end(), [](const auto &event) {
                    return event.event.kind == cycle::ProducerEventKind::ExsiaWorkspaceRelease &&
                        event.event.stripe_id == 0;
                });
                const auto third_acquire = std::find_if(ownership.begin(), ownership.end(), [](const auto &event) {
                    return event.event.kind == cycle::ProducerEventKind::ExsiaWorkspaceAcquire &&
                        event.event.stripe_id == 2;
                });
                require(first_release != ownership.end() && third_acquire != ownership.end() &&
                        first_release->sequence < third_acquire->sequence &&
                        first_release->event.workspace_slot == third_acquire->event.workspace_slot,
                        "ExSIA scratch slot zero reused only after source release");
            }
            for (const auto &event : ownership) {
                require(event.operation_id == fixture.args.cycle_sim_context.operation_id &&
                        event.parent_id.has_value() && event.work_id.has_value() &&
                        event.event.workspace_slot == event.event.stripe_id % 2 &&
                        event.event.row_begin < event.event.row_end &&
                        event.event.source_location.find(':') != std::string::npos,
                        "pipeline producer identity/source binding");
            }
        }
        metadata->execution(node, "Gemmini", "TARGET_NPU", true);
        if (exsia_pipeline) {
            evaluation_lifecycle lifecycle("potal_collection", source_commit, 1,
                                           false, true);
            lifecycle.phase("prefill", {}, 0);
            lifecycle.request_start(0);
            lifecycle.prefill_batch_ready(0, 0);
            lifecycle.begin_dispatch(0);
            lifecycle.end_dispatch(metadata->completed_graph_count(), 0);
            lifecycle.sample(0, 7);
            lifecycle.pipeline_session(*session);
            const auto &events = lifecycle.finish(true, metadata->completed_graph_count());
            const auto path = ggml::gemmini::log::resolve_output_path("log/execution-lifecycle.jsonl");
            require(!path.empty() && ggml::gemmini::log::prepare_output_parent(path),
                    "producer sidecar path");
            std::unique_ptr<FILE, decltype(&std::fclose)> output(
                std::fopen(path.string().c_str(), "wx"), std::fclose);
            require(bool(output), "producer sidecar exclusive create");
            for (const auto &event : events) {
                const auto line = event.dump() + '\n';
                require(std::fwrite(line.data(), 1, line.size(), output.get()) == line.size(),
                        "producer sidecar write");
            }
            require(std::fflush(output.get()) == 0, "producer sidecar flush");
        }
        metadata->finish(true);
        std::printf("CPU_FUNCTIONAL_DISPATCH_PASS mode=%s numerical=PASS actual_rtl_acceptance=NOT_APPLICABLE\n",
                    mode.c_str());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "CPU_FUNCTIONAL_DISPATCH_FAIL %s\n", error.what());
        return 1;
    }
}
