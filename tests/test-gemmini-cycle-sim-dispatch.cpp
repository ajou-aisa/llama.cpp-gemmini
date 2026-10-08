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
#include <fstream>
#include <set>
#include <map>
#include <optional>

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

// P6.2 record-level checks on the dedicated NPU trace (trace version 3). In the paired parent every
// stripe carries a residual and a requested companion: an admitted one runs inside the stripe work
// (no residual child, its RESIDUAL_MERGE requires exactly that work); one the library refused
// (CAPACITY_NONFIT) leaves a planned stripe whose residual is child work bound to it, merged after
// that child. Both outcomes must occur. The compact parent records no companion and binds one
// child per residual stripe.
void verify_paired_trace(const ggml::gemmini::cycle_sim::ProducerParent &compact,
                         const ggml::gemmini::cycle_sim::ProducerParent &paired,
                         size_t stripes) {
    const auto path = ggml::gemmini::log::resolve_output_path(GEMMINI_LOG_DEFAULT_NPU_TRACE_PATH);
    std::ifstream stream(path);
    require(stream.is_open(), "paired trace file");
    struct Stripe {
        std::optional<uint64_t> work;
        std::optional<bool> admitted;
        std::string reason;
        std::set<uint64_t> children;
        size_t merges = 0;
    };
    std::map<uint64_t, Stripe> mine, theirs;
    std::vector<std::vector<uint64_t>> merges;
    for (std::string line; std::getline(stream, line);) {
        if (line.empty()) continue;
        const auto record = nlohmann::json::parse(line);
        const std::string kind = record.value("kind", "");
        if (kind == "NPU_WORK") {
            require(record.value("version", 0) == 3, "paired trace must be version 3");
            const uint64_t operation = record.at("operation_id").get<uint64_t>();
            if (operation != paired.operation_id && operation != compact.operation_id) continue;
            auto &table = operation == paired.operation_id ? mine : theirs;
            const auto stripe = record.at("stripe_id").get<uint64_t>();
            const std::string scope = record.at("scope").get<std::string>();
            const auto work = record.at("work_id").get<uint64_t>();
            if (scope == "residual_compact") {
                require(record.at("paired").is_null(), "residual child carries companion facts");
                table[stripe].children.insert(work);
                continue;
            }
            require(scope == "stripe", "unexpected pipeline work scope");
            table[stripe].work = work;
            const auto &facts = record.at("paired");
            if (facts.is_null()) continue;
            require(operation == paired.operation_id, "compact route recorded a companion");
            const bool admitted = facts.at("admitted").get<bool>();
            require(facts.at("revision") == "pair-loop-tail-v1" && facts.at("requested") == true &&
                    facts.at("route") == (admitted ? "PAIR_LOOP_TAIL" : "COMPACT_FALLBACK") &&
                    (admitted ? facts.at("fallback_reason").is_null()
                              : facts.at("fallback_reason") == "CAPACITY_NONFIT") &&
                    facts.at("residual_rows").get<uint64_t>() > 0 && !facts.at("runs").empty() &&
                    facts.at("capacity").at("accumulator_rows").get<uint64_t>() > 0,
                    "paired companion facts");
            table[stripe].admitted = admitted;
            table[stripe].reason = admitted ? "" : facts.at("fallback_reason").get<std::string>();
        } else if (kind == "NPU_CALL" && record.at("operation_id").get<uint64_t>() == paired.operation_id &&
                   record.value("call_kind", "") == "RESIDUAL_MERGE" &&
                   record.value("stage", "") == "COMPLETE_REQUIRED") {
            merges.push_back(record.at("required_work_ids").get<std::vector<uint64_t>>());
        }
    }
    size_t admitted = 0, rejected = 0, children = 0;
    for (const auto &required : merges) {
        bool owned = false;
        for (auto &[index, stripe] : mine) {
            const std::set<uint64_t> set(required.begin(), required.end());
            if ((stripe.admitted == true && set == std::set<uint64_t>{*stripe.work}) ||
                (stripe.admitted == false && !set.empty() && set == stripe.children)) {
                ++stripe.merges;
                owned = true;
            }
        }
        require(owned, "residual merge requires neither an admitted stripe work nor a fallback child");
    }
    std::map<uint64_t, size_t> bound;
    for (const auto &binding : paired.residual_bindings) ++bound[binding.stripe_id];
    for (const auto &[index, stripe] : mine) {
        require(stripe.work && stripe.admitted.has_value() && stripe.merges == 1,
                "paired parent stripe lacks its companion facts or merge");
        if (*stripe.admitted) {
            ++admitted;
            require(stripe.children.empty() && !bound.count(index),
                    "admitted paired stripe owns a residual child work");
        } else {
            ++rejected;
            children += stripe.children.size();
            require(!stripe.children.empty() && bound[index] == stripe.children.size(),
                    "fallback stripe lacks its bound residual child work");
        }
    }
    require(mine.size() == stripes && admitted >= 1 && rejected >= 1 &&
            paired.work_ids.size() == stripes + children,
            "paired parent must mix admitted and fallback stripes with exact work identities");
    size_t compact_children = 0;
    for (const auto &[index, stripe] : theirs) {
        require(stripe.work && !stripe.admitted.has_value() && !stripe.children.empty(),
                "compact parent stripe lacks its residual child work");
        compact_children += stripe.children.size();
    }
    require(theirs.size() == stripes && compact.residual_bindings.size() == compact_children,
            "compact parent binds one child per residual stripe");
    std::printf("PAIRED_TRACE_PASS stripes=%zu admitted=%zu fallback_capacity_nonfit=%zu "
                "fallback_children=%zu compact_children=%zu merges=%zu\n", stripes, admitted, rejected,
                children, compact_children, merges.size());
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
                mode == "exsia-rmd-pipeline" || mode == "exsia-rmd-cross-parent" ||
                mode == "exsia-paired-pipeline" ||
                mode == "residual" ||
                mode == "residual-runs" ||
                mode == "large-k", "invalid mode");
#if LOG_CYCLE
        require(ggml::gemmini::log::cycle.set_output_path("log/cycle-log.jsonl"),
                "existing CPU cycle-log path");
#endif
        const bool cross_parent = mode == "exsia-rmd-cross-parent";
        // P6.2: the same stripes run first on the compact route, then on the paired route
        // (second parent, same input); outputs must be bit-exact and the paired parent must
        // carry its residual inside Main's stripe work without a residual child.
        const bool paired_pipeline = mode == "exsia-paired-pipeline";
        const bool two_parents = cross_parent || paired_pipeline;
        const bool rmd_pipeline = mode == "exsia-rmd-pipeline" || two_parents;
        const bool exsia_pipeline = mode == "exsia-pipeline" || rmd_pipeline;
        Fixture fixture(cross_parent ? 321 : exsia_pipeline ? 161 : mode == "pipeline" ? 129 : 1,
                        cross_parent ? 321 : exsia_pipeline ? 161 : mode == "pipeline" ? 129 : 3,
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
        const auto first_context = fixture.args.cycle_sim_context;
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
            // Rows carrying outliers (residual rows). The paired mode puts them in every stripe and
            // makes stripe 1's residual three radix lanes wide, so its companion cannot fit the
            // accumulator slot beside Main while those of stripes 0 and 2 do.
            const size_t stripe_rows = fixture.args.activation_rows_per_stripe;
            const auto outlier_row = [&](size_t row) {
                return rmd_pipeline && (row < stripe_rows ||
                                        (paired_pipeline && row < fixture.args.I));
            };
            for (size_t row = 0; row < fixture.args.I; ++row) {
                if (!outlier_row(row)) continue;
                const bool wide = paired_pipeline && row >= stripe_rows && row < 2 * stripe_rows;
                input[row * fixture.args.K] = wide ? 4096.0f : 256.0f;
                input[row * fixture.args.K + 1] = 8.0f;
                input[row * fixture.args.K + 40] = -17.0f;
            }
            ggml_tensor activation_tensor{};
            activation_tensor.type = GGML_TYPE_F32;
            activation_tensor.data = input.data();
            const auto execute_exsia = [&](auto &meta) {
                auto started = ggml::gemmini::im2p_adapter::start_exsia_stripe_pipeline(fixture.args);
                require(started.result.ok(), started.result.message);
                require(started.pipeline && started.pipeline->install_sink().ok(),
                        "ExSIA pipeline sink install");
                ggml::gemmini::quants::act::exsia::ExSIA exsia;
                exsia.set_execution_mode(ggml::gemmini::quants::act::exsia::ExSIAState::ExecutionMode::Sequential);
                const bool quantized = exsia.run(meta, &activation_tensor, fixture.args,
                                                fixture.args.exsia_stripe_ready_sink);
                const auto result = started.pipeline->finish(quantized);
                require(result.result.ok(), result.result.message);
                require(std::all_of(fixture.output.begin(), fixture.output.end(),
                    [&, column = size_t{0}](float value) mutable {
                        const size_t row = column++ / fixture.args.J;
                        return outlier_row(row) ? value > 0.0f && value != -1.0f
                                                : value == float(fixture.args.K);
                    }), "ExSIA numerical publication");
            };
            execute_exsia(producer_meta);
            if (two_parents) {
                const auto first_output = fixture.output;
                metadata->execution(node, "Gemmini", "TARGET_NPU", true);
                session->finish_operation(first_context);
                semantic::capture_graph(graph);
                operation.semantic_context = semantic::context_for(node);
                fixture.args.cycle_sim_context = session->register_operation(node, operation, phase);
                cycle::ScopedContext second_context(fixture.args.cycle_sim_context);
                // Host-stage IDs from the fenced parent cannot belong to this operation.
                fixture.args.cycle_sim_host_dependencies.clear();
                std::fill(fixture.output.begin(), fixture.output.end(), -1.0f);
                if (cross_parent)
                    for (size_t row = 0; row < fixture.args.activation_rows_per_stripe; ++row)
                        input[row * fixture.args.K] = 512.0f;
                else
                    fixture.args.residual_route = ggml::gemmini::residual::ResidualRoute::ws_paired;
                auto &second_meta = fixture.args.act_quant.storage().emplace<
                    ggml::gemmini::quants::act::exsia::Meta>();
                execute_exsia(second_meta);
                if (cross_parent)
                    require(fixture.output != first_output, "second ExSIA parent reused first numerical result");
                else
                    require(fixture.output == first_output,
                            "paired route output differs from the compact route");
                metadata->execution(node, "Gemmini", "TARGET_NPU", true);
                session->finish_operation(fixture.args.cycle_sim_context);
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
        const auto expected_count_per_parent = mode == "pipeline" || exsia_pipeline
            ? (fixture.args.I + fixture.args.activation_rows_per_stripe - 1) /
                fixture.args.activation_rows_per_stripe : 1;
        const auto expected_count = expected_count_per_parent * (two_parents ? 2 : 1);
        const auto dense_observation_count = std::count_if(observations.begin(), observations.end(),
            [](const auto &observation) {
                return observation.geometry.scope == IM2P_GEOMETRY_STRIPE;
            });
        require(!observation_failed &&
                (rmd_pipeline
                    ? dense_observation_count == expected_count && observations.size() > expected_count
                    : observations.size() == expected_count),
                "independent descriptor observer count");
        if (cross_parent)
            require(std::any_of(observed_runs.begin(), observed_runs.end(), [](const auto &runs) {
                        return runs.size() >= 2 && runs[0].original_block_id == 0 &&
                            runs[1].original_block_id == 1 && runs[0].compact_k_count > 0 &&
                            runs[1].compact_k_count > 0;
                    }), "cross-parent producer lost two-run residual");
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
        if (!two_parents) session->finish_operation(fixture.args.cycle_sim_context);
        session->finish();
        if (mode == "pipeline" || exsia_pipeline) {
            const auto parents = session->producer_parents();
            require(parents.size() == (two_parents ? 2u : 1u), "pipeline parent count");
            for (const auto &parent : parents) {
                require(parent.geometry.scope == IM2P_GEOMETRY_STREAM &&
                        parent.geometry.m == fixture.args.I &&
                        parent.work_ids == parent.fence_required_work_ids &&
                        parent.work_ids.size() >= expected_count_per_parent,
                        "pipeline parent final geometry and fence work set");
                const bool paired_parent = paired_pipeline && &parent == &parents.back();
                if (paired_parent) {
                    // Admitted stripes own no child; the fallback stripe's children are checked
                    // against the trace in verify_paired_trace.
                    require(parent.work_ids.size() >= expected_count_per_parent,
                            "paired parent stripe works");
                } else if (rmd_pipeline) {
                    require(!parent.residual_bindings.empty(), "run-aware residual child work missing");
                    for (const auto &binding : parent.residual_bindings)
                        require(binding.dense_parent_id == parent.parent_id &&
                                binding.child_parent_id != binding.dense_parent_id &&
                                binding.stripe_id < expected_count_per_parent &&
                                binding.row_begin < binding.row_end &&
                                std::find(parent.work_ids.begin(), parent.work_ids.end(),
                                          binding.work_id) != parent.work_ids.end(),
                                "residual work lacks exact dense stripe and fence binding");
                }
            }
            if (two_parents)
                require(parents[0].operation_id != parents[1].operation_id &&
                        parents[0].parent_id != parents[1].parent_id &&
                        parents[0].fence_call_id < parents[1].fence_call_id &&
                        parents[0].work_ids.back() < parents[1].work_ids.front(),
                        "cross-parent work/fence identities are not ordered and distinct");
            if (paired_pipeline) verify_paired_trace(parents[0], parents[1], expected_count_per_parent);
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
            if (rmd_pipeline) {
                for (const auto &parent : parents) {
                    // A residual stripe's release requires its dense work, every residual child
                    // bound to it (none for an admitted paired companion) and its merge call.
                    const bool paired_parent = paired_pipeline && &parent == &parents.back();
                    const size_t checked = paired_pipeline ? expected_count_per_parent : 1;
                    for (size_t stripe = 0; stripe < checked; ++stripe) {
                        const auto released = std::find_if(ownership.begin(), ownership.end(),
                            [&](const auto &event) {
                                return event.operation_id == parent.operation_id &&
                                    event.event.kind == cycle::ProducerEventKind::FrontendCapacityRelease &&
                                    event.event.stripe_id == stripe;
                            });
                        const auto children = static_cast<size_t>(std::count_if(
                            parent.residual_bindings.begin(), parent.residual_bindings.end(),
                            [&](const auto &binding) { return binding.stripe_id == stripe; }));
                        require(released != ownership.end() && released->event.rmd_packet &&
                                (paired_parent || children > 0) &&
                                released->required_work_ids.size() == 1 + children &&
                                released->required_call_ids.size() == 1,
                                "residual capacity release requires actual compact work and merge call");
                    }
                }
                require(count(cycle::ProducerEventKind::ResidualHostMergeCompleted) >= parents.size(),
                        "residual merge call missing");
            }
            if (expected_count >= 3 && exsia_pipeline) {
                for (const auto &parent : parents) {
                    const auto first_release = std::find_if(ownership.begin(), ownership.end(), [&](const auto &event) {
                        return event.operation_id == parent.operation_id &&
                            event.event.kind == cycle::ProducerEventKind::ExsiaWorkspaceRelease &&
                            event.event.stripe_id == 0;
                    });
                    const auto third_acquire = std::find_if(ownership.begin(), ownership.end(), [&](const auto &event) {
                        return event.operation_id == parent.operation_id &&
                            event.event.kind == cycle::ProducerEventKind::ExsiaWorkspaceAcquire &&
                            event.event.stripe_id == 2;
                    });
                    require(first_release != ownership.end() && third_acquire != ownership.end() &&
                            first_release->sequence < third_acquire->sequence &&
                            first_release->event.workspace_slot == third_acquire->event.workspace_slot,
                            "ExSIA scratch slot zero reused only after source release");
                }
            }
            for (const auto &event : ownership) {
                require(std::any_of(parents.begin(), parents.end(), [&](const auto &parent) {
                            return event.operation_id == parent.operation_id;
                        }) &&
                        event.parent_id.has_value() && event.work_id.has_value() &&
                        event.event.workspace_slot == event.event.stripe_id % 2 &&
                        event.event.row_begin < event.event.row_end &&
                        event.event.source_location.find(':') != std::string::npos,
                        "pipeline producer identity/source binding");
            }
        }
        if (!two_parents) metadata->execution(node, "Gemmini", "TARGET_NPU", true);
        if (cross_parent)
            require(metadata->completed_graph_count() == 2, "cross-parent semantic graph count");
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
