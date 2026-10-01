#include <gemmini/evaluation_metrics.hpp>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <limits>

namespace evaluation = ggml::gemmini::evaluation;

static std::string read(const std::filesystem::path &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}

template<typename Callback>
static void rejects(Callback callback) {
    bool rejected = false;
    try { callback(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}

#if GGML_GEMMINI_ACT_QUANT_METRICS
static void reference_case(const std::filesystem::path &directory, const std::string &name,
                           size_t m, size_t k, const std::vector<float> &values,
                           uint64_t expected_fp, bool complete = true) {
    evaluation::Config config;
    config.run_id = name;
    config.manifest_sha256 = std::string(64, 'a');
    config.workload_id = "signed-row-original-block-test";
    config.activation_path = (directory / (name + ".jsonl")).string();
    auto session = evaluation::Session::start(config);
    session->chunk(0);
    auto invocation = session->invocation("layer", m, k, values.data());
    for (size_t row = 0; row < m; ++row)
        for (size_t column = 0; column < k; ++column)
            invocation->position(row, column, column == 0, column == 1);
    invocation->finish_activation();
    session->finish(true);
    const auto data = read(config.activation_path);
    assert(data.find("\"definition_status\":\"CONFIRMED_BY_USER\"") != std::string::npos);
    assert(data.find("signed-row-original-bk32-population-2sigma-v1") != std::string::npos);
    assert(data.find("\"valid_positions\":" + std::to_string(m * k)) != std::string::npos);
    assert(data.find("\"potal_selected\":" + std::to_string(m)) != std::string::npos);
    assert(data.find("\"residual_nnz\":" + std::to_string(m)) != std::string::npos);
    if (complete) {
        assert(data.find("\"fp_selected\":" + std::to_string(expected_fp)) != std::string::npos);
        assert(data.find("\"reference_complete\":true") != std::string::npos);
    } else {
        assert(data.find("\"fp_selected\":null") != std::string::npos);
        assert(data.find("\"intersection\":null") != std::string::npos);
        assert(data.find("\"union\":null") != std::string::npos);
        assert(data.find("\"reference_complete\":false") != std::string::npos);
        assert(data.find("NONFINITE_INPUT_UNSPECIFIED") != std::string::npos);
        assert(data.find("\"nonfinite_positions\":1") != std::string::npos);
    }
}

static void confirmed_reference_cases(const std::filesystem::path &directory) {
    reference_case(directory, "strict-equality", 1, 5, {0, 0, 0, 0, 10}, 0);
    reference_case(directory, "population-not-sample-divisor", 1, 6, {0, 0, 0, 0, 2, 10}, 1);
    reference_case(directory, "signed-not-absolute", 1, 8, {-100, 0, 0, 0, 0, 0, 0, 0}, 0);
    std::vector<float> values(32, 0);
    values[0] = 100;
    reference_case(directory, "top1-retained", 1, 32, values, 1);
    values.resize(64, 1000);
    values[0] = 10;
    reference_case(directory, "separate-original-blocks", 1, 64, values, 1);
    reference_case(directory, "separate-original-rows", 2, 32, values, 1);
    values.assign(38, 1000);
    values.resize(76, 0);
    values[38] = 10;
    reference_case(directory, "row-boundary-after-partial-block", 2, 38, values, 1);
    values.assign(37, 0);
    values.back() = 10;
    reference_case(directory, "tail-five-no-padding", 1, 37, values, 0);
    values.assign(38, 0);
    values.back() = 10;
    reference_case(directory, "tail-six-no-padding", 1, 38, values, 1);
    values.assign(33, 0);
    values.back() = 999;
    reference_case(directory, "singleton-tail", 1, 33, values, 0);
    values.assign(64, 0);
    reference_case(directory, "zero-variance", 2, 32, values, 0);
    values[0] = 100;
    values.back() = std::numeric_limits<float>::quiet_NaN();
    reference_case(directory, "nonfinite-block-invalidates-reference", 1, 64, values, 0, false);
    values.back() = std::numeric_limits<float>::infinity();
    reference_case(directory, "infinite-block-invalidates-reference", 1, 64, values, 0, false);
    evaluation::Config obsolete;
    obsolete.run_id = "obsolete";
    obsolete.workload_id = "obsolete";
    obsolete.activation_path = (directory / "obsolete.jsonl").string();
    obsolete.activation_reference_candidate = true;
    rejects([&] { evaluation::Session::start(obsolete); });
    assert(!std::filesystem::exists(obsolete.activation_path));
}
#endif

#if GGML_GEMMINI_SCALE_METRICS
// One SCU workload (2 chunks, 2 layers, 3 invocations, both work types, zero weight) in a given mode.
static std::string scale_workload(const std::filesystem::path &directory, const std::string &mode) {
    evaluation::Config config;
    config.run_id = "scale-" + mode;
    config.workload_id = "scale-workload";
    config.manifest_sha256 = std::string(64, 'b');
    config.scale_path = (directory / ("scale-" + mode + ".jsonl")).string();
    config.scale_aggregate = mode == "aggregate";
    auto session = evaluation::Session::start(config);
    session->chunk(0);
    auto first = session->invocation("blk.0", 1, 64, nullptr);
    first->scale_alignment(0, "DENSE", 0, 0, 8.0, 0.5, 4, 3, 3);
    first->scale_alignment(0, "DENSE", 1, 1, 0.5, 0.5, 0, 0, 5);
    first->scale_alignment(0, "RESIDUAL", 0, 0, 2.0, 0.5, 2, 7, 7);    // same coordinate, other work type
    first->scale_alignment(0, "RESIDUAL", 1, 0, 0, 0.5, 0, 0, 11, true);
    rejects([&] { first->scale_alignment(0, "DENSE", 0, 0, 8.0, 0.5, 4, 3, 3); });       // duplicate
    rejects([&] { first->scale_alignment(0, "DENSE", 2, 0, 8.0, 0.5, 3, 3, 3); });       // scale/offset
    rejects([&] { first->scale_alignment(0, "DENSE", 3, 0, 8.0, 0.5, 4, 2, 3); });       // update count
    rejects([&] { first->scale_alignment(0, "DENSE", 4, 0, NAN, 0.5, 0, 0, 3); });       // nonfinite
    rejects([&] { first->scale_alignment(0, "DENSE", 5, 0, 0, 0.5, 1, 0, 3, true); });   // zero weight
    rejects([&] { first->scale_alignment(0, "OTHER", 6, 0, 0.5, 0.5, 0, 0, 3); });       // work type
    first->finish_activation();
    auto second = session->invocation("blk.1", 1, 32, nullptr);
    second->scale_alignment(0, "DENSE", 0, 0, 32.0, 0.25, 7, 2, 2);
    second->finish_activation();
    session->chunk(1);
    auto third = session->invocation("blk.0", 1, 32, nullptr);
    third->scale_alignment(1, "DENSE", 0, 0, 1.0, 0.25, 2, 4, 4);
    third->finish_activation();
    session->finish(true);
    first.reset(); second.reset(); third.reset(); session.reset();
    return read(config.scale_path);
}
static void scale_aggregate_cases(const std::filesystem::path &directory) {
    const auto detailed = scale_workload(directory, "detailed");
    assert(detailed.find("\"kind\":\"AGGREGATE\"") == std::string::npos);
    assert(detailed.find("\"kind\":\"SCALE_ALIGNMENT\"") != std::string::npos);
    const auto aggregate = scale_workload(directory, "aggregate");
    assert(aggregate.find("SCALE_ALIGNMENT") == std::string::npos);
    assert(aggregate.find("\"schema\":\"im2p-scale-alignment-aggregate\"") != std::string::npos);
    assert(aggregate.find("\"collection_mode\":\"aggregate\"") != std::string::npos);
    const auto has = [&](const std::string &fields) { return aggregate.find(fields) != std::string::npos; };
    // Integer sums per (chunk, layer, work type); DENSE and RESIDUAL apart; zero weight counted, never updated.
    assert(has("\"chunk_id\":0,\"layer\":\"blk.0\",\"work_type\":\"DENSE\",\"delta_w_sum\":4,\"max_delta_w\":4,"
               "\"alignment_count\":2,\"updated_partial_sum_count\":3,\"total_partial_sum_count\":8,"
               "\"zero_weight_count\":0}"));
    assert(has("\"chunk_id\":0,\"layer\":\"blk.0\",\"work_type\":\"RESIDUAL\",\"delta_w_sum\":2,\"max_delta_w\":2,"
               "\"alignment_count\":2,\"updated_partial_sum_count\":7,\"total_partial_sum_count\":18,"
               "\"zero_weight_count\":1}"));
    assert(has("\"chunk_id\":0,\"layer\":\"blk.1\",\"work_type\":\"DENSE\",\"delta_w_sum\":7,\"max_delta_w\":7,"
               "\"alignment_count\":1,\"updated_partial_sum_count\":2,\"total_partial_sum_count\":2,"
               "\"zero_weight_count\":0}"));
    assert(has("\"chunk_id\":1,\"layer\":\"blk.0\",\"work_type\":\"DENSE\",\"delta_w_sum\":2,\"max_delta_w\":2,"
               "\"alignment_count\":1,\"updated_partial_sum_count\":4,\"total_partial_sum_count\":4,"
               "\"zero_weight_count\":0}"));
    assert(has("\"observation_count\":4,\"alignment_count\":6,\"scale_invocation_count\":3}"));
}
#endif

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    evaluation::Config config;
    config.run_id = "test-run";
    config.manifest_sha256 = std::string(64, 'a');
    config.workload_id = "test-workload";
#if !GGML_GEMMINI_SCALE_METRICS
    config.scale_path = (directory / "compiled-out-scale.jsonl").string();
    rejects([&] { evaluation::Session::start(config); });
    assert(!std::filesystem::exists(config.scale_path));
    config.scale_path.clear();
#else
    config.scale_path = (directory / "scale-alignment-metrics.jsonl").string();
#endif
#if !GGML_GEMMINI_ACT_QUANT_METRICS
    config.activation_path = (directory / "compiled-out-act.jsonl").string();
    rejects([&] { evaluation::Session::start(config); });
    assert(!std::filesystem::exists(config.activation_path));
    config.activation_path.clear();
#endif
#if !GGML_GEMMINI_RESIDUAL_METRICS
    config.residual_path = (directory / "compiled-out-res.jsonl").string();
    rejects([&] { evaluation::Session::start(config); });
    assert(!std::filesystem::exists(config.residual_path));
    config.residual_path.clear();
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
    config.activation_path = (directory / "activation-quant-metrics.jsonl").string();
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    config.residual_path = (directory / "residual-path-metrics.jsonl").string();
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    auto invalid = config;
    invalid.manifest_sha256.clear();
    rejects([&] { evaluation::Session::start(invalid); });
#endif
    auto session = evaluation::Session::start(config);
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    assert(session);
    session->chunk(4);
    const float values[] = {-100, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0};
    auto invocation = session->invocation("layer", 2, 8, values);
#if GGML_GEMMINI_SCALE_METRICS
    invocation->scale_alignment(0, "DENSE", 0, 0, 8.0, 0.5, 4, 2, 2);
    invocation->scale_alignment(0, "DENSE", 1, 0, 0.5, 0.5, 0, 0, 2);
    invocation->scale_alignment(0, "DENSE", 2, 0, 0, 0.5, 0, 0, 2, true);
    rejects([&] { invocation->scale_alignment(0, "DENSE", 0, 0, 8.0, 0.5, 4, 2, 2); });
    rejects([&] { invocation->scale_alignment(0, "DENSE", 0, 0, 8, 0.5, 3, 2, 2); });
    rejects([&] { invocation->scale_alignment(0, "DENSE", 0, 0, 8, 0.5, 4, 1, 2); });
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
    std::thread worker([&] {
        invocation->requantized(0, 0);
        invocation->requantized(0, 0);
        for (size_t index = 0; index < 16; ++index)
            invocation->position(index / 8, index % 8, index == 0, index == 3);
    });
    worker.join();
    rejects([&] { invocation->position(0, 0, false, false); });
    rejects([&] { invocation->position(0, 8, false, false); });
    rejects([&] { invocation->requantized(0, 1); });
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    invocation->main_stripe(0, 0, 1, 3, 8);
    invocation->main_stripe(1, 1, 1, 5, 8);
    invocation->radix_stripe(0, 2);
    invocation->radix_stripe(1, 0);
    invocation->compact_work(0, 1, 3, 2, 8, 1, 1, 1,
        {{0, 3, 0, 2}}, {{0, 0}}, 2, 1);
    rejects([&] { invocation->main_stripe(0, 0, 1, 3, 8); });
    rejects([&] { invocation->compact_work(0, 1, 3, 2, 8, 1, 1, 1,
        {{0, 3, 0, 2}}, {{0, 0}}); });
#endif
    invocation->finish_activation();
    session->finish(true);
#if GGML_GEMMINI_SCALE_METRICS
    const auto scale = read(config.scale_path);
    assert(scale.find("\"manifest_sha256\":\"" + config.manifest_sha256 + "\"") != std::string::npos);
    assert(scale.find("\"original_weight_scale\":8") != std::string::npos);
    assert(scale.find("\"aligned_pot_scale\":0.5") != std::string::npos);
    assert(scale.find("\"scu_shift_offset\":4") != std::string::npos);
    assert(scale.find("\"updated_partial_sum_count\":2") != std::string::npos);
    assert(scale.find("\"zero_weight\":true") != std::string::npos);
#endif
    rejects([&] { session->chunk(5); });
#if GGML_GEMMINI_ACT_QUANT_METRICS
    const auto act = read(config.activation_path);
    assert(act.find("\"valid_positions\":16") != std::string::npos);
    assert(act.find("\"fp_selected\":0") != std::string::npos);
    assert(act.find("\"potal_selected\":1") != std::string::npos);
    assert(act.find("\"intersection\":0") != std::string::npos);
    assert(act.find("\"union\":1") != std::string::npos);
    assert(act.find("\"residual_nnz\":1") != std::string::npos);
    assert(act.find("\"unique_actual_requantized_blocks\":1") != std::string::npos);
    assert(act.find("CONFIRMED_BY_USER") != std::string::npos);
#else
    assert(!std::filesystem::exists(directory / "activation-quant-metrics.jsonl"));
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    const auto res = read(config.residual_path);
    assert(res.find("MAIN_STRIPE") != std::string::npos);
    assert(res.find("COMPACT_WORK") != std::string::npos);
    assert(res.find("fp_selected") == std::string::npos);
    assert(res.find("original_k_mask") != std::string::npos);
    assert(res.find("\"radix_limb_count\":2") != std::string::npos);
    assert(res.find("\"zero_limb_pruned_count\":1") != std::string::npos);
#else
    assert(!std::filesystem::exists(directory / "residual-path-metrics.jsonl"));
#endif
    invocation.reset();
    session.reset();
    rejects([&] { evaluation::Session::start(config); });
#if GGML_GEMMINI_ACT_QUANT_METRICS
    confirmed_reference_cases(directory);
#endif
#if GGML_GEMMINI_SCALE_METRICS
    scale_aggregate_cases(directory);
#endif
#else
    assert(!session);
#endif
    return 0;
}
