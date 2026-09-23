#include <gemmini/evaluation_metrics.hpp>

#include <cassert>
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

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    evaluation::Config config;
    config.run_id = "test-run";
    config.workload_id = "test-workload";
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
    auto session = evaluation::Session::start(config);
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
    assert(session);
    session->chunk(4);
    const float values[] = {-100, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0};
    auto invocation = session->invocation("layer", 2, 8, values);
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
    invocation->compact_work(0, 1, 3, 2, 8, 1, 1, 1,
        {{0, 3, 0, 2}}, {{0, 0}});
    rejects([&] { invocation->main_stripe(0, 0, 1, 3, 8); });
    rejects([&] { invocation->compact_work(0, 1, 3, 2, 8, 1, 1, 1,
        {{0, 3, 0, 2}}, {{0, 0}}); });
#endif
    invocation->finish_activation();
    session->finish(true);
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
#else
    assert(!std::filesystem::exists(directory / "residual-path-metrics.jsonl"));
#endif
    invocation.reset();
    session.reset();
    rejects([&] { evaluation::Session::start(config); });
#if GGML_GEMMINI_ACT_QUANT_METRICS
    confirmed_reference_cases(directory);
#endif
#else
    assert(!session);
#endif
    return 0;
}
