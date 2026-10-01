#include <gemmini/evaluation_metrics.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <limits>
#include <tuple>

namespace evaluation = ggml::gemmini::evaluation;
constexpr auto DENSE = evaluation::ScaleWorkType::Dense;
constexpr auto RESIDUAL = evaluation::ScaleWorkType::Residual;

static std::string read(const std::filesystem::path &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}

template<typename Callback>
static void rejects(Callback callback, const char *reason = "") {
    bool rejected = false;
    try { callback(); } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()).find(reason) != std::string::npos;
    }
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
    first->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 3, 3);
    first->scale_alignment(0, DENSE, 1, 1, 0.5, 0.5, 0, 0, 5);
    first->scale_alignment(0, RESIDUAL, 0, 0, 2.0, 0.5, 2, 7, 7);    // same coordinate, other work type
    first->scale_alignment(0, RESIDUAL, 1, 0, 0, 0.5, 0, 0, 11, true);
    rejects([&] { first->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 3, 3); });       // duplicate
    rejects([&] { first->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 3, 3, 3); });       // scale/offset
    rejects([&] { first->scale_alignment(0, DENSE, 3, 0, 8.0, 0.5, 4, 2, 3); });       // update count
    rejects([&] { first->scale_alignment(0, DENSE, 4, 0, NAN, 0.5, 0, 0, 3); });       // nonfinite
    rejects([&] { first->scale_alignment(0, DENSE, 5, 0, 0, 0.5, 1, 0, 3, true); });   // zero weight
    rejects([&] { first->scale_alignment(0, static_cast<evaluation::ScaleWorkType>(2), 6, 0, 0.5, 0.5, 0, 0, 3); });       // work type
    first->finish_activation();
    auto second = session->invocation("blk.1", 1, 32, nullptr);
    second->scale_alignment(0, DENSE, 0, 0, 32.0, 0.25, 7, 2, 2);
    second->finish_activation();
    session->chunk(1);
    auto third = session->invocation("blk.0", 1, 32, nullptr);
    third->scale_alignment(1, DENSE, 0, 0, 1.0, 0.25, 2, 4, 4);
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
static evaluation::Config scale_config(const std::filesystem::path &directory, const std::string &name) {
    evaluation::Config config;
    config.run_id = name;
    config.workload_id = "scale-mutations";
    config.manifest_sha256 = std::string(64, 'c');
    config.scale_path = (directory / (name + ".jsonl")).string();
    config.scale_aggregate = true;
    return config;
}
// Every invalid alignment still rejects in aggregate mode; coordinates differ by work type, stripe, column or block.
static void scale_mutation_cases(const std::filesystem::path &directory) {
    const auto config = scale_config(directory, "scale-mutations");
    auto session = evaluation::Session::start(config);
    session->chunk(0);
    auto invocation = session->invocation("blk.0", 1, 64, nullptr);
    invocation->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 3, 3);
    rejects([&] { invocation->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 3, 3); },
            "duplicate SCU alignment coordinate");
    invocation->scale_alignment(0, RESIDUAL, 0, 0, 8.0, 0.5, 4, 3, 3);   // same coordinate, other work type
    invocation->scale_alignment(1, DENSE, 0, 0, 8.0, 0.5, 4, 3, 3);      // other stripe
    invocation->scale_alignment(0, DENSE, 1, 0, 8.0, 0.5, 4, 3, 3);      // other column
    invocation->scale_alignment(0, DENSE, 0, 1, 8.0, 0.5, 4, 3, 3);      // other original block
    const auto invalid = [&](auto callback) { rejects(callback, "invalid SCU alignment record"); };
    invalid([&] { invocation->scale_alignment(0, static_cast<evaluation::ScaleWorkType>(2), 2, 0, 8.0, 0.5, 4, 3, 3); });
    invalid([&] { invocation->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 32768, 3, 3); });   // shift beyond SCU range
    invalid([&] { invocation->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 4, 4, 3); });       // updated > total
    invalid([&] { invocation->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 4, 0, 0); });       // no partial sums
    invalid([&] { invocation->scale_alignment(0, DENSE, 2, 2, 8.0, 0.5, 4, 3, 3); });       // block beyond K
    invalid([&] { invocation->scale_alignment(0, DENSE, 2, 0, INFINITY, 0.5, 4, 3, 3); }); // nonfinite
    const auto mismatch = [&](auto callback) { rejects(callback, "SCU scale/offset/count mismatch"); };
    mismatch([&] { invocation->scale_alignment(0, DENSE, 2, 0, 0, 0.5, 1, 0, 3, true); });    // zero weight, shift
    mismatch([&] { invocation->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 3, 3, 3); });        // inconsistent scale
    mismatch([&] { invocation->scale_alignment(0, DENSE, 2, 0, 8.0, 0.5, 4, 0, 3); });        // shifted, not updated
    mismatch([&] { invocation->scale_alignment(0, DENSE, 2, 0, 0.5, 0.5, 0, 3, 3); });        // updated, not shifted
    invocation->finish_activation();
    rejects([&] { invocation->finish_activation(); }, "duplicate quantization completion");
    // A sum that would wrap rejects and leaves the sums as they were.
    auto wide = session->invocation("blk.1", 1, 32, nullptr);
    wide->scale_alignment(0, DENSE, 0, 0, 0.5, 0.5, 0, 0, UINT64_MAX);
    rejects([&] { wide->scale_alignment(0, DENSE, 1, 0, 0.5, 0.5, 0, 0, 1); }, "SCU aggregate overflow");
    wide->finish_activation();
    session->chunk(1);
    // An invocation's sums are merged once at its chunk boundary; nothing joins them afterwards.
    rejects([&] { invocation->scale_alignment(0, DENSE, 3, 0, 8.0, 0.5, 4, 3, 3); },
            "SCU alignment after its chunk aggregate");
    session->finish(true);
    rejects([&] { session->finish(true); }, "session already finished");
    invocation.reset(); wide.reset(); session.reset();
    const auto data = read(config.scale_path);
    const auto has = [&](const std::string &fields) { return data.find(fields) != std::string::npos; };
    assert(has("\"chunk_id\":0,\"layer\":\"blk.0\",\"work_type\":\"DENSE\",\"delta_w_sum\":16,\"max_delta_w\":4,"
               "\"alignment_count\":4,\"updated_partial_sum_count\":12,\"total_partial_sum_count\":12,"
               "\"zero_weight_count\":0}"));
    assert(has("\"chunk_id\":0,\"layer\":\"blk.0\",\"work_type\":\"RESIDUAL\",\"delta_w_sum\":4,\"max_delta_w\":4,"
               "\"alignment_count\":1,\"updated_partial_sum_count\":3,\"total_partial_sum_count\":3,"
               "\"zero_weight_count\":0}"));
    assert(has("\"chunk_id\":0,\"layer\":\"blk.1\",\"work_type\":\"DENSE\",\"delta_w_sum\":0,\"max_delta_w\":0,"
               "\"alignment_count\":1,\"updated_partial_sum_count\":0,"
               "\"total_partial_sum_count\":18446744073709551615,\"zero_weight_count\":0}"));
    assert(has("\"success\":true,\"invocation_count\":2,\"observation_count\":3,\"alignment_count\":6,"
               "\"scale_invocation_count\":2}"));
    assert(data.find("\"chunk_id\":1") == std::string::npos);
}
// Concurrent producers of one invocation (stripe workers) under the invocation's SCU lock: each coordinate counted once.
static void scale_concurrency_case(const std::filesystem::path &directory) {
    const auto config = scale_config(directory, "scale-concurrent");
    auto session = evaluation::Session::start(config);
    session->chunk(0);
    auto invocation = session->invocation("blk.0", 1, 64, nullptr);
    std::vector<std::thread> workers;
    for (size_t stripe = 0; stripe < 4; ++stripe)
        workers.emplace_back([&, stripe] {
            for (size_t column = 0; column < 1000; ++column) {
                invocation->scale_alignment(stripe, DENSE, column, 0, 8.0, 0.5, 4, 3, 3);
                invocation->scale_alignment(stripe, RESIDUAL, column, 1, 0, 0.5, 0, 0, 2, true);
            }
        });
    for (auto &worker : workers) worker.join();
    invocation->finish_activation();
    session->finish(true);
    invocation.reset(); session.reset();
    const auto data = read(config.scale_path);
    assert(data.find("\"work_type\":\"DENSE\",\"delta_w_sum\":16000,\"max_delta_w\":4,\"alignment_count\":4000,"
                     "\"updated_partial_sum_count\":12000,\"total_partial_sum_count\":12000,\"zero_weight_count\":0}") !=
           std::string::npos);
    assert(data.find("\"work_type\":\"RESIDUAL\",\"delta_w_sum\":0,\"max_delta_w\":0,\"alignment_count\":4000,"
                     "\"updated_partial_sum_count\":0,\"total_partial_sum_count\":8000,\"zero_weight_count\":4000}") !=
           std::string::npos);
}
// Duplicate detection is exact on a grid with bitmap-word, bitmap-range and size_t edges: in any order, in both
// modes, each coordinate is accepted once and rejected as a duplicate the second time.
static void scale_coordinate_cases(const std::filesystem::path &directory) {
    using Coordinate = std::tuple<evaluation::ScaleWorkType, size_t, size_t, size_t>;  // type, stripe, block, column
    const size_t wide = size_t{1} << 20, top = std::numeric_limits<size_t>::max();
    std::vector<Coordinate> grid;
    for (const auto type : {DENSE, RESIDUAL})
        for (const size_t stripe : {size_t{0}, size_t{1}, top})
            for (const size_t block : {size_t{0}, size_t{1}})
                for (const size_t column : {size_t{0}, size_t{1}, size_t{63}, size_t{64}, size_t{65}, size_t{127},
                                            size_t{128}, wide - 1, wide, wide + 1, top / 2, top})
                    grid.emplace_back(type, stripe, block, column);
    for (const bool aggregate : {true, false}) {
        auto config = scale_config(directory, aggregate ? "scale-grid-aggregate" : "scale-grid-detailed");
        config.scale_aggregate = aggregate;
        auto session = evaluation::Session::start(config);
        session->chunk(0);
        auto invocation = session->invocation("blk.0", 1, 64, nullptr);
        auto order = grid;
        order.insert(order.end(), grid.begin(), grid.end());
        std::shuffle(order.begin(), order.end(), std::mt19937(aggregate ? 7 : 11));
        std::set<Coordinate> seen;
        for (const auto &coordinate : order) {
            const auto align = [&] {
                invocation->scale_alignment(std::get<1>(coordinate), std::get<0>(coordinate), std::get<3>(coordinate),
                                            std::get<2>(coordinate), 8.0, 0.5, 4, 3, 3);
            };
            if (seen.insert(coordinate).second) align();
            else rejects(align, "duplicate SCU alignment coordinate");
        }
        invocation->finish_activation();
        session->finish(true);
        invocation.reset(); session.reset();
        const auto data = read(config.scale_path);
        const auto half = std::to_string(grid.size() / 2);
        if (aggregate) {
            for (const char *type : {"DENSE", "RESIDUAL"})
                assert(data.find(std::string("\"work_type\":\"") + type + "\",\"delta_w_sum\":" +
                                 std::to_string(grid.size() / 2 * 4) + ",\"max_delta_w\":4,\"alignment_count\":" + half +
                                 ",") != std::string::npos);
            assert(data.find("\"alignment_count\":" + std::to_string(grid.size()) + ",\"scale_invocation_count\":1}") !=
                   std::string::npos);
        } else {
            assert(data.find("\"observation_count\":" + std::to_string(grid.size()) + "}") != std::string::npos);
            assert(data.find("\"column\":" + std::to_string(top)) != std::string::npos);
        }
    }
}
// Metric-only terminal lm_head: only a METRIC_PREFILL_256 session may elide it, and each elision is recorded.
static void terminal_lm_head_cases(const std::filesystem::path &directory) {
    auto config = scale_config(directory, "terminal-generation");
    config.workload_id = "E2E_GENERATION_256_128";
    config.terminal_lm_head_metrics_only = true;
    rejects([&] { evaluation::Session::start(config); }, "limited to the METRIC_PREFILL_256");
    assert(!std::filesystem::exists(config.scale_path));
    config = scale_config(directory, "terminal-full");
    config.workload_id = "METRIC_PREFILL_256";
    auto full = evaluation::Session::start(config);
    assert(!full->terminal_lm_head_metrics_only());
    rejects([&] { full->terminal_lm_head_elided(); }, "outside a metric-only session");
    full->finish(true);
    full.reset();
    config = scale_config(directory, "terminal-metrics-only");
    config.workload_id = "METRIC_PREFILL_256";
    config.terminal_lm_head_metrics_only = true;
    auto session = evaluation::Session::start(config);
    assert(session->terminal_lm_head_metrics_only() && session->terminal_lm_head_elisions() == 0);
    session->chunk(0);
    session->terminal_lm_head_elided();
    session->terminal_lm_head_elided();
    assert(session->terminal_lm_head_elisions() == 2);
    session->finish(true);
    rejects([&] { session->terminal_lm_head_elided(); }, "session already finished");
    session.reset();
    // Elision leaves the metric stream exactly as the observations made it.
    const auto stream = read(config.scale_path);
    assert(stream.find("elid") == std::string::npos && stream.find("logits") == std::string::npos);
}
// An invocation whose quantization never completed is never merged: it rejects the chunk boundary of a
// successful run and is left out of a failed one.
static void scale_incomplete_cases(const std::filesystem::path &directory) {
    for (const bool boundary : {true, false}) {
        const auto config = scale_config(directory, boundary ? "scale-incomplete-chunk" : "scale-incomplete-failed");
        auto session = evaluation::Session::start(config);
        session->chunk(0);
        auto complete = session->invocation("blk.0", 1, 32, nullptr);
        complete->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 2, 2);
        complete->finish_activation();
        auto partial = session->invocation("blk.1", 1, 32, nullptr);
        partial->scale_alignment(0, DENSE, 0, 0, 1.0, 0.25, 2, 4, 4);
        if (boundary) rejects([&] { session->chunk(1); }, "SCU aggregate of an incomplete invocation");
        else rejects([&] { session->finish(true); }, "incomplete invocation coverage");
        session->finish(false);
        rejects([&] { partial->scale_alignment(0, DENSE, 1, 0, 1.0, 0.25, 2, 4, 4); },
                "SCU alignment after its chunk aggregate");
        complete.reset(); partial.reset(); session.reset();
        const auto data = read(config.scale_path);
        assert(data.find("\"layer\":\"blk.1\"") == std::string::npos);
        assert((data.find("\"layer\":\"blk.0\"") != std::string::npos) == !boundary);
        assert(data.find("\"success\":false") != std::string::npos);
        assert(data.find(boundary ? "\"alignment_count\":0,\"scale_invocation_count\":0}"
                                  : "\"alignment_count\":1,\"scale_invocation_count\":1}") != std::string::npos);
    }
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
    invocation->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 2, 2);
    invocation->scale_alignment(0, DENSE, 1, 0, 0.5, 0.5, 0, 0, 2);
    invocation->scale_alignment(0, DENSE, 2, 0, 0, 0.5, 0, 0, 2, true);
    rejects([&] { invocation->scale_alignment(0, DENSE, 0, 0, 8.0, 0.5, 4, 2, 2); });
    rejects([&] { invocation->scale_alignment(0, DENSE, 0, 0, 8, 0.5, 3, 2, 2); });
    rejects([&] { invocation->scale_alignment(0, DENSE, 0, 0, 8, 0.5, 4, 1, 2); });
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
    scale_mutation_cases(directory);
    scale_coordinate_cases(directory);
    scale_concurrency_case(directory);
    terminal_lm_head_cases(directory);
    scale_incomplete_cases(directory);
#endif
#else
    assert(!session);
#endif
    return 0;
}
