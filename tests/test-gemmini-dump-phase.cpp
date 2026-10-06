#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-gemmini.h"
#include "dump/dump_tensor.hpp"
#include "dump/scale/scale_dump_dispatch.hpp"
#include <gemmini/performance.hpp>
#include "../ggml/src/ggml-gemmini-utils/src/trace-metadata.hpp"
#include "json.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace perf = ggml::gemmini::performance;
namespace log  = ggml::gemmini::log;
static void check(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Graph {
    ggml_backend_t        backend = ggml_backend_gemmini_init();
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         out     = nullptr;
    int                   rows;
    explicit Graph(int n) : rows(n) {
        check(backend != nullptr, "backend allocation");
        ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
        check(ctx != nullptr, "context allocation");
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, rows);
        ggml_set_name(x, "attn_norm-0");
        out = ggml_mul_mat(ctx, w, x);
        ggml_set_name(out, "phase\"\\fixture");
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "tensor allocation");
        std::vector<float> weights(64, 1.0f), input(32 * rows, 2.0f);
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    }
    ~Graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    void run(const char * phase, uint64_t step, uint64_t request, uint64_t operation) {
        const auto context = perf::capture_context();
        check(context.request_id == request && context.operation_id == operation,
              "coherent context");
        if (operation) {
            check(std::string(perf::phase_name(context.phase)) == phase, "captured phase");
            check(context.decode_ordinal == (std::string(phase) == "decode" ? step - 1 : 0),
                  "captured ordinal");
            const std::string expected = "{\"request_id\":" + std::to_string(request) +
                                         ",\"operation_id\":" + std::to_string(operation) +
                                         ",\"phase\":\"" + phase + "\",\"included\":true}";
            check(perf::serialize_context(context) == expected, "context byte schema");
        }
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> result(2 * rows);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        for (float value : result)
            check(std::fabs(value - 64.0f) < 0.001f, "numerical result");
        const auto dump = log::dump_get_context();
        check(dump.graph_max_I == static_cast<uint32_t>(rows), "actual max I");
        log::dump_tensor("phase-fixture", out);
        std::fflush(nullptr);
        std::ifstream input(log::resolve_output_path("dump-phase.jsonl"));
        std::string   line, last;
        while (std::getline(input, line))
            if (!line.empty())
                last = line;
        const auto row = nlohmann::json::parse(last);
        std::printf("phase=%s step=%llu request=%llu operation=%llu maxI=%u\n",
                    row.at("phase").get<std::string>().c_str(),
                    static_cast<unsigned long long>(row.at("step_id").get<uint64_t>()),
                    static_cast<unsigned long long>(request),
                    static_cast<unsigned long long>(operation),
                    dump.graph_max_I);
        check(row.at("phase") == phase && row.at("step_id") == step, "dump phase/step");
        check(row.at("tensor") == "phase\"\\fixture", "dump escaping");
    }
};
#if LOG_DUMP_SCALE && !defined(_WIN32)
static nlohmann::json read_last(const char * path) {
    std::ifstream input(log::resolve_output_path(path));
    std::string   line, last;
    while (std::getline(input, line))
        if (!line.empty())
            last = line;
    return nlohmann::json::parse(last);
}
static void check_dump_outputs() {
    Graph       graph(1);
    const float values[] = {3.25f, -2.0f};
    ggml_backend_tensor_set(graph.out, values, 0, sizeof(values));
    check(log::dump_tensor.set_output_path("tensor-default.jsonl"), "tensor default path");
    log::dump_tensor("escaped\"layer", graph.out);
    std::fflush(nullptr);
    const auto first = read_last("tensor-default.jsonl");
    check(first.at("data") == nlohmann::json::array({{3.25, -2.0}}), "default tensor values");
    check(first.at("layer") == "escaped\"layer", "default tensor escaping");
    auto descriptors = [] {
        int count = 0;
        for (int fd = 0; fd < 256; ++fd)
            count += fcntl(fd, F_GETFD) != -1;
        return count;
    };
    const int before = descriptors();
    for (int i = 0; i < 10; ++i)
        log::dump_tensor(log::file("nested/tensor.jsonl"), "target", graph.out);
    check(descriptors() == before, "per-call tensor file close");
    check(read_last("nested/tensor.jsonl").at("data") == first.at("data"), "target tensor values");
    log::dump_tensor(log::file("../escape.jsonl"), "invalid", graph.out);
    check(!std::filesystem::exists("escape.jsonl"), "invalid tensor target rejected");
    std::ofstream(log::resolve_output_path("blocked")) << "file, not directory";
    log::dump_tensor(log::file("blocked/tensor.jsonl"), "failed", graph.out);
    check(!std::filesystem::exists(log::resolve_output_path("blocked/tensor.jsonl")),
          "tensor open failure");
    check(!log::dump_tensor.set_output_path("../invalid-default.jsonl"),
          "invalid default rejected");
    log::dump_tensor.set_output(nullptr);
    log::dump_tensor("disabled", graph.out);

    namespace scale = ggml::gemmini::log::scale;
    scale::DumpMeta meta{};
    meta.layer = "scale\"layer";
    meta.K     = 64;
    meta.I     = 1;
    meta.J     = 2;
    scale::ScaleTableView  view{values, 1, 2, 32};
    scale::GroupDumpConfig config{};
    const int              scale_before = descriptors();
    for (int i = 0; i < 10; ++i) {
        const auto result =
            scale::dump_scale_groups(log::file("nested/scale.jsonl"), meta, view, config);
        check(result.success && result.value_count == 2, "scale result");
    }
    check(descriptors() == scale_before, "per-call scale file close");
    const auto scale_row = read_last("nested/scale.jsonl");
    check(scale_row.at("layer") == "scale\"layer", "scale escaping");
    check(scale_row.at("data") == first.at("data"), "scale values");
    check(!scale::dump_scale_groups(log::file("blocked/scale.jsonl"), meta, view, config).success,
          "scale open failure");
    check(!scale::dump_scale_groups(log::file("../escape-scale.jsonl"), meta, view, config).success,
          "invalid scale target rejected");
    const auto path    = log::resolve_output_path("scale-default.jsonl");
    FILE *     capture = std::fopen(path.c_str(), "w");
    check(capture != nullptr, "scale capture");
    const int saved = dup(STDERR_FILENO);
    check(saved >= 0 && dup2(fileno(capture), STDERR_FILENO) >= 0, "scale redirect");
    const auto default_result = scale::dump_scale_groups({}, meta, view, config);
    std::fflush(stderr);
    const int restored = dup2(saved, STDERR_FILENO);
    close(saved);
    std::fclose(capture);
    check(restored >= 0 && default_result.success, "scale default write");
    check(read_last("scale-default.jsonl") == scale_row, "default and targeted scale equality");
    std::puts("PASS tensor/scale default,target,escaping,values,close,invalid target");
}
#endif
int main(int argc, char ** argv) try {
#if LOG_DUMP_SCALE && !defined(_WIN32)
    if (argc == 2 && std::string(argv[1]) == "--outputs") {
        check_dump_outputs();
        return 0;
    }
#else
    (void)argc;
    (void)argv;
#endif
    check(log::dump_tensor.set_output_path("dump-phase.jsonl"), "dump output");
    perf::Measurement measurement;
    measurement.kind            = perf::Measurement::Kind::cpu;
    measurement.sequence        = 901;
    measurement.cycles          = 0;
    const auto           legacy = perf::serialize_measurement(measurement);
    const nlohmann::json fields = {{"inference_context", nullptr},
                                   {"operator_context", {{"segment_id", 42}}}};
    auto decorated = nlohmann::json::parse(perf::serialize_measurement(measurement, fields));
    check(decorated.at("cycles") == 0, "zero measurement");
    check(decorated.at("inference_context").is_null(), "captured empty measurement context");
    check(decorated.at("operator_context").at("segment_id") == 42, "measurement metadata hook");
    decorated.erase("operator_context");
    decorated.erase("inference_context");
    check(decorated.dump() == legacy, "measurement legacy bytes");
    std::ofstream("measurement-legacy.json") << legacy << '\n';
    perf::reset();
    Graph a(1), b(3);
    perf::begin_operation(perf::Phase::decode, 1);
    a.run("unknown", 0, 0, 0);
    for (int request = 0; request < 2; ++request) {
        perf::start_request(100);
        const auto id = perf::capture_context().request_id;
        perf::begin_operation(perf::Phase::prefill, 101);
        auto op = perf::capture_context().operation_id;
        a.run("prefill", 1, id, op);
        b.run("prefill", 1, id, op);
        perf::end_operation(102, true);
        a.run("unknown", 0, id, 0);
        perf::begin_operation(perf::Phase::decode, 103);
        op = perf::capture_context().operation_id;
        a.run("decode", 2, id, op);
        b.run("decode", 2, id, op);
        perf::end_operation(104, true);
        perf::begin_operation(perf::Phase::decode, 105);
        op = perf::capture_context().operation_id;
        b.run("decode", 3, id, op);
        perf::end_operation(106, false);
        perf::finish_request(107);
        b.run("unknown", 0, 0, 0);
    }
    perf::finish_recording();
    std::puts("PASS real graph request phases");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
