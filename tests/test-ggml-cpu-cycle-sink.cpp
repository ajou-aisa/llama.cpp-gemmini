#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <gemmini/log.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return 1;
    const bool preserve = argc == 3 && std::string(argv[2]) == "--preserve";
    const std::filesystem::path root = std::filesystem::absolute(argv[1]);
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root / "work", error);
    if (error) return 2;
    std::filesystem::current_path(root / "work", error);
    if (error) return 3;
    const auto selected = root / "selected-cycle.jsonl";
    if (!gemmini_log_cycle_set_output_path(selected.c_str())) return 4;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return 6;
    ggml_tensor * lhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * rhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * sum = ggml_add(context, lhs, rhs);
    ggml_set_name(sum, "sink_probe");
    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    ggml_build_forward_expand(graph, sum);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) return 7;
    const std::array<float, 4> left{1, 2, 3, 4};
    const std::array<float, 4> right{5, 6, 7, 8};
    ggml_backend_tensor_set(lhs, left.data(), 0, sizeof(left));
    ggml_backend_tensor_set(rhs, right.data(), 0, sizeof(right));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    gemmini_log_cycle_set_output(stderr);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    if (status != GGML_STATUS_SUCCESS) return 8;

    const std::string output = read_file(selected);
    const auto default_path = root / "work/output/log/cycle-log.jsonl";
    if (output.find("\"name\":\"cpu.add\"") == std::string::npos ||
        std::filesystem::exists(default_path)) return 9;
    if (!preserve) std::filesystem::remove_all(root, error);
    return error ? 10 : 0;
}
