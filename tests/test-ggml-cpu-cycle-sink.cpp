#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <gemmini/cycle_reader.hpp>
#include <gemmini/layer.h>
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
    std::array<char, 64> layer{};
    gemmini_get_layer("Qcur-7", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.qcur") return 11;
    gemmini_get_layer("Qcur-7 (reshaped)", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.qcur") return 14;
    gemmini_get_layer("cache_k_l7 (view) (copy of Kcur-7 (reshaped))",
                      layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.cache_k_l7") return 15;
    gemmini_get_layer("blk.3.attn.q_proj", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.3.attn.q_proj") return 12;
    gemmini_get_layer("inp_embd", layer.data(), layer.size());
    if (std::string(layer.data()) != "inp_embd") return 13;

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
    ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return 6;
    ggml_tensor * lhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * rhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * sum = ggml_add(context, lhs, rhs);
    ggml_set_name(sum, "attn_norm-7");
    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    ggml_build_forward_expand(graph, sum);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) return 7;
    const std::array<float, 4> left{1, 2, 3, 4};
    const std::array<float, 4> right{5, 6, 7, 8};
    ggml_backend_tensor_set(lhs, left.data(), 0, sizeof(left));
    ggml_backend_tensor_set(rhs, right.data(), 0, sizeof(right));
    ggml::gemmini::cycle::reset_read_count_for_test();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    const std::uint64_t sample_count = ggml::gemmini::cycle::read_count_for_test();
    gemmini_log_cycle_set_output(stderr);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    if (status != GGML_STATUS_SUCCESS) return 8;

    const std::string output = read_file(selected);
    const auto default_path = root / "work/output/log/cycle-log.jsonl";
    if (sample_count != 2 ||
        output.find("\"version\":2") == std::string::npos ||
        output.find("\"op\":\"cpu.add\"") == std::string::npos ||
        output.find("\"layer\":\"blk.7.attn_norm\"") == std::string::npos ||
        output.find("\"run_id\":null") != std::string::npos ||
        output.find("\"stripe_id\":null") == std::string::npos ||
        output.find("\"node_id\":0") == std::string::npos ||
        output.find("\"worker_id\":0") == std::string::npos ||
        output.find("\"stripe_id\":0") != std::string::npos ||
#if defined(__linux__) && defined(__aarch64__)
        output.find("\"source\":\"linux_perf_cpu_cycles\",\"unit\":\"cycle\"") == std::string::npos ||
#else
        output.find("\"source\":\"host_tick\",\"unit\":\"tick\"") == std::string::npos ||
#endif
        std::filesystem::exists(default_path)) return 9;
    if (!preserve) std::filesystem::remove_all(root, error);
    return error ? 10 : 0;
}
