#pragma once

#include "ggml-backend.h"
#include "ggml-metal-quantized.h"
#include "json.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

struct evaluation_metal {
    using stats_fn = ggml_metal_quantized_stats (*)();
    using reset_fn = void (*)();
    using placement_key = std::tuple<std::string, std::string, std::string, std::string>;

    stats_fn stats = nullptr;
    reset_fn reset = nullptr;
    std::map<placement_key, uint64_t> placements;
    uint64_t observed = 0;
    uint64_t on_metal = 0;
    uint64_t prefill = 0;
    uint64_t decode = 0;
    bool decoding = false;

    void initialize() {
        auto reg = ggml_backend_reg_by_name("Metal");
        if (!reg) throw std::runtime_error("Metal backend unavailable");
        stats = reinterpret_cast<stats_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_metal_quantized_get_stats"));
        reset = reinterpret_cast<reset_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_metal_quantized_reset_stats"));
        if (!stats || !reset) throw std::runtime_error("Metal quantized execution counters unavailable");
    }

    void begin() {
        reset();
        placements.clear();
        observed = on_metal = prefill = decode = 0;
        decoding = false;
    }

    static std::string device(const ggml_tensor * tensor) {
        auto buffer = tensor->buffer;
        if (!buffer && tensor->view_src) buffer = tensor->view_src->buffer;
        if (!buffer) return "unallocated";
        auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
        return dev ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) : "unknown";
    }

    static bool observe(ggml_tensor * tensor, bool ask, void * user) {
        if (!ask) return true;
        if ((tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_MUL_MAT_ID) ||
                !tensor->src[0] || !ggml_is_quantized(tensor->src[0]->type)) return false;
        auto & state = *static_cast<evaluation_metal *>(user);
        const auto input = device(tensor->src[0]);
        const auto output = device(tensor);
        ++state.observed;
        ++(state.decoding ? state.decode : state.prefill);
        state.on_metal += input == "Metal" && output == "Metal";
        ++state.placements[{tensor->src[0]->name, ggml_type_name(tensor->src[0]->type), input, output}];
        return false;
    }

    nlohmann::ordered_json evidence(bool complete) const {
        const auto counters = stats();
        const uint64_t completed = counters.block_calls + counters.hp1_calls;
        auto tensors = nlohmann::ordered_json::array();
        for (const auto & entry : placements) {
            tensors.push_back({{"weight", std::get<0>(entry.first)}, {"type", std::get<1>(entry.first)},
                {"weight_backend", std::get<2>(entry.first)}, {"output_backend", std::get<3>(entry.first)},
                {"calls", entry.second}});
        }
        const bool verified = complete && observed > 0 && observed == on_metal && observed == completed &&
            counters.failed_calls == 0 && counters.fallback_calls == 0 &&
            counters.dense_launches == completed && counters.merge_launches == counters.residual_launches &&
            (counters.block_calls == 0 || counters.residual_launches == 0);
        return {{"schema", "metal-quantized-execution"}, {"version", 1}, {"backend", "Metal"},
            {"producer", "cpu"}, {"complete", complete}, {"placement_verified", verified},
            {"activation_bits", GGML_GEMMINI_ACTIVATION_BITS}, {"weight_bits", GGML_GEMMINI_WEIGHT_BITS},
            {"dim", GGML_GEMMINI_DIM}, {"activation_mode", EVALUATION_ACTIVATION_MODE},
            {"rmd_enabled", GGML_GEMMINI_ENABLE_RMD != 0},
            {"observed_quantized_matmuls", observed}, {"metal_quantized_matmuls", on_metal},
            {"prefill_matmuls", prefill}, {"decode_matmuls", decode},
            {"block_calls", counters.block_calls}, {"hp1_calls", counters.hp1_calls},
            {"dense_launches", counters.dense_launches}, {"residual_launches", counters.residual_launches},
            {"merge_launches", counters.merge_launches}, {"failed_calls", counters.failed_calls},
            {"fallback_calls", counters.fallback_calls}, {"tensor_placement", tensors},
            {"dense_timing_includes", counters.block_calls > 0 ? "BLOCK_DOT_CORRECTION_AND_RESTORATION" : "HP1_SCU_AND_DENSE_RESTORATION"},
            {"residual_merge_policy", "ONE_MERGE_PER_HP1_RESIDUAL_REQUEST"},
            {"scheduler_observer", "ask_only_with_split_synchronization"},
            {"seconds", {{"producer", counters.producer_seconds}, {"transfer", counters.transfer_seconds},
                {"dense_gpu", counters.dense_gpu_seconds}, {"residual_gpu", counters.residual_gpu_seconds},
                {"merge_gpu", counters.merge_gpu_seconds}, {"matmul_total", counters.total_seconds}}}};
    }
};
