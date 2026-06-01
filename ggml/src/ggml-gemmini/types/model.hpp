#pragma once

#include <cstdint>

namespace ggml::gemmini::types
{
enum class ModelType : uint8_t {
    unknown = 0,
    gpt2,
    llama_3_2_1b,
    gemma,
    qwen,
};

inline const char *to_string(ModelType t) {
    switch (t) {
        case ModelType::gpt2: return "gpt2";
        case ModelType::llama_3_2_1b: return "llama-3.2-1b";
        case ModelType::gemma: return "gemma";
        case ModelType::qwen:  return "qwen";
        default:               return "unknown";
    }
}
} // namespace ggml::gemmini::types
