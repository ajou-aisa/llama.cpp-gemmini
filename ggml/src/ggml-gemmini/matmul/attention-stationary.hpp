#pragma once

#include "ggml.h"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
#include "../../ggml-cuda/cpu-exact/ggml-cuda-potal-attention.h"
#endif

namespace ggml::gemmini {

// Absolute block scales survive V-tail changes; relative carriers do not.
struct AttentionBlock {
    std::array<int8_t, 32> low{}, high{};
    int16_t theta = 0;
    int64_t sum = 0;
};

struct AttentionStationary {
    size_t k = 0, n = 0, kp = 0;
    size_t column_stride = 0;
    size_t packed_columns() const { return column_stride ? column_stride : n; }
    std::vector<int16_t> base;
    std::vector<uint32_t> carriers;
    std::array<std::vector<int32_t>, 2> codes;
    std::vector<int64_t> colsum;
    std::vector<std::vector<AttentionBlock>> columns;
    bool packed_valid = false;
    uint64_t epoch = 0, revision = 0;
    uint32_t frontier = 0;
    const void * source = nullptr;
    ggml_type type = GGML_TYPE_F32;
    std::array<size_t, 4> strides{};
#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
    mutable std::unique_ptr<ggml_cuda_potal_attention_weights,
        decltype(&ggml_cuda_potal_attention_destroy)> gpu{nullptr, ggml_cuda_potal_attention_destroy};
#endif
};

struct AttentionCache {
    // Owned by one backend, rather than process-global/model-shared state.
    uint64_t instance = 0;
    std::map<std::tuple<uint32_t, size_t, size_t, bool>, AttentionStationary> entries;
};

struct AttentionPreparation {
    uint64_t quantized_blocks = 0, reused_blocks = 0;
};

struct AttentionOperand {
    const ggml_tensor & tensor;
    size_t head, batch;
    bool qk;
    bool reuse;
    size_t begin, end;
};

AttentionPreparation prepare_attention_stationary(
    const AttentionOperand & input, AttentionStationary & output, int workers);

} // namespace ggml::gemmini
