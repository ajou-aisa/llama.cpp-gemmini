#pragma once

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ggml::gemmini {

enum class q8_h1_layout_tag : uint8_t {
    k_by_j,
    transposed_or_permuted,
};

struct q8_h1_root_identity {
    uint64_t generation = 0;
    size_t root_offset = 0;
    std::string root_name;
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<size_t, GGML_MAX_DIMS> nb = {};
    q8_h1_layout_tag layout = q8_h1_layout_tag::k_by_j;

    bool operator==(const q8_h1_root_identity & other) const noexcept;
};

struct q8_h1_artifact_metadata {
    q8_h1_root_identity identity;
    size_t logical_k = 0;
    size_t global_j = 0;
    size_t blocks_per_row = 0;
    size_t storage_bytes = 0;
    bool immutable = false;
};

struct q8_h1_artifact_lease;

class q8_h1_artifact_handle {
public:
    q8_h1_artifact_handle() = default;

    bool valid() const noexcept;
    const q8_h1_artifact_metadata * metadata() const noexcept;
    const q8_h1_root_identity * identity() const noexcept;
    const void * root_data() const noexcept;
    size_t root_size() const noexcept;

private:
    q8_h1_artifact_handle(
        q8_h1_artifact_metadata metadata,
        size_t root_size,
        std::shared_ptr<q8_h1_artifact_lease> lease) noexcept;

    q8_h1_artifact_metadata metadata_;
    size_t root_size_ = 0;
    std::shared_ptr<q8_h1_artifact_lease> lease_;

    friend std::optional<q8_h1_artifact_handle> acquire_q8_h1_artifact(const ggml_tensor * tensor);
};

ggml_backend_buffer_type_t gemmini_buffer_type();
ggml_backend_buffer_type_t gemmini_buffer_type(ggml_backend_dev_t device);
// External aliases do not own ptr. Release all Q8_H1 handles before ggml_backend_buffer_free().
ggml_backend_buffer_t gemmini_buffer_from_host_ptr(void * ptr, size_t size);
ggml_backend_buffer_t gemmini_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size);

bool is_gemmini_buffer(ggml_backend_buffer_t buffer);
uint64_t gemmini_buffer_generation(ggml_backend_buffer_t buffer);
size_t q8_h1_artifact_registry_count(ggml_backend_buffer_t buffer);

std::optional<q8_h1_artifact_handle> acquire_q8_h1_artifact(const ggml_tensor * tensor);

}
