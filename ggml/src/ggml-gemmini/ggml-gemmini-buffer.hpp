#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini {

ggml_backend_buffer_type_t gemmini_buffer_type();
ggml_backend_buffer_type_t gemmini_buffer_type(ggml_backend_dev_t device);
// External aliases do not own ptr; the caller keeps the mapping alive until buffer release.
ggml_backend_buffer_t gemmini_buffer_from_host_ptr(void * ptr, size_t size);
ggml_backend_buffer_t
gemmini_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size);

bool     is_gemmini_buffer(ggml_backend_buffer_t buffer);
uint64_t gemmini_buffer_generation(ggml_backend_buffer_t buffer);

} // namespace ggml::gemmini
