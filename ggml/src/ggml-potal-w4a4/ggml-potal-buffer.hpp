#pragma once

// The POTAL buffer type: ordinary aligned host memory (the CPU reads and writes
// it directly) with its own identity, so llama.cpp's model loader lists the
// POTAL device as an ACCEL weight buffer candidate (make_cpu_buft_list() skips
// an ACCEL device whose default buffer type is the CPU one).
//
//   owned    alloc_buffer: ggml_aligned_malloc, freed with the buffer.
//   alias    potal_buffer_from_host_ptr (the loader's mmap path): references
//            caller memory, never frees it; the caller keeps the mapping alive
//            until the buffer is released.
//   identity every buffer gets a nonzero generation, unique for the process,
//            so a buffer allocated later at a reused address is distinct.

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace ggml::potal {

ggml_backend_buffer_type_t potal_buffer_type(ggml_backend_dev_t device);

// nullptr if ptr is not TENSOR_ALIGNMENT-aligned.
ggml_backend_buffer_t potal_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size);

bool is_potal_buffer(ggml_backend_buffer_t buffer);

// 0 for a buffer that is not a (non-empty) POTAL buffer.
uint64_t potal_buffer_generation(ggml_backend_buffer_t buffer);

} // namespace ggml::potal
