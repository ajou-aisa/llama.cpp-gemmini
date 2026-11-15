// POTAL host buffer type, modelled on ggml-gemmini-buffer.cpp. See ggml-potal-buffer.hpp.

#include "ggml-potal-buffer.hpp"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "log.hpp"

#include <atomic>
#include <cstring>

namespace ggml::potal {

namespace {

std::atomic<uint64_t> next_generation{ 1 };

struct buffer_context {
    uint8_t * data;
    size_t    size;
    uint64_t  generation;
    bool      owns_data;
};

void * potal_buffer_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<buffer_context *>(buffer->context)->data;
}

void potal_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<buffer_context *>(buffer->context);
    if (context->owns_data) {
        ggml_aligned_free(context->data, context->size);
    }
    delete context;
    buffer->context = nullptr;
}

void potal_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset,
                                size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
    GGML_UNUSED(buffer);
}

void potal_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                             size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    GGML_UNUSED(buffer);
}

void potal_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                             size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
    GGML_UNUSED(buffer);
}

bool potal_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    GGML_UNUSED(buffer);
    return false;
}

void potal_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<buffer_context *>(buffer->context);
    std::memset(context->data, value, context->size);
}

const ggml_backend_buffer_i potal_buffer_i = {
    /* .free_buffer     = */ potal_buffer_free,
    /* .get_base        = */ potal_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ potal_buffer_memset_tensor,
    /* .set_tensor      = */ potal_buffer_set_tensor,
    /* .get_tensor      = */ potal_buffer_get_tensor,
    /* .cpy_tensor      = */ potal_buffer_cpy_tensor,
    /* .clear           = */ potal_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t make_buffer(ggml_backend_buffer_type_t buft, void * data, size_t size, bool owns_data) {
    const uint64_t generation = next_generation.fetch_add(1, std::memory_order_relaxed);
    GGML_ASSERT(generation != 0 && "POTAL buffer generation counter exhausted");
    auto * context = new buffer_context{ static_cast<uint8_t *>(data), size, generation, owns_data };
    return ggml_backend_buffer_init(buft, potal_buffer_i, context, size);
}

const char * potal_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "POTAL";
}

ggml_backend_buffer_t potal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        ggml::potal::log::failure("potal.backend", "[POTAL_BACKEND] %s: failed to allocate buffer of size %zu",
                                  __func__, size);
        return nullptr;
    }
    return make_buffer(buft, data, size, true);
}

size_t potal_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

bool potal_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

} // namespace

ggml_backend_buffer_type_t potal_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device != nullptr);
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name       = */ potal_buffer_type_get_name,
            /* .alloc_buffer   = */ potal_buffer_type_alloc_buffer,
            /* .get_alignment  = */ potal_buffer_type_get_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ potal_buffer_type_is_host,
        },
        /* .device  = */ device,
        /* .context = */ nullptr,
    };
    GGML_ASSERT(buft.device == device); // one POTAL device
    return &buft;
}

ggml_backend_buffer_t potal_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size) {
    if (reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT != 0) {
        ggml::potal::log::failure("potal.backend", "[POTAL_BACKEND] %s: host pointer %p is not %d-byte aligned",
                                  __func__, ptr, TENSOR_ALIGNMENT);
        return nullptr;
    }
    return make_buffer(potal_buffer_type(device), ptr, size, false);
}

bool is_potal_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.get_base == potal_buffer_get_base;
}

uint64_t potal_buffer_generation(ggml_backend_buffer_t buffer) {
    return is_potal_buffer(buffer) ? static_cast<buffer_context *>(buffer->context)->generation : 0;
}

} // namespace ggml::potal
