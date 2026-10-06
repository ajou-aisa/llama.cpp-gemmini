#include "ggml-gemmini-buffer.hpp"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <utility>

namespace ggml::gemmini {

namespace {

std::atomic<uint64_t> next_buffer_generation{1};

uint64_t allocate_buffer_generation() {
    const uint64_t generation = next_buffer_generation.fetch_add(1, std::memory_order_relaxed);
    if (generation == 0) {
        GGML_ABORT("Gemmini buffer generation counter exhausted");
    }
    return generation;
}

struct buffer_state {
    buffer_state(uint8_t * data, size_t size, uint64_t generation, bool owns_data)
        : data(data), size(size), generation(generation), owns_data(owns_data) {}

    uint8_t * data;
    size_t    size;
    uint64_t  generation;
    bool      owns_data;

    ~buffer_state() {
        if (owns_data) {
            ggml_aligned_free(data, size);
        }
    }
};

struct buffer_context {
    std::shared_ptr<buffer_state> state;
};

buffer_context * get_buffer_context(ggml_backend_buffer_t buffer) {
    if (!is_gemmini_buffer(buffer)) {
        return nullptr;
    }
    return static_cast<buffer_context *>(buffer->context);
}

void * gemmini_buffer_get_base(ggml_backend_buffer_t buffer) {
    buffer_context * context = get_buffer_context(buffer);
    return context ? context->state->data : nullptr;
}

void gemmini_buffer_free(ggml_backend_buffer_t buffer) {
    buffer_context * context = get_buffer_context(buffer);
    if (context == nullptr) {
        return;
    }

    delete context;
    buffer->context = nullptr;
}

void gemmini_buffer_memset_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
    GGML_UNUSED(buffer);
}

void gemmini_buffer_set_tensor(ggml_backend_buffer_t buffer,
                               ggml_tensor *         tensor,
                               const void *          data,
                               size_t                offset,
                               size_t                size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    GGML_UNUSED(buffer);
}

void gemmini_buffer_get_tensor(ggml_backend_buffer_t buffer,
                               const ggml_tensor *   tensor,
                               void *                data,
                               size_t                offset,
                               size_t                size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
    GGML_UNUSED(buffer);
}

bool gemmini_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                               const ggml_tensor *   source,
                               ggml_tensor *         destination) {
    if (ggml_backend_buffer_is_host(source->buffer)) {
        std::memcpy(destination->data, source->data, ggml_nbytes(source));
        return true;
    }
    GGML_UNUSED(buffer);
    return false;
}

void gemmini_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    buffer_context * context = get_buffer_context(buffer);
    GGML_ASSERT(context != nullptr);
    std::memset(context->state->data, value, context->state->size);
}

const ggml_backend_buffer_i gemmini_buffer_i = {
    /* .free_buffer     = */ gemmini_buffer_free,
    /* .get_base        = */ gemmini_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ gemmini_buffer_memset_tensor,
    /* .set_tensor      = */ gemmini_buffer_set_tensor,
    /* .get_tensor      = */ gemmini_buffer_get_tensor,
    /* .cpy_tensor      = */ gemmini_buffer_cpy_tensor,
    /* .clear           = */ gemmini_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t make_gemmini_buffer(ggml_backend_buffer_type_t buffer_type,
                                          void *                     data,
                                          size_t                     size,
                                          bool                       owns_data) {
    auto state = std::make_shared<buffer_state>(
        static_cast<uint8_t *>(data), size, allocate_buffer_generation(), owns_data);
    auto * context = new buffer_context{std::move(state)};
    return ggml_backend_buffer_init(buffer_type, gemmini_buffer_i, context, size);
}

const char * gemmini_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(buffer_type);
    return "GEMMINI";
}

ggml_backend_buffer_t gemmini_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type,
                                                       size_t                     size) {
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return nullptr;
    }
    return make_gemmini_buffer(buffer_type, data, size, true);
}

size_t gemmini_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(buffer_type);
    return TENSOR_ALIGNMENT;
}

bool gemmini_buffer_type_is_host(ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(buffer_type);
    return true;
}

ggml_backend_buffer_type make_gemmini_buffer_type(ggml_backend_dev_t device) {
    return {
        /* .iface   = */ {
            /* .get_name         = */ gemmini_buffer_type_get_name,
            /* .alloc_buffer     = */ gemmini_buffer_type_alloc_buffer,
            /* .get_alignment    = */ gemmini_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ gemmini_buffer_type_is_host,
        },
        /* .device  = */ device,
        /* .context = */ nullptr,
    };
}

} // namespace

ggml_backend_buffer_type_t gemmini_buffer_type() {
    static ggml_backend_buffer_type buffer_type = make_gemmini_buffer_type(nullptr);
    return &buffer_type;
}

ggml_backend_buffer_type_t gemmini_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device != nullptr);
    static ggml_backend_buffer_type buffer_type = make_gemmini_buffer_type(device);
    GGML_ASSERT(buffer_type.device == device);
    return &buffer_type;
}

ggml_backend_buffer_t gemmini_buffer_from_host_ptr(void * ptr, size_t size) {
    GGML_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT == 0 &&
                "buffer pointer must be aligned");
    return make_gemmini_buffer(gemmini_buffer_type(), ptr, size, false);
}

ggml_backend_buffer_t
gemmini_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size) {
    GGML_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT == 0 &&
                "buffer pointer must be aligned");
    return make_gemmini_buffer(gemmini_buffer_type(device), ptr, size, false);
}

bool is_gemmini_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.get_base == gemmini_buffer_get_base;
}

uint64_t gemmini_buffer_generation(ggml_backend_buffer_t buffer) {
    buffer_context * context = get_buffer_context(buffer);
    return context ? context->state->generation : 0;
}

} // namespace ggml::gemmini
