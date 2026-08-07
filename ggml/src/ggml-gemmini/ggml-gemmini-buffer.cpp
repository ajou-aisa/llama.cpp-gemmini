#include "ggml-gemmini-buffer.hpp"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace ggml::gemmini {

struct q8_h1_artifact_entry {
    q8_h1_artifact_metadata metadata;
    size_t root_size = 0;
};

namespace {

std::atomic<uint64_t> next_buffer_generation { 1 };

uint64_t allocate_buffer_generation() {
    const uint64_t generation = next_buffer_generation.fetch_add(1, std::memory_order_relaxed);
    if (generation == 0) {
        GGML_ABORT("Gemmini buffer generation counter exhausted");
    }
    return generation;
}

bool checked_add(size_t left, size_t right, size_t & result) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_mul(size_t left, size_t right, size_t & result) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool positive_dimension_to_size(int64_t dimension, size_t & result) {
    if (dimension <= 0 ||
        static_cast<uint64_t>(dimension) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    result = static_cast<size_t>(dimension);
    return true;
}

bool tensor_span(const ggml_tensor * tensor, size_t & span) {
    if (tensor == nullptr || tensor->type < 0 || tensor->type >= GGML_TYPE_COUNT) {
        return false;
    }

    const int64_t block_size_i64 = ggml_blck_size(tensor->type);
    if (block_size_i64 <= 0) {
        return false;
    }

    const size_t block_size = static_cast<size_t>(block_size_i64);
    size_t first_dimension = 0;
    if (!positive_dimension_to_size(tensor->ne[0], first_dimension) || first_dimension % block_size != 0 ||
        !checked_mul(first_dimension / block_size, tensor->nb[0], span)) {
        return false;
    }

    for (int dimension_index = 1; dimension_index < GGML_MAX_DIMS; ++dimension_index) {
        size_t dimension = 0;
        size_t stride_span = 0;
        if (!positive_dimension_to_size(tensor->ne[dimension_index], dimension) ||
            !checked_mul(dimension - 1, tensor->nb[dimension_index], stride_span) ||
            !checked_add(span, stride_span, span)) {
            return false;
        }
    }

    return true;
}

q8_h1_layout_tag layout_tag_for(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE
        ? q8_h1_layout_tag::transposed_or_permuted
        : q8_h1_layout_tag::k_by_j;
}

void hash_combine(size_t & seed, size_t value) {
    seed ^= value + 0x9e3779b9U + (seed << 6) + (seed >> 2);
}

struct q8_h1_root_identity_hash {
    size_t operator()(const q8_h1_root_identity & identity) const noexcept {
        size_t result = std::hash<uint64_t> {}(identity.generation);
        hash_combine(result, std::hash<size_t> {}(identity.root_offset));
        hash_combine(result, std::hash<std::string> {}(identity.root_name));
        hash_combine(result, std::hash<int> {}(static_cast<int>(identity.type)));
        for (const int64_t dimension : identity.ne) {
            hash_combine(result, std::hash<int64_t> {}(dimension));
        }
        for (const size_t stride : identity.nb) {
            hash_combine(result, std::hash<size_t> {}(stride));
        }
        hash_combine(result, std::hash<uint8_t> {}(static_cast<uint8_t>(identity.layout)));
        return result;
    }
};

struct buffer_state {
    buffer_state(uint8_t * data, size_t size, uint64_t generation, bool owns_data) :
        data(data),
        size(size),
        generation(generation),
        owns_data(owns_data) {
    }

    uint8_t * data;
    size_t size;
    uint64_t generation;
    bool owns_data;
    std::mutex mutex;
    size_t acquisition_count = 0;
    std::unordered_map<q8_h1_root_identity, std::shared_ptr<q8_h1_artifact_entry>, q8_h1_root_identity_hash> artifacts;

    ~buffer_state() {
        if (owns_data) {
            ggml_aligned_free(data, size);
        }
    }
};

struct buffer_context {
    std::shared_ptr<buffer_state> state;
};

struct tensor_root_info {
    q8_h1_artifact_metadata metadata;
    size_t root_size;
    size_t tensor_offset;
    size_t tensor_size;
};

bool tensor_offset_in_buffer(
    const buffer_state & state,
    const ggml_tensor * tensor,
    size_t span,
    size_t & offset) {
    if (state.data == nullptr || tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    const uintptr_t base_address = reinterpret_cast<uintptr_t>(state.data);
    const uintptr_t tensor_address = reinterpret_cast<uintptr_t>(tensor->data);
    if (tensor_address < base_address) {
        return false;
    }

    const uintptr_t address_offset = tensor_address - base_address;
    if (address_offset > std::numeric_limits<size_t>::max()) {
        return false;
    }
    offset = static_cast<size_t>(address_offset);
    return offset <= state.size && span <= state.size - offset;
}

bool q8_h1_root_info(const buffer_state & state, const ggml_tensor * tensor, tensor_root_info & info) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_H1 || tensor->buffer == nullptr) {
        return false;
    }

    const ggml_tensor * root = tensor->view_src ? tensor->view_src : tensor;
    if (root->type != GGML_TYPE_Q8_H1 || root->view_src != nullptr || root->view_offs != 0 ||
        root->buffer != tensor->buffer) {
        return false;
    }

    size_t root_size = 0;
    size_t tensor_size = 0;
    size_t root_offset = 0;
    size_t tensor_offset = 0;
    if (!tensor_span(root, root_size) || !tensor_span(tensor, tensor_size) ||
        !tensor_offset_in_buffer(state, root, root_size, root_offset) ||
        !tensor_offset_in_buffer(state, tensor, tensor_size, tensor_offset)) {
        return false;
    }

    if (tensor != root &&
        (tensor->view_src != root || tensor_offset < root_offset || tensor_offset - root_offset != tensor->view_offs)) {
        return false;
    }

    size_t logical_k = 0;
    size_t global_j = 0;
    if (!positive_dimension_to_size(root->ne[0], logical_k) ||
        !positive_dimension_to_size(root->ne[1], global_j)) {
        return false;
    }

    const int64_t block_size_i64 = ggml_blck_size(root->type);
    if (block_size_i64 <= 0) {
        return false;
    }
    const size_t block_size = static_cast<size_t>(block_size_i64);
    if (logical_k % block_size != 0) {
        return false;
    }

    info.metadata.identity.generation = state.generation;
    info.metadata.identity.root_offset = root_offset;
    info.metadata.identity.root_name = root->name;
    info.metadata.identity.type = root->type;
    for (int dimension_index = 0; dimension_index < GGML_MAX_DIMS; ++dimension_index) {
        info.metadata.identity.ne[dimension_index] = root->ne[dimension_index];
        info.metadata.identity.nb[dimension_index] = root->nb[dimension_index];
    }
    info.metadata.identity.layout = layout_tag_for(root);
    info.metadata.logical_k = logical_k;
    info.metadata.global_j = global_j;
    info.metadata.blocks_per_row = logical_k / block_size;
    info.metadata.storage_bytes = root_size;
    info.root_size = root_size;
    info.tensor_offset = tensor_offset;
    info.tensor_size = tensor_size;
    return true;
}

bool compatible_logical_layout(const tensor_root_info & info, const ggml_tensor * tensor) {
    if (info.metadata.identity.layout != q8_h1_layout_tag::k_by_j ||
        tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE ||
        info.tensor_offset != info.metadata.identity.root_offset ||
        info.tensor_size != info.root_size) {
        return false;
    }

    for (int dimension_index = 0; dimension_index < GGML_MAX_DIMS; ++dimension_index) {
        if (tensor->ne[dimension_index] != info.metadata.identity.ne[dimension_index] ||
            tensor->nb[dimension_index] != info.metadata.identity.nb[dimension_index]) {
            return false;
        }
    }
    return true;
}

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

    if (!context->state->owns_data) {
        std::lock_guard<std::mutex> lock(context->state->mutex);
        // Host aliases have no mapping owner token. Destroying one while a handle exists would dangle root_data().
        if (context->state->acquisition_count != 0) {
            GGML_ABORT("Gemmini external buffer destroyed with outstanding Q8_H1 artifact handles");
        }
    }

    delete context;
    buffer->context = nullptr;
}

enum ggml_status gemmini_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_H1 || tensor->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }

    buffer_context * context = get_buffer_context(buffer);
    tensor_root_info info;
    if (context == nullptr || !q8_h1_root_info(*context->state, tensor, info)) {
        return GGML_STATUS_FAILED;
    }

    std::lock_guard<std::mutex> lock(context->state->mutex);
    auto artifact = context->state->artifacts.find(info.metadata.identity);
    if (artifact == context->state->artifacts.end()) {
        auto entry = std::make_shared<q8_h1_artifact_entry>();
        entry->metadata = std::move(info.metadata);
        entry->root_size = info.root_size;
        q8_h1_root_identity identity = entry->metadata.identity;
        context->state->artifacts.emplace(std::move(identity), std::move(entry));
    }
    return GGML_STATUS_SUCCESS;
}

void gemmini_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
    GGML_UNUSED(buffer);
}

void gemmini_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    GGML_UNUSED(buffer);
}

void gemmini_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
    GGML_UNUSED(buffer);
}

bool gemmini_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * destination) {
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
    /* .init_tensor     = */ gemmini_buffer_init_tensor,
    /* .memset_tensor   = */ gemmini_buffer_memset_tensor,
    /* .set_tensor      = */ gemmini_buffer_set_tensor,
    /* .get_tensor      = */ gemmini_buffer_get_tensor,
    /* .cpy_tensor      = */ gemmini_buffer_cpy_tensor,
    /* .clear           = */ gemmini_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t make_gemmini_buffer(
    ggml_backend_buffer_type_t buffer_type,
    void * data,
    size_t size,
    bool owns_data) {
    auto state = std::make_shared<buffer_state>(
        static_cast<uint8_t *>(data), size, allocate_buffer_generation(), owns_data);
    auto * context = new buffer_context { std::move(state) };
    return ggml_backend_buffer_init(buffer_type, gemmini_buffer_i, context, size);
}

const char * gemmini_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(buffer_type);
    return "GEMMINI";
}

ggml_backend_buffer_t gemmini_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
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

}

struct q8_h1_artifact_lease {
    explicit q8_h1_artifact_lease(std::shared_ptr<buffer_state> state) : state(std::move(state)) {
    }

    ~q8_h1_artifact_lease() {
        std::lock_guard<std::mutex> lock(state->mutex);
        GGML_ASSERT(state->acquisition_count != 0);
        --state->acquisition_count;
    }

    std::shared_ptr<buffer_state> state;
};

bool q8_h1_root_identity::operator==(const q8_h1_root_identity & other) const noexcept {
    return generation == other.generation &&
        root_offset == other.root_offset &&
        root_name == other.root_name &&
        type == other.type &&
        ne == other.ne &&
        nb == other.nb &&
        layout == other.layout;
}

q8_h1_artifact_handle::q8_h1_artifact_handle(
    q8_h1_artifact_metadata metadata,
    size_t root_size,
    std::shared_ptr<q8_h1_artifact_lease> lease) noexcept :
    metadata_(std::move(metadata)),
    root_size_(root_size),
    lease_(std::move(lease)) {
}

bool q8_h1_artifact_handle::valid() const noexcept {
    return lease_ != nullptr;
}

const q8_h1_artifact_metadata * q8_h1_artifact_handle::metadata() const noexcept {
    return valid() ? &metadata_ : nullptr;
}

const q8_h1_root_identity * q8_h1_artifact_handle::identity() const noexcept {
    const q8_h1_artifact_metadata * artifact_metadata = metadata();
    return artifact_metadata ? &artifact_metadata->identity : nullptr;
}

const void * q8_h1_artifact_handle::root_data() const noexcept {
    const q8_h1_root_identity * root_identity = identity();
    return root_identity ? lease_->state->data + root_identity->root_offset : nullptr;
}

size_t q8_h1_artifact_handle::root_size() const noexcept {
    return valid() ? root_size_ : 0;
}

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
    GGML_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return make_gemmini_buffer(gemmini_buffer_type(), ptr, size, false);
}

ggml_backend_buffer_t gemmini_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size) {
    GGML_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return make_gemmini_buffer(gemmini_buffer_type(device), ptr, size, false);
}

bool is_gemmini_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr &&
        buffer->iface.get_base == gemmini_buffer_get_base &&
        buffer->iface.init_tensor == gemmini_buffer_init_tensor;
}

uint64_t gemmini_buffer_generation(ggml_backend_buffer_t buffer) {
    buffer_context * context = get_buffer_context(buffer);
    return context ? context->state->generation : 0;
}

size_t q8_h1_artifact_registry_count(ggml_backend_buffer_t buffer) {
    buffer_context * context = get_buffer_context(buffer);
    if (context == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(context->state->mutex);
    return context->state->artifacts.size();
}

std::optional<q8_h1_artifact_handle> acquire_q8_h1_artifact(const ggml_tensor * tensor) {
    if (tensor == nullptr || !is_gemmini_buffer(tensor->buffer)) {
        return std::nullopt;
    }

    buffer_context * context = get_buffer_context(tensor->buffer);
    tensor_root_info info;
    if (context == nullptr || !q8_h1_root_info(*context->state, tensor, info) ||
        !compatible_logical_layout(info, tensor)) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(context->state->mutex);
    const auto artifact = context->state->artifacts.find(info.metadata.identity);
    if (artifact == context->state->artifacts.end()) {
        return std::nullopt;
    }

    q8_h1_artifact_metadata metadata = artifact->second->metadata;
    metadata.immutable = ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    auto lease = std::make_shared<q8_h1_artifact_lease>(context->state);
    ++context->state->acquisition_count;
    return q8_h1_artifact_handle(std::move(metadata), artifact->second->root_size, std::move(lease));
}

}
