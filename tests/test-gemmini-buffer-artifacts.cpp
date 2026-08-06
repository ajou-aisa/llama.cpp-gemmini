#include <ggml.h>
#include <ggml-backend.h>

#include "../ggml/src/ggml-gemmini/ggml-gemmini-buffer.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr int64_t kLogicalK = 64;
constexpr int64_t kGlobalJ = 3;
constexpr size_t kContextTensorSlots = 32;
constexpr size_t kAliasStorageBytes = 4096;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

ggml_context * make_context() {
    const ggml_init_params params = {
        kContextTensorSlots * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    return ggml_init(params);
}

ggml_tensor * make_q8_h1_tensor(ggml_context * context, int64_t global_j, const char * name) {
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_Q8_H1, kLogicalK, global_j);
    if (tensor != nullptr) {
        ggml_set_name(tensor, name);
    }
    return tensor;
}

bool allocate_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, size_t offset = 0) {
    if (buffer == nullptr || tensor == nullptr) {
        return false;
    }

    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
    return ggml_backend_tensor_alloc(buffer, tensor, base + offset) == GGML_STATUS_SUCCESS;
}

bool initialize_view(ggml_tensor * tensor) {
    return tensor != nullptr && ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS;
}

bool check_artifact_metadata(
    const ggml::gemmini::q8_h1_artifact_handle & handle,
    const ggml_tensor * root,
    ggml_backend_buffer_t buffer,
    bool immutable,
    const char * label) {
    const auto * metadata = handle.metadata();
    const auto * identity = handle.identity();
    if (!check(handle.valid(), label) ||
        !check(metadata != nullptr, "artifact metadata is present") ||
        !check(identity != nullptr, "artifact identity is present")) {
        return false;
    }

    const auto * base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(buffer));
    const auto * root_data = static_cast<const uint8_t *>(root->data);
    const size_t expected_offset = static_cast<size_t>(root_data - base);
    const size_t block_size = static_cast<size_t>(ggml_blck_size(GGML_TYPE_Q8_H1));
    bool ok = true;
    ok = check(metadata->identity.generation == ggml::gemmini::gemmini_buffer_generation(buffer), "metadata generation matches buffer") && ok;
    ok = check(metadata->identity.root_offset == expected_offset, "metadata root offset matches allocation") && ok;
    ok = check(metadata->identity.root_name == root->name, "metadata root name matches tensor") && ok;
    ok = check(metadata->identity.type == GGML_TYPE_Q8_H1, "metadata type is Q8_H1") && ok;
    ok = check(metadata->identity.ne[0] == kLogicalK && metadata->identity.ne[1] == root->ne[1], "metadata shape matches root") && ok;
    ok = check(metadata->identity.nb[0] == root->nb[0] && metadata->identity.nb[1] == root->nb[1], "metadata strides match root") && ok;
    ok = check(metadata->identity.layout == ggml::gemmini::q8_h1_layout_tag::k_by_j, "metadata layout is K-by-J") && ok;
    ok = check(metadata->logical_k == static_cast<size_t>(kLogicalK), "metadata logical_k matches") && ok;
    ok = check(metadata->global_j == static_cast<size_t>(root->ne[1]), "metadata global_j matches") && ok;
    ok = check(metadata->blocks_per_row == static_cast<size_t>(kLogicalK) / block_size, "metadata blocks_per_row matches") && ok;
    ok = check(metadata->storage_bytes == ggml_nbytes(root), "metadata storage_bytes matches") && ok;
    ok = check(metadata->immutable == immutable, "metadata immutable matches usage") && ok;
    ok = check(handle.root_data() == root->data, "artifact root_data aliases root tensor") && ok;
    ok = check(handle.root_size() == ggml_nbytes(root), "artifact root_size matches root tensor") && ok;
    return ok;
}

bool same_metadata(
    const ggml::gemmini::q8_h1_artifact_handle & left,
    const ggml::gemmini::q8_h1_artifact_handle & right) {
    const auto * left_metadata = left.metadata();
    const auto * right_metadata = right.metadata();
    return left_metadata != nullptr && right_metadata != nullptr &&
        left_metadata->identity == right_metadata->identity &&
        left_metadata->logical_k == right_metadata->logical_k &&
        left_metadata->global_j == right_metadata->global_j &&
        left_metadata->blocks_per_row == right_metadata->blocks_per_row &&
        left_metadata->storage_bytes == right_metadata->storage_bytes &&
        left_metadata->immutable == right_metadata->immutable;
}

bool test_owned_buffer_io_and_destruction() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "owned test context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "owned-root");
    ggml_tensor * copy = make_q8_h1_tensor(context, kGlobalJ, "owned-copy");
    if (!check(root != nullptr && copy != nullptr, "owned test tensors initialize")) {
        ggml_free(context);
        return false;
    }

    const size_t alignment = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    const size_t slot_size = align_up(ggml_nbytes(root), alignment);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), 2 * slot_size);
    if (!check(buffer != nullptr, "owned Gemmini buffer allocates")) {
        ggml_free(context);
        return false;
    }

    bool ok = true;
    ok = check(ggml::gemmini::is_gemmini_buffer(buffer), "owned buffer identifies as Gemmini") && ok;
    ok = check(ggml_backend_buffer_get_alignment(buffer) == alignment, "owned buffer reports type alignment") && ok;
    ok = check(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)) % alignment == 0, "owned buffer base is aligned") && ok;
    ok = check(allocate_tensor(buffer, root), "owned root allocation registers") && ok;
    ok = check(allocate_tensor(buffer, copy, slot_size), "owned copy allocation registers") && ok;

    const size_t root_size = ggml_nbytes(root);
    std::vector<uint8_t> expected(root_size);
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<uint8_t>((index * 17U + 5U) % 251U);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, expected.size());
    ok = check(std::memcmp(root->data, expected.data(), expected.size()) == 0, "owned set writes CPU-accessible bytes") && ok;

    std::vector<uint8_t> round_trip(root_size);
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "owned get reads set bytes") && ok;

    ggml_backend_tensor_memset(root, 0xA5, 1, root_size - 2);
    std::memset(expected.data() + 1, 0xA5, root_size - 2);
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "owned memset updates CPU-accessible bytes") && ok;

    ggml_backend_tensor_copy(root, copy);
    ok = check(std::memcmp(copy->data, expected.data(), expected.size()) == 0, "owned tensor copy preserves bytes") && ok;

    ggml_backend_buffer_clear(buffer, 0x3C);
    const auto * base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t index = 0; index < ggml_backend_buffer_get_size(buffer); ++index) {
        ok = check(base[index] == 0x3C, "owned clear updates every buffer byte") && ok;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_host_alias_parity_and_destruction() {
    alignas(64) std::array<uint8_t, kAliasStorageBytes> storage = {};
    ggml_context * context = make_context();
    if (!check(context != nullptr, "host alias context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "host-alias-root");
    ggml_backend_buffer_t buffer = ggml::gemmini::gemmini_buffer_from_host_ptr(storage.data(), storage.size());
    if (!check(root != nullptr && buffer != nullptr, "host alias objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = true;
    ok = check(ggml_backend_buffer_get_base(buffer) == storage.data(), "host alias base is caller storage") && ok;
    ok = check(allocate_tensor(buffer, root), "host alias root allocation registers") && ok;

    std::vector<uint8_t> expected(ggml_nbytes(root));
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<uint8_t>(index + 9U);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, expected.size());
    ok = check(std::memcmp(storage.data(), expected.data(), expected.size()) == 0, "host alias set is zero-copy") && ok;

    std::vector<uint8_t> round_trip(expected.size());
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "host alias get reads caller storage") && ok;

    auto handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    ok = check(handle.has_value(), "host alias acquires artifact") && ok;
    if (handle) {
        ok = check_artifact_metadata(*handle, root, buffer, false, "host alias handle is valid") && ok;
        storage[0] = 0x7E;
        ok = check(*static_cast<const uint8_t *>(handle->root_data()) == 0x7E, "host alias handle observes caller writes") && ok;
        handle.reset();
    }

    ggml_backend_buffer_free(buffer);
    storage[0] = 0xD3;
    ok = check(storage[0] == 0xD3, "host alias destruction leaves caller storage owned by caller") && ok;
    ggml_free(context);
    return ok;
}

bool test_mmap_alias_metadata_is_advisory() {
    alignas(64) std::array<uint8_t, kAliasStorageBytes> storage = {};
    for (size_t index = 0; index < storage.size(); ++index) {
        storage[index] = static_cast<uint8_t>((index * 29U + 11U) % 253U);
    }

    ggml_context * context = make_context();
    if (!check(context != nullptr, "mmap alias context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "mmap-alias-root");
    ggml_backend_buffer_t buffer = ggml::gemmini::gemmini_buffer_from_host_ptr(storage.data(), storage.size());
    if (!check(root != nullptr && buffer != nullptr, "mmap alias objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    const size_t root_size = ggml_nbytes(root);
    const std::vector<uint8_t> snapshot(storage.begin(), storage.begin() + root_size);
    bool ok = true;
    ok = check(allocate_tensor(buffer, root), "mmap alias root allocation registers") && ok;
    ok = check(std::memcmp(storage.data(), snapshot.data(), snapshot.size()) == 0, "mmap alias registration copies no bytes") && ok;

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    ok = check(handle.has_value(), "mmap alias acquires artifact") && ok;
    if (handle) {
        ok = check_artifact_metadata(*handle, root, buffer, true, "mmap alias handle is valid") && ok;
        storage[1] ^= 0xFFU;
        ok = check(static_cast<const uint8_t *>(handle->root_data())[1] == storage[1], "mmap alias handle observes caller changes") && ok;
        const uint8_t replacement = 0xA4;
        ggml_backend_tensor_set(root, &replacement, 0, sizeof(replacement));
        ok = check(storage[0] == replacement, "immutable metadata does not enforce host-memory read-only") && ok;
        handle.reset();
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_repeated_acquire_identity_and_metadata() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "repeated acquire context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "repeated-root");
    ggml_backend_buffer_t buffer = root == nullptr ? nullptr :
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), ggml_nbytes(root));
    if (!check(root != nullptr && buffer != nullptr, "repeated acquire objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root), "repeated acquire root allocation registers");
    auto first = ggml::gemmini::acquire_q8_h1_artifact(root);
    auto second = ggml::gemmini::acquire_q8_h1_artifact(root);
    ok = check(first.has_value() && second.has_value(), "repeated acquire returns handles") && ok;
    if (first && second) {
        ok = check_artifact_metadata(*first, root, buffer, false, "first repeated handle is valid") && ok;
        ok = check(*first->identity() == *second->identity(), "repeated acquire returns equal identity") && ok;
        ok = check(same_metadata(*first, *second), "repeated acquire returns equal metadata") && ok;
        ok = check(first->root_data() == second->root_data(), "repeated acquire returns one root data identity") && ok;
    }
    ok = check(ggml::gemmini::q8_h1_artifact_registry_count(buffer) == 1, "repeated acquire keeps one registry entry") && ok;

    first.reset();
    second.reset();
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_view_canonicalization_and_safe_rejection() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "view test context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "view-root");
    if (!check(root != nullptr, "view root initializes")) {
        ggml_free(context);
        return false;
    }

    const size_t alignment = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    const size_t slot_size = align_up(ggml_nbytes(root), alignment);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), 2 * slot_size);
    if (!check(buffer != nullptr, "view buffer allocates")) {
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root, slot_size), "view root allocates at nonzero offset");
    ggml_tensor * direct_alias = ggml_view_2d(context, root, kLogicalK, kGlobalJ, root->nb[1], 0);
    ggml_tensor * nested_alias = ggml_view_2d(context, direct_alias, kLogicalK, kGlobalJ, root->nb[1], 0);
    ggml_tensor * overflow_alias = ggml_view_2d(context, root, kLogicalK, kGlobalJ, root->nb[1], 0);
    ok = check(initialize_view(direct_alias), "direct zero-offset view initializes") && ok;
    ok = check(initialize_view(nested_alias), "nested zero-offset view initializes") && ok;
    ok = check(initialize_view(overflow_alias), "overflow test view initializes") && ok;

    auto root_handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    auto direct_handle = ggml::gemmini::acquire_q8_h1_artifact(direct_alias);
    auto nested_handle = ggml::gemmini::acquire_q8_h1_artifact(nested_alias);
    ok = check(root_handle.has_value() && direct_handle.has_value() && nested_handle.has_value(), "root and nested views acquire artifacts") && ok;
    if (root_handle && direct_handle && nested_handle) {
        ok = check_artifact_metadata(*root_handle, root, buffer, false, "nonzero-offset root metadata is valid") && ok;
        ok = check(*root_handle->identity() == *direct_handle->identity(), "zero-offset view uses canonical root identity") && ok;
        ok = check(*root_handle->identity() == *nested_handle->identity(), "nested view uses canonical root identity") && ok;
        ok = check(root_handle->root_data() == direct_handle->root_data(), "zero-offset view uses canonical root data") && ok;
        ok = check(root_handle->root_data() == nested_handle->root_data(), "nested view uses canonical root data") && ok;
    }

    overflow_alias->nb[1] = std::numeric_limits<size_t>::max();
    ok = check(!ggml::gemmini::acquire_q8_h1_artifact(overflow_alias).has_value(), "overflowing view stride rejects safely") && ok;
    ok = check(ggml::gemmini::q8_h1_artifact_registry_count(buffer) == 1, "views add no duplicate artifact metadata") && ok;

    root_handle.reset();
    direct_handle.reset();
    nested_handle.reset();
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_incompatible_aliases_reject() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "incompatible alias context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "incompatible-root");
    ggml_backend_buffer_t buffer = root == nullptr ? nullptr :
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), ggml_nbytes(root));
    if (!check(root != nullptr && buffer != nullptr, "incompatible alias objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root), "incompatible alias root allocation registers");
    ggml_tensor * transposed = ggml_transpose(context, root);
    ggml_tensor * permuted = ggml_permute(context, root, 1, 0, 2, 3);
    ggml_tensor * strided = ggml_view_2d(context, root, kLogicalK, kGlobalJ, root->nb[1] + root->nb[0], 0);
    ok = check(initialize_view(transposed), "transpose view initializes") && ok;
    ok = check(initialize_view(permuted), "permuted view initializes") && ok;
    ok = check(initialize_view(strided), "strided view initializes") && ok;
    ok = check(!ggml::gemmini::acquire_q8_h1_artifact(transposed).has_value(), "transpose alias rejects") && ok;
    ok = check(!ggml::gemmini::acquire_q8_h1_artifact(permuted).has_value(), "permuted alias rejects") && ok;
    ok = check(!ggml::gemmini::acquire_q8_h1_artifact(strided).has_value(), "incompatible stride alias rejects") && ok;
    ok = check(ggml::gemmini::q8_h1_artifact_registry_count(buffer) == 1, "rejected aliases add no metadata") && ok;

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_distinct_roots_and_multiple_metadata() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "multiple roots context initializes")) {
        return false;
    }

    ggml_tensor * first = make_q8_h1_tensor(context, kGlobalJ, "first-root");
    ggml_tensor * second = make_q8_h1_tensor(context, kGlobalJ, "second-root");
    ggml_tensor * third = make_q8_h1_tensor(context, 2, "third-root");
    if (!check(first != nullptr && second != nullptr && third != nullptr, "multiple root tensors initialize")) {
        ggml_free(context);
        return false;
    }

    const size_t alignment = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    const size_t slot_size = align_up(ggml_nbytes(first), alignment);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), 3 * slot_size);
    if (!check(buffer != nullptr, "multiple root buffer allocates")) {
        ggml_free(context);
        return false;
    }

    bool ok = true;
    ok = check(allocate_tensor(buffer, first), "first root allocates") && ok;
    ok = check(allocate_tensor(buffer, second, slot_size), "second root allocates") && ok;
    ok = check(allocate_tensor(buffer, third, 2 * slot_size), "third root allocates") && ok;
    auto first_handle = ggml::gemmini::acquire_q8_h1_artifact(first);
    auto second_handle = ggml::gemmini::acquire_q8_h1_artifact(second);
    auto third_handle = ggml::gemmini::acquire_q8_h1_artifact(third);
    ok = check(first_handle.has_value() && second_handle.has_value() && third_handle.has_value(), "multiple roots acquire artifacts") && ok;
    if (first_handle && second_handle && third_handle) {
        ok = check_artifact_metadata(*first_handle, first, buffer, false, "first root metadata is valid") && ok;
        ok = check_artifact_metadata(*second_handle, second, buffer, false, "second root metadata is valid") && ok;
        ok = check_artifact_metadata(*third_handle, third, buffer, false, "third root metadata is valid") && ok;
        ok = check(!(*first_handle->identity() == *second_handle->identity()), "same shape roots at different offsets and names are distinct") && ok;
        ok = check(first_handle->identity()->root_offset != second_handle->identity()->root_offset, "same shape roots retain distinct offsets") && ok;
        ok = check(first_handle->identity()->root_name != second_handle->identity()->root_name, "same shape roots retain distinct names") && ok;
    }
    ok = check(ggml::gemmini::q8_h1_artifact_registry_count(buffer) == 3, "multiple roots produce exact registry count") && ok;

    first_handle.reset();
    second_handle.reset();
    third_handle.reset();
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_buffer_generations_are_unique() {
    const size_t size = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    ggml_backend_buffer_t first = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), size);
    ggml_backend_buffer_t second = ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), size);
    if (!check(first != nullptr && second != nullptr, "generation buffers allocate")) {
        ggml_backend_buffer_free(first);
        ggml_backend_buffer_free(second);
        return false;
    }

    const uint64_t first_generation = ggml::gemmini::gemmini_buffer_generation(first);
    const uint64_t second_generation = ggml::gemmini::gemmini_buffer_generation(second);
    const bool ok = check(first_generation != 0 && second_generation != 0, "buffer generations are nonzero") &&
        check(first_generation != second_generation, "successive buffers receive distinct generations");
    ggml_backend_buffer_free(first);
    ggml_backend_buffer_free(second);
    return ok;
}

bool test_concurrent_acquire_keeps_one_identity() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "concurrent acquire context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "concurrent-root");
    ggml_backend_buffer_t buffer = root == nullptr ? nullptr :
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), ggml_nbytes(root));
    if (!check(root != nullptr && buffer != nullptr, "concurrent acquire objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root), "concurrent acquire root allocation registers");
    constexpr size_t worker_count = 4;
    std::array<std::optional<ggml::gemmini::q8_h1_artifact_handle>, worker_count> handles;
    std::array<std::thread, worker_count> workers;
    std::atomic<size_t> ready { 0 };
    std::atomic<bool> start { false };
    for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        workers[worker_index] = std::thread([&, worker_index] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            handles[worker_index] = ggml::gemmini::acquire_q8_h1_artifact(root);
        });
    }
    while (ready.load(std::memory_order_acquire) != worker_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread & worker : workers) {
        worker.join();
    }

    for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        ok = check(handles[worker_index].has_value(), "concurrent acquire returns a handle") && ok;
        if (handles[worker_index] && handles[0]) {
            ok = check(*handles[worker_index]->identity() == *handles[0]->identity(), "concurrent acquire returns one identity") && ok;
            ok = check(handles[worker_index]->root_data() == handles[0]->root_data(), "concurrent acquire returns one root data pointer") && ok;
        }
    }
    ok = check(ggml::gemmini::q8_h1_artifact_registry_count(buffer) == 1, "concurrent acquire preserves one registry entry") && ok;

    for (auto & handle : handles) {
        handle.reset();
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return ok;
}

bool test_owned_handle_survives_buffer_free() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "owned handle context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "owned-handle-root");
    ggml_backend_buffer_t buffer = root == nullptr ? nullptr :
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), ggml_nbytes(root));
    if (!check(root != nullptr && buffer != nullptr, "owned handle objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root), "owned handle root allocation registers");
    std::vector<uint8_t> expected(ggml_nbytes(root));
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<uint8_t>((index * 7U + 3U) % 255U);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, expected.size());
    auto handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    ok = check(handle.has_value(), "owned handle acquires artifact") && ok;
    if (handle) {
        ok = check_artifact_metadata(*handle, root, buffer, false, "owned handle metadata is valid") && ok;
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ok = check(handle->valid(), "owned handle remains valid after buffer free") && ok;
        ok = check(std::memcmp(handle->root_data(), expected.data(), expected.size()) == 0, "owned handle retains storage after buffer free") && ok;
        handle.reset();
    } else {
        ggml_backend_buffer_free(buffer);
    }

    ggml_free(context);
    return ok;
}

#if !defined(_WIN32)
int external_alias_free_with_handle_child() {
    alignas(64) std::array<uint8_t, kAliasStorageBytes> storage = {};
    ggml_context * context = make_context();
    ggml_tensor * root = context == nullptr ? nullptr : make_q8_h1_tensor(context, kGlobalJ, "external-death-root");
    ggml_backend_buffer_t buffer = root == nullptr ? nullptr : ggml::gemmini::gemmini_buffer_from_host_ptr(storage.data(), storage.size());
    if (context == nullptr || root == nullptr || buffer == nullptr || !allocate_tensor(buffer, root)) {
        _exit(2);
    }

    auto handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    if (!handle) {
        _exit(3);
    }

    ggml_backend_buffer_free(buffer);
    _exit(0);
}
#endif

bool test_external_alias_handle_lifetime() {
    alignas(64) std::array<uint8_t, kAliasStorageBytes> storage = {};
    ggml_context * context = make_context();
    if (!check(context != nullptr, "external lifetime context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_h1_tensor(context, kGlobalJ, "external-lifetime-root");
    ggml_backend_buffer_t buffer = ggml::gemmini::gemmini_buffer_from_host_ptr(storage.data(), storage.size());
    if (!check(root != nullptr && buffer != nullptr, "external lifetime objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = check(allocate_tensor(buffer, root), "external lifetime root allocation registers");
    auto handle = ggml::gemmini::acquire_q8_h1_artifact(root);
    ok = check(handle.has_value(), "external lifetime acquires handle") && ok;
    handle.reset();
    ggml_backend_buffer_free(buffer);
    storage[0] = 0x42;
    ok = check(storage[0] == 0x42, "released external handle permits buffer free") && ok;
    ggml_free(context);

#if defined(_WIN32)
    std::fprintf(stderr, "SKIP: external alias death test requires POSIX\n");
#else
    const pid_t child = fork();
    if (!check(child >= 0, "external lifetime child starts")) {
        return false;
    }
    if (child == 0) {
        return external_alias_free_with_handle_child();
    }

    int status = 0;
    const pid_t waited_child = waitpid(child, &status, 0);
    ok = check(waited_child == child, "external lifetime child waits") && ok;
    ok = check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "external alias free with live handle aborts safely") && ok;
#endif
    return ok;
}

}

int main() {
    const bool ok =
        test_owned_buffer_io_and_destruction() &&
        test_host_alias_parity_and_destruction() &&
        test_mmap_alias_metadata_is_advisory() &&
        test_repeated_acquire_identity_and_metadata() &&
        test_view_canonicalization_and_safe_rejection() &&
        test_incompatible_aliases_reject() &&
        test_distinct_roots_and_multiple_metadata() &&
        test_buffer_generations_are_unique() &&
        test_concurrent_acquire_keeps_one_identity() &&
        test_owned_handle_survives_buffer_free() &&
        test_external_alias_handle_lifetime();
    std::printf("test-gemmini-buffer-artifacts: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
