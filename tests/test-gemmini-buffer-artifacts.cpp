#include <ggml.h>
#include <ggml-backend.h>

#include "../ggml/src/ggml-gemmini/ggml-gemmini-buffer.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t kLogicalK           = 64;
constexpr int64_t kGlobalJ            = 3;
constexpr size_t  kContextTensorSlots = 32;
constexpr size_t  kAliasStorageBytes  = 4096;

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

ggml_tensor * make_q8_0_tensor(ggml_context * context, int64_t global_j, const char * name) {
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, kLogicalK, global_j);
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

bool test_owned_buffer_io_and_destruction() {
    ggml_context * context = make_context();
    if (!check(context != nullptr, "owned test context initializes")) {
        return false;
    }

    ggml_tensor * root = make_q8_0_tensor(context, kGlobalJ, "owned-root");
    ggml_tensor * copy = make_q8_0_tensor(context, kGlobalJ, "owned-copy");
    if (!check(root != nullptr && copy != nullptr, "owned test tensors initialize")) {
        ggml_free(context);
        return false;
    }

    const size_t alignment = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    const size_t slot_size = align_up(ggml_nbytes(root), alignment);
    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), 2 * slot_size);
    if (!check(buffer != nullptr, "owned Gemmini buffer allocates")) {
        ggml_free(context);
        return false;
    }

    bool ok = true;
    ok =
        check(ggml::gemmini::is_gemmini_buffer(buffer), "owned buffer identifies as Gemmini") && ok;
    ok = check(ggml_backend_buffer_get_alignment(buffer) == alignment,
               "owned buffer reports type alignment") &&
         ok;
    ok = check(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)) % alignment == 0,
               "owned buffer base is aligned") &&
         ok;
    ok = check(allocate_tensor(buffer, root), "owned root allocation registers") && ok;
    ok = check(allocate_tensor(buffer, copy, slot_size), "owned copy allocation registers") && ok;

    const size_t         root_size = ggml_nbytes(root);
    std::vector<uint8_t> expected(root_size);
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<uint8_t>((index * 17U + 5U) % 251U);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, expected.size());
    ok = check(std::memcmp(root->data, expected.data(), expected.size()) == 0,
               "owned set writes CPU-accessible bytes") &&
         ok;

    std::vector<uint8_t> round_trip(root_size);
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "owned get reads set bytes") && ok;

    ggml_backend_tensor_memset(root, 0xA5, 1, root_size - 2);
    std::memset(expected.data() + 1, 0xA5, root_size - 2);
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "owned memset updates CPU-accessible bytes") && ok;

    ggml_backend_tensor_copy(root, copy);
    ok = check(std::memcmp(copy->data, expected.data(), expected.size()) == 0,
               "owned tensor copy preserves bytes") &&
         ok;

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
    ggml_context *                                      context = make_context();
    if (!check(context != nullptr, "host alias context initializes")) {
        return false;
    }

    ggml_tensor *         root = make_q8_0_tensor(context, kGlobalJ, "host-alias-root");
    ggml_backend_buffer_t buffer =
        ggml::gemmini::gemmini_buffer_from_host_ptr(storage.data(), storage.size());
    if (!check(root != nullptr && buffer != nullptr, "host alias objects initialize")) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    bool ok = true;
    ok      = check(ggml_backend_buffer_get_base(buffer) == storage.data(),
                    "host alias base is caller storage") &&
              ok;
    ok      = check(allocate_tensor(buffer, root), "host alias root allocation registers") && ok;

    std::vector<uint8_t> expected(ggml_nbytes(root));
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<uint8_t>(index + 9U);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, expected.size());
    ok = check(std::memcmp(storage.data(), expected.data(), expected.size()) == 0,
               "host alias set is zero-copy") &&
         ok;

    std::vector<uint8_t> round_trip(expected.size());
    ggml_backend_tensor_get(root, round_trip.data(), 0, round_trip.size());
    ok = check(round_trip == expected, "host alias get reads caller storage") && ok;

    ggml_backend_buffer_free(buffer);
    storage[0] = 0xD3;
    ok =
        check(storage[0] == 0xD3, "host alias destruction leaves caller storage owned by caller") &&
        ok;
    ggml_free(context);
    return ok;
}

bool test_buffer_generations_are_unique() {
    const size_t size = ggml_backend_buft_get_alignment(ggml::gemmini::gemmini_buffer_type());
    ggml_backend_buffer_t first =
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), size);
    ggml_backend_buffer_t second =
        ggml_backend_buft_alloc_buffer(ggml::gemmini::gemmini_buffer_type(), size);
    if (!check(first != nullptr && second != nullptr, "generation buffers allocate")) {
        ggml_backend_buffer_free(first);
        ggml_backend_buffer_free(second);
        return false;
    }

    const uint64_t first_generation  = ggml::gemmini::gemmini_buffer_generation(first);
    const uint64_t second_generation = ggml::gemmini::gemmini_buffer_generation(second);
    const bool     ok =
        check(first_generation != 0 && second_generation != 0, "buffer generations are nonzero") &&
        check(first_generation != second_generation,
              "successive buffers receive distinct generations");
    ggml_backend_buffer_free(first);
    ggml_backend_buffer_free(second);
    return ok;
}

} // namespace

int main() {
    const bool ok = test_owned_buffer_io_and_destruction() &&
                    test_host_alias_parity_and_destruction() &&
                    test_buffer_generations_are_unique();
    std::printf("test-gemmini-buffer-artifacts: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
