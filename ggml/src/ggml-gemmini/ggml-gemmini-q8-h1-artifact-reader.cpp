#include "ggml-gemmini-q8-h1-artifact-reader.hpp"

#include "quants/common/fp16_util.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace ggml::gemmini::detail {
namespace {
constexpr char GQ8H1_MAGIC[8] = {'G', 'Q', '8', 'H', '1', 0, 1, 0};
constexpr uint32_t GQ8H1_VERSION_MAJOR = 1;
constexpr uint32_t GQ8H1_VERSION_MINOR = 0;
constexpr size_t GQ8H1_BLOCK_SIZE = 32;
// ponytail: bounded whole-file load; switch to streaming if sidecars need to exceed this.
constexpr size_t GQ8H1_MAX_ARTIFACT_BYTES = 512ull * 1024ull * 1024ull;

struct artifact_reader {
    const std::vector<uint8_t> & bytes;
    size_t offset = 0;
    size_t remaining() const { return bytes.size() - offset; }
    bool can_read(size_t size) const { return size <= remaining(); }
    bool done() const { return offset == bytes.size(); }
    bool read_bytes(void * dst, size_t size) {
        if (!can_read(size)) {
            return false;
        }
        if (size > 0) {
            std::memcpy(dst, bytes.data() + offset, size);
        }
        offset += size;
        return true;
    }
    template <typename T>
    bool read_pod(T & value) { return read_bytes(&value, sizeof(value)); }
};
bool fail_parse(q8_h1_artifact_store & store, std::string * error, const char * message) {
    store.clear();
    if (error != nullptr) { *error = message; }
    return false;
}
bool checked_mul(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) { return false; }
    out = lhs * rhs;
    return true;
}
bool checked_add(size_t lhs, size_t rhs, size_t & out) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) { return false; }
    out = lhs + rhs;
    return true;
}
bool checked_product_3(size_t a, size_t b, size_t c, size_t & out) {
    size_t ab = 0;
    return checked_mul(a, b, ab) && checked_mul(ab, c, out);
}
bool checked_u64_to_size(uint64_t value, size_t & out) {
    if (value > std::numeric_limits<size_t>::max()) { return false; }
    out = static_cast<size_t>(value);
    return true;
}
bool checked_i64_to_size(int64_t value, size_t & out) {
    if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) { return false; }
    out = static_cast<size_t>(value);
    return true;
}
bool read_artifact_file(const std::string & path, std::vector<uint8_t> & bytes, q8_h1_artifact_store & store, std::string * error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail_parse(store, error, "failed to open artifact");
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) {
        return fail_parse(store, error, "failed to measure artifact");
    }
    if (static_cast<uint64_t>(size) > static_cast<uint64_t>(GQ8H1_MAX_ARTIFACT_BYTES)) {
        return fail_parse(store, error, "artifact too large");
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (!bytes.empty() && !file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size))) {
        return fail_parse(store, error, "failed to read artifact");
    }
    return true;
}
bool read_tensor_name(artifact_reader & reader, q8_h1_artifact_store & store, std::string * error, std::string & name) {
    uint32_t name_len = 0;
    if (!reader.read_pod(name_len)) {
        return fail_parse(store, error, "truncated tensor name length");
    }
    const size_t name_size = static_cast<size_t>(name_len);
    if (!reader.can_read(name_size)) {
        return fail_parse(store, error, "truncated tensor name");
    }
    name.resize(name_size, '\0');
    reader.read_bytes(name.data(), name.size());
    if (store.find(name) != store.end()) {
        return fail_parse(store, error, "duplicate tensor name");
    }
    return true;
}
bool read_tensor_geometry(
        artifact_reader & reader,
        q8_h1_artifact_store & store,
        std::string * error,
        q8_h1_artifact_tensor & tensor) {
    uint64_t logical_rows = 0, k = 0, blocks_per_row = 0, qs_bytes = 0, subs_bytes = 0, sups_bytes = 0, z_bytes = 0;
    if (!reader.read_pod(logical_rows) || !reader.read_pod(k) || !reader.read_pod(blocks_per_row) ||
        !reader.read_pod(qs_bytes) || !reader.read_pod(subs_bytes) || !reader.read_pod(sups_bytes) || !reader.read_pod(z_bytes)) {
        return fail_parse(store, error, "truncated tensor metadata");
    }
    size_t expected_rows = 0;
    if (!checked_product_3(static_cast<size_t>(tensor.dims[1]), static_cast<size_t>(tensor.dims[2]), static_cast<size_t>(tensor.dims[3]), expected_rows)) {
        return fail_parse(store, error, "tensor row count overflow");
    }
    size_t logical_rows_size = 0, k_size = 0, blocks_per_row_size = 0;
    if (!checked_u64_to_size(logical_rows, logical_rows_size) || !checked_u64_to_size(k, k_size) || !checked_u64_to_size(blocks_per_row, blocks_per_row_size)) {
        return fail_parse(store, error, "tensor geometry overflow");
    }
    if (logical_rows == 0 || k == 0 || k != static_cast<uint64_t>(tensor.dims[0]) || logical_rows_size != expected_rows ||
        k % GQ8H1_BLOCK_SIZE != 0 || blocks_per_row_size != k_size / GQ8H1_BLOCK_SIZE) {
        return fail_parse(store, error, "impossible tensor geometry");
    }

    size_t expected_qs_bytes = 0, expected_subs_bytes = 0, expected_sups_bytes = 0, expected_z_bytes = 0;
    if (!checked_mul(logical_rows_size, k_size, expected_qs_bytes) ||
        !checked_mul(logical_rows_size, blocks_per_row_size, expected_subs_bytes) ||
        !checked_mul(logical_rows_size, sizeof(uint16_t), expected_sups_bytes) ||
        !checked_mul(logical_rows_size, sizeof(uint16_t), expected_z_bytes)) {
        return fail_parse(store, error, "tensor payload size overflow");
    }
    if (qs_bytes != expected_qs_bytes || subs_bytes != expected_subs_bytes || sups_bytes != expected_sups_bytes || z_bytes != expected_z_bytes) {
        return fail_parse(store, error, "invalid tensor payload sizes");
    }
    tensor.logical_rows = logical_rows_size;
    tensor.k = k_size;
    tensor.blocks_per_row = blocks_per_row_size;
    return true;
}
bool read_tensor_payload(
        artifact_reader & reader,
        q8_h1_artifact_store & store,
        std::string * error,
        q8_h1_artifact_tensor & tensor) {
    size_t qs_bytes = 0, subs_bytes = 0, sups_bytes = 0, z_bytes = 0;
    if (!checked_mul(tensor.logical_rows, tensor.k, qs_bytes) ||
        !checked_mul(tensor.logical_rows, tensor.blocks_per_row, subs_bytes) ||
        !checked_mul(tensor.logical_rows, sizeof(uint16_t), sups_bytes) ||
        !checked_mul(tensor.logical_rows, sizeof(uint16_t), z_bytes)) {
        return fail_parse(store, error, "tensor payload size overflow");
    }
    size_t payload_bytes = 0;
    if (!checked_add(qs_bytes, subs_bytes, payload_bytes) || !checked_add(payload_bytes, sups_bytes, payload_bytes) || !checked_add(payload_bytes, z_bytes, payload_bytes)) {
        return fail_parse(store, error, "tensor payload size overflow");
    }
    if (!reader.can_read(payload_bytes)) {
        return fail_parse(store, error, "truncated tensor payload");
    }
    tensor.qs.resize(qs_bytes);
    tensor.subs.resize(subs_bytes);
    tensor.sups_f32.reserve(tensor.logical_rows);
    tensor.z.resize(tensor.logical_rows);
    std::vector<uint16_t> sups_fp16(tensor.logical_rows, 0);
    if (!reader.read_bytes(tensor.qs.data(), tensor.qs.size()) ||
        !reader.read_bytes(tensor.subs.data(), tensor.subs.size()) ||
        !reader.read_bytes(sups_fp16.data(), sups_bytes) ||
        !reader.read_bytes(tensor.z.data(), z_bytes)) {
        return fail_parse(store, error, "truncated tensor payload");
    }
    for (uint16_t raw : sups_fp16) {
        const float value = quants::fp16_to_fp32(raw);
        if (!std::isfinite(value)) {
            return fail_parse(store, error, "non-finite row scale");
        }
        tensor.sups_f32.push_back(value);
    }
    return true;
}
bool read_tensor(artifact_reader & reader, q8_h1_artifact_store & store, std::string * error) {
    std::string name;
    if (!read_tensor_name(reader, store, error, name)) {
        return false;
    }
    q8_h1_artifact_tensor tensor;
    for (size_t dim = 0; dim < tensor.dims.size(); ++dim) {
        if (!reader.read_pod(tensor.dims[dim])) {
            return fail_parse(store, error, "truncated tensor dims");
        }
        if (tensor.dims[dim] <= 0) {
            return fail_parse(store, error, "invalid tensor dims");
        }
    }

    return read_tensor_geometry(reader, store, error, tensor) &&
           read_tensor_payload(reader, store, error, tensor) &&
           static_cast<bool>(store.emplace(std::move(name), std::move(tensor)).second);
}
}
bool dims_to_geometry(
        const std::array<int64_t, GGML_MAX_DIMS> & dims,
        size_t & logical_rows,
        size_t & k,
        size_t & blocks_per_row) {
    size_t dim0 = 0, dim1 = 0, dim2 = 0, dim3 = 0;
    if (!checked_i64_to_size(dims[0], dim0) || !checked_i64_to_size(dims[1], dim1) || !checked_i64_to_size(dims[2], dim2) ||
        !checked_i64_to_size(dims[3], dim3) || !checked_product_3(dim1, dim2, dim3, logical_rows) || dim0 % GQ8H1_BLOCK_SIZE != 0) {
        return false;
    }
    k = dim0;
    blocks_per_row = k / GQ8H1_BLOCK_SIZE;
    return logical_rows > 0 && blocks_per_row > 0;
}
bool load_q8_h1_artifact_impl(
        const std::string & path,
        q8_h1_artifact_store & store,
        std::string * error) {
    std::vector<uint8_t> bytes;
    if (!read_artifact_file(path, bytes, store, error)) {
        return false;
    }
    artifact_reader reader{bytes};
    char magic[sizeof(GQ8H1_MAGIC)] = {};
    uint32_t version_major = 0, version_minor = 0;
    uint64_t tensor_count = 0;
    if (!reader.read_bytes(magic, sizeof(magic))) {
        return fail_parse(store, error, "truncated artifact header");
    }
    if (std::memcmp(magic, GQ8H1_MAGIC, sizeof(magic)) != 0) {
        return fail_parse(store, error, "bad artifact magic");
    }
    if (!reader.read_pod(version_major) || !reader.read_pod(version_minor) || !reader.read_pod(tensor_count)) {
        return fail_parse(store, error, "truncated artifact version header");
    }
    if (version_major != GQ8H1_VERSION_MAJOR || version_minor != GQ8H1_VERSION_MINOR) {
        return fail_parse(store, error, "unsupported artifact version");
    }
    for (uint64_t tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx) {
        if (!read_tensor(reader, store, error)) {
            return false;
        }
    }
    if (!reader.done()) {
        return fail_parse(store, error, "unexpected trailing bytes");
    }
    return true;
}
}
