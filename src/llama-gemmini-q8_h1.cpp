#include "llama-gemmini-q8_h1.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t GEMMINI_Q8_0_BLOCK_SIZE = 32;
constexpr size_t GEMMINI_Q8_0_BLOCK_BYTES = sizeof(ggml_fp16_t) + GEMMINI_Q8_0_BLOCK_SIZE;
constexpr float GEMMINI_Q8_H1_SCALE_BINS = 255.0f;

uint8_t gemmini_q8_h1_sub(float scale, float sups, uint16_t z) {
    if (!std::isfinite(scale) || !std::isfinite(sups) || sups <= 0.0f) {
        return 0;
    }

    const double effective = std::round(static_cast<double>(scale) / static_cast<double>(sups));
    const double shifted = effective - static_cast<double>(z);
    const double clamped = std::min(255.0, std::max(0.0, shifted));
    return static_cast<uint8_t>(clamped);
}

struct gemmini_q8_h1_tensor {
    std::vector<int8_t> qs;
    std::vector<uint8_t> subs;
    std::vector<ggml_fp16_t> sups;
    std::vector<uint16_t> z;
};

struct q8_0_tensor_view {
    const void * data;
    int64_t ne0;
    int64_t logical_rows;
};

bool checked_mul_size(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checked_logical_rows(const ggml_tensor * tensor, int64_t & out) {
    if (!tensor || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] <= 0) {
        return false;
    }
    if (tensor->ne[1] > std::numeric_limits<int64_t>::max() / tensor->ne[2]) {
        return false;
    }
    const int64_t rows_12 = tensor->ne[1] * tensor->ne[2];
    if (rows_12 > std::numeric_limits<int64_t>::max() / tensor->ne[3]) {
        return false;
    }
    out = rows_12 * tensor->ne[3];
    return true;
}

bool gemmini_q8_h1_from_q8_0(q8_0_tensor_view src, gemmini_q8_h1_tensor & dst) {
    if (!src.data || src.ne0 <= 0 || src.logical_rows <= 0 || src.ne0 % static_cast<int64_t>(GEMMINI_Q8_0_BLOCK_SIZE) != 0) {
        return false;
    }

    const size_t k = static_cast<size_t>(src.ne0);
    const size_t rows = static_cast<size_t>(src.logical_rows);
    const size_t blocks_per_row = k / GEMMINI_Q8_0_BLOCK_SIZE;
    const auto * bytes = static_cast<const uint8_t *>(src.data);
    size_t qs_count = 0;
    size_t subs_count = 0;
    if (!checked_mul_size(rows, k, qs_count) || !checked_mul_size(rows, blocks_per_row, subs_count)) {
        return false;
    }

    dst.qs.assign(qs_count, 0);
    dst.subs.assign(subs_count, 0);
    dst.sups.assign(rows, 0);
    dst.z.assign(rows, 0);

    std::vector<float> scales(blocks_per_row, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        float min_scale = std::numeric_limits<float>::infinity();
        float max_scale = -std::numeric_limits<float>::infinity();

        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t block_index = row * blocks_per_row + block;
            const uint8_t * block_ptr = bytes + block_index * GEMMINI_Q8_0_BLOCK_BYTES;

            ggml_fp16_t scale_fp16 = 0;
            std::memcpy(&scale_fp16, block_ptr, sizeof(scale_fp16));
            const float scale = ggml_fp16_to_fp32(scale_fp16);
            if (!std::isfinite(scale)) {
                return false;
            }

            scales[block] = scale;
            min_scale = std::min(min_scale, scale);
            max_scale = std::max(max_scale, scale);
            std::memcpy(dst.qs.data() + row * k + block * GEMMINI_Q8_0_BLOCK_SIZE,
                    block_ptr + sizeof(ggml_fp16_t), GEMMINI_Q8_0_BLOCK_SIZE);
        }

        const float scale_range = max_scale - min_scale;
        if (!std::isfinite(scale_range) || scale_range <= 0.0f) {
            dst.sups[row] = min_scale > 0.0f && std::isfinite(min_scale) ? ggml_fp32_to_fp16(min_scale) : 0;
            dst.z[row] = min_scale > 0.0f && std::isfinite(min_scale) ? 1 : 0;
            continue;
        }

        const float range_sups = scale_range / GEMMINI_Q8_H1_SCALE_BINS;
        const float offset_sups = min_scale > 0.0f ? min_scale / static_cast<float>(std::numeric_limits<uint16_t>::max()) : 0.0f;
        const float sups = std::max(range_sups, offset_sups);
        if (!std::isfinite(sups) || sups <= 0.0f) {
            dst.sups[row] = min_scale > 0.0f && std::isfinite(min_scale) ? ggml_fp32_to_fp16(min_scale) : 0;
            dst.z[row] = min_scale > 0.0f && std::isfinite(min_scale) ? 1 : 0;
            continue;
        }

        const double z_value = std::round(static_cast<double>(min_scale) / static_cast<double>(sups));
        const uint16_t z = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, z_value)));
        dst.sups[row] = ggml_fp32_to_fp16(sups);
        dst.z[row] = z;

        const float stored_sups = ggml_fp16_to_fp32(dst.sups[row]);
        for (size_t block = 0; block < blocks_per_row; ++block) {
            dst.subs[row * blocks_per_row + block] = gemmini_q8_h1_sub(scales[block], stored_sups, z);
        }
    }

    return true;
}

void gemmini_write_bytes(std::ofstream & file, const void * data, size_t size) {
    if (size > 0) {
        file.write(static_cast<const char *>(data), size);
    }
}

void gemmini_write_u32(std::ofstream & file, uint32_t value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void gemmini_write_u64(std::ofstream & file, uint64_t value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void gemmini_write_i64(std::ofstream & file, int64_t value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

}

gemmini_q8_h1_artifact_writer::gemmini_q8_h1_artifact_writer(const char * path) {
    if (!path) {
        return;
    }
    file = std::ofstream(path, std::ios::binary);
    file.exceptions(std::ofstream::failbit);
    const char magic[8] = {'G', 'Q', '8', 'H', '1', 0, 1, 0};
    file.write(magic, sizeof(magic));
    gemmini_write_u32(file, 1);
    gemmini_write_u32(file, 0);
    gemmini_write_u64(file, 0);
}

void gemmini_q8_h1_artifact_writer::add_tensor(const std::string & name, const ggml_tensor * tensor, const void * data) {
    if (!file.is_open()) {
        return;
    }

    int64_t logical_rows = 0;
    if (!checked_logical_rows(tensor, logical_rows)) {
        throw std::runtime_error("invalid Gemmini Q8_H1 tensor shape " + name);
    }
    if (tensor->ne[0] <= 0 || tensor->ne[0] % static_cast<int64_t>(GEMMINI_Q8_0_BLOCK_SIZE) != 0) {
        throw std::runtime_error("invalid Gemmini Q8_H1 tensor K dimension " + name);
    }
    gemmini_q8_h1_tensor q8_h1;
    const q8_0_tensor_view view { data, tensor->ne[0], logical_rows };
    if (!gemmini_q8_h1_from_q8_0(view, q8_h1)) {
        throw std::runtime_error("failed to build Gemmini Q8_H1 artifact tensor " + name);
    }

    gemmini_write_u32(file, static_cast<uint32_t>(name.size()));
    gemmini_write_bytes(file, name.data(), name.size());
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        gemmini_write_i64(file, tensor->ne[i]);
    }
    gemmini_write_u64(file, static_cast<uint64_t>(logical_rows));
    gemmini_write_u64(file, static_cast<uint64_t>(tensor->ne[0]));
    gemmini_write_u64(file, static_cast<uint64_t>(tensor->ne[0] / static_cast<int64_t>(GEMMINI_Q8_0_BLOCK_SIZE)));
    gemmini_write_u64(file, static_cast<uint64_t>(q8_h1.qs.size() * sizeof(q8_h1.qs[0])));
    gemmini_write_u64(file, static_cast<uint64_t>(q8_h1.subs.size() * sizeof(q8_h1.subs[0])));
    gemmini_write_u64(file, static_cast<uint64_t>(q8_h1.sups.size() * sizeof(q8_h1.sups[0])));
    gemmini_write_u64(file, static_cast<uint64_t>(q8_h1.z.size() * sizeof(q8_h1.z[0])));
    gemmini_write_bytes(file, q8_h1.qs.data(), q8_h1.qs.size() * sizeof(q8_h1.qs[0]));
    gemmini_write_bytes(file, q8_h1.subs.data(), q8_h1.subs.size() * sizeof(q8_h1.subs[0]));
    gemmini_write_bytes(file, q8_h1.sups.data(), q8_h1.sups.size() * sizeof(q8_h1.sups[0]));
    gemmini_write_bytes(file, q8_h1.z.data(), q8_h1.z.size() * sizeof(q8_h1.z[0]));
    ++tensor_count;
}

void gemmini_q8_h1_artifact_writer::finish() {
    if (!file.is_open()) {
        return;
    }
    file.seekp(16);
    gemmini_write_u64(file, tensor_count);
    file.close();
}
