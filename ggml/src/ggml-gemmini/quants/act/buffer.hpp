#pragma once

#include "../../ggml-gemmini-config.hpp"
#include <gemmini_params.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants::act {

struct QuantizedActivationBuffer {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    uint8_t                               bits             = 8;
    size_t                                rows             = 0;
    size_t                                cols             = 0;
    size_t                                row_stride_bytes = 0;
    size_t                                row_offset       = 0;

    bool allocate(size_t r, size_t c, uint8_t b) {
        if (r == 0 || c == 0 || (b != 4 && b != 8 && b != 16))
            return false;

        size_t staged_row_stride = c;
        if (b == 16) {
            if (c > std::numeric_limits<size_t>::max() / sizeof(int16_t))
                return false;
            staged_row_stride = c * sizeof(int16_t);
        }
        if (r > std::numeric_limits<size_t>::max() / staged_row_stride)
            return false;
        const size_t byte_count = r * staged_row_stride;
        if (byte_count > std::vector<uint8_t>{}.max_size())
            return false;

        std::shared_ptr<std::vector<uint8_t>> staged_bytes;
        try {
            staged_bytes = std::make_shared<std::vector<uint8_t>>(byte_count, 0);
        } catch (const std::bad_alloc &) {
            return false;
        } catch (const std::length_error &) {
            return false;
        }

        bytes            = std::move(staged_bytes);
        bits             = b;
        rows             = r;
        cols             = c;
        row_stride_bytes = staged_row_stride;
        row_offset       = 0;
        return true;
    }

    bool valid() const {
        return bytes != nullptr && !bytes->empty() && bits != 0 && rows != 0 && cols != 0;
    }

    int32_t get(size_t row, size_t col) const {
        if (!valid() || row >= rows || col >= cols)
            return 0;
        const size_t actual_row  = row_offset + row;
        const size_t byte_offset = actual_row * row_stride_bytes;
        if (bits == 4 || bits == 8) {
            const int8_t v = static_cast<int8_t>((*bytes)[byte_offset + col]);
            return static_cast<int32_t>(v);
        } else { // 16-bit
            const size_t idx = byte_offset + col * 2;
            int16_t      v   = 0;
            std::memcpy(&v, &(*bytes)[idx], sizeof(v));
            return static_cast<int32_t>(v);
        }
    }

    bool set(size_t row, size_t col, int32_t value) {
        if (!valid() || row >= rows || col >= cols)
            return false;
        int32_t qmin = -(int32_t{1} << (bits - 1));
        int32_t qmax = (int32_t{1} << (bits - 1)) - 1;
        if (value < qmin || value > qmax)
            return false;
        const size_t actual_row  = row_offset + row;
        const size_t byte_offset = actual_row * row_stride_bytes;
        if (bits == 4 || bits == 8) {
            (*bytes)[byte_offset + col] = static_cast<uint8_t>(value);
        } else { // 16-bit
            const size_t idx = byte_offset + col * 2;
            int16_t      v   = static_cast<int16_t>(value);
            std::memcpy(&(*bytes)[idx], &v, sizeof(v));
        }
        return true;
    }

    QuantizedActivationBuffer slice_rows(size_t begin, size_t count) const {
        QuantizedActivationBuffer s;
        s.bytes            = bytes;
        s.bits             = bits;
        s.rows             = count;
        s.cols             = cols;
        s.row_stride_bytes = row_stride_bytes;
        s.row_offset       = row_offset + begin;
        return s;
    }

    void zero_fill() {
        if (bytes)
            std::fill(bytes->begin(), bytes->end(), 0);
    }

    const uint8_t * raw_data() const {
        if (!bytes || (row_stride_bytes != 0 &&
                       row_offset > std::numeric_limits<size_t>::max() / row_stride_bytes)) {
            return nullptr;
        }
        const size_t offset = row_offset * row_stride_bytes;
        return offset <= bytes->size() ? bytes->data() + offset : nullptr;
    }

    size_t raw_size() const {
        if (!bytes || (row_stride_bytes != 0 &&
                       row_offset > std::numeric_limits<size_t>::max() / row_stride_bytes)) {
            return 0;
        }
        const size_t offset = row_offset * row_stride_bytes;
        return offset <= bytes->size() ? bytes->size() - offset : 0;
    }

    // Raw transport conversion accepts the configured width and legacy one-byte tests.
    operator elem_t *() {
        return (bits == GGML_GEMMINI_ACTIVATION_BITS || (bits == 8 && sizeof(elem_t) == 1))
                   ? reinterpret_cast<elem_t *>(const_cast<uint8_t *>(raw_data()))
                   : nullptr;
    }
    operator const elem_t *() const {
        return (bits == GGML_GEMMINI_ACTIVATION_BITS || (bits == 8 && sizeof(elem_t) == 1))
                   ? reinterpret_cast<const elem_t *>(raw_data())
                   : nullptr;
    }
};

} // namespace ggml::gemmini::quants::act
