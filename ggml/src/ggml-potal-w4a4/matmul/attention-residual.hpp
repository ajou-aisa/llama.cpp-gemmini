#pragma once

#include "attention.hpp"
#include "residual/stripe-residual.hpp"
#include "../../ggml-npu-common/quantization/common/math.hpp"

#include <memory>
#include <vector>

namespace ggml::potal::attention::detail {

// The paired path consumes run/lane rows directly; preserve ascending K and
// lane/source-row order without a temporary compact StripePacket.
inline std::shared_ptr<const residual::decoded_packet> decode_upper(
        const stream & activation, size_t row_begin, size_t rows, role operation) {
    constexpr size_t block_k = 32;
    constexpr int bits = 4, radix = 16, bias = 8;
    namespace rmd = ggml::npu::rmd;
    const auto require = [](bool valid, potal_status status = POTAL_ERR_INVALID_ARGUMENT) {
        if (!valid) throw status;
    };
    const auto product = [&](size_t a, size_t b) {
        require(!a || b <= SIZE_MAX / a, POTAL_ERR_CAPACITY);
        return a * b;
    };
    require(activation.k && activation.k <= activation.kp && activation.kp % block_k == 0);
    require(rows && rows <= UINT16_MAX && activation.k <= UINT32_MAX &&
            row_begin <= activation.rows && rows <= activation.rows - row_begin &&
            activation.codes.size() == product(activation.rows, activation.kp) &&
            activation.limbs[0].size() == activation.codes.size());
    auto out = std::make_shared<residual::decoded_packet>();
    struct event { uint32_t k; uint16_t row; uint8_t lane; int8_t digit; };
    std::vector<event> events;
    const size_t capacity = rmd::balanced_radix_contract(bits).lane_capacity;
    std::vector<uint32_t> k_masks(activation.kp / block_k, 0);
    std::vector<size_t> row_map(product(capacity, rows), SIZE_MAX);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < activation.k; ++k) {
            const size_t index = (row_begin + row) * activation.kp + k;
            int64_t value = (int64_t(activation.codes[index]) - activation.limbs[0][index] -
                             (operation == role::pv ? bias : 0)) / radix;
            uint8_t lane = 0;
            while (value) {
                require(lane < capacity, POTAL_ERR_CAPACITY);
                const int32_t digit = ggml::npu::lowdigit(uint64_t(value), bits);
                value = (value - digit) / radix;
                if (digit) {
                    events.push_back({uint32_t(k), uint16_t(row), lane, int8_t(digit)});
                    k_masks[k / block_k] |= uint32_t{1} << (k % block_k);
                    row_map[lane * rows + row] = 0;
                }
                ++lane;
            }
        }
    }
    if (events.empty()) return out;
    std::vector<size_t> k_map(activation.kp, SIZE_MAX);
    for (size_t block = 0; block < k_masks.size(); ++block) {
        if (!k_masks[block]) continue;
        residual::run run;
        run.block_id = uint32_t(block);
        for (size_t k = 0; k < block_k; ++k) {
            if (!(k_masks[block] & (uint32_t{1} << k))) continue;
            run.local_k.push_back(uint16_t(k));
            k_map[block * block_k + k] = out->k_compact++;
        }
        out->runs.push_back(std::move(run));
    }
    for (size_t lane = 0; lane < capacity; ++lane) {
        for (size_t row = 0; row < rows; ++row) {
            auto & mapped = row_map[lane * rows + row];
            if (mapped == SIZE_MAX) continue;
            mapped = out->rows.size();
            out->rows.push_back({uint8_t(lane), uint32_t(row)});
        }
    }
    out->digits.assign(product(out->rows.size(), out->k_compact), 0);
    for (const auto & e : events)
        out->digits[row_map[e.lane * rows + e.row] * out->k_compact + k_map[e.k]] = e.digit;
    return out;
}

} // namespace ggml::potal::attention::detail
