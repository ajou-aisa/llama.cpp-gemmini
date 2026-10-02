#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace rmd = ggml::gemmini::rmd;
struct Event { size_t row, k; int32_t value; };
struct Input { std::string name; size_t rows, k; std::vector<Event> events; };
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<uint64_t> canonical(const rmd::StripePacket &p) {
    std::vector<uint64_t> result;
    const auto scalar = [&](auto value) { result.push_back(static_cast<uint64_t>(value)); };
    const auto sequence = [&](const auto &values) {
        scalar(values.size());
        for (auto value : values) scalar(value);
    };
    scalar(p.version); scalar(p.digit_bits); scalar(p.lane_capacity); scalar(p.digit_storage);
    scalar(p.stripe_id); scalar(p.row_begin); scalar(p.row_count); scalar(p.logical_k);
    scalar(p.logical_j); scalar(p.j_padded); scalar(p.block_size); scalar(p.array_dim);
    scalar(p.activation_value_count); scalar(p.residual_event_count);
    scalar(p.residual_min); scalar(p.residual_max); scalar(p.required_planes);
    scalar(p.digit_nnz); scalar(p.active_original_rows); scalar(p.active_original_rows_valid);
    scalar(p.residual_observations_valid); scalar(p.total_output_values);
    sequence(p.k_indices); sequence(p.stacked_activation.signed_int8);
    sequence(p.stacked_activation.signed_int16); scalar(p.blocks.size());
    for (const auto &b : p.blocks) {
        scalar(b.block_id); scalar(b.global_k_begin); scalar(b.compact_k_count);
        scalar(b.padded_k_count); scalar(b.active_lane_mask); scalar(b.active_lane_count);
        sequence(b.lane_ids); sequence(b.lane_k_masks); scalar(b.k_index_offset);
        scalar(b.activation_offset); scalar(b.activation_byte_offset);
        scalar(b.activation_byte_count); scalar(b.output_value_offset);
        scalar(b.rows_padded); scalar(b.lane_stride_values); scalar(b.groups.size());
        for (const auto &g : b.groups) {
            sequence(g.lane_positions); sequence(g.row_offsets); sequence(g.row_ids);
            scalar(g.k_mask); scalar(g.padded_k_count); scalar(g.activation_offset);
            scalar(g.activation_byte_offset); scalar(g.activation_byte_count);
        }
    }
    return result;
}

// Independent oracle reads raw payload and reconstructs original coordinates and integers.
// It calls neither implementation's decomposition nor packet-digit reader.
void check_source(const rmd::StripePacket &packet, const Input &input) {
    std::vector<int64_t> actual(input.rows * input.k, 0), expected(actual.size(), 0);
    for (const auto &e : input.events) expected.at(e.row * input.k + e.k) = e.value;
    for (const auto &block : packet.blocks) for (const auto &group : block.groups) {
        std::vector<size_t> indices;
        for (size_t k = 0; k < 32; ++k)
            if (group.k_mask & (uint32_t{1} << k)) indices.push_back(block.global_k_begin + k);
        for (size_t lane = 0; lane < group.lane_positions.size(); ++lane) {
            const auto original_lane = block.lane_ids[group.lane_positions[lane]];
            const int64_t multiplier = int64_t{1} << (packet.digit_bits * original_lane);
            for (size_t row = group.row_offsets[lane]; row < group.row_offsets[lane + 1]; ++row)
                for (size_t k = 0; k < indices.size(); ++k) {
                    const auto digit = packet.stacked_activation.signed_int8.at(
                        group.activation_offset + row * group.padded_k_count + k);
                    actual.at(group.row_ids.at(row) * input.k + indices[k]) += digit * multiplier;
                }
        }
    }
    require(actual == expected, "independent source reconstruction failed: " + input.name);
}


rmd::StripePacketHandle verify(rmd::RmdBitmapBuilder &bitmap, const Input &input,
                               uint8_t bits, size_t stride) {
    std::vector<uint64_t> mask((input.rows * stride + 63) / 64, 0);
    for (const auto &event : input.events) {
        const size_t cell = event.row * stride + event.k;
        mask[cell / 64] |= uint64_t{1} << (cell % 64);
    }
    rmd::RmdStripeBuilder legacy;
    legacy.reset(3, 7, input.rows, input.k, 65, bits);
    bitmap.reset(3, 7, input.rows, input.k, 65, bits, mask, stride);
    for (const auto &event : input.events) {
        require(legacy.add_residual(event.row, event.k, event.value), "legacy add");
        if (event.value) require(bitmap.emit(event.row, event.k, event.value), "bitmap emit");
    }
    std::fill(mask.begin(), mask.end(), 0);
    const auto expected = legacy.finish(), actual = bitmap.finish();
    require(legacy.status() == rmd::RmdStatus::success &&
            bitmap.status() == rmd::RmdStatus::success, "finish status");
    require(bool(expected) == bool(actual), "empty mismatch");
    if (actual) {
        require(canonical(*expected) == canonical(*actual), "packet mismatch");
        check_source(*actual, input);
        require(canonical(*bitmap.finish()) == canonical(*actual), "repeat finish mismatch");
    }
    return actual;
}

int main() {
    try {
        rmd::RmdBitmapBuilder reused;
        rmd::StripePacketHandle retained;
        std::vector<uint64_t> retained_bytes;
        for (uint8_t bits : {4, 8}) {
            for (size_t rows : {1, 3, 47, 64, 65, 129}) {
                for (size_t k : {1, 31, 32, 33, 64, 67}) {
                    Input input{"edges", rows, k, {}};
                    const int32_t values[] = {0, 1, -1, 8, -9, 128, -129, 257,
                                             65536, INT32_MIN, INT32_MAX};
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t col = 0; col < k; ++col)
                            if ((row + col) % 3 == 0)
                                input.events.push_back({row, col, values[(row + col) % std::size(values)]});
                    const auto packet = verify(reused, input, bits, rmd::align_up(k, 32));
                    if (packet && !retained) {
                        retained = packet;
                        retained_bytes = canonical(*packet);
                    }
                }
            }
            for (unsigned seed = 0; seed < 30; ++seed) {
                std::mt19937 rng(seed);
                Input input{"random", 1 + seed % 19, 1 + seed * 3, {}};
                for (size_t row = 0; row < input.rows; ++row)
                    for (size_t k = 0; k < input.k; ++k)
                        if (rng() % 7 == 0)
                            input.events.push_back({row, k, static_cast<int32_t>(rng())});
                verify(reused, input, bits, input.k + 3);
            }
            verify(reused, Input{"empty", 3, 65, {}}, bits, 96);
            verify(reused, Input{"selected zero", 2, 32, {{0, 2, 0}, {1, 3, 0}}}, bits, 32);
        }
        require(retained && canonical(*retained) == retained_bytes, "retained packet mutated");
        std::vector<uint64_t> mask(1, uint64_t{1} << 1);
        reused.reset(0, 0, 1, 64, 16, 4, mask, 64);
        require(reused.emit(0, 1, 257), "initial emission");
        mask[0] |= uint64_t{1} << 35;
        require(reused.emit(0, 35, -129), "forward mask promotion");
        mask.clear();
        require(bool(reused.finish()), "finish must not borrow selection");
        mask.assign(1, ~uint64_t{0});
        reused.reset(0, 0, 1, 64, 16, 4, mask, 64);
        require(reused.emit(0, 5, 1) && !reused.emit(0, 4, 1), "unordered input rejected");
        require(!reused.finish() && reused.status() == rmd::RmdStatus::invalid_arguments,
                "failed builder must not publish");
        reused.reset(0, 0, 1, 64, 16, 4, mask, 64);
        require(reused.emit(0, 5, 1) && !reused.emit(0, 5, 1), "duplicate rejected");
        reused.reset(0, 0, 1, 64, 16, 16, mask, 64);
        require(reused.status() == rmd::RmdStatus::invalid_arguments, "unsupported bitmap radix");
        mask.clear();
        reused.reset(0, 0, 1, 64, 16, 4, mask, 64);
        require(reused.status() == rmd::RmdStatus::invalid_arguments, "missing selection rejected");
        std::cout << "PASS: bitmap/legacy exact packets, independent reconstruction, lifetime and failure boundaries\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
