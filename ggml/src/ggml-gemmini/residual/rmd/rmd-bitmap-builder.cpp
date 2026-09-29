#include "rmd-bitmap-builder.hpp"
#include "rmd-builder.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

namespace ggml::gemmini::rmd {
namespace {

unsigned population(uint32_t mask) {
    return static_cast<unsigned>(__builtin_popcount(mask));
}

size_t tiles(size_t count) {
    return count / kArrayDim + (count % kArrayDim != 0);
}

bool fits_u32(size_t value) {
    return value <= std::numeric_limits<uint32_t>::max();
}

template<unsigned Bits>
uint8_t decompose(int32_t residual, std::array<int8_t, kMaxNativeRadixLanes> &packed,
                  uint16_t &mask, size_t &count,
                  std::array<uint32_t, kMaxNativeRadixLanes> &lane_masks, uint32_t k_bit,
                  uint64_t *lane_rows, size_t row_words, uint64_t row_bit) {
    constexpr int64_t radix = int64_t{1} << Bits;
    int64_t quotient = residual;
    uint8_t lane = 0;
    while (quotient != 0) {
        int64_t digit = static_cast<uint64_t>(quotient) & (radix - 1);
        if (digit >= radix / 2) digit -= radix;
        quotient = (quotient - digit) / radix;
        if (digit != 0) {
            mask |= uint16_t{1} << lane;
            packed[count++] = static_cast<int8_t>(digit);
            lane_masks[lane] |= k_bit;
            lane_rows[lane * row_words] |= row_bit;
        }
        ++lane;
    }
    return lane;
}

// Select by padded GEMM work using the original stripe height. Row compaction
// happens after this partition decision, preserving its independent contract.
std::array<uint8_t, kMaxNativeRadixLanes> partition(const BlockDescriptor &block,
                                                  size_t row_count) {
    std::array<uint8_t, kMaxNativeRadixLanes> best{};
    std::array<uint8_t, kMaxNativeRadixLanes> assignment{};
    std::array<uint32_t, kMaxNativeRadixLanes> masks{};
    std::array<size_t, kMaxNativeRadixLanes> counts{};
    const size_t budget = tiles(block.compact_k_count);
    if (budget == 1 || block.active_lane_count == 1) return best;
    size_t best_cost = budget * tiles(block.active_lane_count * row_count);
    masks[0] = block.lane_k_masks[block.lane_ids[0]];
    counts[0] = 1;
    auto visit = [&](auto &&self, size_t lane, size_t groups) -> void {
        if (lane == block.active_lane_count) {
            size_t used = 0;
            size_t cost = 0;
            for (size_t group = 0; group < groups; ++group) {
                const size_t k_tiles = tiles(population(masks[group]));
                used += k_tiles;
                cost += k_tiles * tiles(counts[group] * row_count);
            }
            if (used <= budget && cost < best_cost) {
                best_cost = cost;
                best = assignment;
            }
            return;
        }
        for (size_t group = 0; group < groups + (groups < budget); ++group) {
            const uint32_t old_mask = masks[group];
            masks[group] |= block.lane_k_masks[block.lane_ids[lane]];
            ++counts[group];
            assignment[lane] = static_cast<uint8_t>(group);
            self(self, lane + 1, std::max(groups, group + 1));
            --counts[group];
            masks[group] = old_mask;
        }
    };
    visit(visit, 1, 1);
    return best;
}

}

void RmdBitmapBuilder::reset(size_t stripe_id, size_t row_begin, size_t rows,
                               size_t k, size_t j, uint8_t bits,
                               const std::vector<uint64_t> &selection, size_t stride) {
    selection_ = &selection;
    stride_ = stride;
    cursor_ = 0;
    limb_masks_.clear();
    digits_.clear();
    metadata_ = StripePacket{};
    metadata_.stripe_id = stripe_id;
    metadata_.row_begin = row_begin;
    metadata_.row_count = rows;
    metadata_.logical_k = k;
    metadata_.logical_j = j;
    metadata_.j_padded = align_up(j, kArrayDim);
    metadata_.digit_bits = bits;
    metadata_.lane_capacity = bits == 4 ? 9 : 5;
    metadata_.digit_storage = DigitStorage::signed_int8;
#if LOG_CYCLE
    metadata_.active_original_rows_valid = true;
#endif
    metadata_.residual_observations_valid = true;
    status_ = RmdStatus::success;
    if ((bits != 4 && bits != 8) || !rows || !k || !j || stride < k) {
        status_ = RmdStatus::invalid_arguments;
        return;
    }
    if (!fits_u32(k) || !metadata_.j_padded ||
        align_up(rows, kArrayDim) > std::numeric_limits<uint16_t>::max() ||
        rows > std::numeric_limits<uint16_t>::max() ||
        row_begin > std::numeric_limits<size_t>::max() - rows ||
        stride > std::numeric_limits<size_t>::max() / rows) {
        status_ = RmdStatus::overflow;
        return;
    }
    const size_t cells = rows * stride;
    if (selection.size() < cells / 64 + (cells % 64 != 0)) {
        status_ = RmdStatus::invalid_arguments;
        return;
    }
    try {
        const size_t blocks = k / kBlockSize + (k % kBlockSize != 0);
        row_words_ = (rows + 63) / 64;
        block_lane_masks_.assign(blocks, {});
        lane_rows_.assign(blocks * metadata_.lane_capacity * row_words_, 0);
#if LOG_CYCLE
        row_seen_.assign(rows, 0);
#endif
        nonzero_mask_.assign(cells / 64 + (cells % 64 != 0), 0);
    } catch (const std::bad_alloc &) {
        status_ = RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        status_ = RmdStatus::overflow;
    }
}

bool RmdBitmapBuilder::emit(size_t row, size_t k, int32_t residual) {
    if (status_ != RmdStatus::success) return false;
    if (row >= metadata_.row_count || k >= metadata_.logical_k) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }
    const size_t cell = row * stride_ + k;
    if (cell < cursor_ || !((*selection_)[cell / 64] & (uint64_t{1} << (cell % 64)))) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }
    cursor_ = cell + 1;
    if (residual == 0) return true;
    std::array<int8_t, kMaxNativeRadixLanes> packed{};
    uint16_t mask = 0;
    size_t count = 0;
    const size_t block = k / kBlockSize;
    const uint32_t k_bit = uint32_t{1} << (k % kBlockSize);
    uint64_t *lane_rows = lane_rows_.data() + block * metadata_.lane_capacity * row_words_ + row / 64;
    const uint64_t row_bit = uint64_t{1} << (row % 64);
    const uint8_t lane_count = metadata_.digit_bits == 4
        ? decompose<4>(residual, packed, mask, count, block_lane_masks_[block], k_bit,
                       lane_rows, row_words_, row_bit)
        : decompose<8>(residual, packed, mask, count, block_lane_masks_[block], k_bit,
                       lane_rows, row_words_, row_bit);
    try {
        limb_masks_.push_back(mask);
        digits_.insert(digits_.end(), packed.begin(), packed.begin() + count);
    } catch (const std::bad_alloc &) {
        status_ = RmdStatus::allocation_failure;
        return false;
    } catch (const std::length_error &) {
        status_ = RmdStatus::overflow;
        return false;
    }
    nonzero_mask_[cell / 64] |= uint64_t{1} << (cell % 64);
    if (!metadata_.residual_event_count) {
        metadata_.residual_min = residual;
        metadata_.residual_max = residual;
    } else {
        metadata_.residual_min = std::min(metadata_.residual_min, residual);
        metadata_.residual_max = std::max(metadata_.residual_max, residual);
    }
    ++metadata_.residual_event_count;
#if LOG_CYCLE
    row_seen_[row] = 1;
#endif
    metadata_.digit_nnz += count;
    metadata_.required_planes = std::max(metadata_.required_planes, lane_count);
    return true;
}

StripePacketHandle RmdBitmapBuilder::finish() {
    if (status_ != RmdStatus::success || digits_.empty()) {
        return nullptr;
    }
    try {
        const size_t block_count = block_lane_masks_.size();
        std::vector<uint32_t> block_indices(block_count, 0);
        std::vector<uint32_t> row_ranks(lane_rows_.size(), 0);
        std::vector<std::array<uint8_t, kMaxNativeRadixLanes>> lane_groups(block_count);
        auto packet = std::make_shared<StripePacket>(metadata_);
        packet->blocks.reserve(block_count);
        packet->k_indices.reserve(std::min(digits_.size(), block_count * kBlockSize));
        const size_t rows_padded = align_up(metadata_.row_count, kArrayDim);
        if (packet->j_padded > std::numeric_limits<uint32_t>::max() / rows_padded) {
            status_ = RmdStatus::overflow;
            return nullptr;
        }

        size_t activation_values = 0;
        for (size_t block_id = 0; block_id < block_count; ++block_id) {
            BlockDescriptor block;
            block.lane_k_masks = block_lane_masks_[block_id];
            block.block_id = static_cast<uint32_t>(block_id);
            block.global_k_begin = static_cast<uint32_t>(block_id * kBlockSize);
            uint32_t block_mask = 0;
            for (size_t lane = 0; lane < packet->lane_capacity; ++lane) {
                block_mask |= block.lane_k_masks[lane];
                if (block.lane_k_masks[lane]) block.active_lane_mask |= uint16_t{1} << lane;
            }
            if (!block_mask) continue;
            block_indices[block_id] = static_cast<uint32_t>(packet->blocks.size());
            for (size_t lane = 0; lane < packet->lane_capacity; ++lane) {
                if (block.active_lane_mask & (uint16_t{1} << lane)) {
                    block.lane_ids[block.active_lane_count++] = static_cast<uint8_t>(lane);
                }
            }
            block.compact_k_count = static_cast<uint16_t>(population(block_mask));
            block.padded_k_count = static_cast<uint16_t>(align_up(block.compact_k_count, kArrayDim));
            if (!fits_u32(packet->k_indices.size()) ||
                !fits_u32(activation_values) ||
                !fits_u32(packet->total_output_values)) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            block.k_index_offset = static_cast<uint32_t>(packet->k_indices.size());
            for (size_t k = 0; k < kBlockSize; ++k) {
                if (block_mask & (uint32_t{1} << k)) {
                    packet->k_indices.push_back(static_cast<uint16_t>(k));
                }
            }
            block.activation_offset = static_cast<uint32_t>(activation_values);
            block.activation_byte_offset = block.activation_offset;
            block.output_value_offset = static_cast<uint32_t>(packet->total_output_values);
            block.rows_padded = static_cast<uint16_t>(rows_padded);
            block.lane_stride_values = static_cast<uint32_t>(rows_padded * packet->j_padded);
            const size_t output_count = size_t{block.lane_stride_values} * block.active_lane_count;
            if (output_count > std::numeric_limits<uint32_t>::max() - packet->total_output_values) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            packet->total_output_values += output_count;
            const auto assignment = partition(block, packet->row_count);
            const size_t group_count = *std::max_element(assignment.begin(),
                assignment.begin() + block.active_lane_count) + 1;
            block.groups.resize(group_count);
            for (size_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                auto &group = block.groups[assignment[lane_position]];
                group.lane_positions.push_back(static_cast<uint8_t>(lane_position));
                group.k_mask |= block.lane_k_masks[block.lane_ids[lane_position]];
            }
            for (size_t group_id = 0; group_id < block.groups.size(); ++group_id) {
                auto &group = block.groups[group_id];
                group.padded_k_count = static_cast<uint16_t>(align_up(population(group.k_mask), kArrayDim));
                size_t group_row_count = 0;
                for (uint8_t position : group.lane_positions) {
                    const size_t base = (block_id * packet->lane_capacity + block.lane_ids[position]) * row_words_;
                    for (size_t word = 0; word < row_words_; ++word)
                        group_row_count += static_cast<size_t>(__builtin_popcountll(lane_rows_[base + word]));
                }
                group.row_ids.reserve(group_row_count);
                for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
                    const uint8_t lane = block.lane_ids[group.lane_positions[group_lane]];
                    group.row_offsets[group_lane] = static_cast<uint32_t>(group.row_ids.size());
                    lane_groups[block_id][lane] = static_cast<uint8_t>(group_id);
                    const size_t base = (block_id * packet->lane_capacity + lane) * row_words_;
                    for (size_t word = 0; word < row_words_; ++word) {
                        row_ranks[base + word] = static_cast<uint32_t>(group.row_ids.size());
                        uint64_t rows = lane_rows_[base + word];
                        while (rows) {
                            group.row_ids.push_back(static_cast<uint16_t>(word * 64 + __builtin_ctzll(rows)));
                            rows &= rows - 1;
                        }
                    }
                }
                group.row_offsets[group.lane_positions.size()] = static_cast<uint32_t>(group.row_ids.size());
                const size_t group_rows = align_up(group.row_ids.size(), kArrayDim);
                const size_t value_count = group_rows * group.padded_k_count;
                if (value_count > std::numeric_limits<uint32_t>::max() - activation_values) {
                    status_ = RmdStatus::overflow;
                    return nullptr;
                }
                group.activation_offset = static_cast<uint32_t>(activation_values);
                group.activation_byte_offset = group.activation_offset;
                group.activation_byte_count = static_cast<uint32_t>(value_count);
                activation_values += value_count;
            }
            block.activation_byte_count = static_cast<uint32_t>(activation_values - block.activation_byte_offset);
            packet->blocks.push_back(std::move(block));

        }
        packet->stacked_activation.signed_int8.resize(activation_values, 0);
        size_t residual_index = 0, digit_index = 0;
        for (size_t word = 0; word < nonzero_mask_.size(); ++word) {
            uint64_t nonzero = nonzero_mask_[word];
            while (nonzero) {
                const size_t cell = word * 64 + static_cast<size_t>(__builtin_ctzll(nonzero));
                nonzero &= nonzero - 1;
                const size_t row = cell / stride_, k = cell % stride_;
                if (residual_index >= limb_masks_.size()) {
                    status_ = RmdStatus::invalid_arguments;
                    return nullptr;
                }
                uint32_t lanes = limb_masks_[residual_index++];
                const size_t block_id = k / kBlockSize;
                if (block_indices[block_id] >= packet->blocks.size() ||
                    packet->blocks[block_indices[block_id]].block_id != block_id) {
                    status_ = RmdStatus::invalid_arguments;
                    return nullptr;
                }
                const auto &block = packet->blocks[block_indices[block_id]];
                const uint32_t lower_k = (uint32_t{1} << (k % kBlockSize)) - 1;
                while (lanes) {
                    const size_t lane = static_cast<size_t>(__builtin_ctz(lanes));
                    const size_t index = (block_id * packet->lane_capacity + lane) * row_words_ + row / 64;
                    if (!(block.lane_k_masks[lane] & (uint32_t{1} << (k % kBlockSize))) ||
                        !(lane_rows_[index] & (uint64_t{1} << (row % 64)))) {
                        status_ = RmdStatus::invalid_arguments;
                        return nullptr;
                    }
                    const auto &group = block.groups[lane_groups[block_id][lane]];
                    const size_t compact_row = row_ranks[index] +
                        __builtin_popcountll(lane_rows_[index] & ((uint64_t{1} << (row % 64)) - 1));
                    const size_t compact_k = population(group.k_mask & lower_k);
                    packet->stacked_activation.signed_int8[
                        group.activation_offset + compact_row * group.padded_k_count + compact_k] =
                        digits_[digit_index++];
                    lanes &= lanes - 1;
                }
            }
        }
        if (residual_index != limb_masks_.size() || digit_index != digits_.size()) {
            status_ = RmdStatus::invalid_arguments;
            return nullptr;
        }
        packet->activation_value_count = packet->stacked_activation.signed_int8.size();
#if LOG_CYCLE
        packet->active_original_rows = static_cast<size_t>(
            std::count(row_seen_.begin(), row_seen_.end(), uint8_t{1}));
#endif
        status_ = validate_packet(*packet);
        if (status_ != RmdStatus::success) return nullptr;
        return packet;
    } catch (const std::bad_alloc &) {
        status_ = RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        status_ = RmdStatus::overflow;
    }
    return nullptr;
}

}
