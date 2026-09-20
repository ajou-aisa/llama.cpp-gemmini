#include "rmd-executor.hpp"

#include "rmd-builder.hpp"
#include "rmd-compose.hpp"
#include "rmd-im2p-executor.hpp"
#if CYCLE_SIM
#include <im2p_cycle_sim.hpp>
#endif

#include "../../ggml-gemmini-args.h"

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
#include <im2p_sim.h>
#endif
#include "../../quants/common/hp1_scu.hpp"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace ggml::gemmini::rmd {

static_assert(kNativeWeightScaleGroup == QK8_0);
static_assert(sizeof(elem_t) == GGML_GEMMINI_ACTIVATION_STORAGE_BYTES,
              "native activation staging must match elem_t storage");
static_assert(sizeof(elem_t) == GGML_GEMMINI_WEIGHT_STORAGE_BYTES,
              "native weight staging must match elem_t storage");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 16
                  ? std::is_same_v<elem_t, int16_t>
                  : std::is_same_v<elem_t, int8_t>,
              "native operand staging must use signed width-native elem_t");
static_assert(std::is_integral_v<acc_t> && std::is_signed_v<acc_t> &&
                  sizeof(acc_t) * 8 >= 32,
              "native SRMD requires a signed accumulator of at least 32 bits");

namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;

bool checked_mul_i64(int64_t lhs, int64_t rhs, int64_t &out) {
  const __int128 product =
      static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
  if (product > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
      product < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
    return false;
  }
  out = static_cast<int64_t>(product);
  return true;
}

bool checked_add_i64(int64_t lhs, int64_t rhs, int64_t &out) {
  const __int128 sum = static_cast<__int128>(lhs) + static_cast<__int128>(rhs);
  if (sum > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
      sum < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
    return false;
  }
  out = static_cast<int64_t>(sum);
  return true;
}

struct WeightGatherCounts {
  size_t values = 0;
  size_t baseline_address_resolutions = 0;
  size_t address_resolutions = 0;
};

class WeightGather {
public:
  WeightGather(const ggml_gemmini_args_t &args,
               const wroute::WeightRoutePlan &plan)
      : args_(args), plan_(plan) {}

  bool valid() const {
    return plan_.valid && wroute::weight_route_status(
                              plan_, wroute::WeightExecutionPath::Compact) ==
                              wroute::WeightRouteStatus::Success;
  }

  template <typename TileElement>
  RmdStatus fill_tile(uint32_t block_id, const uint16_t *local_k,
                      size_t valid_k, size_t col_base, size_t valid_cols,
                      TileElement *tile, size_t tile_stride,
                      WeightGatherCounts &counts) const {
    static_assert(std::is_integral_v<TileElement> &&
                      std::is_signed_v<TileElement>,
                  "compact tiles require signed integer elements");
    if (!valid() || tile == nullptr || tile_stride < valid_cols) {
      return RmdStatus::execution_failed;
    }

    std::array<int32_t, kArrayDim * kArrayDim> staged{};
    size_t address_resolutions = 0;
    if (wreader::read_code_tile_validated(args_, plan_, block_id, local_k,
                                          valid_k, col_base, valid_cols,
                                          staged.data(), address_resolutions) !=
        wreader::WeightReaderStatus::Success) {
      return RmdStatus::execution_failed;
    }
    for (size_t k = 0; k < valid_k; ++k) {
      for (size_t col = 0; col < valid_cols; ++col) {
        if (staged[k * kArrayDim + col] <
                static_cast<int64_t>(std::numeric_limits<TileElement>::min()) ||
            staged[k * kArrayDim + col] >
                static_cast<int64_t>(std::numeric_limits<TileElement>::max())) {
          return RmdStatus::overflow;
        }
      }
    }
    for (size_t k = 0; k < valid_k; ++k) {
      std::copy_n(staged.data() + k * kArrayDim, valid_cols,
                  tile + k * tile_stride);
    }

    counts.values = valid_k * valid_cols;
    counts.baseline_address_resolutions = counts.values;
    counts.address_resolutions = address_resolutions;
    return RmdStatus::success;
  }

  RmdStatus validate_native_packet(const StripePacket &packet) const {
    std::array<elem_t, kArrayDim * kArrayDim> tile{};
    for (const BlockDescriptor &block : packet.blocks) {
      for (size_t k_base = 0; k_base < block.compact_k_count;
           k_base += kArrayDim) {
        const size_t valid_k = std::min(
            kArrayDim, static_cast<size_t>(block.compact_k_count) - k_base);
        const uint16_t *local_k =
            packet.k_indices.data() + block.k_index_offset + k_base;
        for (size_t col_base = 0; col_base < packet.logical_j;
             col_base += kArrayDim) {
          const size_t valid_cols =
              std::min(kArrayDim, packet.logical_j - col_base);
          WeightGatherCounts ignored{};
          const RmdStatus status =
              fill_tile(block.block_id, local_k, valid_k, col_base, valid_cols,
                        tile.data(), kArrayDim, ignored);
          if (status != RmdStatus::success) {
            return status;
          }
        }
      }
    }
    return RmdStatus::success;
  }

private:
  const ggml_gemmini_args_t &args_;
  const wroute::WeightRoutePlan &plan_;
};

enum class CompactExecutorBackend : uint8_t {
  gemmini_ws,
  im2p_sim,
  checked_software,
};

// Owns one matrix in the selected native mvin representation. Logical strides
// remain element-based; only the transport bytes differ by configured width.
class NativeOperandBuffer {
public:
  RmdStatus assign(const int32_t *values, size_t count) {
    if (values == nullptr || count == 0) {
      return RmdStatus::invalid_arguments;
    }
    constexpr int32_t qmin =
        -(int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1));
    constexpr int32_t qmax =
        (int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1;
    for (size_t index = 0; index < count; ++index) {
      if (values[index] < qmin || values[index] > qmax) {
        return RmdStatus::overflow;
      }
    }
    try {
#if GGML_GEMMINI_ACTIVATION_BITS == 4
      const size_t byte_count = count / 2 + count % 2;
      packed_int4_.assign(byte_count, uint8_t{0});
      for (size_t index = 0; index < count; ++index) {
        const uint8_t nibble = static_cast<uint8_t>(values[index]) & 0x0fu;
        packed_int4_[index / 2] |=
            static_cast<uint8_t>(nibble << ((index % 2) * 4));
      }
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
      signed_int8_.resize(count);
      for (size_t index = 0; index < count; ++index) {
        signed_int8_[index] = static_cast<int8_t>(values[index]);
      }
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
      signed_int16_.resize(count);
      for (size_t index = 0; index < count; ++index) {
        signed_int16_[index] = static_cast<int16_t>(values[index]);
      }
#else
#error "unsupported native Gemmini activation width"
#endif
    } catch (const std::bad_alloc &) {
      return RmdStatus::allocation_failure;
    }
    return RmdStatus::success;
  }

  const elem_t *data() const {
#if GGML_GEMMINI_ACTIVATION_BITS == 4
    return reinterpret_cast<const elem_t *>(packed_int4_.data());
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
    return reinterpret_cast<const elem_t *>(signed_int8_.data());
#else
    return reinterpret_cast<const elem_t *>(signed_int16_.data());
#endif
  }

private:
  std::vector<uint8_t> packed_int4_;
  std::vector<int8_t> signed_int8_;
  std::vector<int16_t> signed_int16_;
};

} // namespace

#if defined(GGML_GEMMINI_TESTING)
RmdStatus gather_weight_tile_for_test(const ggml_gemmini_args_t &args,
                                      uint32_t block_id,
                                      const uint16_t *local_k, size_t valid_k,
                                      size_t col_base, size_t valid_cols,
                                      elem_t *tile, size_t tile_stride,
                                      RmdExecutionMetrics *metrics) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
    return RmdStatus::unsupported_route;
  }
  const WeightGather weights(args, plan);
  if (!weights.valid()) {
    return RmdStatus::unsupported_route;
  }
  WeightGatherCounts counts{};
  const RmdStatus gather_status =
      weights.fill_tile(block_id, local_k, valid_k, col_base, valid_cols, tile,
                        tile_stride, counts);
  if (gather_status != RmdStatus::success) {
    return gather_status;
  }
  if (metrics != nullptr) {
    metrics->weight_values_gathered += counts.values;
    metrics->weight_baseline_address_resolutions +=
        counts.baseline_address_resolutions;
    metrics->weight_address_resolutions += counts.address_resolutions;
  }
  return RmdStatus::success;
}

RmdStatus gather_wide_weight_tile_for_test(
    const ggml_gemmini_args_t &args, uint32_t block_id, const uint16_t *local_k,
    size_t valid_k, size_t col_base, size_t valid_cols, int32_t *tile,
    size_t tile_stride, RmdExecutionMetrics *metrics) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
    return RmdStatus::unsupported_route;
  }
  const WeightGather weights(args, plan);
  if (!weights.valid()) {
    return RmdStatus::unsupported_route;
  }
  WeightGatherCounts counts{};
  const RmdStatus gather_status =
      weights.fill_tile(block_id, local_k, valid_k, col_base, valid_cols, tile,
                        tile_stride, counts);
  if (gather_status != RmdStatus::success) {
    return gather_status;
  }
  if (metrics != nullptr) {
    metrics->weight_values_gathered += counts.values;
    metrics->weight_baseline_address_resolutions +=
        counts.baseline_address_resolutions;
    metrics->weight_address_resolutions += counts.address_resolutions;
  }
  return RmdStatus::success;
}

RmdStatus repeat_weight_tile_gather_for_test(
    const ggml_gemmini_args_t &args, uint32_t block_count,
    const uint16_t *local_k, size_t valid_k, size_t col_base, size_t valid_cols,
    size_t iterations, uint64_t &checksum, RmdExecutionMetrics &metrics) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
    return RmdStatus::unsupported_route;
  }
  const WeightGather weights(args, plan);
  if (!weights.valid()) {
    return RmdStatus::unsupported_route;
  }

  std::array<elem_t, kArrayDim * kArrayDim> tile{};
  uint64_t local_checksum = 0;
  RmdExecutionMetrics local_metrics{};
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    for (uint32_t block_id = 0; block_id < block_count; ++block_id) {
      WeightGatherCounts counts{};
      const RmdStatus gather_status =
          weights.fill_tile(block_id, local_k, valid_k, col_base, valid_cols,
                            tile.data(), kArrayDim, counts);
      if (gather_status != RmdStatus::success) {
        return gather_status;
      }
      for (size_t k = 0; k < valid_k; ++k) {
        for (size_t col = 0; col < valid_cols; ++col) {
          local_checksum += static_cast<uint8_t>(tile[k * kArrayDim + col]);
        }
      }
      local_metrics.weight_values_gathered += counts.values;
      local_metrics.weight_baseline_address_resolutions +=
          counts.baseline_address_resolutions;
      local_metrics.weight_address_resolutions += counts.address_resolutions;
    }
  }
  checksum = local_checksum;
  metrics = local_metrics;
  return RmdStatus::success;
}

RmdStatus repeat_scalar_weight_tile_gather_for_test(
    const ggml_gemmini_args_t &args, uint32_t block_count,
    const uint16_t *local_k, size_t valid_k, size_t col_base, size_t valid_cols,
    size_t iterations, uint64_t &checksum) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  if (!plan.valid || !wroute::route_supports_integer_block_scale(plan) ||
      plan.weight_bits != 8 || local_k == nullptr || valid_k == 0 ||
      valid_k > kArrayDim || valid_cols == 0 || valid_cols > kArrayDim ||
      col_base > args.J || valid_cols > args.J - col_base) {
    return RmdStatus::unsupported_route;
  }

  std::array<elem_t, kArrayDim * kArrayDim> tile{};
  uint64_t local_checksum = 0;
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    for (uint32_t block_id = 0; block_id < block_count; ++block_id) {
      for (size_t k = 0; k < valid_k; ++k) {
        const size_t global_k =
            static_cast<size_t>(block_id) * kBlockSize + local_k[k];
        if (local_k[k] >= kBlockSize || global_k >= args.K) {
          return RmdStatus::execution_failed;
        }
        for (size_t col = 0; col < valid_cols; ++col) {
          const wreader::WeightCodeResult code =
              wreader::read_code(args, plan, col_base + col, global_k);
          if (!code.ok() || code.value < std::numeric_limits<elem_t>::min() ||
              code.value > std::numeric_limits<elem_t>::max()) {
            return RmdStatus::execution_failed;
          }
          tile[k * kArrayDim + col] = static_cast<elem_t>(code.value);
        }
      }
      for (size_t k = 0; k < valid_k; ++k) {
        for (size_t col = 0; col < valid_cols; ++col) {
          local_checksum += static_cast<uint8_t>(tile[k * kArrayDim + col]);
        }
      }
    }
  }
  checksum = local_checksum;
  return RmdStatus::success;
}
#endif

RmdStatus RmdOutputAssembler::begin(const StripePacket &packet) {
  packet_ = &packet;
  output_ = nullptr;
  correction_ = nullptr;
  correction_values_.clear();
  m_tiles_ = (packet.row_count + kArrayDim - 1) / kArrayDim;
  j_tiles_ = (packet.logical_j + kArrayDim - 1) / kArrayDim;
  if (m_tiles_ == 0 || j_tiles_ == 0) {
    return RmdStatus::invalid_packet;
  }

  try {
    tile_offset_.assign(packet.blocks.size(), 0);
    size_t cursor = 0;
    for (size_t index = 0; index < packet.blocks.size(); ++index) {
      tile_offset_[index] = cursor;
      cursor += static_cast<size_t>(packet.blocks[index].active_lane_count) *
                m_tiles_ * j_tiles_;
    }
    expected_ = cursor;
    submitted_ = 0;
    seen_.assign(expected_, 0);
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }
  return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::begin(const StripePacket &packet,
                                    CompressedOutput &output) {
  const RmdStatus validation = validate_packet(packet);
  return validation == RmdStatus::success ? begin_validated(packet, output)
                                          : validation;
}

RmdStatus RmdOutputAssembler::begin_validated(const StripePacket &packet,
                                              CompressedOutput &output) {
  const RmdStatus status = begin(packet);
  if (status != RmdStatus::success)
    return status;
  try {
    output.domain = CompressedOutput::Domain::block_scaled_int64;
    output.j_padded = packet.j_padded;
    output.values.assign(packet.total_output_values, OutputValue{0});
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }
  output_ = &output;
  return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::begin(const StripePacket &packet,
                                    Correction &correction) {
  const RmdStatus validation = validate_packet(packet);
  return validation == RmdStatus::success ? begin_validated(packet, correction)
                                          : validation;
}

RmdStatus RmdOutputAssembler::begin_validated(const StripePacket &packet,
                                              Correction &correction) {
  const RmdStatus status = begin(packet);
  if (status != RmdStatus::success)
    return status;
  size_t count = 0;
  if (__builtin_mul_overflow(packet.row_count, packet.logical_j, &count)) {
    return RmdStatus::overflow;
  }
  try {
    correction_values_.assign(count, __int128{0});
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }
  correction_ = &correction;
  return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::submit(const PhysicalTile &tile) {
  if (packet_ == nullptr || (output_ == nullptr && correction_ == nullptr) ||
      tile.values == nullptr) {
    return RmdStatus::invalid_arguments;
  }
  if (tile.packet_block_index >= packet_->blocks.size()) {
    return RmdStatus::invalid_arguments;
  }
  const BlockDescriptor &block = packet_->blocks[tile.packet_block_index];
  if (tile.lane_position >= block.active_lane_count ||
      tile.lane_id != block.lane_ids[tile.lane_position] ||
      tile.m_tile >= m_tiles_ || tile.j_tile >= j_tiles_) {
    return RmdStatus::invalid_arguments;
  }

  const size_t row_base = static_cast<size_t>(tile.m_tile) * kArrayDim;
  const size_t col_base = static_cast<size_t>(tile.j_tile) * kArrayDim;
  const size_t expected_rows =
      std::min(kArrayDim, packet_->row_count - row_base);
  const size_t expected_cols =
      std::min(kArrayDim, packet_->logical_j - col_base);
  if (tile.valid_rows != expected_rows || tile.valid_cols != expected_cols) {
    return RmdStatus::invalid_arguments;
  }

  const size_t slot =
      tile_offset_[tile.packet_block_index] +
      (static_cast<size_t>(tile.lane_position) * m_tiles_ + tile.m_tile) *
          j_tiles_ +
      tile.j_tile;
  if (slot >= seen_.size()) {
    return RmdStatus::invalid_arguments;
  }
  if (seen_[slot] != 0) {
    return RmdStatus::invalid_arguments; // duplicate tile
  }
  seen_[slot] = 1;
  ++submitted_;

  if (correction_ != nullptr) {
    // Pruned lane positions are storage indices; the original lane ID sets
    // radix weight. Multiply by a positive power of two instead of
    // left-shifting a negative value.
    const __int128 place = __int128{1} << (packet_->digit_bits * tile.lane_id);
    for (size_t row = 0; row < tile.valid_rows; ++row) {
      const size_t destination =
          (row_base + row) * packet_->logical_j + col_base;
      for (size_t col = 0; col < tile.valid_cols; ++col) {
        __int128 contribution = tile.values[row * kArrayDim + col];
        // Lane/block terms may exceed INT64 and cancel; narrow only in
        // finish().
        __int128 &total = correction_values_[destination + col];
        if (__builtin_mul_overflow(contribution, place, &contribution) ||
            __builtin_add_overflow(total, contribution, &total)) {
          return RmdStatus::overflow;
        }
      }
    }
    return RmdStatus::success;
  }

  const size_t lane_base =
      block.output_value_offset +
      static_cast<size_t>(tile.lane_position) * block.lane_stride_values;
  for (size_t row = 0; row < tile.valid_rows; ++row) {
    const size_t destination =
        lane_base + (row_base + row) * output_->j_padded + col_base;
    if (destination + tile.valid_cols > output_->values.size()) {
      return RmdStatus::invalid_packet;
    }
    std::copy_n(tile.values + row * kArrayDim, tile.valid_cols,
                output_->values.begin() + static_cast<ptrdiff_t>(destination));
  }
  return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::finish() {
  if (packet_ == nullptr) {
    return RmdStatus::invalid_arguments;
  }
  RmdStatus status =
      submitted_ == expected_ ? RmdStatus::success : RmdStatus::invalid_packet;
  if (status == RmdStatus::success && correction_ != nullptr) {
    try {
      BlockScaledInt64Correction staged;
      staged.values.resize(correction_values_.size());
      for (size_t index = 0; index < correction_values_.size(); ++index) {
        const __int128 value = correction_values_[index];
        if (value > std::numeric_limits<int64_t>::max() ||
            value < std::numeric_limits<int64_t>::min()) {
          status = RmdStatus::overflow;
          break;
        }
        staged.values[index] = static_cast<int64_t>(value);
      }
      if (status == RmdStatus::success) {
        *correction_ = std::move(staged);
      }
    } catch (const std::bad_alloc &) {
      status = RmdStatus::allocation_failure;
    }
  }
  packet_ = nullptr;
  output_ = nullptr;
  correction_ = nullptr;
  correction_values_.clear();
  seen_.clear();
  tile_offset_.clear();
  return status;
}

void collect_packet_metrics(const StripePacket &packet,
                            RmdExecutionMetrics &metrics) {
  metrics.active_blocks = packet.blocks.size();
  metrics.active_lanes = 0;
  metrics.compact_k_count = 0;
  metrics.padded_k_count = 0;
  metrics.block_padding_zeros = 0;
  metrics.row_padding_zeros = 0;
  metrics.compressed_output_values = packet.total_output_values;
  metrics.packet_bytes =
      packet.blocks.size() * sizeof(BlockDescriptor) +
      packet.k_indices.size() * sizeof(uint16_t) +
      packet.stacked_activation.packed_int4.size() +
      packet.stacked_activation.signed_int8.size() * sizeof(int8_t) +
      packet.stacked_activation.signed_int16.size() * sizeof(int16_t) +
      sizeof(StripePacket);

  const size_t m_tiles = (packet.row_count + kArrayDim - 1) / kArrayDim;
  const size_t j_tiles = (packet.logical_j + kArrayDim - 1) / kArrayDim;
  metrics.physical_tile_count = 0;
  metrics.matmul_call_count = 0;
  metrics.lane_group_count = 0;
  metrics.baseline_stacked_i_tile_count = 0;
  metrics.stacked_i_tile_count = 0;
  metrics.weight_values_gathered = 0;
  metrics.weight_baseline_address_resolutions = 0;
  metrics.weight_address_resolutions = 0;

  for (const BlockDescriptor &block : packet.blocks) {
    metrics.active_lanes += block.active_lane_count;
    metrics.compact_k_count += block.compact_k_count;
    metrics.padded_k_count += block.padded_k_count;
    metrics.physical_tile_count +=
        static_cast<size_t>(block.active_lane_count) * m_tiles * j_tiles;
    metrics.baseline_stacked_i_tile_count +=
        (block.padded_k_count / kArrayDim) * block.active_lane_count * m_tiles *
        j_tiles;
    metrics.packet_bytes += block.groups.size() * sizeof(LaneGroupDescriptor);
    for (const LaneGroupDescriptor &group : block.groups) {
      metrics.packet_bytes += group.lane_positions.size() * sizeof(uint8_t);
      const size_t valid_rows = group.lane_positions.size() * packet.row_count;
      const size_t stacked_rows = align_up(valid_rows, kArrayDim);
      const size_t k_pad =
          group.padded_k_count -
          static_cast<size_t>(__builtin_popcount(group.k_mask));
      metrics.block_padding_zeros += stacked_rows * k_pad;
      metrics.row_padding_zeros +=
          (stacked_rows - valid_rows) * group.padded_k_count;
    }
  }
  metrics.j_padding_zeros = (packet.j_padded - packet.logical_j) *
                            packet.row_count * metrics.active_lanes;
}

static RmdStatus validate_execution_request(const ggml_gemmini_args_t &args,
                                            const StripePacket &packet) {
  const RmdStatus validation = validate_packet(packet);
  if (validation != RmdStatus::success) {
    return validation;
  }
  if (packet.logical_j != args.J || packet.logical_k != args.K) {
    return RmdStatus::invalid_arguments;
  }
  return RmdStatus::success;
}

static RmdStatus compact_plan_status(const ggml_gemmini_args_t &args,
                                     const wroute::WeightRoutePlan &plan) {
  if (!plan.valid) {
    if (plan.route == wroute::WeightRouteKind::HP1 &&
        wreader::validate(args, plan) ==
            wreader::WeightReaderStatus::ScaleOverflow) {
      return RmdStatus::overflow;
    }
    return RmdStatus::unsupported_route;
  }
  if (wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) !=
      wroute::WeightRouteStatus::Success) {
    return RmdStatus::unsupported_route;
  }
  // A populated activation buffer identifies the runtime artifact width. Some
  // focused packet tests intentionally omit it, but a present identity may
  // never disagree with the weight reader selected for this compact request.
  if (args.A.valid() && args.A.bits != plan.weight_bits) {
    return RmdStatus::unsupported_route;
  }
  return RmdStatus::success;
}

namespace detail {
struct RmdAssemblerAccess {
  template <typename Output>
  static RmdStatus begin(RmdOutputAssembler &assembler,
                         const StripePacket &packet, Output &output) {
    return assembler.begin_validated(packet, output);
  }
};
} // namespace detail

template <CompactExecutorBackend Backend, typename Output>
RmdStatus execute_rmd_stripe_impl(
    const ggml_gemmini_args_t &args, const StripePacket &packet,
    const wroute::WeightRoutePlan &plan, Output &output,
    RmdExecutionMetrics *metrics, im2p_sim_t *im2p_sim = nullptr,
    Im2pProviderTestFault im2p_fault = Im2pProviderTestFault::none,
    const Im2pFullExecutor *executor = nullptr) {
  const WeightGather weights(args, plan);
  if (!weights.valid())
    return RmdStatus::unsupported_route;
  const bool hp1_scu = plan.hp1_carriers;
  if (hp1_scu && (packet.digit_bits != 4 && packet.digit_bits != 8))
    return RmdStatus::unsupported_route;
  if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
    if (hp1_scu)
      return RmdStatus::unsupported_route; // no raw-WS fallback
  }
#if defined(IM2P_FPGA_ARCH_GEMMINI_HP1) ||                                     \
    defined(IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1)
  if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
    if (!hp1_scu)
      return RmdStatus::unsupported_route;
  }
#endif
  if constexpr (Backend != CompactExecutorBackend::checked_software) {
    if (packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
        plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
        packet.digit_bits != plan.weight_bits) {
      return RmdStatus::unsupported_route;
    }
    // Validated immutable native H1/HP1 storage and matching widths already
    // bound every code. Other layouts still need the pre-dispatch scan.
    if (!plan.native_weight_blocks ||
        (plan.route != wroute::WeightRouteKind::H1 &&
         plan.route != wroute::WeightRouteKind::HP1)) {
      const RmdStatus native_status = weights.validate_native_packet(packet);
      if (native_status != RmdStatus::success) {
        return native_status;
      }
    }
  }

#if CYCLE_SIM
  const auto event_context = Backend == CompactExecutorBackend::im2p_sim
      ? args.cycle_sim_context : cycle_sim::Context{};
  std::vector<uint64_t> packet_work_ids;
  im2p::gemmini::cycle_sim::WorkCollector packet_collector(packet_work_ids, true);
  im2p::gemmini::cycle_sim::StageCall prepare_call(event_context, cycle_sim::CallKind::ResidualPrepare);
  im2p::gemmini::cycle_sim::HostStageScope preparation(event_context,
      "im2p.residual_preparation", "POTAL_HOST", "llama.cpp-gemmini",
      "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_impl",
      args.matmul_layer.c_str(), {}, args.cycle_sim_host_dependencies, true);
#endif
  Output staged_output;
  RmdOutputAssembler assembler;
  const RmdStatus begin_status =
      detail::RmdAssemblerAccess::begin(assembler, packet, staged_output);
  if (begin_status != RmdStatus::success) {
    return begin_status;
  }

  const size_t m_tiles = (packet.row_count + kArrayDim - 1) / kArrayDim;
  const size_t j_tiles = (packet.logical_j + kArrayDim - 1) / kArrayDim;
  size_t matmul_call_count = 0;
  size_t lane_group_count = 0;
  size_t stacked_i_tile_count = 0;
  size_t weight_values_gathered = 0;
  size_t weight_baseline_address_resolutions = 0;
  size_t weight_address_resolutions = 0;
  RmdExecutionMetrics staged_metrics{};

  size_t max_stacked_rows = 0;
  for (const BlockDescriptor &block : packet.blocks) {
    for (const LaneGroupDescriptor &group : block.groups) {
      const size_t stacked_rows =
          align_up(group.lane_positions.size() * packet.row_count, kArrayDim);
      max_stacked_rows = std::max(max_stacked_rows, stacked_rows);
    }
  }

  std::vector<OutputValue> stacked_values;
  std::vector<int32_t> weight_tile;
  NativeOperandBuffer native_weight_tile;
  std::vector<acc_t> ws_values;
  std::vector<OutputValue> im2p_values;
  std::vector<int8_t> unpacked_int4;
  std::vector<uint64_t> block_scales;
  std::vector<uint32_t> hp1_carriers;
  std::vector<int32_t> compact_weights;
  detail::Im2pProviderStatsAggregate im2p_stats{};
  try {
    stacked_values.assign(max_stacked_rows * kArrayDim, OutputValue{0});
    weight_tile.assign(kArrayDim * kArrayDim, int32_t{0});
    if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
      ws_values.assign(max_stacked_rows * kArrayDim, acc_t{0});
    } else if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
      im2p_values.assign(max_stacked_rows * kArrayDim, OutputValue{0});
    }
    block_scales.assign(kArrayDim, uint64_t{0});
    if (hp1_scu) {
      hp1_carriers.resize(kArrayDim);
      compact_weights.resize(kBlockSize * kArrayDim);
    }
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }

#if CYCLE_SIM
  preparation.finish();
  auto packet_host_dependencies = args.cycle_sim_host_dependencies;
  if (preparation.id()) packet_host_dependencies.push_back(*preparation.id());
  prepare_call.finish();
#endif
  for (size_t block_index = 0; block_index < packet.blocks.size();
       ++block_index) {
#if CYCLE_SIM
    im2p::gemmini::cycle_sim::HostStageScope block_preparation(event_context,
        "im2p.residual_block_preparation", "POTAL_HOST", "llama.cpp-gemmini",
        "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_impl",
        args.matmul_layer.c_str(), {}, packet_host_dependencies, true);
#endif
    const BlockDescriptor &block = packet.blocks[block_index];
    lane_group_count += block.groups.size();
    std::array<std::array<uint16_t, kBlockSize>, kMaxNativeRadixLanes>
        group_k{};
    std::array<size_t, kMaxNativeRadixLanes> group_k_counts{};
    for (size_t group_index = 0; group_index < block.groups.size();
         ++group_index) {
      for (uint32_t remaining = block.groups[group_index].k_mask;
           remaining != 0; remaining &= remaining - 1) {
        group_k[group_index][group_k_counts[group_index]++] =
            static_cast<uint16_t>(__builtin_ctz(remaining));
      }
    }
    if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
      if (packet.digit_bits == 4) {
        // IM2P consumes one signed byte per A4 digit, unlike the packed packet.
        // Decode once per block and reuse across all J tiles and lane groups.
        try {
          unpacked_int4.resize(
              static_cast<size_t>(block.activation_byte_count) * 2);
        } catch (const std::bad_alloc &) {
          return RmdStatus::allocation_failure;
        }
        for (size_t index = 0; index < unpacked_int4.size(); ++index) {
          const uint8_t packed =
              packet.stacked_activation
                  .packed_int4[block.activation_byte_offset + index / 2];
          // Even digits occupy the low nibble; XOR/subtract sign-extends INT4.
          const uint8_t raw = (packed >> ((index % 2) * 4)) & 15;
          unpacked_int4[index] = static_cast<int8_t>((raw ^ 8) - 8);
        }
      }
    }

#if CYCLE_SIM
    block_preparation.finish();
    auto block_host_dependencies = packet_host_dependencies;
    if (block_preparation.id()) block_host_dependencies.push_back(*block_preparation.id());
#endif
    for (size_t j_tile = 0; j_tile < j_tiles; ++j_tile) {
#if CYCLE_SIM
      im2p::gemmini::cycle_sim::HostStageScope column_preparation(event_context,
          "im2p.residual_carrier_preparation", "POTAL_HOST", "llama.cpp-gemmini",
          "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_impl",
          args.matmul_layer.c_str(), {}, block_host_dependencies, true);
#endif
      const size_t col_base = j_tile * kArrayDim;
      const size_t valid_cols =
          std::min(kArrayDim, packet.logical_j - col_base);

      // Preserve the original block carrier. No expanded HP1 factor is
      // computed on the production NPU path (including exponents >= 63).
      for (size_t col = 0; col < valid_cols; ++col) {
        if (hp1_scu) {
          const auto metadata = wreader::read_hp1_carrier_validated(
              args, plan, col_base + col, block.block_id);
          if (!metadata.ok())
            return RmdStatus::unsupported_route;
          hp1_carriers[col] = metadata.carrier;
        } else {
          block_scales[col] =
              plan.scales.row_header_mode || plan.scales.scalar_mode ||
                      plan.scales.channel_mode
                  ? 1
                  : wreader::read_scale_validated(args, plan, col_base + col,
                                                  block.block_id)
                        .integer_block_scale;
        }
      }

#if CYCLE_SIM
      column_preparation.finish();
      auto column_host_dependencies = block_host_dependencies;
      if (column_preparation.id()) column_host_dependencies.push_back(*column_preparation.id());
#endif
      for (size_t group_index = 0; group_index < block.groups.size();
           ++group_index) {
#if CYCLE_SIM
        im2p::gemmini::cycle_sim::HostStageScope gather(event_context,
            "im2p.residual_gather", "POTAL_HOST", "llama.cpp-gemmini",
            "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_impl",
            args.matmul_layer.c_str(), {}, column_host_dependencies, true);
#endif
        const LaneGroupDescriptor &group = block.groups[group_index];
        const size_t k_tiles = group.padded_k_count / kArrayDim;
        const size_t stacked_rows =
            align_up(group.lane_positions.size() * packet.row_count, kArrayDim);
        const size_t stacked_value_count = stacked_rows * kArrayDim;
        std::fill_n(stacked_values.begin(), stacked_value_count,
                    OutputValue{0});
#if CYCLE_SIM
        const size_t required_begin = im2p::gemmini::cycle_sim::work_count();
#endif

        if (hp1_scu) {
          const size_t compact_k = group_k_counts[group_index];
          const size_t logical_rows =
              group.lane_positions.size() * packet.row_count;
          // The gather primitive is DIM-sized, not the NPU request.
          // Append every selected row before the ONE logical K<=32 call.
          for (size_t base = 0; base < compact_k; base += kArrayDim) {
            const size_t count = std::min(kArrayDim, compact_k - base);
            WeightGatherCounts gathered{};
            const auto status = weights.fill_tile(
                block.block_id, group_k[group_index].data() + base, count,
                col_base, valid_cols, compact_weights.data() + base * kArrayDim,
                kArrayDim, gathered);
            if (status != RmdStatus::success)
              return status;
            weight_values_gathered += gathered.values;
            weight_baseline_address_resolutions +=
                gathered.baseline_address_resolutions;
            weight_address_resolutions += gathered.address_resolutions;
          }
#if CYCLE_SIM
          gather.finish();
          auto dot_host_dependencies = column_host_dependencies;
          if (gather.id()) dot_host_dependencies.push_back(*gather.id());
#endif
          ++matmul_call_count;
          stacked_i_tile_count += stacked_rows / kArrayDim;
          if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
            const size_t offset = group.activation_offset;
            const detail::Im2pCompactDot dot{
                packet.digit_bits,
                packet.digit_bits == 8
                    ? static_cast<const void *>(
                          packet.stacked_activation.signed_int8.data() + offset)
                    : static_cast<const void *>(unpacked_int4.data() + offset -
                                                block.activation_offset),
                logical_rows,
                group.padded_k_count,
                compact_weights.data(),
                valid_cols,
                kArrayDim,
                compact_k,
                hp1_carriers.data(),
                block.block_id,
                args.optrace_context,
                args.optrace_context ? args.matmul_layer : std::string{},
                packet.row_begin, packet.row_count, packet.stripe_id,
                col_base, group_index
#if CYCLE_SIM
                , args.cycle_sim_context, dot_host_dependencies
#endif
            };
            if (im2p_fault == Im2pProviderTestFault::cancel_after_first_dot &&
                staged_metrics.im2p_dot_calls != 0)
              return RmdStatus::execution_failed;
            const auto status = detail::execute_im2p_compact_dot(
                im2p_sim, dot, stacked_values.data(), kArrayDim, im2p_stats,
                im2p_fault, executor);
            if (status != RmdStatus::success)
              return status;
            ++staged_metrics.im2p_dot_calls;
            // Legacy fault names remain test-only rollback probes;
            // there is no production host K-add or HP1 scale multiply.
            if (im2p_fault == Im2pProviderTestFault::k_accumulation_overflow ||
                im2p_fault == Im2pProviderTestFault::block_scale_overflow)
              return RmdStatus::overflow;
          } else if constexpr (Backend ==
                               CompactExecutorBackend::checked_software) {
            const auto digit = [&](size_t row, size_t k) -> int32_t {
              const size_t index =
                  group.activation_offset + row * group.padded_k_count + k;
              if (packet.digit_bits == 8)
                return packet.stacked_activation.signed_int8[index];
              const uint8_t raw =
                  (packet.stacked_activation.packed_int4[index / 2] >>
                   ((index % 2) * 4)) &
                  15;
              return int32_t(raw ^ 8) - 8;
            };
            const size_t fragment_k = std::min(kArrayDim, kBlockSize);
            for (size_t row = 0; row < logical_rows; ++row) {
              for (size_t col = 0; col < valid_cols; ++col) {
                int32_t acc = 0;
                for (size_t base = 0; base < compact_k; base += fragment_k) {
                  int64_t partial = 0;
                  for (size_t k = base;
                       k < std::min(compact_k, base + fragment_k); ++k)
                    partial += int64_t(digit(row, k)) *
                               compact_weights[k * kArrayDim + col];
                  if (partial < INT32_MIN || partial > INT32_MAX)
                    return RmdStatus::overflow;
                  const auto q = quants::hp1::apply_validated(
                      static_cast<int32_t>(partial), hp1_carriers[col]);
                  acc = base == 0 ? q : quants::hp1::accumulate(acc, q);
                }
                stacked_values[row * kArrayDim + col] = acc;
              }
            }
          }
        } else
          for (size_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const size_t k_base = k_tile * kArrayDim;
            const size_t valid_k =
                group_k_counts[group_index] > k_base
                    ? std::min(kArrayDim, group_k_counts[group_index] - k_base)
                    : 0;
            if (valid_k == 0) {
              continue;
            }
            ++matmul_call_count;
            stacked_i_tile_count += stacked_rows / kArrayDim;
            WeightGatherCounts gather_counts{};
            const RmdStatus gather_status = weights.fill_tile(
                block.block_id, group_k[group_index].data() + k_base, valid_k,
                col_base, valid_cols, weight_tile.data(), kArrayDim,
                gather_counts);
            if (gather_status != RmdStatus::success) {
              return gather_status;
            }
            weight_values_gathered += gather_counts.values;
            weight_baseline_address_resolutions +=
                gather_counts.baseline_address_resolutions;
            weight_address_resolutions += gather_counts.address_resolutions;
            const size_t activation_offset = group.activation_offset + k_base;
            if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
              std::fill_n(im2p_values.begin(), stacked_value_count,
                          OutputValue{0});
              const detail::Im2pCompactDot dot{
                  packet.digit_bits,
                  packet.digit_bits == 16
                      ? static_cast<const void *>(
                            packet.stacked_activation.signed_int16.data() +
                            activation_offset)
                  : packet.digit_bits == 8
                      ? static_cast<const void *>(
                            packet.stacked_activation.signed_int8.data() +
                            activation_offset)
                      : static_cast<const void *>(unpacked_int4.data() +
                                                  activation_offset -
                                                  block.activation_offset),
                  // Keep padded storage, but execute only the original lane
                  // rows.
                  group.lane_positions.size() * packet.row_count,
                  group.padded_k_count *
                      (packet.digit_bits == 16 ? sizeof(int16_t) : 1),
                  weight_tile.data(),
                  valid_cols,
                  kArrayDim,
                  valid_k,
              };
              if (im2p_fault == Im2pProviderTestFault::cancel_after_first_dot &&
                  staged_metrics.im2p_dot_calls != 0) {
                return RmdStatus::execution_failed;
              }
              const RmdStatus dot_status = detail::execute_im2p_compact_dot(
                  im2p_sim, dot, im2p_values.data(), kArrayDim, im2p_stats,
                  im2p_fault, executor);
              if (dot_status != RmdStatus::success) {
                return dot_status;
              }
              if (im2p_fault ==
                  Im2pProviderTestFault::k_accumulation_overflow) {
                std::fill_n(im2p_values.begin(), stacked_value_count,
                            staged_metrics.im2p_dot_calls == 0
                                ? std::numeric_limits<OutputValue>::max()
                                : OutputValue{1});
              }
              ++staged_metrics.im2p_dot_calls;
              for (size_t row = 0; row < stacked_rows; ++row) {
                OutputValue *accumulator =
                    stacked_values.data() + row * kArrayDim;
                for (size_t col = 0; col < valid_cols; ++col) {
                  if (!checked_add_i64(accumulator[col],
                                       im2p_values[row * kArrayDim + col],
                                       accumulator[col])) {
                    return RmdStatus::overflow;
                  }
                }
              }
            } else if constexpr (Backend ==
                                 CompactExecutorBackend::gemmini_ws) {
              const RmdStatus staging_status = native_weight_tile.assign(
                  weight_tile.data(), weight_tile.size());
              if (staging_status != RmdStatus::success) {
                return staging_status;
              }
              const elem_t *native_activation_tile = nullptr;
#if GGML_GEMMINI_ACTIVATION_BITS == 4
              native_activation_tile = reinterpret_cast<const elem_t *>(
                  packet.stacked_activation.packed_int4.data() +
                  activation_offset / 2);
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
              native_activation_tile =
                  packet.stacked_activation.signed_int8.data() +
                  activation_offset;
#else
              native_activation_tile =
                  packet.stacked_activation.signed_int16.data() +
                  activation_offset;
#endif
              const elem_t *native_weight = native_weight_tile.data();
              if (native_activation_tile == nullptr ||
                  native_weight == nullptr) {
                return RmdStatus::execution_failed;
              }
              ++staged_metrics.ws_call_count;
              std::fill_n(ws_values.begin(), stacked_value_count, acc_t{0});
              tiled_matmul(stacked_rows, valid_cols, valid_k,
                           native_activation_tile, native_weight, nullptr,
                           ws_values.data(), group.padded_k_count, kArrayDim, 0,
                           kArrayDim, 1.0f, 1.0f, 1.0f, NO_ACTIVATION,
                           ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, false, 1, 1,
                           1, false, false, true, false, 0, WS);
              for (size_t row = 0; row < stacked_rows; ++row) {
                OutputValue *accumulator =
                    stacked_values.data() + row * kArrayDim;
                for (size_t col = 0; col < valid_cols; ++col) {
                  if (!checked_add_i64(accumulator[col],
                                       ws_values[row * kArrayDim + col],
                                       accumulator[col])) {
                    return RmdStatus::overflow;
                  }
                }
              }
            } else {
              const auto accumulate = [&](auto read_digit) {
                for (size_t row = 0; row < stacked_rows; ++row) {
                  OutputValue *accumulator =
                      stacked_values.data() + row * kArrayDim;
                  for (size_t k = 0; k < valid_k; ++k) {
                    const int64_t digit =
                        read_digit(row * group.padded_k_count + k);
                    if (digit == 0) {
                      continue;
                    }
                    const int32_t *weight_row =
                        weight_tile.data() + k * kArrayDim;
                    for (size_t col = 0; col < valid_cols; ++col) {
                      int64_t product = 0;
                      if (!checked_mul_i64(digit, weight_row[col], product) ||
                          !checked_add_i64(accumulator[col], product,
                                           accumulator[col])) {
                        return RmdStatus::overflow;
                      }
                    }
                  }
                }
                return RmdStatus::success;
              };
              RmdStatus dot_status;
              if (packet.digit_bits == 8) {
                const int8_t *activation =
                    packet.stacked_activation.signed_int8.data() +
                    activation_offset;
                dot_status =
                    accumulate([&](size_t index) { return activation[index]; });
              } else if (packet.digit_bits == 16) {
                const int16_t *activation =
                    packet.stacked_activation.signed_int16.data() +
                    activation_offset;
                dot_status =
                    accumulate([&](size_t index) { return activation[index]; });
              } else {
                const uint8_t *activation =
                    packet.stacked_activation.packed_int4.data() +
                    activation_offset / 2;
                dot_status = accumulate([&](size_t index) {
                  const uint8_t raw =
                      (activation[index / 2] >> ((index % 2) * 4)) & 15;
                  return static_cast<int32_t>(raw ^ 8) - 8;
                });
              }
              if (dot_status != RmdStatus::success)
                return dot_status;
            }
          }

#if defined(GGML_GEMMINI_TESTING)
        WsCallObservation observation{};
        if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
          observation.rows = stacked_rows;
          observation.cols = valid_cols;
          observation.k = group_k_counts[group_index];
          observation.lane_id = block.lane_ids[group.lane_positions.front()];
          int32_t first_activation = 0;
          const uint16_t first_k = group_k[group_index][0];
          const auto compact_begin =
              packet.k_indices.begin() + block.k_index_offset;
          const size_t first_compact = static_cast<size_t>(
              std::lower_bound(compact_begin,
                               compact_begin + block.compact_k_count, first_k) -
              compact_begin);
          if (read_packet_digit(packet, block, group.lane_positions.front(), 0,
                                first_compact,
                                first_activation) != RmdStatus::success) {
            return RmdStatus::invalid_packet;
          }
          observation.first_activation = static_cast<elem_t>(first_activation);
          observation.first_weight = static_cast<elem_t>(weight_tile.front());
          observation.raw_value = stacked_values.front();
          if (metrics != nullptr) {
            for (size_t group_lane = 0;
                 group_lane < group.lane_positions.size(); ++group_lane) {
              staged_metrics.raw_lane_values.push_back(
                  stacked_values[group_lane * packet.row_count * kArrayDim]);
            }
          }
          for (size_t row = 0; row < stacked_rows; ++row) {
            for (size_t col = 0; col < valid_cols; ++col) {
              if (stacked_values[row * kArrayDim + col] != 0) {
                ++observation.raw_nonzero_count;
              }
            }
          }
          observation.block_scale = block_scales.front();
        }
#endif
        // Only explicitly selected legacy/non-HP1 routes use host block
        // scaling. Production HP1 lane values already passed SCU+Sat32.
        if (!hp1_scu) {
          if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
            if (im2p_fault == Im2pProviderTestFault::block_scale_overflow) {
              std::fill_n(stacked_values.begin(), stacked_value_count,
                          std::numeric_limits<OutputValue>::max());
              std::fill_n(block_scales.begin(), valid_cols, uint64_t{2});
            }
          }
          for (size_t row = 0; row < stacked_rows; ++row) {
            OutputValue *accumulator = stacked_values.data() + row * kArrayDim;
            for (size_t col = 0; col < valid_cols; ++col) {
              if (block_scales[col] >
                      static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max()) ||
                  !checked_mul_i64(accumulator[col],
                                   static_cast<int64_t>(block_scales[col]),
                                   accumulator[col])) {
                return RmdStatus::overflow;
              }
            }
          }

        } // legacy block scale only

#if defined(GGML_GEMMINI_TESTING)
        if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
          observation.scaled_value = stacked_values.front();
          observation.compressed_value = observation.scaled_value;
          const BalancedRadixContract radix =
              balanced_radix_contract(packet.digit_bits);
          int64_t place = 1;
          for (uint8_t lane = 0; lane < observation.lane_id; ++lane) {
            if (!checked_mul_i64(place, static_cast<int64_t>(radix.radix),
                                 place)) {
              return RmdStatus::overflow;
            }
          }
          if (!checked_mul_i64(observation.scaled_value, place,
                               observation.composed_value)) {
            return RmdStatus::overflow;
          }
          if (metrics != nullptr) {
            staged_metrics.ws_observations.push_back(observation);
          }
        }
#endif
#if CYCLE_SIM
        im2p::gemmini::cycle_sim::StageCall recompose_call(
            event_context, cycle_sim::CallKind::ResidualRecompose, required_begin);
        im2p::gemmini::cycle_sim::HostStageScope recomposition(event_context,
            std::is_same_v<Output, Correction> ? "im2p.residual_radix_recomposition"
                                             : "im2p.residual_output_assembly",
            "POTAL_HOST", "llama.cpp-gemmini",
            "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_impl",
            args.matmul_layer.c_str(), im2p::gemmini::cycle_sim::work_ids_since(required_begin), {}, true);
#endif
        for (size_t group_lane = 0; group_lane < group.lane_positions.size();
             ++group_lane) {
          const uint8_t lane_position = group.lane_positions[group_lane];
          // Input lanes touch; only the group tail has physical row padding.
          // The assembler still writes each logical lane's padded output
          // stride.
          const size_t lane_row_base = group_lane * packet.row_count;
          for (size_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
            const size_t row_base = m_tile * kArrayDim;
            const size_t valid_rows =
                std::min(kArrayDim, packet.row_count - row_base);
            PhysicalTile tile{};
            tile.packet_block_index = static_cast<uint32_t>(block_index);
            tile.lane_position = lane_position;
            tile.lane_id = block.lane_ids[lane_position];
            tile.m_tile = static_cast<uint32_t>(m_tile);
            tile.j_tile = static_cast<uint32_t>(j_tile);
            tile.valid_rows = static_cast<uint16_t>(valid_rows);
            tile.valid_cols = static_cast<uint16_t>(valid_cols);
            tile.values =
                stacked_values.data() + (lane_row_base + row_base) * kArrayDim;
            const RmdStatus submit_status = assembler.submit(tile);
            if (submit_status != RmdStatus::success) {
              return submit_status;
            }
          }
        }
#if CYCLE_SIM
        recomposition.finish();
        recompose_call.finish();
#endif
      }
    }
  }

  const RmdStatus finish_status = assembler.finish();
  if (finish_status != RmdStatus::success) {
    return finish_status;
  }
  if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
    if (args.optrace_context) {
      // Independently accumulated execution metric, not a writer callback
      // count. A failed packet cannot finalize as a successful traced run.
      args.optrace_context->session->independent_count(
          *args.optrace_context, args.matmul_layer, "residual",
          staged_metrics.im2p_dot_calls);
    }
  }
#if CYCLE_SIM
  if (event_context) {
    try { event_context.session->ensure_healthy(); }
    catch (...) { return RmdStatus::execution_failed; }
  }
#endif
  output = std::move(staged_output);
  if (metrics != nullptr) {
    collect_packet_metrics(packet, staged_metrics);
    if constexpr (std::is_same_v<Output, Correction>) {
      staged_metrics.compressed_output_values = 0;
    }
    staged_metrics.matmul_call_count = matmul_call_count;
    staged_metrics.lane_group_count = lane_group_count;
    staged_metrics.stacked_i_tile_count = stacked_i_tile_count;
    staged_metrics.weight_values_gathered = weight_values_gathered;
    staged_metrics.weight_baseline_address_resolutions =
        weight_baseline_address_resolutions;
    staged_metrics.weight_address_resolutions = weight_address_resolutions;
    staged_metrics.packet_call_count = 1;
    if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
      staged_metrics.ws_call_count = matmul_call_count;
    } else if constexpr (Backend == CompactExecutorBackend::im2p_sim) {
      staged_metrics.im2p_stats = im2p_stats.stats;
    }
    *metrics = std::move(staged_metrics);
  }
  return RmdStatus::success;
}

template <typename Output>
static RmdStatus execute_rmd_stripe_im2p_output(
    im2p_sim_t *sim, const ggml_gemmini_args_t &args,
    const StripePacket &packet, Output &output, RmdExecutionMetrics *metrics,
    const wroute::WeightRoutePlan *shared_plan = nullptr,
    const Im2pFullExecutor *executor = nullptr) {
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                       \
    !defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
  (void)sim;
  (void)args;
  (void)packet;
  (void)output;
  (void)metrics;
  (void)shared_plan;
  (void)executor;
  return RmdStatus::unsupported_route;
#else
  if (executor != nullptr ? (executor->execute == nullptr &&
                             executor->execute_planned == nullptr)
                          : sim == nullptr) {
    return RmdStatus::invalid_arguments;
  }
  const wroute::WeightRoutePlan plan =
      shared_plan != nullptr
          ? *shared_plan
          : wroute::resolve_weight_route_plan(
                args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  const RmdStatus plan_status = compact_plan_status(args, plan);
  if (plan_status != RmdStatus::success) {
    return plan_status;
  }
  const RmdStatus validation = validate_execution_request(args, packet);
  if (validation != RmdStatus::success) {
    return validation;
  }
  if ((plan.route != wroute::WeightRouteKind::H1 &&
       plan.route != wroute::WeightRouteKind::HP1 &&
       plan.route != wroute::WeightRouteKind::Q8ChannelDirect &&
       plan.route != wroute::WeightRouteKind::Q8ChannelSidecar) ||
      packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
      plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
      packet.digit_bits != plan.weight_bits) {
    return RmdStatus::unsupported_route;
  }
  return execute_rmd_stripe_impl<CompactExecutorBackend::im2p_sim>(
      args, packet, plan, output, metrics, sim, Im2pProviderTestFault::none,
      executor);
#endif
}

template <typename Execute>
RmdStatus
execute_block_correction(const ggml_gemmini_args_t &args,
                         const StripePacket &packet, Correction &output,
                         RmdExecutionMetrics *metrics, Execute execute,
                         [[maybe_unused]] bool npu_residual = false) {
#if CYCLE_SIM
  const auto event_context = npu_residual ? args.cycle_sim_context : cycle_sim::Context{};
  std::vector<uint64_t> packet_work_ids;
  im2p::gemmini::cycle_sim::WorkCollector packet_collector(packet_work_ids, true);
  const size_t required_begin = im2p::gemmini::cycle_sim::work_count();
#endif
  CompressedOutput compressed;
  RmdExecutionMetrics staged_metrics;
  RmdExecutionMetrics *const staged =
      metrics != nullptr ? &staged_metrics : nullptr;
  const RmdStatus status = execute(compressed, staged);
  if (status != RmdStatus::success)
    return status;
  Correction staged_output = BlockScaledInt64Correction{};
#if CYCLE_SIM
  im2p::gemmini::cycle_sim::StageCall recompose_call(
      event_context, cycle_sim::CallKind::ResidualRecompose, required_begin);
  im2p::gemmini::cycle_sim::HostStageScope recomposition(event_context,
      "im2p.residual_block_recomposition", "POTAL_HOST", "llama.cpp-gemmini",
      "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_block_correction",
      args.matmul_layer.c_str(), im2p::gemmini::cycle_sim::work_ids_since(required_begin), {}, true);
#endif
  const RmdStatus compose =
      compose_block_rmd_output(args, packet, compressed, staged_output);
#if CYCLE_SIM
  recomposition.finish(compose == RmdStatus::success);
#endif
  if (compose != RmdStatus::success)
    return compose;
#if CYCLE_SIM
  recompose_call.finish();
  if (event_context) {
    try { event_context.session->ensure_healthy(); }
    catch (...) { return RmdStatus::execution_failed; }
  }
#endif
  output.swap(staged_output);
  if (metrics != nullptr) {
    staged_metrics.compressed_output_values = 0;
    *metrics = std::move(staged_metrics);
  }
  return RmdStatus::success;
}

RmdStatus execute_rmd_stripe_im2p(im2p_sim_t *sim,
                                  const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  CompressedOutput &output,
                                  RmdExecutionMetrics *metrics,
                                  const Im2pFullExecutor *executor) {
  return execute_rmd_stripe_im2p_output(sim, args, packet, output, metrics,
                                        nullptr, executor);
}

RmdStatus execute_rmd_stripe_im2p(im2p_sim_t *sim,
                                  const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  Correction &output,
                                  RmdExecutionMetrics *metrics,
                                  const Im2pFullExecutor *executor) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, output, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_im2p_output(sim, args, packet, compressed,
                                                staged, nullptr, executor);
        }, true);
  }
  return execute_rmd_stripe_im2p_output(sim, args, packet, output, metrics,
                                        nullptr, executor);
}

template <typename Output>
static RmdStatus execute_rmd_stripe_ws_output(
    const ggml_gemmini_args_t &args, const StripePacket &packet, Output &output,
    RmdExecutionMetrics *metrics,
    const wroute::WeightRoutePlan *shared_plan = nullptr) {
  const wroute::WeightRoutePlan plan =
      shared_plan != nullptr
          ? *shared_plan
          : wroute::resolve_weight_route_plan(
                args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  const RmdStatus plan_status = compact_plan_status(args, plan);
  if (plan_status != RmdStatus::success) {
    return plan_status;
  }
  const RmdStatus validation = validate_execution_request(args, packet);
  if (validation != RmdStatus::success) {
    return validation;
  }

#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                       \
    !defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART) && !defined(__riscv)
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return RmdStatus::unsupported_route;
  }
#endif

  if (plan.route == wroute::WeightRouteKind::H1 ||
      plan.route == wroute::WeightRouteKind::HP1) {
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    im2p_sim_t *sim = im2p_sim_create();
    if (sim == nullptr) {
      return RmdStatus::allocation_failure;
    }
    const RmdStatus status =
        execute_rmd_stripe_impl<CompactExecutorBackend::im2p_sim>(
            args, packet, plan, output, metrics, sim);
    im2p_sim_destroy(sim);
    return status;
#elif defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
    return RmdStatus::unsupported_route;
#else
    if (plan.hp1_carriers)
      return RmdStatus::unsupported_route;
    return execute_rmd_stripe_impl<CompactExecutorBackend::checked_software>(
        args, packet, plan, output, metrics);
#endif
  }

#if !defined(__riscv)
  (void)output;
  (void)metrics;
  return RmdStatus::unsupported_route;
#else
  return execute_rmd_stripe_impl<CompactExecutorBackend::gemmini_ws>(
      args, packet, plan, output, metrics);
#endif
}

RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t &args,
                                const StripePacket &packet,
                                CompressedOutput &output,
                                RmdExecutionMetrics *metrics) {
  return execute_rmd_stripe_ws_output(args, packet, output, metrics);
}

RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t &args,
                                const StripePacket &packet, Correction &output,
                                RmdExecutionMetrics *metrics) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, output, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_ws_output(args, packet, compressed, staged);
        });
  }
  return execute_rmd_stripe_ws_output(args, packet, output, metrics);
}

namespace detail {
RmdStatus execute_rmd_stripe_ws_with_weights(const ggml_gemmini_args_t &args,
                                             const StripePacket &packet,
                                             Correction &correction,
                                             RmdWeightPreparation &weights,
                                             RmdExecutionMetrics *metrics) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, correction, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_ws_output(args, packet, compressed, staged,
                                              &weights.route_plan(args));
        });
  }
  return execute_rmd_stripe_ws_output(args, packet, correction, metrics,
                                      &weights.route_plan(args));
}

RmdStatus execute_rmd_stripe_im2p_with_weights(im2p_sim_t *sim,
                                               const ggml_gemmini_args_t &args,
                                               const StripePacket &packet,
                                               Correction &correction,
                                               RmdWeightPreparation &weights,
                                               RmdExecutionMetrics *metrics) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, correction, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_im2p_output(
              sim, args, packet, compressed, staged, &weights.route_plan(args));
        }, true);
  }
  return execute_rmd_stripe_im2p_output(sim, args, packet, correction, metrics,
                                        &weights.route_plan(args));
}
} // namespace detail

#if defined(GGML_GEMMINI_TESTING)
template <typename Output>
static RmdStatus
execute_rmd_stripe_reference_output(const ggml_gemmini_args_t &args,
                                    const StripePacket &packet, Output &output,
                                    RmdExecutionMetrics *metrics) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  const RmdStatus plan_status = compact_plan_status(args, plan);
  if (plan_status != RmdStatus::success) {
    return plan_status;
  }
  const RmdStatus validation = validate_execution_request(args, packet);
  if (validation != RmdStatus::success) {
    return validation;
  }
  return execute_rmd_stripe_impl<CompactExecutorBackend::checked_software>(
      args, packet, plan, output, metrics);
}

RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t &args,
                                       const StripePacket &packet,
                                       CompressedOutput &output,
                                       RmdExecutionMetrics *metrics) {
  return execute_rmd_stripe_reference_output(args, packet, output, metrics);
}

RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t &args,
                                       const StripePacket &packet,
                                       Correction &output,
                                       RmdExecutionMetrics *metrics) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, output, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_reference_output(args, packet, compressed,
                                                     staged);
        });
  }
  return execute_rmd_stripe_reference_output(args, packet, output, metrics);
}

template <typename Output>
static RmdStatus execute_rmd_stripe_im2p_for_test_output(
    im2p_sim_t *sim, const ggml_gemmini_args_t &args,
    const StripePacket &packet, Output &output, RmdExecutionMetrics *metrics,
    Im2pProviderTestFault fault) {
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
  (void)sim;
  (void)args;
  (void)packet;
  (void)output;
  (void)metrics;
  (void)fault;
  return RmdStatus::unsupported_route;
#else
  if (sim == nullptr)
    return RmdStatus::invalid_arguments;
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
  const RmdStatus plan_status = compact_plan_status(args, plan);
  if (plan_status != RmdStatus::success)
    return plan_status;
  const RmdStatus validation = validate_execution_request(args, packet);
  if (validation != RmdStatus::success)
    return validation;
  if ((plan.route != wroute::WeightRouteKind::H1 &&
       plan.route != wroute::WeightRouteKind::HP1 &&
       plan.route != wroute::WeightRouteKind::Q8ChannelDirect &&
       plan.route != wroute::WeightRouteKind::Q8ChannelSidecar) ||
      packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
      plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
      packet.digit_bits != plan.weight_bits) {
    return RmdStatus::unsupported_route;
  }
  return execute_rmd_stripe_impl<CompactExecutorBackend::im2p_sim>(
      args, packet, plan, output, metrics, sim, fault);
#endif
}

RmdStatus execute_rmd_stripe_im2p_for_test(im2p_sim_t *sim,
                                           const ggml_gemmini_args_t &args,
                                           const StripePacket &packet,
                                           CompressedOutput &output,
                                           RmdExecutionMetrics *metrics,
                                           Im2pProviderTestFault fault) {
  return execute_rmd_stripe_im2p_for_test_output(sim, args, packet, output,
                                                 metrics, fault);
}

RmdStatus execute_rmd_stripe_im2p_for_test(im2p_sim_t *sim,
                                           const ggml_gemmini_args_t &args,
                                           const StripePacket &packet,
                                           Correction &output,
                                           RmdExecutionMetrics *metrics,
                                           Im2pProviderTestFault fault) {
  if (std::holds_alternative<quants::act::block::Meta>(
          args.act_quant.storage())) {
    return execute_block_correction(
        args, packet, output, metrics,
        [&](CompressedOutput &compressed, RmdExecutionMetrics *staged) {
          return execute_rmd_stripe_im2p_for_test_output(
              sim, args, packet, compressed, staged, fault);
        });
  }
  return execute_rmd_stripe_im2p_for_test_output(sim, args, packet, output,
                                                 metrics, fault);
}

RmdStatus execute_rmd_stripe_gemmini_for_test(const ggml_gemmini_args_t &args,
                                              const StripePacket &packet,
                                              CompressedOutput &output,
                                              RmdExecutionMetrics *metrics) {
  const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
      args, wroute::WeightScaleInfoMode::Residual);
  if (packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
      plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
      packet.digit_bits != plan.weight_bits) {
    return RmdStatus::unsupported_route;
  }
  const RmdStatus plan_status = compact_plan_status(args, plan);
  if (plan_status != RmdStatus::success) {
    return plan_status;
  }
  const RmdStatus validation = validate_execution_request(args, packet);
  if (validation != RmdStatus::success) {
    return validation;
  }
  return execute_rmd_stripe_impl<CompactExecutorBackend::gemmini_ws>(
      args, packet, plan, output, metrics);
}
#endif

} // namespace ggml::gemmini::rmd
