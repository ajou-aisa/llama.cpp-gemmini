#pragma once

// Balanced Radix-256 Residual Matrix Decomposition (RMD).
//
// A residual value r is decomposed into at most four signed INT8 digits:
//
//     r = d0 + 2^8 * d1 + 2^16 * d2 + 2^24 * d3,   d_l in [-128, 127]
//
// Each digit index l is a "lane". Lane l contributes 256^l to the reconstruction.
// The digits of one stripe are packed per original weight block so that the NPU can
// gather weight rows with block-local K indices and apply the block's integer scale
// exactly once.

#include "../../ggml-gemmini-config.hpp"

#include <gemmini_params.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

namespace ggml::gemmini::rmd {

constexpr size_t kArrayDim = DIM;
constexpr size_t kNativeWeightScaleGroup = 32;
constexpr size_t kBlockSize = kNativeWeightScaleGroup;
constexpr size_t kMaxLanes = 4;

static_assert(kArrayDim > 0, "Gemmini DIM must be positive");
static_assert(kBlockSize > 0, "native Q8 weight group must be positive");
static_assert(kBlockSize % kArrayDim == 0 || kArrayDim % kBlockSize == 0,
              "Gemmini DIM and native Q8 weight group must divide one another");

constexpr uint32_t kPacketVersion = 1;

using OutputValue = int64_t;

// CPU-direct H1/HP1 corrections have consumed the integer block factor but not
// the per-column floating factor. H0 corrections have consumed their arbitrary
// floating block factors in double precision. The variant keeps those domains
// distinct until the common composition path handles them.
struct BlockScaledInt64Correction {
    std::vector<OutputValue> values;
};

struct PreScaledFloat64Correction {
    std::vector<double> values;
};

// The correction remains tagged through execution, radix composition, and the
// final destination merge. Integer-block-scaled and pre-scaled floating values
// are intentionally not implicitly convertible to one another.
using Correction = std::variant<BlockScaledInt64Correction, PreScaledFloat64Correction>;
using DirectOutput = Correction;

inline size_t correction_size(const Correction & correction) {
    return std::visit([](const auto & typed) { return typed.values.size(); }, correction);
}

inline bool correction_empty(const Correction & correction) {
    return correction_size(correction) == 0;
}

inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value == 0)
        return value;
    const size_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    const size_t padding = alignment - remainder;
    return value > std::numeric_limits<size_t>::max() - padding ? 0 : value + padding;
}

// One original weight block that carries at least one residual digit in this stripe.
struct BlockDescriptor {
    uint32_t block_id = 0;        // original weight block index (K / kBlockSize)
    uint32_t global_k_begin = 0;  // block_id * kBlockSize

    uint16_t compact_k_count = 0; // selected K within this block
    uint16_t padded_k_count = 0;  // compact_k_count aligned to kArrayDim

    uint8_t active_lane_mask = 0; // bit l set when lane l carries a nonzero digit
    uint8_t active_lane_count = 0;
    std::array<uint8_t, kMaxLanes> lane_ids{}; // lane position -> actual lane id

    uint32_t k_index_offset = 0;    // into StripePacket::k_indices (block-local K)
    uint32_t activation_offset = 0; // into StripePacket::stacked_activation

    uint32_t output_value_offset = 0; // into CompressedOutput::values
    uint16_t rows_padded = 0;         // row_count aligned to kArrayDim
    uint32_t lane_stride_values = 0;  // rows_padded * j_padded
};

// Immutable, self-contained description of one stripe's residual work.
// The packet never borrows ExSIA slot memory: it owns every buffer it exposes.
struct StripePacket {
    uint32_t version = kPacketVersion;

    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_count = 0;
    size_t logical_k = 0;
    size_t logical_j = 0;
    size_t j_padded = 0;

    size_t block_size = kBlockSize;
    size_t array_dim = kArrayDim;

    std::vector<BlockDescriptor> blocks;
    std::vector<uint16_t> k_indices;       // block-local K, ascending inside each block
    std::vector<int8_t> stacked_activation; // block / lane position / padded row / padded K

    size_t total_output_values = 0;
};

using StripePacketHandle = std::shared_ptr<const StripePacket>;

struct CompressedOutput {
    enum class Domain : uint8_t {
        block_scaled_int64,
    };

    Domain domain = Domain::block_scaled_int64;
    size_t j_padded = 0;
    std::vector<OutputValue> values;
};

enum class RmdStatus : uint8_t {
    success,
    invalid_arguments,
    invalid_packet,
    residual_too_wide,   // q4 != 0: the value does not fit four balanced digits
    unsupported_route,   // weight route has no integer block scale
    overflow,
    allocation_failure,
    execution_failed,
};

const char * rmd_status_message(RmdStatus status);

// Balanced radix-256 digits of one residual. `lanes` is always kMaxLanes wide;
// unused high lanes are zero.
struct BalancedDigits {
    std::array<int8_t, kMaxLanes> digits{};
    uint8_t lane_mask = 0;
};

// Returns false when the value needs a fifth digit (explicit failure, never a
// silent truncation).
bool decompose_balanced_radix256(int32_t residual, BalancedDigits & out);

// Reconstructs a residual from its digits; used by tests and the reference path.
int64_t compose_balanced_radix256(const BalancedDigits & digits);

}
